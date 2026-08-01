from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
from typing import Protocol

from etils import epath
import jax
import numpy as np
import orbax.checkpoint as ocp

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils

_CHECKPOINT_CONCURRENT_GB = 4


class _SequentialArrayHandler(ocp.type_handlers.ArrayHandler):
    """Bound device-to-host checkpoint transfers to one array at a time."""

    async def serialize(self, values, infos, args=None):
        commit_futures = []
        for index, (value, info) in enumerate(zip(values, infos, strict=True)):
            value_args = None if args is None else [args[index]]
            futures = await super().serialize([value], [info], value_args)
            for commit_future in futures:
                await asyncio.to_thread(commit_future.result)
            commit_futures.extend(futures)
        return commit_futures


def _type_handler_registry() -> ocp.type_handlers.TypeHandlerRegistry:
    handlers = ocp.type_handlers
    return handlers.create_type_handler_registry(
        (int, handlers.ScalarHandler()),
        (float, handlers.ScalarHandler()),
        (bytes, handlers.ScalarHandler()),
        (np.number, handlers.ScalarHandler()),
        (np.ndarray, handlers.NumpyHandler()),
        (jax.Array, _SequentialArrayHandler()),
        (str, handlers.StringHandler()),
    )


def _pytree_checkpoint_handler() -> ocp.PyTreeCheckpointHandler:
    return ocp.PyTreeCheckpointHandler(
        save_concurrent_gb=_CHECKPOINT_CONCURRENT_GB,
        restore_concurrent_gb=_CHECKPOINT_CONCURRENT_GB,
        type_handler_registry=_type_handler_registry(),
    )


def _checkpoint_item_handlers(*, params_only: bool) -> dict:
    handlers = {
        "assets": CallbackHandler(),
        "params": _pytree_checkpoint_handler(),
    }
    if not params_only:
        handlers["train_state"] = _pytree_checkpoint_handler()
        handlers["data_loader"] = _pytree_checkpoint_handler()
    return handlers


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str,
    *,
    keep_period: int | None,
    overwrite: bool,
    resume: bool,
    params_only: bool,
) -> tuple[ocp.CheckpointManager, bool]:
    if params_only and resume:
        raise ValueError("params-only checkpoints cannot resume training")
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers=_checkpoint_item_handlers(params_only=params_only),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    *,
    params_only: bool,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    if params_only:
        items = {
            "assets": save_assets,
            "params": {"params": params},
        }
    else:
        items = {
            "assets": save_assets,
            "train_state": train_state,
            "params": {"params": params},
            "data_loader": data_loader.state_dict(),
        }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
                "data_loader": data_loader.state_dict(),
            },
        )
    data_loader.load_state_dict(restored["data_loader"])
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        await asyncio.to_thread(self.save, directory, args)
        return []

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
