"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

from collections.abc import Sequence
import dataclasses

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


def keep_norm_field(key: str, value) -> bool:
    if key == "images" or key.startswith("observation.images"):
        return False
    return not np.issubdtype(np.asarray(value).dtype, np.str_)


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {key: value for key, value in x.items() if keep_norm_field(key, value)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    episodes: Sequence[int] | None = None,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config, episodes=episodes)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    single_epoch = max_frames is None or max_frames >= len(dataset)
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = (len(dataset) + batch_size - 1) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        single_epoch=single_epoch,
        drop_last=not single_epoch,
        num_batches=None if single_epoch else num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def exact_statistics(values: Sequence[np.ndarray]) -> normalize.NormStats:
    if not values:
        raise ValueError("cannot compute exact statistics without values")
    arrays = [np.asarray(value).reshape(-1, np.asarray(value).shape[-1]) for value in values]
    combined = np.concatenate(arrays, axis=0)
    if combined.shape[0] < 2:
        raise ValueError("cannot compute exact statistics from fewer than two vectors")
    q01, q99 = np.quantile(combined, (0.01, 0.99), axis=0)
    return normalize.NormStats(
        mean=combined.mean(axis=0, dtype=np.float64),
        std=combined.std(axis=0, dtype=np.float64),
        q01=q01,
        q99=q99,
    )


def valid_action_values(batch: dict) -> np.ndarray:
    actions = np.asarray(batch["actions"])
    action_mask = batch.get("action_mask")
    if action_mask is None:
        raise ValueError("exact action statistics require action_is_pad-derived action_mask")
    mask = np.asarray(action_mask, dtype=np.bool_)
    if mask.shape != actions.shape:
        raise ValueError(f"action mask shape mismatch: {mask.shape} != {actions.shape}")
    valid_steps = mask.any(axis=-1)
    if not np.all(mask[valid_steps]):
        raise ValueError("exact action statistics require the same valid dimensions at every supervised step")
    return actions[valid_steps]


def main(
    config_name: str,
    max_frames: int | None = None,
    repo_id: str | None = None,
    *,
    exact: bool = False,
):
    config = _config.get_config(config_name)
    if repo_id is not None:
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, repo_id=repo_id))
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            episodes=config.train_episodes,
            max_frames=max_frames,
        )

    keys = ("state", "actions")
    stats = {key: normalize.RunningStats() for key in keys}
    exact_values: dict[str, list[np.ndarray]] = {key: [] for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        if exact:
            exact_values["state"].append(np.asarray(batch["state"]))
            exact_values["actions"].append(valid_action_values(batch))
            continue
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    if exact:
        norm_stats = {key: exact_statistics(values) for key, values in exact_values.items()}
    else:
        norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
