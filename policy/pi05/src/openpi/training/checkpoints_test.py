from __future__ import annotations

import asyncio
from pathlib import Path

from etils import epath
import jax
import orbax.checkpoint as ocp

from openpi.training import checkpoints


def test_checkpoint_handlers_bound_host_transfer_memory() -> None:
    handler = checkpoints._pytree_checkpoint_handler()  # noqa: SLF001

    assert handler._save_concurrent_bytes == 3_000_000_000  # noqa: SLF001
    assert handler._restore_concurrent_bytes == 3_000_000_000  # noqa: SLF001
    array_handler = handler._type_handler_registry.get(jax.Array)  # noqa: SLF001
    assert isinstance(array_handler, checkpoints._SequentialArrayHandler)  # noqa: SLF001


def test_sequential_array_handler_waits_before_next_transfer(monkeypatch) -> None:
    events = []

    class CompletedFuture:
        def result(self) -> None:
            events.append("wait")

    async def fake_serialize(self, values, infos, args=None):
        del self, infos, args
        events.append(("serialize", values[0]))
        return [CompletedFuture()]

    monkeypatch.setattr(ocp.type_handlers.ArrayHandler, "serialize", fake_serialize)
    values = [object(), object()]
    result = asyncio.run(
        checkpoints._SequentialArrayHandler().serialize(  # noqa: SLF001
            values, [object(), object()], [object(), object()]
        )
    )

    assert result == []
    assert events == [
        ("serialize", values[0]),
        "wait",
        ("serialize", values[1]),
        "wait",
    ]


def test_save_state_uses_disposable_pinned_host_transfers(monkeypatch) -> None:
    class CheckpointManager:
        saved = None

        def save(self, step, *, args) -> None:
            self.saved = (step, args)

    monkeypatch.setattr(
        checkpoints, "_split_params", lambda state: ("train-state", "ema-params")
    )
    manager = CheckpointManager()
    checkpoints.save_state(manager, object(), object(), 1)

    step, args = manager.saved
    assert step == 1
    assert set(args.keys()) == {"assets", "train_state", "params"}
    assert isinstance(args["assets"], checkpoints.CallbackSave)
    assert args["train_state"].item == "train-state"
    assert args["params"].item == {"params": "ema-params"}
    assert args["train_state"].enable_pinned_host_transfer is True
    assert args["params"].enable_pinned_host_transfer is True


def test_callback_handler_completes_before_async_save_returns(tmp_path: Path) -> None:
    callback_paths: list[epath.Path] = []
    directory = epath.Path(tmp_path)
    args = checkpoints.CallbackSave(callback_paths.append)

    pending = asyncio.run(checkpoints.CallbackHandler().async_save(directory, args))

    assert pending == []
    assert callback_paths == [directory]
