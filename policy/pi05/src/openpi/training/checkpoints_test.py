import asyncio
from pathlib import Path

from etils import epath
import jax
import orbax.checkpoint as ocp
import pytest

from openpi.training import checkpoints


def test_checkpoint_item_handlers_match_storage_mode():
    assert set(checkpoints._checkpoint_item_handlers(params_only=True)) == {"assets", "params"}  # noqa: SLF001
    assert set(checkpoints._checkpoint_item_handlers(params_only=False)) == {  # noqa: SLF001
        "assets",
        "params",
        "train_state",
        "data_loader",
    }


def test_checkpoint_pytree_handlers_limit_host_transfer_concurrency():
    handlers = checkpoints._checkpoint_item_handlers(params_only=False)  # noqa: SLF001

    for name in ("params", "train_state", "data_loader"):
        handler = handlers[name]
        assert handler._save_concurrent_bytes == 6_000_000_000  # noqa: SLF001
        assert handler._restore_concurrent_bytes == 6_000_000_000  # noqa: SLF001


def test_checkpoint_pytree_handlers_use_sequential_array_transfers():
    handler = checkpoints._pytree_checkpoint_handler()  # noqa: SLF001
    array_handler = handler._type_handler_registry.get(jax.Array)  # noqa: SLF001

    assert isinstance(array_handler, checkpoints._SequentialArrayHandler)  # noqa: SLF001


def test_sequential_array_handler_waits_before_next_transfer(monkeypatch):
    events = []

    class CompletedFuture:
        def result(self):
            events.append("wait")

    async def fake_serialize(self, values, infos, args=None):
        del self, infos, args
        events.append(("serialize", values[0]))
        return [CompletedFuture()]

    monkeypatch.setattr(ocp.type_handlers.ArrayHandler, "serialize", fake_serialize)
    values = [object(), object()]
    result = asyncio.run(
        checkpoints._SequentialArrayHandler().serialize(values, [object(), object()], [object(), object()])  # noqa: SLF001
    )

    assert result == []
    assert events == [("serialize", values[0]), "wait", ("serialize", values[1]), "wait"]


def test_save_state_enables_pinned_host_transfer(monkeypatch):
    class CheckpointManager:
        saved = None

        def save(self, step, *, args):
            self.saved = (step, args)

    class DataLoader:
        def state_dict(self):
            return {"cursor": 32}

    monkeypatch.setattr(checkpoints, "_split_params", lambda state: ("train-state", "ema-params"))
    manager = CheckpointManager()
    checkpoints.save_state(manager, object(), DataLoader(), 1, params_only=False)

    step, args = manager.saved
    assert step == 1
    assert set(args.keys()) == {"assets", "train_state", "params", "data_loader"}
    assert isinstance(args["assets"], checkpoints.CallbackSave)
    assert args["train_state"].item == "train-state"
    assert args["params"].item == {"params": "ema-params"}
    assert args["data_loader"].item == {"cursor": 32}
    for name in ("train_state", "params", "data_loader"):
        assert args[name].enable_pinned_host_transfer is True


def test_params_only_checkpoint_rejects_resume(tmp_path: Path):
    with pytest.raises(ValueError, match="cannot resume"):
        checkpoints.initialize_checkpoint_dir(
            epath.Path(tmp_path),
            keep_period=None,
            overwrite=False,
            resume=True,
            params_only=True,
        )


def test_callback_handler_waits_for_save(tmp_path: Path):
    saved_paths = []
    args = checkpoints.CallbackSave(lambda path: saved_paths.append(Path(str(path))))
    result = asyncio.run(checkpoints.CallbackHandler().async_save(epath.Path(tmp_path), args))
    assert result == []
