import asyncio
from pathlib import Path

from etils import epath
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
        assert handler._save_concurrent_bytes == 1_000_000_000  # noqa: SLF001
        assert handler._restore_concurrent_bytes == 1_000_000_000  # noqa: SLF001


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
