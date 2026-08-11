from __future__ import annotations

from pathlib import Path
from runpy import run_path

import pytest


resolve_checkpoint_path = run_path(
    str(
        Path(__file__).parents[1]
        / "policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/common/checkpoint_path.py"
    )
)["resolve_checkpoint_path"]


def test_resolve_checkpoint_path_preserves_absolute_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "3000.ckpt"
    checkpoint.write_bytes(b"checkpoint")

    assert resolve_checkpoint_path({"checkpoint_path": str(checkpoint)}) == checkpoint


@pytest.mark.parametrize(
    "arguments",
    (
        {},
        {"checkpoint_path": ""},
        {"checkpoint_path": Path("checkpoint.ckpt")},
        {"checkpoint_path": "checkpoint.ckpt"},
    ),
)
def test_resolve_checkpoint_path_rejects_missing_or_relative_path(arguments) -> None:
    with pytest.raises(ValueError):
        resolve_checkpoint_path(arguments)


def test_resolve_checkpoint_path_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint_path({"checkpoint_path": str(tmp_path / "missing.ckpt")})
