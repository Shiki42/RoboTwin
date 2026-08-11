from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def resolve_checkpoint_path(arguments: Mapping[str, Any]) -> Path:
    raw_path = arguments.get("checkpoint_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("checkpoint_path must be a non-empty string")
    checkpoint = Path(raw_path)
    if not checkpoint.is_absolute():
        raise ValueError(f"checkpoint_path must be absolute: {checkpoint}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint file does not exist: {checkpoint}")
    return checkpoint
