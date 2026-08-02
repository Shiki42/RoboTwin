from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import os
import pathlib
import random
import shutil
from typing import Any

import numpy as np
import safetensors.torch
import torch
from torch import nn


def move_to_device(
    value: Any,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> Any:
    """Recursively move an OpenPI observation tree to a torch device."""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value).to(device, non_blocking=non_blocking)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: move_to_device(getattr(value, field.name), device, non_blocking=non_blocking)
            for field in dataclasses.fields(value)
        }
        return dataclasses.replace(value, **updates)
    if isinstance(value, Mapping):
        return type(value)(
            (key, move_to_device(item, device, non_blocking=non_blocking)) for key, item in value.items()
        )
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device, non_blocking=non_blocking) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device, non_blocking=non_blocking) for item in value]
    return value


def learning_rate(
    step: int,
    *,
    warmup_steps: int,
    peak_lr: float,
    decay_steps: int,
    end_lr: float,
) -> float:
    if step < 0:
        raise ValueError("step must be non-negative")
    if warmup_steps < 0 or decay_steps < warmup_steps:
        raise ValueError("expected 0 <= warmup_steps <= decay_steps")
    if warmup_steps > 0 and step < warmup_steps:
        initial_lr = peak_lr / (warmup_steps + 1)
        return initial_lr + (peak_lr - initial_lr) * step / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
    cosine = 0.5 * (1 + np.cos(np.pi * progress))
    return float(end_lr + (peak_lr - end_lr) * cosine)


def seed_everything(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def load_pretrained(
    model: nn.Module,
    checkpoint: pathlib.Path,
    *,
    allowed_missing_prefixes: Sequence[str] = (),
) -> tuple[list[str], list[str]]:
    checkpoint = pathlib.Path(checkpoint)
    if checkpoint.name != "model.safetensors":
        checkpoint = checkpoint / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    missing, unexpected = safetensors.torch.load_model(
        model,
        checkpoint,
        strict=False,
        device="cpu",
    )
    disallowed_missing = [
        key for key in missing if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if disallowed_missing or unexpected:
        raise ValueError(f"pretrained state mismatch: missing={disallowed_missing}, unexpected={unexpected}")
    return missing, unexpected


def _checkpoint_model(model: nn.Module) -> nn.Module:
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def initialize_ema(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().to(dtype=torch.float32).clone()
        for name, parameter in _checkpoint_model(model).named_parameters()
    }


@torch.no_grad()
def update_ema(
    ema_state: Mapping[str, torch.Tensor],
    model: nn.Module,
    decay: float,
) -> None:
    if not 0.0 < decay < 1.0:
        raise ValueError("EMA decay must be between zero and one")
    parameters = dict(_checkpoint_model(model).named_parameters())
    if parameters.keys() != ema_state.keys():
        raise ValueError("EMA state does not match model parameters")
    for name, parameter in parameters.items():
        ema_state[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)


def _load_model_file(model: nn.Module, path: pathlib.Path, device: torch.device) -> None:
    missing, unexpected = safetensors.torch.load_model(model, path, strict=True, device=str(device))
    if missing or unexpected:
        raise ValueError(f"checkpoint state mismatch: missing={missing}, unexpected={unexpected}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    checkpoint_root: pathlib.Path,
    metadata: Mapping[str, Any],
    extra_files: Mapping[pathlib.Path, str | bytes] | None = None,
    ema_state: Mapping[str, torch.Tensor] | None = None,
) -> pathlib.Path:
    if global_step < 1:
        raise ValueError("global_step must be positive")
    checkpoint_root = pathlib.Path(checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    final_dir = checkpoint_root / str(global_step)
    temporary_dir = checkpoint_root / f".tmp-{global_step}-{os.getpid()}"
    if final_dir.exists():
        raise FileExistsError(final_dir)
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir()

    try:
        checkpoint_model = _checkpoint_model(model)
        model_path = temporary_dir / "model.safetensors"
        if ema_state is None:
            safetensors.torch.save_model(checkpoint_model, model_path)
        else:
            parameters = dict(checkpoint_model.named_parameters())
            if parameters.keys() != ema_state.keys():
                raise ValueError("EMA state does not match model parameters")
            training_model_path = temporary_dir / "training_model.safetensors"
            safetensors.torch.save_model(checkpoint_model, training_model_path)
            try:
                with torch.no_grad():
                    for name, parameter in parameters.items():
                        parameter.copy_(ema_state[name])
                safetensors.torch.save_model(checkpoint_model, model_path)
            finally:
                device = next(checkpoint_model.parameters()).device
                _load_model_file(checkpoint_model, training_model_path, device)
            safetensors.torch.save_file(
                dict(ema_state),
                temporary_dir / "ema_state.safetensors",
            )
        torch.save(
            {
                "global_step": global_step,
                "optimizer": optimizer.state_dict(),
                "rng": capture_rng_state(),
                "metadata": dict(metadata),
            },
            temporary_dir / "training_state.pt",
        )
        for path, contents in (extra_files or {}).items():
            relative = pathlib.Path(path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"checkpoint extra file must be relative: {relative}")
            output_path = temporary_dir / relative
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(contents if isinstance(contents, bytes) else contents.encode())
        temporary_dir.rename(final_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
    return final_dir


def prune_checkpoints(checkpoint_root: pathlib.Path, *, current_step: int, keep_period: int | None) -> None:
    if current_step < 1:
        raise ValueError("current_step must be positive")
    if keep_period is not None and keep_period < 1:
        raise ValueError("keep_period must be positive")
    checkpoint_root = pathlib.Path(checkpoint_root)
    for checkpoint in checkpoint_root.iterdir():
        if not checkpoint.is_dir() or not checkpoint.name.isdigit():
            continue
        step = int(checkpoint.name)
        if step == current_step:
            continue
        if keep_period is not None and step % keep_period == 0:
            continue
        shutil.rmtree(checkpoint)


def latest_checkpoint(checkpoint_root: pathlib.Path) -> pathlib.Path:
    checkpoint_root = pathlib.Path(checkpoint_root)
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(checkpoint_root)
    steps = [int(path.name) for path in checkpoint_root.iterdir() if path.is_dir() and path.name.isdigit()]
    if not steps:
        raise FileNotFoundError(f"no checkpoints under {checkpoint_root}")
    return checkpoint_root / str(max(steps))


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint: pathlib.Path,
    *,
    device: torch.device,
    load_ema: bool = False,
) -> tuple[int, dict[str, Any], dict[str, torch.Tensor] | None]:
    checkpoint = pathlib.Path(checkpoint)
    if checkpoint.name.isdigit() is False:
        checkpoint = latest_checkpoint(checkpoint)
    model_path = checkpoint / ("training_model.safetensors" if load_ema else "model.safetensors")
    state_path = checkpoint / "training_state.pt"
    if not model_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(f"incomplete checkpoint: {checkpoint}")
    _load_model_file(_checkpoint_model(model), model_path, device)
    state = torch.load(state_path, map_location=device, weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    restore_rng_state(state["rng"])
    ema_state = None
    if load_ema:
        ema_path = checkpoint / "ema_state.safetensors"
        if not ema_path.is_file():
            raise FileNotFoundError(f"incomplete EMA checkpoint: {checkpoint}")
        ema_state = safetensors.torch.load_file(ema_path, device=str(device))
        if ema_state.keys() != dict(_checkpoint_model(model).named_parameters()).keys():
            raise ValueError("EMA checkpoint does not match model parameters")
    return int(state["global_step"]), dict(state["metadata"]), ema_state
