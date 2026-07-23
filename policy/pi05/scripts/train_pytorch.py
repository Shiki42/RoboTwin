"""Single-GPU, full-state PyTorch trainer for matched PI0.5/CASM runs."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pathlib
import re
import shutil
import time
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import torch
import wandb

from openpi.models import pi0_config
from openpi.models_pytorch import pi0_pytorch
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
from openpi.training import data_loader as _data
from openpi.training import pytorch_training

logger = logging.getLogger(__name__)


def _required_code_commit() -> str:
    code_commit = os.environ.get("PARALLELVLA_CODE_COMMIT", "")
    if re.fullmatch(r"[0-9a-f]{40}", code_commit) is None:
        raise ValueError("PARALLELVLA_CODE_COMMIT must be a full lowercase Git commit SHA")
    return code_commit


def _config_signature(config: _config.TrainConfig) -> dict[str, Any]:
    model = config.model
    return {
        "config_name": config.name,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "seed": config.seed,
        "model": {
            "pi05": getattr(model, "pi05", False),
            "casm_mode": getattr(model, "casm_mode", "none"),
            "paligemma_variant": getattr(model, "paligemma_variant", None),
            "action_expert_variant": getattr(model, "action_expert_variant", None),
            "action_dim": model.action_dim,
            "action_horizon": model.action_horizon,
        },
        "dataset_repo": config.data.repo_id,
        "dataset_revision": os.environ.get("PARALLELVLA_DATASET_REVISION"),
        "base_sha256": os.environ.get("PI05_BASE_SHA256"),
        "code_commit": _required_code_commit(),
    }


def _prepare_checkpoint_root(config: _config.TrainConfig) -> pathlib.Path:
    root = config.checkpoint_dir
    if config.resume:
        if not root.is_dir():
            raise FileNotFoundError(root)
        return root
    if root.exists():
        if not config.overwrite:
            raise FileExistsError(root)
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def _build_model(config: _config.TrainConfig, device: torch.device) -> pi0_pytorch.PI0Pytorch:
    if not isinstance(config.model, pi0_config.Pi0Config):
        raise TypeError("PyTorch trainer requires Pi0Config")
    if "lora" in config.model.paligemma_variant or "lora" in config.model.action_expert_variant:
        raise ValueError("PyTorch trainer requires full PI0.5 variants; LoRA is not implemented")
    model_config = dataclasses.replace(config.model, dtype=config.pytorch_training_precision)
    model = pi0_pytorch.PI0Pytorch(model_config).to(device)
    model.gradient_checkpointing_enable()
    return model


def _build_optimizer(config: _config.TrainConfig, model: torch.nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.lr_schedule.peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )


def _require_run_receipts(config: _config.TrainConfig) -> pathlib.Path:
    if config.pytorch_weight_path is None:
        raise ValueError("pytorch_weight_path is required")
    base = pathlib.Path(config.pytorch_weight_path)
    if not (base / "model.safetensors").is_file():
        raise FileNotFoundError(base / "model.safetensors")
    required_environment = (
        "PI05_BASE_SHA256",
        "PARALLELVLA_DATASET_REVISION",
        "PARALLELVLA_DATASET_RECEIPT",
    )
    missing = [name for name in required_environment if not os.environ.get(name)]
    if missing:
        raise ValueError(f"missing run receipt environment: {missing}")
    dataset_receipt = pathlib.Path(os.environ["PARALLELVLA_DATASET_RECEIPT"])
    if not dataset_receipt.is_file():
        raise FileNotFoundError(dataset_receipt)
    return base


def _initialize_pretrained(model: pi0_pytorch.PI0Pytorch, base: pathlib.Path) -> None:
    allowed = ("phase_gate.",) if model.casm_mode == "visual_phase_gate" else ()
    missing, unexpected = pytorch_training.load_pretrained(
        model,
        base,
        allowed_missing_prefixes=allowed,
    )
    logger.info("loaded base weights; missing=%s unexpected=%s", missing, unexpected)


def _checkpoint_files(data_config: _config.DataConfig, manifest: dict[str, Any]) -> dict[pathlib.Path, str]:
    files = {pathlib.Path("run_manifest.json"): json.dumps(manifest, indent=2, sort_keys=True)}
    if data_config.norm_stats is None or data_config.asset_id is None:
        raise ValueError("normalization stats and asset_id are required for training")
    files[pathlib.Path("assets") / data_config.asset_id / "norm_stats.json"] = _normalize.serialize_json(
        data_config.norm_stats
    )
    return files


def _append_metrics(path: pathlib.Path, metrics: dict[str, float | int]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(metrics, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _train_step(
    model: pi0_pytorch.PI0Pytorch,
    optimizer: torch.optim.AdamW,
    observation,
    actions: torch.Tensor,
    config: _config.TrainConfig,
    global_step: int,
) -> dict[str, float]:
    lr = pytorch_training.learning_rate(
        global_step,
        warmup_steps=config.lr_schedule.warmup_steps,
        peak_lr=config.lr_schedule.peak_lr,
        decay_steps=config.lr_schedule.decay_steps,
        end_lr=config.lr_schedule.decay_lr,
    )
    for group in optimizer.param_groups:
        group["lr"] = lr

    if model.casm_mode == "visual_phase_gate":
        losses, auxiliary = model(observation, actions, return_aux=True)
    else:
        losses = model(observation, actions)
        auxiliary = {}
    loss = losses.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite loss at step {global_step}: {loss}")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        config.optimizer.clip_gradient_norm,
        error_if_nonfinite=True,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    metrics = {
        "loss": float(loss.detach().cpu()),
        "learning_rate": lr,
        "gradient_norm": float(gradient_norm.detach().cpu()),
    }
    metrics.update({name: float(value.detach().cpu()) for name, value in auxiliary.items()})
    return metrics


def _init_wandb(config: _config.TrainConfig, manifest: dict[str, Any], run_id: str | None):
    if not config.wandb_enabled:
        return None
    if wandb.api.api_key is None:
        raise RuntimeError("W&B is enabled but no login credential is available")
    return wandb.init(
        project=config.project_name,
        name=config.exp_name,
        id=run_id,
        resume="must" if run_id else None,
        config=manifest,
        save_code=False,
        settings=wandb.Settings(disable_code=True),
    )


def train(config: _config.TrainConfig) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("this trainer intentionally supports one GPU only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training; use static preflight for CPU checks")
    if config.num_workers != 0:
        raise ValueError("full-state PyTorch training requires num_workers=0")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    pytorch_training.seed_everything(config.seed)
    checkpoint_root = _prepare_checkpoint_root(config)
    base = _require_run_receipts(config)
    manifest = _config_signature(config)
    loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    data_config = loader.data_config()
    extra_files = _checkpoint_files(data_config, manifest)
    model = _build_model(config, device)
    optimizer = _build_optimizer(config, model)

    global_step = 0
    resume_metadata: dict[str, Any] = {}
    if config.resume:
        global_step, resume_metadata = pytorch_training.load_checkpoint(
            model,
            optimizer,
            checkpoint_root,
            device=device,
        )
        if resume_metadata["config_signature"] != manifest:
            raise ValueError("resume config signature does not match checkpoint")
        loader.load_state_dict(resume_metadata["data_loader_state"])
    else:
        _initialize_pretrained(model, base)

    run = _init_wandb(config, manifest, resume_metadata.get("wandb_run_id"))
    metrics_path = checkpoint_root / "metrics.jsonl"
    iterator = iter(loader)
    model.train()
    last_log_time = time.monotonic()

    while global_step < config.num_train_steps:
        observation, actions = next(iterator)
        observation = pytorch_training.move_to_device(observation, device)
        actions = actions.to(device=device, dtype=torch.float32)
        metrics = _train_step(model, optimizer, observation, actions, config, global_step)
        global_step += 1
        metrics["step"] = global_step
        if global_step % config.log_interval == 0 or global_step == 1:
            torch.cuda.synchronize(device)
            now = time.monotonic()
            metrics["seconds_per_log_interval"] = now - last_log_time
            last_log_time = now
            _append_metrics(metrics_path, metrics)
            logger.info("step=%d loss=%.6f grad_norm=%.6f", global_step, metrics["loss"], metrics["gradient_norm"])
            if run is not None:
                run.log(metrics, step=global_step)

        should_save = global_step % config.save_interval == 0 or global_step == config.num_train_steps
        if should_save:
            metadata = {
                "config_signature": manifest,
                "data_loader_state": loader.state_dict(),
                "wandb_run_id": None if run is None else run.id,
            }
            pytorch_training.save_checkpoint(
                model,
                optimizer,
                global_step=global_step,
                checkpoint_root=checkpoint_root,
                metadata=metadata,
                extra_files=extra_files,
            )
            pytorch_training.prune_checkpoints(
                checkpoint_root,
                current_step=global_step,
                keep_period=config.keep_period,
            )

    if run is not None:
        run.finish()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(_config.cli())


if __name__ == "__main__":
    main()
