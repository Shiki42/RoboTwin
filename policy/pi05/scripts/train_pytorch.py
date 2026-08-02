"""Single-GPU, full-state PyTorch trainer for matched PI0.5/CASM runs."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json
import logging
import os
import pathlib
import re
import shutil
import time
from typing import Any, Literal

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import torch
import wandb

from openpi.models import pi0_config
from openpi.models_pytorch import pi0_pytorch
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
from openpi.training import data_loader as _data
from openpi.training import performance as _performance
from openpi.training import pytorch_training

logger = logging.getLogger(__name__)


def _required_code_commit() -> str:
    code_commit = os.environ.get("PARALLELVLA_CODE_COMMIT", "")
    if re.fullmatch(r"[0-9a-f]{40}", code_commit) is None:
        raise ValueError("PARALLELVLA_CODE_COMMIT must be a full lowercase Git commit SHA")
    return code_commit


def config_signature(config: _config.TrainConfig) -> dict[str, Any]:
    model = config.model
    return {
        "config_name": config.name,
        "batch_size": config.batch_size * config.gradient_accumulation_steps,
        "microbatch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "pytorch_training_precision": config.pytorch_training_precision,
        "ema_decay": config.ema_decay,
        "num_workers": config.num_workers,
        "prefetch_factor": config.prefetch_factor,
        "persistent_workers": config.persistent_workers,
        "pin_memory": config.pin_memory,
        "pytorch_compile_mode": config.pytorch_compile_mode,
        "pytorch_attention_implementation": config.pytorch_attention_implementation,
        "pytorch_fused_optimizer": config.pytorch_fused_optimizer,
        "pytorch_gradient_checkpointing": config.pytorch_gradient_checkpointing,
        "pytorch_gradient_checkpointing_scope": config.pytorch_gradient_checkpointing_scope,
        "pytorch_trainable_scope": config.pytorch_trainable_scope,
        "seed": config.seed,
        "inactive_action_weight": config.data.inactive_action_weight,
        "lr_schedule": dataclasses.asdict(config.lr_schedule),
        "optimizer": dataclasses.asdict(config.optimizer),
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


def build_model(config: _config.TrainConfig, device: torch.device) -> pi0_pytorch.PI0Pytorch:
    if not isinstance(config.model, pi0_config.Pi0Config):
        raise TypeError("PyTorch trainer requires Pi0Config")
    if "lora" in config.model.paligemma_variant or "lora" in config.model.action_expert_variant:
        raise ValueError("PyTorch trainer requires full PI0.5 variants; LoRA is not implemented")
    model_config = dataclasses.replace(config.model, dtype=config.pytorch_training_precision)
    model = pi0_pytorch.PI0Pytorch(model_config).to(device)
    model.set_attention_implementation(config.pytorch_attention_implementation)
    if config.pytorch_gradient_checkpointing:
        model.gradient_checkpointing_enable(scope=config.pytorch_gradient_checkpointing_scope)
    else:
        model.gradient_checkpointing_disable()
    return model


def configure_trainable_parameters(
    model: torch.nn.Module,
    scope: Literal["all", "action_expert_and_gate"],
) -> tuple[str, ...]:
    if scope == "all":
        prefixes: tuple[str, ...] | None = None
    elif scope == "action_expert_and_gate":
        prefixes = (
            "paligemma_with_expert.gemma_expert.",
            "action_in_proj.",
            "action_out_proj.",
            "time_mlp_in.",
            "time_mlp_out.",
            "phase_gate.",
        )
    else:
        raise ValueError(f"unsupported PyTorch trainable scope: {scope}")
    trainable_names = []
    for name, parameter in model.named_parameters():
        trainable = prefixes is None or name.startswith(prefixes)
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
    if not trainable_names:
        raise ValueError(f"PyTorch trainable scope selected no parameters: {scope}")
    return tuple(trainable_names)


def build_optimizer(config: _config.TrainConfig, model: torch.nn.Module) -> torch.optim.AdamW:
    trainable_names = configure_trainable_parameters(
        model,
        config.pytorch_trainable_scope,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in parameters)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info(
        "PyTorch trainable scope=%s parameters=%d/%d tensors=%d",
        config.pytorch_trainable_scope,
        trainable_count,
        total_count,
        len(trainable_names),
    )
    return torch.optim.AdamW(
        parameters,
        lr=config.lr_schedule.peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
        fused=config.pytorch_fused_optimizer,
    )


def require_run_receipts(config: _config.TrainConfig) -> pathlib.Path:
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


def initialize_pretrained(model: pi0_pytorch.PI0Pytorch, base: pathlib.Path) -> None:
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


def train_step(
    model: pi0_pytorch.PI0Pytorch,
    optimizer: torch.optim.AdamW,
    batches: Sequence[tuple[Any, torch.Tensor]],
    config: _config.TrainConfig,
    global_step: int,
    timer: _performance.CudaStageTimer | None = None,
    ema_state: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    if len(batches) != config.gradient_accumulation_steps:
        raise ValueError("batch count must match gradient accumulation steps")
    lr = pytorch_training.learning_rate(
        global_step,
        warmup_steps=config.lr_schedule.warmup_steps,
        peak_lr=config.lr_schedule.peak_lr,
        decay_steps=config.lr_schedule.decay_steps,
        end_lr=config.lr_schedule.decay_lr,
    )
    for group in optimizer.param_groups:
        group["lr"] = lr

    optimizer.zero_grad(set_to_none=True)
    metric_sums: dict[str, float] = {}
    accumulation_steps = len(batches)
    for observation, actions in batches:
        if timer is not None:
            timer.start("forward")
        if model.casm_mode == "visual_phase_gate":
            losses, auxiliary = model(observation, actions, return_aux=True)
        else:
            losses = model(observation, actions)
            auxiliary = {}
        loss = losses.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {global_step}: {loss}")
        if timer is not None:
            timer.end("forward")
            timer.start("backward")
        (loss / accumulation_steps).backward()
        if timer is not None:
            timer.end("backward")
        metric_sums["loss"] = metric_sums.get("loss", 0.0) + float(loss.detach().cpu())
        for name, value in auxiliary.items():
            metric_sums[name] = metric_sums.get(name, 0.0) + float(value.detach().cpu())

    if timer is not None:
        timer.start("optimizer")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        config.optimizer.clip_gradient_norm,
        error_if_nonfinite=True,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if config.ema_decay is not None:
        if ema_state is None:
            raise ValueError("EMA state is required when EMA decay is configured")
        pytorch_training.update_ema(ema_state, model, config.ema_decay)
    elif ema_state is not None:
        raise ValueError("EMA state was provided without an EMA decay")
    if timer is not None:
        timer.end("optimizer")
    metrics = {
        "learning_rate": lr,
        "gradient_norm": float(gradient_norm.detach().cpu()),
    }
    metrics.update({name: value / accumulation_steps for name, value in metric_sums.items()})
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
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    pytorch_training.seed_everything(config.seed)
    checkpoint_root = _prepare_checkpoint_root(config)
    base = require_run_receipts(config)
    manifest = config_signature(config)
    loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    data_config = loader.data_config()
    extra_files = _checkpoint_files(data_config, manifest)
    model = build_model(config, device)
    optimizer = build_optimizer(config, model)

    ema_state = None
    global_step = 0
    resume_metadata: dict[str, Any] = {}
    if config.resume:
        global_step, resume_metadata, ema_state = pytorch_training.load_checkpoint(
            model,
            optimizer,
            checkpoint_root,
            device=device,
            load_ema=config.ema_decay is not None,
        )
        if resume_metadata["config_signature"] != manifest:
            raise ValueError("resume config signature does not match checkpoint")
        loader.load_state_dict(resume_metadata["data_loader_state"])
    else:
        initialize_pretrained(model, base)
        if config.ema_decay is not None:
            ema_state = pytorch_training.initialize_ema(model)
    if config.pytorch_compile_mode != "none":
        model.compile(mode=config.pytorch_compile_mode)

    run = _init_wandb(config, manifest, resume_metadata.get("wandb_run_id"))
    metrics_path = checkpoint_root / "metrics.jsonl"
    iterator = iter(loader)
    model.train()
    last_log_time = time.monotonic()
    performance_receipt = None
    performance_path = os.environ.get("PARALLELVLA_PERFORMANCE_RECEIPT")
    record_batch_sha256 = os.environ.get("PARALLELVLA_RECORD_BATCH_SHA256") == "1"
    if performance_path:
        performance_receipt = _performance.TimingReceipt(
            pathlib.Path(performance_path),
            warmup_steps=int(os.environ.get("PARALLELVLA_PERFORMANCE_WARMUP_STEPS", "5")),
            metadata={**manifest, "images_per_sample": 3, "record_batch_sha256": record_batch_sha256},
        )

    try:
        while global_step < config.num_train_steps:
            step_started = time.perf_counter()
            cpu_batches = []
            data_wait_ms = 0.0
            for _ in range(config.gradient_accumulation_steps):
                data_started = time.perf_counter()
                cpu_batches.append(next(iterator))
                data_wait_ms += (time.perf_counter() - data_started) * 1000
            batch_sha256 = _performance.tree_sha256(cpu_batches) if record_batch_sha256 else None
            cuda_timer = _performance.CudaStageTimer() if performance_receipt is not None else None
            batches = []
            for cpu_observation, cpu_actions in cpu_batches:
                if cuda_timer is not None:
                    cuda_timer.start("h2d")
                observation = pytorch_training.move_to_device(cpu_observation, device, non_blocking=True)
                actions = cpu_actions.to(device=device, dtype=torch.float32, non_blocking=True)
                if cuda_timer is not None:
                    cuda_timer.end("h2d")
                batches.append((observation, actions))
            metrics = train_step(model, optimizer, batches, config, global_step, cuda_timer, ema_state)
            cuda_timings = cuda_timer.resolve_ms() if cuda_timer is not None else {}
            global_step += 1
            metrics["step"] = global_step

            logging_started = time.perf_counter()
            if global_step % config.log_interval == 0 or global_step == 1:
                torch.cuda.synchronize(device)
                now = time.monotonic()
                metrics["seconds_per_log_interval"] = now - last_log_time
                last_log_time = now
                _append_metrics(metrics_path, metrics)
                logger.info(
                    "step=%d loss=%.6f grad_norm=%.6f",
                    global_step,
                    metrics["loss"],
                    metrics["gradient_norm"],
                )
                if run is not None:
                    run.log(metrics, step=global_step)
            logging_ms = (time.perf_counter() - logging_started) * 1000

            checkpoint_started = time.perf_counter()
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
                    ema_state=ema_state,
                )
                pytorch_training.prune_checkpoints(
                    checkpoint_root,
                    current_step=global_step,
                    keep_period=config.keep_period,
                )
            checkpoint_ms = (time.perf_counter() - checkpoint_started) * 1000
            if performance_receipt is not None:
                receipt_row = {
                    "step": global_step,
                    "data_wait_ms": data_wait_ms,
                    **cuda_timings,
                    "logging_ms": logging_ms,
                    "checkpoint_ms": checkpoint_ms,
                    "checkpoint_saved": should_save,
                    "step_total_ms": (time.perf_counter() - step_started) * 1000,
                }
                if batch_sha256 is not None:
                    receipt_row["batch_sha256"] = batch_sha256
                performance_receipt.append(receipt_row)
    finally:
        if performance_receipt is not None:
            summary_path = performance_receipt.close()
            logger.info("performance timing summary: %s", summary_path)

    if run is not None:
        run.finish()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(_config.cli())


if __name__ == "__main__":
    main()
