"""Single-GPU, full-state PyTorch trainer for matched PI0.5/CASM runs."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import hashlib
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

from openpi.models import gemma as _gemma
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


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_split_signature(config: _config.TrainConfig) -> dict[str, Any] | None:
    if config.train_episodes is None:
        return None
    receipt_value = os.environ.get("PARALLELVLA_SPLIT_RECEIPT")
    if not receipt_value:
        raise ValueError("PARALLELVLA_SPLIT_RECEIPT is required for episode-subset training")
    receipt_path = pathlib.Path(receipt_value)
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    revision = os.environ.get("PARALLELVLA_DATASET_REVISION")
    if payload.get("dataset_repo") != config.data.repo_id:
        raise ValueError("split receipt dataset repo does not match config")
    if payload.get("dataset_revision") != revision:
        raise ValueError("split receipt dataset revision does not match run")
    train = tuple(payload.get("train_episodes", ()))
    validation = tuple(payload.get("validation_episodes", ()))
    unused = tuple(payload.get("unused_episodes", ()))
    if train != config.train_episodes:
        raise ValueError("split receipt training episodes do not match config")
    if validation != config.validation_episodes:
        raise ValueError("split receipt validation episodes do not match config")
    total_episodes = payload.get("total_episodes")
    if not isinstance(total_episodes, int) or total_episodes < 1:
        raise ValueError("split receipt total_episodes must be positive")
    if sorted((*train, *validation, *unused)) != list(range(total_episodes)):
        raise ValueError("split receipt does not partition every source episode exactly once")
    return {
        "receipt_sha256": _sha256(receipt_path),
        "total_episodes": total_episodes,
        "train_episodes": list(train),
        "validation_episodes": list(validation),
        "unused_episodes": list(unused),
        "validation_interval": config.validation_interval,
        "validation_rng_seed": config.seed + 1,
    }


def _normalizer_signature(
    config: _config.TrainConfig,
    episode_split: dict[str, Any] | None,
) -> dict[str, str] | None:
    if episode_split is None:
        return None
    receipt_value = os.environ.get("PARALLELVLA_NORM_STATS_RECEIPT")
    if not receipt_value:
        raise ValueError("PARALLELVLA_NORM_STATS_RECEIPT is required for episode-subset training")
    receipt_path = pathlib.Path(receipt_value)
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if payload.get("dataset_repo") != config.data.repo_id:
        raise ValueError("normalizer receipt dataset repo does not match config")
    if payload.get("dataset_revision") != os.environ.get("PARALLELVLA_DATASET_REVISION"):
        raise ValueError("normalizer receipt dataset revision does not match run")
    if tuple(payload.get("train_episodes", ())) != config.train_episodes:
        raise ValueError("normalizer receipt training episodes do not match config")
    if payload.get("split_receipt_sha256") != episode_split["receipt_sha256"]:
        raise ValueError("normalizer receipt is not bound to the current episode split")
    statistics = payload.get("statistics")
    if statistics != "exact_concat_train_only_valid_action_steps_v1":
        raise ValueError("normalizer receipt does not record exact train-only statistics")
    normalizer_path = config.assets_dirs / config.data.repo_id / "norm_stats.json"
    if not normalizer_path.is_file():
        raise FileNotFoundError(normalizer_path)
    normalizer_sha256 = _sha256(normalizer_path)
    if payload.get("normalizer_sha256") != normalizer_sha256:
        raise ValueError("normalizer receipt SHA-256 does not match norm_stats.json")
    inventory_sha256 = payload.get("input_inventory_sha256")
    if not isinstance(inventory_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", inventory_sha256) is None:
        raise ValueError("normalizer receipt requires a lowercase input inventory SHA-256")
    return {
        "receipt_sha256": _sha256(receipt_path),
        "normalizer_sha256": normalizer_sha256,
        "input_inventory_sha256": inventory_sha256,
        "statistics": statistics,
    }


def _lora_signature(variant: str) -> dict[str, float | int] | None:
    adapters = _gemma.get_config(variant).lora_configs
    if not adapters:
        return None
    settings = {(adapter.rank, adapter.alpha) for adapter in adapters.values()}
    if len(settings) != 1:
        raise ValueError(f"LoRA settings do not match for {variant}")
    rank, alpha = settings.pop()
    return {"rank": rank, "alpha": alpha}


def config_signature(config: _config.TrainConfig) -> dict[str, Any]:
    model = config.model
    episode_split = _episode_split_signature(config)
    return {
        "config_name": config.name,
        "batch_size": config.batch_size * config.gradient_accumulation_steps,
        "microbatch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "pytorch_training_precision": config.pytorch_training_precision,
        "pytorch_compute_precision": config.pytorch_compute_precision,
        "ema_decay": config.ema_decay,
        "num_workers": config.num_workers,
        "validation_num_workers": config.validation_num_workers,
        "validation_persistent_workers": False,
        "prefetch_factor": config.prefetch_factor,
        "persistent_workers": config.persistent_workers,
        "pin_memory": config.pin_memory,
        "pytorch_compile_mode": config.pytorch_compile_mode,
        "pytorch_attention_implementation": config.pytorch_attention_implementation,
        "pytorch_fused_optimizer": config.pytorch_fused_optimizer,
        "pytorch_gradient_checkpointing": config.pytorch_gradient_checkpointing,
        "pytorch_gradient_checkpointing_scope": config.pytorch_gradient_checkpointing_scope,
        "pytorch_trainable_scope": config.pytorch_trainable_scope,
        "pytorch_action_prompt_mode": config.pytorch_action_prompt_mode,
        "training_objective": {
            "all": "action_policy_v1",
            "action_expert_and_gate": "action_policy_v1",
            "lora": "action_policy_v1",
            "subtask_head": "detached_subtask_ce_only_v1",
            "subtask_head_and_projector": (
                "teacher_forced_factorized_action_plus_subtask_shared_projector_v1"
                if config.pytorch_action_prompt_mode == "teacher_forced_joint_subtask"
                else "action_plus_subtask_shared_projector_v1"
            ),
        }[config.pytorch_trainable_scope],
        "seed": config.seed,
        "episode_split": episode_split,
        "normalizer": _normalizer_signature(config, episode_split),
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
            "aux_subtask": {
                "classes": getattr(model, "pytorch_aux_subtask_classes", 0),
                "hidden_dim": getattr(model, "pytorch_aux_subtask_hidden_dim", None),
                "state_dim": getattr(model, "pytorch_aux_subtask_state_dim", None),
                "loss_weight": getattr(model, "pytorch_aux_subtask_loss_weight", None),
                "class_weights": getattr(model, "pytorch_aux_subtask_class_weights", None),
                "stop_gradient": getattr(model, "pytorch_aux_subtask_stop_gradient", None),
            },
            "lora": {
                "paligemma": _lora_signature(model.paligemma_variant),
                "action_expert": _lora_signature(model.action_expert_variant),
            },
        },
        "action_loss_normalization": "valid_dimension_weighted_v1",
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
    if config.pytorch_training_precision != "float32":
        raise ValueError("PyTorch training requires float32 master parameters")
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
    scope: Literal[
        "all",
        "action_expert_and_gate",
        "lora",
        "subtask_head",
        "subtask_head_and_projector",
    ],
) -> tuple[str, ...]:
    frozen_lora_prefixes = (
        "paligemma_with_expert.paligemma.model.language_model.",
        "paligemma_with_expert.gemma_expert.",
    )
    adapter_names = {name for name, _ in model.named_parameters() if name.endswith((".lora_a", ".lora_b"))}
    if scope == "all":

        def selected(name: str) -> bool:
            return True

    elif scope == "action_expert_and_gate":
        prefixes = (
            "paligemma_with_expert.gemma_expert.",
            "action_in_proj.",
            "action_out_proj.",
            "time_mlp_in.",
            "time_mlp_out.",
            "phase_gate.",
        )

        def selected(name: str) -> bool:
            return name.startswith(prefixes)

    elif scope == "lora":
        if not adapter_names:
            raise ValueError("LoRA trainable scope requires injected adapter parameters")

        def selected(name: str) -> bool:
            return name in adapter_names or not name.startswith(frozen_lora_prefixes)

    elif scope == "subtask_head":

        def selected(name: str) -> bool:
            return name.startswith("subtask_head.")

    elif scope == "subtask_head_and_projector":
        prefixes = (
            "subtask_head.",
            "paligemma_with_expert.paligemma.model.multi_modal_projector.",
        )

        def selected(name: str) -> bool:
            return name.startswith(prefixes)

    else:
        raise ValueError(f"unsupported PyTorch trainable scope: {scope}")
    trainable_names = []
    for name, parameter in model.named_parameters():
        trainable = selected(name)
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
    required_environment = [
        "PI05_BASE_SHA256",
        "PARALLELVLA_DATASET_REVISION",
        "PARALLELVLA_DATASET_RECEIPT",
    ]
    if config.train_episodes is not None:
        required_environment.extend(
            (
                "PARALLELVLA_SPLIT_RECEIPT",
                "PARALLELVLA_NORM_STATS_RECEIPT",
            )
        )
    missing = [name for name in required_environment if not os.environ.get(name)]
    if missing:
        raise ValueError(f"missing run receipt environment: {missing}")
    dataset_receipt = pathlib.Path(os.environ["PARALLELVLA_DATASET_RECEIPT"])
    if not dataset_receipt.is_file():
        raise FileNotFoundError(dataset_receipt)
    if config.train_episodes is not None:
        episode_split = _episode_split_signature(config)
        _normalizer_signature(config, episode_split)
    return base


def initialize_pretrained(model: pi0_pytorch.PI0Pytorch, base: pathlib.Path) -> None:
    allowed = ["phase_gate."] if model.casm_mode == "visual_phase_gate" else []
    if getattr(model, "aux_subtask_classes", 0):
        allowed.append("subtask_head.")
    allowed.extend(name for name, _ in model.named_parameters() if name.endswith((".lora_a", ".lora_b")))
    missing, unexpected = pytorch_training.load_pretrained(
        model,
        base,
        allowed_missing_prefixes=tuple(allowed),
    )
    logger.info("loaded base weights; missing=%s unexpected=%s", missing, unexpected)


def _checkpoint_files(data_config: _config.DataConfig, manifest: dict[str, Any]) -> dict[pathlib.Path, str]:
    files = {pathlib.Path("run_manifest.json"): json.dumps(manifest, indent=2, sort_keys=True)}
    if manifest.get("episode_split") is not None:
        split_receipt = pathlib.Path(os.environ["PARALLELVLA_SPLIT_RECEIPT"])
        normalizer_receipt = pathlib.Path(os.environ["PARALLELVLA_NORM_STATS_RECEIPT"])
        files[pathlib.Path("receipts/episode_split.json")] = split_receipt.read_text(encoding="utf-8")
        files[pathlib.Path("receipts/normalizer.json")] = normalizer_receipt.read_text(encoding="utf-8")
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


def _action_loss_total(
    losses: torch.Tensor,
    action_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if action_mask is None:
        return losses.sum(), losses.new_tensor(losses.numel())
    weights = action_mask.to(device=losses.device, dtype=losses.dtype).sum(dim=-1)
    if weights.shape != losses.shape:
        raise ValueError(f"action loss/mask shape mismatch: {losses.shape} != {weights.shape}")
    total_weight = weights.sum()
    if total_weight <= 0:
        raise ValueError("action loss mask has no supervised dimensions")
    return (losses * weights).sum(), total_weight


def _mean_action_loss(losses: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
    total, weight = _action_loss_total(losses, action_mask)
    return total / weight


def forward_losses(
    model: pi0_pytorch.PI0Pytorch,
    observation: Any,
    actions: torch.Tensor,
    config: _config.TrainConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    use_autocast = config.pytorch_compute_precision == "bfloat16"
    with torch.autocast(
        device_type=actions.device.type,
        dtype=torch.bfloat16,
        enabled=use_autocast,
    ):
        if getattr(model, "casm_mode", "none") == "visual_phase_gate" or getattr(model, "aux_subtask_classes", 0):
            return model(observation, actions, return_aux=True)
        return model(observation, actions), {}


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
        if config.pytorch_trainable_scope == "subtask_head":
            with torch.autocast(
                device_type=actions.device.type,
                dtype=torch.bfloat16,
                enabled=config.pytorch_compute_precision == "bfloat16",
            ):
                auxiliary = model(observation, subtask_only=True)
                loss = model.aux_subtask_loss_weight * auxiliary["subtask_loss"]
        else:
            losses, auxiliary = forward_losses(model, observation, actions, config)
            if model.casm_mode == "visual_phase_gate":
                loss = losses.mean()
            else:
                action_mask = getattr(observation, "action_mask", None)
                if config.pytorch_trainable_scope == "lora" and action_mask is None:
                    raise ValueError("LoRA training requires action_is_pad-derived supervision mask")
                loss = _mean_action_loss(losses, action_mask)
                if getattr(model, "aux_subtask_classes", 0):
                    auxiliary = {**auxiliary, "action_loss": loss.detach()}
            if getattr(model, "aux_subtask_classes", 0):
                loss = loss + model.aux_subtask_loss_weight * auxiliary["subtask_loss"]
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
    parameters_with_grad = (parameter for parameter in model.parameters() if parameter.grad is not None)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters_with_grad,
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


@torch.no_grad()
def evaluate_validation_loss(
    model: pi0_pytorch.PI0Pytorch,
    loader: _data.DataLoader,
    device: torch.device,
    *,
    seed: int,
    config: _config.TrainConfig,
) -> dict[str, float | int]:
    rng_state = pytorch_training.capture_rng_state()
    was_training = model.training
    started = time.perf_counter()
    loss_total = torch.zeros((), device=device, dtype=torch.float32)
    weight_total = torch.zeros((), device=device, dtype=torch.float32)
    batch_count = 0
    sample_count = 0
    try:
        pytorch_training.seed_everything(seed)
        model.eval()
        for cpu_observation, cpu_actions in loader:
            observation = pytorch_training.move_to_device(cpu_observation, device, non_blocking=True)
            actions = cpu_actions.to(device=device, dtype=torch.float32, non_blocking=True)
            losses, _ = forward_losses(model, observation, actions, config)
            action_mask = getattr(observation, "action_mask", None)
            batch_loss, batch_weight = _action_loss_total(losses, action_mask)
            loss_total += batch_loss
            weight_total += batch_weight
            batch_count += 1
            sample_count += actions.shape[0]
        if batch_count == 0:
            raise ValueError("validation loader produced no batches")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return {
            "validation_loss": float((loss_total / weight_total).cpu()),
            "validation_batches": batch_count,
            "validation_samples": sample_count,
            "validation_duration_s": time.perf_counter() - started,
            "validation_rng_seed": seed,
        }
    finally:
        pytorch_training.restore_rng_state(rng_state)
        model.train(was_training)


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
    validation_loader = None
    if config.validation_episodes:
        validation_loader = _data.create_data_loader(config, split="validation", framework="pytorch")
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
    validation_metrics_path = checkpoint_root / "validation_metrics.jsonl"
    iterator = iter(loader)
    model.train()
    performance_receipt = None
    performance_path = os.environ.get("PARALLELVLA_PERFORMANCE_RECEIPT")
    record_batch_sha256 = os.environ.get("PARALLELVLA_RECORD_BATCH_SHA256") == "1"
    if performance_path:
        performance_receipt = _performance.TimingReceipt(
            pathlib.Path(performance_path),
            warmup_steps=int(os.environ.get("PARALLELVLA_PERFORMANCE_WARMUP_STEPS", "5")),
            metadata={**manifest, "images_per_sample": 3, "record_batch_sha256": record_batch_sha256},
        )

    if validation_loader is not None and global_step == 0:
        validation_metrics = evaluate_validation_loss(
            model,
            validation_loader,
            device,
            seed=config.seed + 1,
            config=config,
        )
        validation_metrics["step"] = 0
        _append_metrics(validation_metrics_path, validation_metrics)
        logger.info("step=0 validation_loss=%.6f", validation_metrics["validation_loss"])
        if run is not None:
            run.log(validation_metrics, step=0)
    last_log_time = time.monotonic()

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

            validation_started = time.perf_counter()
            should_validate = validation_loader is not None and (
                global_step % config.validation_interval == 0 or global_step == config.num_train_steps
            )
            if should_validate:
                validation_metrics = evaluate_validation_loss(
                    model,
                    validation_loader,
                    device,
                    seed=config.seed + 1,
                    config=config,
                )
                validation_metrics["step"] = global_step
                _append_metrics(validation_metrics_path, validation_metrics)
                logger.info("step=%d validation_loss=%.6f", global_step, validation_metrics["validation_loss"])
                if run is not None:
                    run.log(validation_metrics, step=global_step)
            validation_ms = (time.perf_counter() - validation_started) * 1000

            checkpoint_started = time.perf_counter()
            should_save = global_step % config.save_interval == 0 or global_step == config.num_train_steps
            if should_save:
                checkpoint_files = dict(extra_files)
                if validation_metrics_path.is_file():
                    checkpoint_files[pathlib.Path("validation_metrics.jsonl")] = validation_metrics_path.read_text()
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
                    extra_files=checkpoint_files,
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
                    "validation_ms": validation_ms,
                    "validation_run": should_validate,
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
