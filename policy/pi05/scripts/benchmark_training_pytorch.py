"""End-to-end PyTorch PI0.5/CASM benchmark without checkpoint writes."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pathlib
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import torch

from openpi.training import config as _config
from openpi.training import data_loader as _data
from openpi.training import performance as _performance
from openpi.training import pytorch_training
from scripts import benchmark_model_pytorch as _model_benchmark
from scripts import train_pytorch as _trainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--assets-base-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=87431)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--record-batch-sha256", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "reduce-overhead", "max-autotune"),
        default="none",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa"),
        default="sdpa",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fused-optimizer", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.warmup_steps < 0 or args.measured_steps < 1:
        raise ValueError("benchmark requires non-negative warmup and positive measured steps")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient accumulation steps must be positive")
    if args.num_workers < 0:
        raise ValueError("number of workers must be non-negative")
    if args.prefetch_factor < 1:
        raise ValueError("prefetch factor must be positive")
    if args.persistent_workers and args.num_workers == 0:
        raise ValueError("persistent workers require at least one worker")


def _build_config(args: argparse.Namespace, assets_base_dir: str) -> _config.TrainConfig:
    return dataclasses.replace(
        _config.get_config(args.config_name),
        assets_base_dir=assets_base_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        pytorch_compile_mode=args.compile_mode,
        pytorch_attention_implementation=args.attention_implementation,
        pytorch_gradient_checkpointing=args.gradient_checkpointing,
        pytorch_fused_optimizer=args.fused_optimizer,
        seed=args.seed,
    )


def _step(
    iterator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: _config.TrainConfig,
    device: torch.device,
    step: int,
    *,
    record_batch_sha256: bool,
    ema_state: dict[str, torch.Tensor] | None,
) -> tuple[dict[str, float | int | str], str | None]:
    wall_started = time.perf_counter()
    cpu_batches = []
    data_wait_ms = 0.0
    for _ in range(config.gradient_accumulation_steps):
        data_started = time.perf_counter()
        cpu_batches.append(next(iterator))
        data_wait_ms += (time.perf_counter() - data_started) * 1000
    batch_hash = _performance.tree_sha256(cpu_batches) if record_batch_sha256 else None
    timer = _performance.CudaStageTimer()
    batches = []
    for cpu_observation, cpu_actions in cpu_batches:
        timer.start("h2d")
        observation = pytorch_training.move_to_device(cpu_observation, device, non_blocking=True)
        actions = cpu_actions.to(device=device, dtype=torch.float32, non_blocking=True)
        timer.end("h2d")
        batches.append((observation, actions))
    metrics = _trainer.train_step(model, optimizer, batches, config, step - 1, timer, ema_state)
    cuda_timings = timer.resolve_ms()
    row: dict[str, float | int | str] = {
        "step": step,
        "data_wait_ms": data_wait_ms,
        **cuda_timings,
        "step_total_ms": (time.perf_counter() - wall_started) * 1000,
        **metrics,
    }
    if batch_hash is not None:
        row["batch_sha256"] = batch_hash
    return row, batch_hash


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    summary_path = args.output.with_suffix(".summary.json")
    if args.output.exists() or summary_path.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    assets_base_dir = _performance.resolve_benchmark_assets_base_dir(args.assets_base_dir)
    config = _build_config(args, assets_base_dir)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    pytorch_training.seed_everything(config.seed)
    total_steps = args.warmup_steps + args.measured_steps
    loader = _data.create_data_loader(
        config,
        framework="pytorch",
        shuffle=True,
        num_batches=total_steps * config.gradient_accumulation_steps,
    )

    initialization_started = time.perf_counter()
    model = _trainer.build_model(config, device)
    optimizer = _trainer.build_optimizer(config, model)
    base = _trainer.require_run_receipts(config)
    _trainer.initialize_pretrained(model, base)
    initialization_s = time.perf_counter() - initialization_started
    ema_state = pytorch_training.initialize_ema(model) if config.ema_decay is not None else None
    attention_backends = _model_benchmark.attention_implementations(model)

    compile_registration_started = time.perf_counter()
    if config.pytorch_compile_mode != "none":
        model.compile(mode=config.pytorch_compile_mode)
    compile_registration_s = time.perf_counter() - compile_registration_started

    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.compiler.reset()
    iterator = iter(loader)
    metadata = {
        **_trainer.config_signature(config),
        "framework": "pytorch",
        "mode": "end-to-end",
        "images_per_sample": 3,
        "run_index": args.run_index,
        "record_batch_sha256": args.record_batch_sha256,
        "assets_base_dir": assets_base_dir,
    }
    receipt = _performance.TimingReceipt(args.output, warmup_steps=args.warmup_steps, metadata=metadata)
    first_step_s = None
    first_batch_hash = None
    try:
        for step in range(1, total_steps + 1):
            row, batch_hash = _step(
                iterator,
                model,
                optimizer,
                config,
                device,
                step,
                record_batch_sha256=args.record_batch_sha256,
                ema_state=ema_state,
            )
            if first_step_s is None:
                first_step_s = float(row["step_total_ms"]) / 1000
                first_batch_hash = batch_hash
            if not all(math.isfinite(float(row[name])) for name in ("loss", "gradient_norm")):
                raise FloatingPointError(f"non-finite benchmark metrics at step {step}: {row}")
            receipt.append(row)
    finally:
        receipt.close()

    summary = json.loads(summary_path.read_text())
    summary.update(
        {
            "initialization_s": initialization_s,
            "compile_registration_s": compile_registration_s,
            "first_compiled_step_s": first_step_s,
            "first_batch_sha256": first_batch_hash,
            "final_data_loader_state": loader.state_dict(),
            "attention_implementations": attention_backends,
            "sdpa_flags": {
                "flash": torch.backends.cuda.flash_sdp_enabled(),
                "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math": torch.backends.cuda.math_sdp_enabled(),
                "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
            },
            "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "dynamo_counters": _model_benchmark.dynamo_counters(),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
