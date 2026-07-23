"""Model-only PyTorch PI0.5/CASM benchmark on one fixed GPU-resident batch."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pathlib
import time
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import torch
from scripts import train_pytorch as _trainer

from openpi.training import config as _config
from openpi.training import data_loader as _data
from openpi.training import performance as _performance
from openpi.training import pytorch_training


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--assets-base-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=87431)
    parser.add_argument("--run-index", type=int, required=True)
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
    parser.add_argument("--channels-last", action="store_true")
    return parser.parse_args()


def _attention_implementations(model: torch.nn.Module) -> dict[str, int]:
    implementations: dict[str, int] = {}
    for module in model.modules():
        config = getattr(module, "config", None)
        implementation = getattr(config, "_attn_implementation", None)
        if implementation is not None:
            key = str(implementation)
            implementations[key] = implementations.get(key, 0) + 1
    return implementations


def _dynamo_counters() -> dict[str, dict[str, int]]:
    counters = torch._dynamo.utils.counters  # noqa: SLF001
    return {
        group: {str(name): int(value) for name, value in values.items()} for group, values in counters.items() if values
    }


def _fixed_batch(config: _config.TrainConfig) -> tuple[Any, torch.Tensor, str]:
    loader_config = dataclasses.replace(
        config,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    loader = _data.create_data_loader(
        loader_config,
        framework="pytorch",
        shuffle=True,
        num_batches=1,
    )
    observation, actions = next(iter(loader))
    batch_hash = _performance.tree_sha256((observation, actions))
    return observation, actions, batch_hash


def main() -> None:
    args = _parse_args()
    summary_path = args.output.with_suffix(".summary.json")
    if args.output.exists() or summary_path.exists():
        raise FileExistsError(args.output)
    if args.warmup_steps < 0 or args.measured_steps < 1:
        raise ValueError("benchmark requires non-negative warmup and positive measured steps")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    assets_base_dir = _performance.resolve_benchmark_assets_base_dir(args.assets_base_dir)
    config = dataclasses.replace(
        _config.get_config(args.config_name),
        assets_base_dir=assets_base_dir,
        batch_size=args.batch_size,
        pytorch_compile_mode=args.compile_mode,
        pytorch_attention_implementation=args.attention_implementation,
        pytorch_gradient_checkpointing=args.gradient_checkpointing,
        pytorch_fused_optimizer=args.fused_optimizer,
        seed=args.seed,
    )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    pytorch_training.seed_everything(config.seed)

    observation, actions, batch_hash = _fixed_batch(config)
    observation = pytorch_training.move_to_device(observation, device)
    actions = actions.to(device=device, dtype=torch.float32)
    if args.channels_last:
        observation = dataclasses.replace(
            observation,
            images={k: v.contiguous(memory_format=torch.channels_last) for k, v in observation.images.items()},
        )
    torch.cuda.synchronize(device)

    initialization_started = time.perf_counter()
    model = _trainer.build_model(config, device)
    optimizer = _trainer.build_optimizer(config, model)
    base = _trainer.require_run_receipts(config)
    _trainer.initialize_pretrained(model, base)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    initialization_s = time.perf_counter() - initialization_started
    attention_implementations = _attention_implementations(model)

    compile_registration_started = time.perf_counter()
    if config.pytorch_compile_mode != "none":
        model.compile(mode=config.pytorch_compile_mode)
    compile_registration_s = time.perf_counter() - compile_registration_started

    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.compiler.reset()
    metadata = {
        **_trainer.config_signature(config),
        "framework": "pytorch",
        "mode": "model-only",
        "batch_sha256": batch_hash,
        "images_per_sample": 3,
        "run_index": args.run_index,
        "channels_last": args.channels_last,
        "assets_base_dir": assets_base_dir,
    }
    receipt = _performance.TimingReceipt(
        args.output,
        warmup_steps=args.warmup_steps,
        metadata=metadata,
    )
    first_step_s = None
    total_steps = args.warmup_steps + args.measured_steps
    try:
        for step in range(1, total_steps + 1):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            wall_started = time.perf_counter()
            start_event.record()
            metrics = _trainer.train_step(model, optimizer, observation, actions, config, step - 1)
            end_event.record()
            end_event.synchronize()
            wall_s = time.perf_counter() - wall_started
            if first_step_s is None:
                first_step_s = wall_s
            if not all(math.isfinite(value) for value in metrics.values()):
                raise FloatingPointError(f"non-finite benchmark metrics at step {step}: {metrics}")
            receipt.append(
                {
                    "step": step,
                    "compute_cuda_ms": start_event.elapsed_time(end_event),
                    "step_total_ms": wall_s * 1000,
                    **metrics,
                }
            )
    finally:
        receipt.close()

    summary = json.loads(summary_path.read_text())
    summary.update(
        {
            "initialization_s": initialization_s,
            "compile_registration_s": compile_registration_s,
            "first_compiled_step_s": first_step_s,
            "batch_sha256": batch_hash,
            "attention_implementations": attention_implementations,
            "sdpa_flags": {
                "flash": torch.backends.cuda.flash_sdp_enabled(),
                "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math": torch.backends.cuda.math_sdp_enabled(),
                "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
            },
            "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "dynamo_counters": _dynamo_counters(),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
