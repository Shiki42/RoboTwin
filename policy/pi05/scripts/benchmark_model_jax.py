"""Model-only JAX PI0.5/CASM benchmark on one fixed GPU-resident batch."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import math
import os
import pathlib
import re
import time


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
        "--xla-preallocate",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--xla-memory-fraction", type=float, default=0.9)
    return parser.parse_args()


def _require_receipts() -> dict[str, str]:
    required = (
        "PI05_BASE_CHECKPOINT",
        "PARALLELVLA_CODE_COMMIT",
        "PARALLELVLA_DATASET_RECEIPT",
        "PARALLELVLA_DATASET_REVISION",
    )
    values = {name: os.environ.get(name, "") for name in required}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(f"missing benchmark receipt environment: {missing}")
    if re.fullmatch(r"[0-9a-f]{40}", values["PARALLELVLA_CODE_COMMIT"]) is None:
        raise ValueError("PARALLELVLA_CODE_COMMIT must be a full lowercase Git SHA")
    if not pathlib.Path(values["PARALLELVLA_DATASET_RECEIPT"]).is_file():
        raise FileNotFoundError(values["PARALLELVLA_DATASET_RECEIPT"])
    return values


def _initialize_jax_runtime(jax_module) -> None:
    """Load JAX CUDA libraries before the PyTorch-backed input loader."""
    startup_key = jax_module.random.key(0)
    jax_module.block_until_ready(startup_key)


def main() -> None:
    args = _parse_args()
    summary_path = args.output.with_suffix(".summary.json")
    if args.output.exists() or summary_path.exists():
        raise FileExistsError(args.output)
    if args.warmup_steps < 0 or args.measured_steps < 1:
        raise ValueError("benchmark requires non-negative warmup and positive measured steps")
    if not 0 < args.xla_memory_fraction <= 1:
        raise ValueError("XLA memory fraction must be in (0, 1]")

    os.environ.pop("JAX_PLATFORMS", None)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(args.xla_preallocate).lower()
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.xla_memory_fraction)

    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    _initialize_jax_runtime(jax)
    import train as _trainer  # noqa: PLC0415

    from openpi.training import config as _config  # noqa: PLC0415
    from openpi.training import data_loader as _data  # noqa: PLC0415
    from openpi.training import performance as _performance  # noqa: PLC0415
    from openpi.training import sharding  # noqa: PLC0415

    environment = _require_receipts()
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError(f"JAX did not initialize a GPU backend: {jax.devices()}")

    assets_base_dir = _performance.resolve_benchmark_assets_base_dir(args.assets_base_dir)
    config = dataclasses.replace(
        _config.get_config(args.config_name),
        assets_base_dir=assets_base_dir,
        batch_size=args.batch_size,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        wandb_enabled=False,
        seed=args.seed,
    )
    loader = _data.create_data_loader(
        config,
        framework="pytorch",
        shuffle=True,
        num_batches=1,
    )
    cpu_batch = next(iter(loader))
    batch_hash = _performance.tree_sha256(cpu_batch)
    numpy_batch = jax.tree.map(np.asarray, cpu_batch)

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    initialization_started = time.perf_counter()
    with sharding.set_mesh(mesh):
        state, state_sharding = _trainer.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    batch = jax.tree.map(lambda value: jax.device_put(value, data_sharding), numpy_batch)
    jax.block_until_ready(batch)
    initialization_s = time.perf_counter() - initialization_started

    ptrain_step = jax.jit(
        functools.partial(_trainer.train_step, config),
        in_shardings=(replicated_sharding, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    compile_started = time.perf_counter()
    with sharding.set_mesh(mesh):
        compiled_step = ptrain_step.lower(train_rng, state, batch).compile()
    compile_s = time.perf_counter() - compile_started

    metadata = {
        "framework": "jax",
        "mode": "model-only",
        "config_name": config.name,
        "batch_size": args.batch_size,
        "batch_sha256": batch_hash,
        "images_per_sample": 3,
        "run_index": args.run_index,
        "seed": config.seed,
        "code_commit": environment["PARALLELVLA_CODE_COMMIT"],
        "dataset_revision": environment["PARALLELVLA_DATASET_REVISION"],
        "xla_preallocate": args.xla_preallocate,
        "assets_base_dir": assets_base_dir,
        "xla_memory_fraction": args.xla_memory_fraction,
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
            wall_started = time.perf_counter()
            with sharding.set_mesh(mesh):
                state, info = compiled_step(train_rng, state, batch)
            jax.block_until_ready((state, info))
            wall_s = time.perf_counter() - wall_started
            if first_step_s is None:
                first_step_s = wall_s
            host_info = jax.device_get(info)
            metrics = {name: float(np.asarray(value)) for name, value in host_info.items()}
            if not all(math.isfinite(value) for value in metrics.values()):
                raise FloatingPointError(f"non-finite benchmark metrics at step {step}: {metrics}")
            receipt.append(
                {
                    "step": step,
                    "step_total_ms": wall_s * 1000,
                    **metrics,
                }
            )
    finally:
        receipt.close()

    device = jax.devices("gpu")[0]
    memory_stats = device.memory_stats() or {}
    summary = json.loads(summary_path.read_text())
    summary.update(
        {
            "jax_version": jax.__version__,
            "initialization_s": initialization_s,
            "compile_s": compile_s,
            "first_compiled_step_s": first_step_s,
            "batch_sha256": batch_hash,
            "device": str(device),
            "device_memory_stats": {
                str(name): int(value) for name, value in memory_stats.items() if isinstance(value, int)
            },
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
