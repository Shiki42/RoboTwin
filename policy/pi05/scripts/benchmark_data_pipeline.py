"""Benchmark PI0.5/CASM data loading and video decoding with raw receipts."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import dataclasses
import json
import pathlib
import time
from typing import Any

import numpy as np
import torch

from openpi.training import config as _config
from openpi.training import data_loader as _data
from openpi.training import performance as _performance


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--mode", choices=("loader", "decode-only"), default="loader")
    parser.add_argument("--backend", choices=("pyav", "torchcodec"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=87431)
    parser.add_argument("--measured-steps", type=int, default=100)
    parser.add_argument("--run-index", type=int, required=True)
    return parser.parse_args()


def _replace_backend(config: _config.TrainConfig, backend: str) -> _config.TrainConfig:
    base = config.data.base_config or _config.DataConfig()
    data = dataclasses.replace(config.data, base_config=dataclasses.replace(base, video_backend=backend))
    return dataclasses.replace(config, data=data)


def _leaves(value: Any) -> Iterable[Any]:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _leaves(getattr(value, field.name))
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
        return
    if isinstance(value, tuple | list):
        for item in value:
            yield from _leaves(item)
        return
    if value is not None:
        yield value


def _batch_bytes(batch: Any) -> int:
    return sum(
        int(value.numel() * value.element_size()) if isinstance(value, torch.Tensor) else np.asarray(value).nbytes
        for value in _leaves(batch)
    )


def _all_tensors_pinned(batch: Any) -> bool:
    tensors = [value for value in _leaves(batch) if isinstance(value, torch.Tensor)]
    return bool(tensors) and all(value.is_pinned() for value in tensors)


def _benchmark_loader(
    config: _config.TrainConfig,
    *,
    total_steps: int,
    receipt: _performance.TimingReceipt,
) -> tuple[int, bool]:
    loader = _data.create_data_loader(
        config,
        framework="pytorch",
        shuffle=True,
        num_batches=total_steps,
    )
    iterator = iter(loader)
    bytes_per_batch = 0
    pinned = False
    for step in range(1, total_steps + 1):
        started = time.perf_counter()
        batch = next(iterator)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if step == 1:
            bytes_per_batch = _batch_bytes(batch)
            pinned = _all_tensors_pinned(batch)
        receipt.append({"step": step, "data_wait_ms": elapsed_ms, "step_total_ms": elapsed_ms})
    return bytes_per_batch, pinned


def _benchmark_decode_only(
    config: _config.TrainConfig,
    *,
    total_steps: int,
    receipt: _performance.TimingReceipt,
) -> tuple[int, bool]:
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _data.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    plan = _data.DeterministicBatchSampler(dataset, batch_size=config.batch_size, seed=config.seed)
    iterator = iter(plan)
    bytes_per_batch = 0
    for step in range(1, total_steps + 1):
        started = time.perf_counter()
        items = [dataset[index] for index in next(iterator)]
        elapsed_ms = (time.perf_counter() - started) * 1000
        plan.mark_consumed()
        if step == 1:
            bytes_per_batch = _batch_bytes(items)
        receipt.append({"step": step, "decode_ms": elapsed_ms, "step_total_ms": elapsed_ms})
    return bytes_per_batch, False


def main() -> None:
    args = _parse_args()
    if args.output.exists() or args.output.with_suffix(".summary.json").exists():
        raise FileExistsError(args.output)
    if args.warmup_steps < 0 or args.measured_steps < 1:
        raise ValueError("benchmark requires non-negative warmup and positive measured steps")
    if args.mode == "decode-only" and args.num_workers != 0:
        raise ValueError("decode-only mode benchmarks the decoder in the calling process")

    config = _replace_backend(_config.get_config(args.config_name), args.backend)
    config = dataclasses.replace(
        config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        seed=args.seed,
    )
    metadata = {
        "framework": "pytorch",
        "mode": args.mode,
        "backend": args.backend,
        "batch_size": args.batch_size,
        "images_per_sample": 3,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "persistent_workers": args.persistent_workers,
        "pin_memory": args.pin_memory,
        "run_index": args.run_index,
        "seed": args.seed,
    }
    receipt = _performance.TimingReceipt(
        args.output,
        warmup_steps=args.warmup_steps,
        metadata=metadata,
    )
    total_steps = args.warmup_steps + args.measured_steps
    benchmark = _benchmark_loader if args.mode == "loader" else _benchmark_decode_only
    try:
        bytes_per_batch, pinned = benchmark(config, total_steps=total_steps, receipt=receipt)
    finally:
        summary_path = receipt.close()

    summary = json.loads(summary_path.read_text())
    summary["bytes_per_batch"] = bytes_per_batch
    summary["all_tensors_pinned"] = pinned
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
