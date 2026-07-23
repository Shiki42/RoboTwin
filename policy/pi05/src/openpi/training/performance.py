"""Low-overhead stage timing receipts for PyTorch training."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import dataclasses
import hashlib
import json
import pathlib
import statistics
from typing import Any

import numpy as np
import torch

CUDA_STAGE_NAMES = ("h2d", "forward", "backward", "optimizer")


def resolve_benchmark_assets_base_dir(path: pathlib.Path) -> str:
    """Return an existing benchmark assets directory as an absolute path."""
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    return str(resolved)


class CudaStageTimer:
    """Record ordered CUDA stages and synchronize once when resolving them."""

    def __init__(self) -> None:
        self._events = {
            name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for name in CUDA_STAGE_NAMES
        }

    def start(self, name: str) -> None:
        self._events[name][0].record()

    def end(self, name: str) -> None:
        self._events[name][1].record()

    def resolve_ms(self) -> dict[str, float]:
        self._events[CUDA_STAGE_NAMES[-1]][1].synchronize()
        return {f"{name}_ms": start.elapsed_time(end) for name, (start, end) in self._events.items()}


class TimingReceipt:
    """Append raw JSONL timing rows and emit percentile summaries."""

    def __init__(
        self,
        path: pathlib.Path,
        *,
        warmup_steps: int,
        metadata: Mapping[str, Any],
        flush_interval: int = 10,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup steps must be non-negative")
        if flush_interval < 1:
            raise ValueError("flush interval must be positive")
        self._path = path
        self._warmup_steps = warmup_steps
        self._metadata = dict(metadata)
        self._flush_interval = flush_interval
        self._rows_seen = 0
        self._measured: list[dict[str, float | int | str]] = []
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("w", encoding="utf-8")

    def append(self, row: Mapping[str, float | int | str]) -> None:
        record = dict(row)
        self._rows_seen += 1
        record["receipt_index"] = self._rows_seen
        record["phase"] = "warmup" if self._rows_seen <= self._warmup_steps else "measured"
        self._stream.write(json.dumps(record, sort_keys=True) + "\n")
        if self._rows_seen % self._flush_interval == 0:
            self._stream.flush()
        if record["phase"] == "measured":
            self._measured.append(record)

    def close(self) -> pathlib.Path:
        self._stream.flush()
        self._stream.close()
        summary_path = self._path.with_suffix(".summary.json")
        summary = summarize_rows(self._measured)
        summary["throughput"] = _throughput(self._measured, self._metadata)
        steady_rows = [row for row in self._measured if not row.get("checkpoint_saved", False)]
        if len(steady_rows) != len(self._measured):
            summary["steady_throughput_without_checkpoint"] = _throughput(steady_rows, self._metadata)
            summary["checkpoint_rows"] = len(self._measured) - len(steady_rows)
        summary["metadata"] = self._metadata
        summary["raw_receipt"] = str(self._path)
        summary["warmup_steps"] = self._warmup_steps
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary_path


def tree_sha256(value: Any) -> str:
    """Hash a nested CPU batch including paths, shapes, dtypes, and values."""
    digest = hashlib.sha256()
    for path, leaf in _tree_leaves(value):
        digest.update(path.encode())
        if isinstance(leaf, torch.Tensor):
            tensor = leaf.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
            continue
        array = np.asarray(leaf)
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _tree_leaves(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _tree_leaves(getattr(value, field.name), f"{path}.{field.name}")
        return
    if isinstance(value, Mapping):
        for key in sorted(value):
            yield from _tree_leaves(value[key], f"{path}[{key!r}]")
        return
    if isinstance(value, tuple | list):
        for index, item in enumerate(value):
            yield from _tree_leaves(item, f"{path}[{index}]")
        return
    if value is not None:
        yield path, value


def summarize_rows(rows: list[Mapping[str, float | int | str]]) -> dict[str, Any]:
    if not rows:
        return {"measured_steps": 0, "timings_ms": {}}
    timing_fields = sorted(
        key for key in rows[0] if key.endswith("_ms") and all(isinstance(row.get(key), int | float) for row in rows)
    )
    timings = {field: summarize_values([float(row[field]) for row in rows]) for field in timing_fields}
    return {"measured_steps": len(rows), "timings_ms": timings}


def _throughput(
    rows: list[Mapping[str, float | int | str]],
    metadata: Mapping[str, Any],
) -> dict[str, float]:
    if not rows or "step_total_ms" not in rows[0]:
        return {}
    duration_s = sum(float(row["step_total_ms"]) for row in rows) / 1000
    steps_per_second = len(rows) / duration_s
    batch_size = int(metadata.get("batch_size", 0))
    images_per_sample = int(metadata.get("images_per_sample", 0))
    return {
        "duration_s": duration_s,
        "steps_per_second": steps_per_second,
        "steps_per_minute": steps_per_second * 60,
        "samples_per_second": steps_per_second * batch_size,
        "images_per_second": steps_per_second * batch_size * images_per_sample,
    }


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(ordered),
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p99": _percentile(ordered, 0.99),
    }


def _percentile(ordered: list[float], quantile: float) -> float:
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
