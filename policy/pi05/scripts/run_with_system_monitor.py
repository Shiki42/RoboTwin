"""Run one benchmark while recording raw GPU and process resource samples."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import platform
import subprocess
import time
from typing import Any

import psutil
import pynvml

from openpi.training import performance


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=0.2)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    return args


def _process_sample(root: psutil.Process) -> dict[str, float | int]:
    processes = [root]
    with contextlib.suppress(psutil.Error):
        processes.extend(root.children(recursive=True))
    cpu_percent = 0.0
    rss_bytes = 0
    threads = 0
    live_processes = 0
    for process in processes:
        try:
            cpu_percent += process.cpu_percent(interval=None)
            rss_bytes += process.memory_info().rss
            threads += process.num_threads()
            live_processes += 1
        except psutil.Error:
            continue
    return {
        "process_cpu_percent": cpu_percent,
        "process_rss_bytes": rss_bytes,
        "process_threads": threads,
        "process_count": live_processes,
    }


def _gpu_sample(handle: Any) -> dict[str, float | int]:
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    sample: dict[str, float | int] = {
        "gpu_util_percent": utilization.gpu,
        "gpu_memory_used_bytes": memory.used,
        "gpu_memory_total_bytes": memory.total,
    }
    with contextlib.suppress(pynvml.NVMLError_NotSupported):
        sample["gpu_power_watts"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
    return sample


def _summarize(samples: list[dict[str, float | int]], metadata: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "gpu_util_percent",
        "gpu_memory_used_bytes",
        "gpu_power_watts",
        "process_cpu_percent",
        "process_rss_bytes",
        "process_threads",
        "process_count",
    )
    summary = {
        field: performance.summarize_values([float(sample[field]) for sample in samples if field in sample])
        for field in fields
    }
    summary["sample_count"] = len(samples)
    summary["metadata"] = metadata
    return summary


def main() -> None:
    args = _parse_args()
    summary_path = args.output.with_suffix(".summary.json")
    if args.output.exists() or summary_path.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    metadata = {
        "command": args.command,
        "interval_seconds": args.interval_seconds,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "gpu_name": pynvml.nvmlDeviceGetName(handle),
        "gpu_uuid": pynvml.nvmlDeviceGetUUID(handle),
        "driver_version": pynvml.nvmlSystemGetDriverVersion(),
        "gpu_memory_total_bytes": pynvml.nvmlDeviceGetMemoryInfo(handle).total,
    }
    started_at = time.time()
    child = subprocess.Popen(args.command)
    process = psutil.Process(child.pid)
    process.cpu_percent(interval=None)
    samples: list[dict[str, float | int]] = []
    with args.output.open("w", encoding="utf-8") as stream:
        while child.poll() is None:
            sample = {
                "elapsed_seconds": time.time() - started_at,
                "timestamp_unix": time.time(),
                **_gpu_sample(handle),
                **_process_sample(process),
            }
            samples.append(sample)
            stream.write(json.dumps(sample, sort_keys=True) + "\n")
            stream.flush()
            time.sleep(args.interval_seconds)
    return_code = child.wait()
    metadata.update(
        {
            "return_code": return_code,
            "wall_seconds": time.time() - started_at,
        }
    )
    summary_path.write_text(
        json.dumps(_summarize(samples, metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pynvml.nvmlShutdown()
    if return_code:
        raise SystemExit(return_code)
    print(summary_path)


if __name__ == "__main__":
    main()
