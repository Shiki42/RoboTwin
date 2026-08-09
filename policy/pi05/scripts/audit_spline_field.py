"""Audit B-spline action reconstruction directly from a native LeRobot parquet dataset."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from openpi import spline_field

ACTION_GROUPS = {
    "left_arm": slice(0, 6),
    "left_gripper": slice(6, 7),
    "right_arm": slice(7, 13),
    "right_gripper": slice(13, 14),
}


def _metrics(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "p50": float(np.percentile(flat, 50)),
        "p90": float(np.percentile(flat, 90)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(np.max(flat)),
        "rmse": float(np.sqrt(np.mean(np.square(flat)))),
    }


def _load_chunks(dataset_dir: pathlib.Path, control_horizon: int) -> tuple[np.ndarray, int]:
    parquet_files = sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no episode parquet files under {dataset_dir}")
    chunks = []
    frame_count = 0
    offsets = np.arange(control_horizon)[None, :]
    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file, columns=["action"])
        actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"expected action shape [frames, 14], got {actions.shape} in {parquet_file}")
        if not np.all(np.isfinite(actions)):
            raise ValueError(f"non-finite action in {parquet_file}")
        frame_count += actions.shape[0]
        indices = np.minimum(np.arange(actions.shape[0])[:, None] + offsets, actions.shape[0] - 1)
        chunks.append(actions[indices])
    return np.concatenate(chunks, axis=0), frame_count


def _load_action_quantiles(norm_stats_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(norm_stats_path.read_text(encoding="utf-8"))
    stats = payload["norm_stats"]["actions"]
    q01 = np.asarray(stats["q01"], dtype=np.float64)
    q99 = np.asarray(stats["q99"], dtype=np.float64)
    if q01.shape != (14,) or q99.shape != (14,):
        raise ValueError(f"expected 14D action quantiles, got {q01.shape} and {q99.shape}")
    return q01, q99


def _candidate_audit(
    normalized_chunks: np.ndarray,
    physical_chunks: np.ndarray,
    q01: np.ndarray,
    q99: np.ndarray,
    *,
    control_points: int,
    control_horizon: int,
    degree: int,
    regularization: float,
) -> dict[str, Any]:
    coefficients = spline_field.encode_actions(
        normalized_chunks,
        control_horizon=control_horizon,
        control_points=control_points,
        degree=degree,
        regularization=regularization,
    )
    decoded = spline_field.decode_actions(
        coefficients,
        control_horizon=control_horizon,
        control_points=control_points,
        degree=degree,
        regularization=regularization,
    )
    scale = q99 - q01 + 1e-6
    decoded_physical = (decoded.astype(np.float64) + 1.0) / 2.0 * scale + q01
    normalized_error = np.abs(decoded.astype(np.float64) - normalized_chunks)
    physical_error = np.abs(decoded_physical - physical_chunks)
    group_metrics = {}
    for name, action_slice in ACTION_GROUPS.items():
        group_metrics[name] = {
            "normalized_abs_error": _metrics(normalized_error[..., action_slice]),
            "physical_abs_error": _metrics(physical_error[..., action_slice]),
        }
    endpoint_error = normalized_error[..., [0, -1], :]
    basis, projector = spline_field.spline_matrices(
        control_horizon,
        control_points,
        degree,
        regularization,
    )
    return {
        "control_points": control_points,
        "compression_ratio": control_horizon / control_points,
        "finite": bool(np.isfinite(coefficients).all() and np.isfinite(decoded).all()),
        "basis_partition_max_error": float(np.max(np.abs(basis.sum(axis=1) - 1.0))),
        "projector_condition_number": float(np.linalg.cond(projector @ projector.T)),
        "all_dimensions": {
            "normalized_abs_error": _metrics(normalized_error),
            "physical_abs_error": _metrics(physical_error),
        },
        "endpoint_normalized_abs_error": _metrics(endpoint_error),
        "groups": group_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=pathlib.Path, required=True)
    parser.add_argument("--norm-stats", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--control-horizon", type=int, default=50)
    parser.add_argument("--control-points", type=int, nargs="+", default=[8, 12, 16, 24])
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--regularization", type=float, default=1e-6)
    args = parser.parse_args()

    physical_chunks, frame_count = _load_chunks(args.dataset_dir, args.control_horizon)
    q01, q99 = _load_action_quantiles(args.norm_stats)
    scale = q99 - q01 + 1e-6
    normalized_chunks = ((physical_chunks.astype(np.float64) - q01) / scale * 2.0 - 1.0).astype(np.float32)
    candidates = [
        _candidate_audit(
            normalized_chunks,
            physical_chunks,
            q01,
            q99,
            control_points=control_points,
            control_horizon=args.control_horizon,
            degree=args.degree,
            regularization=args.regularization,
        )
        for control_points in args.control_points
    ]
    receipt = {
        "schema_version": 1,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "norm_stats": str(args.norm_stats.resolve()),
        "episode_count": len(list((args.dataset_dir / "data").glob("chunk-*/episode_*.parquet"))),
        "frame_count": frame_count,
        "chunk_count": int(normalized_chunks.shape[0]),
        "control_horizon": args.control_horizon,
        "degree": args.degree,
        "regularization": args.regularization,
        "selected_control_points": 16,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
