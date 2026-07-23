#!/usr/bin/env python3
"""Audit a local LeRobot put-object-cabinet dataset before CASM training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq

REQUIRED_FEATURES = {
    "action": (14,),
    "observation.state": (14,),
    "observation.arm_active_mask": (2,),
    "observation.phase_one_hot": (2,),
    "observation.images.cam_high": None,
    "observation.images.cam_left_wrist": None,
    "observation.images.cam_right_wrist": None,
}
FIXED_ROLE_PATTERNS = (
    r"fixed arm roles?",
    r"fixed arm assignment",
    r"roles? never swap",
    r"left arm .* object.*right arm .* drawer",
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _feature_shape(feature: dict) -> tuple[int, ...]:
    return tuple(int(value) for value in feature["shape"])


def _as_matrix(column, width: int, name: str) -> np.ndarray:
    values = np.asarray(column.to_pylist())
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"{name} must have shape [frames, {width}], got {values.shape}")
    return values


def _validate_binary_one_hot(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.isin(values, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0/1 values")
    if not np.all(values.sum(axis=1) == 1):
        raise ValueError(f"{name} must be one-hot on every frame")


def _validate_binary_mask(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.isin(values, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0/1 values")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _boundary_counts(phase: np.ndarray, horizon: int = 50) -> tuple[int, int]:
    phase_id = np.argmax(phase, axis=1)
    starts = np.arange(len(phase_id))[:, None]
    offsets = np.arange(horizon)[None, :]
    indices = np.minimum(starts + offsets, len(phase_id) - 1)
    consistent = phase_id[indices] == phase_id[:, None]
    return int(consistent.sum()), int(consistent.size - consistent.sum())


def audit_dataset(
    dataset_dir: Path,
    *,
    expected_episodes: int,
    expected_repo: str,
    reject_fixed_roles: bool,
) -> dict:
    info_path = dataset_dir / "meta/info.json"
    episodes_path = dataset_dir / "meta/episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError("dataset must contain meta/info.json and meta/episodes.jsonl")

    info = _load_json(info_path)
    features = info.get("features", {})
    missing = sorted(REQUIRED_FEATURES - features.keys())
    if missing:
        raise ValueError(f"missing required dataset features: {missing}")
    for name, expected_shape in REQUIRED_FEATURES.items():
        if expected_shape is not None and _feature_shape(features[name]) != expected_shape:
            raise ValueError(f"{name} has shape {_feature_shape(features[name])}, expected {expected_shape}")

    episode_records = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    metadata_episodes = int(info["total_episodes"])
    if metadata_episodes != expected_episodes or len(episode_records) != expected_episodes:
        raise ValueError(
            f"expected {expected_episodes} episodes, info has {metadata_episodes}, "
            f"episodes.jsonl has {len(episode_records)}"
        )

    tagged_prompts = [
        {"episode_index": record["episode_index"], "task": task}
        for record in episode_records
        for task in record.get("tasks", [])
        if "[PHASE=" in task
    ]
    if tagged_prompts:
        raise ValueError(f"ground-truth phase tags leak through task prompts: {tagged_prompts[:3]}")

    parquet_files = sorted((dataset_dir / "data").glob("**/*.parquet"))
    if len(parquet_files) != expected_episodes:
        raise ValueError(f"expected {expected_episodes} parquet files, found {len(parquet_files)}")

    readme_path = dataset_dir / "README.md"
    readme = readme_path.read_text() if readme_path.is_file() else ""
    fixed_role_matches = [
        pattern for pattern in FIXED_ROLE_PATTERNS if re.search(pattern, readme, flags=re.IGNORECASE | re.DOTALL)
    ]
    if reject_fixed_roles and fixed_role_matches:
        raise ValueError(f"README declares fixed arm roles: {fixed_role_matches}")

    phase_counts = np.zeros(2, dtype=np.int64)
    arm_mask_counts = {"00": 0, "01": 0, "10": 0, "11": 0}
    phase_consistent_targets = 0
    cross_boundary_targets = 0
    total_frames = 0
    columns = [
        "action",
        "observation.state",
        "observation.arm_active_mask",
        "observation.phase_one_hot",
    ]
    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path, columns=columns)
        action = _as_matrix(table["action"], 14, "action")
        state = _as_matrix(table["observation.state"], 14, "observation.state")
        phase = _as_matrix(table["observation.phase_one_hot"], 2, "observation.phase_one_hot")
        arm_mask = _as_matrix(table["observation.arm_active_mask"], 2, "observation.arm_active_mask")
        if not np.isfinite(action).all() or not np.isfinite(state).all():
            raise ValueError(f"non-finite action/state values in {parquet_path}")
        _validate_binary_one_hot(phase, "observation.phase_one_hot")
        _validate_binary_mask(arm_mask, "observation.arm_active_mask")
        phase_counts += phase.astype(np.int64).sum(axis=0)
        consistent, crossing = _boundary_counts(phase)
        phase_consistent_targets += consistent
        cross_boundary_targets += crossing
        for row in arm_mask.astype(np.int8):
            arm_mask_counts[f"{row[0]}{row[1]}"] += 1
        total_frames += table.num_rows

    if (phase_counts == 0).any():
        raise ValueError(f"sync and async must both be present, got counts {phase_counts.tolist()}")
    if total_frames != int(info["total_frames"]):
        raise ValueError(f"frame count mismatch: parquet={total_frames}, info={info['total_frames']}")

    return {
        "status": "passed",
        "dataset_repo": expected_repo,
        "dataset_dir": str(dataset_dir.resolve()),
        "episodes": expected_episodes,
        "frames": total_frames,
        "phase_convention": {"sync": [1, 0], "async": [0, 1]},
        "phase_counts": {"sync": int(phase_counts[0]), "async": int(phase_counts[1])},
        "gate_positive_weight": float(phase_counts[0] / phase_counts[1]),
        "arm_active_mask_counts": arm_mask_counts,
        "action_horizon_boundary_mask": {
            "horizon": 50,
            "phase_consistent_targets": phase_consistent_targets,
            "cross_boundary_targets_masked": cross_boundary_targets,
            "masked_fraction": cross_boundary_targets / (phase_consistent_targets + cross_boundary_targets),
        },
        "readme_fixed_role_matches": fixed_role_matches,
        "phase_prompt_tag_matches": 0,
        "metadata_sha256": {
            "meta/info.json": _sha256(info_path),
            "meta/episodes.jsonl": _sha256(episodes_path),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--expected-repo", required=True)
    parser.add_argument("--expected-episodes", type=int, default=50)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--reject-fixed-roles", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    receipt = audit_dataset(
        args.dataset_dir,
        expected_episodes=args.expected_episodes,
        expected_repo=args.expected_repo,
        reject_fixed_roles=args.reject_fixed_roles,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
