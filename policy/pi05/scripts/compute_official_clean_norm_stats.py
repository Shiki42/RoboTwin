"""Compute exact π0.5 normalizers from a converted official-clean dataset."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as parquet
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as config_module

JOINT_DELTA_MASK = np.asarray([True, True, True, True, True, True, False, True, True, True, True, True, True, False])


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_normalization_samples(
    states: np.ndarray,
    actions: np.ndarray,
    action_horizon: int,
    *,
    allow_synthetic_fixture: bool,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if states.shape != actions.shape or states.ndim != 2 or states.shape[1] != 14:
        raise ValueError(f"expected matching [frames, 14] state/action arrays, got {states.shape} and {actions.shape}")
    if len(states) < action_horizon and not allow_synthetic_fixture:
        raise ValueError("short episodes require allow_synthetic_fixture=true")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("normalization inputs contain non-finite values")

    action_samples = []
    for start_index, state in enumerate(states):
        chunk = actions[start_index : start_index + action_horizon].copy()
        chunk[:, JOINT_DELTA_MASK] -= state[JOINT_DELTA_MASK]
        action_samples.append(chunk)
    return states, np.concatenate(action_samples, axis=0)


def exact_stats(values: np.ndarray) -> tuple[normalize.NormStats, int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 14 or len(values) < 2:
        raise ValueError(f"expected at least two 14-D samples, got {values.shape}")
    stats = normalize.NormStats(
        mean=np.mean(values, axis=0),
        std=np.std(values, axis=0),
        q01=np.quantile(values, 0.01, axis=0),
        q99=np.quantile(values, 0.99, axis=0),
    )
    return stats, len(values)


def load_episode(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = parquet.read_table(path, columns=["observation.state", "action"])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    return states, actions


def dataset_inventory_sha256(dataset_root: Path) -> str:
    aggregate = hashlib.sha256()
    for path in sorted(item for item in dataset_root.rglob("*") if item.is_file()):
        relative_path = path.relative_to(dataset_root).as_posix()
        digest = sha256_path(path)
        aggregate.update(f"{relative_path}\t{path.stat().st_size}\t{digest}\n".encode())
    return aggregate.hexdigest()


def validate_dataset_receipt(dataset_root: Path, receipt_path: Path) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("format") != "robotwin2_official_clean_to_lerobot_v1":
        raise ValueError(f"unexpected dataset receipt format: {receipt.get('format')}")
    expected_root = Path(receipt["output"]["root"]).resolve()
    if dataset_root.resolve() != expected_root:
        raise ValueError(f"dataset root does not match receipt: {dataset_root.resolve()} != {expected_root}")
    inventory_sha256 = dataset_inventory_sha256(dataset_root)
    if inventory_sha256 != receipt["output"]["inventory_sha256"]:
        raise ValueError("dataset inventory SHA-256 does not match conversion receipt")
    return receipt


def compute(
    dataset_root: Path,
    dataset_receipt: Path,
    assets_base_dir: Path,
    receipt_path: Path,
    *,
    config_name: str = "pi05_putcab_official_clean50_full",
) -> None:
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite normalizer receipt: {receipt_path}")
    receipt = validate_dataset_receipt(dataset_root, dataset_receipt)
    train_config = dataclasses.replace(
        config_module.get_config(config_name),
        assets_base_dir=str(assets_base_dir),
    )
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.repo_id != receipt["output"]["repo_id"]:
        raise ValueError(f"config repo ID does not match receipt: {data_config.repo_id}")

    parquet_paths = sorted(dataset_root.glob("data/chunk-*/episode_*.parquet"))
    if len(parquet_paths) != receipt["output"]["episodes"]:
        raise ValueError(f"expected {receipt['output']['episodes']} episode parquet files, got {len(parquet_paths)}")
    state_samples = []
    action_samples = []
    episode_rows = []
    for episode_index, parquet_path in enumerate(parquet_paths):
        states, actions = load_episode(parquet_path)
        states, chunk_actions = episode_normalization_samples(
            states,
            actions,
            train_config.model.action_horizon,
            allow_synthetic_fixture=False,
        )
        state_samples.append(states)
        action_samples.append(chunk_actions)
        episode_rows.append(
            {
                "episode_index": episode_index,
                "frames": len(states),
                "valid_action_targets": len(chunk_actions),
                "parquet": parquet_path.relative_to(dataset_root).as_posix(),
                "parquet_sha256": sha256_path(parquet_path),
            }
        )

    all_states = np.concatenate(state_samples, axis=0)
    all_actions = np.concatenate(action_samples, axis=0)
    state_stats, state_count = exact_stats(all_states)
    action_stats, action_count = exact_stats(all_actions)
    norm_stats = {"state": state_stats, "actions": action_stats}
    output_dir = train_config.assets_dirs / data_config.repo_id
    output_path = output_dir / "norm_stats.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite normalizer: {output_path}")
    normalize.save(output_dir, norm_stats)

    normalizer_receipt = {
        "format": "robotwin2_official_clean_exact_normalizer_v1",
        "config_name": config_name,
        "dataset": {
            "root": str(dataset_root.resolve()),
            "repo_id": data_config.repo_id,
            "conversion_receipt": str(dataset_receipt.resolve()),
            "conversion_receipt_sha256": sha256_path(dataset_receipt),
            "inventory_sha256": receipt["output"]["inventory_sha256"],
        },
        "normalizer": {
            "path": str(output_path.resolve()),
            "sha256": sha256_path(output_path),
            "state_samples": state_count,
            "valid_action_targets": action_count,
            "action_horizon": train_config.model.action_horizon,
            "joint_action_semantics": "delta for dims 0:6 and 7:13; absolute grippers",
            "quantiles": "numpy.quantile exact concatenated valid targets",
        },
        "episodes": episode_rows,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(normalizer_receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    tyro.cli(compute)
