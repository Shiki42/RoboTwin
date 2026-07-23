"""Compute PI0.5 putcab normalization statistics without JAX or video decoding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq

ACTION_DIM = 14
DELTA_ACTION_DIMS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)


def _episode_action_chunks(
    states: np.ndarray,
    actions: np.ndarray,
    action_horizon: int,
    *,
    allow_synthetic_fixture: bool = False,
) -> np.ndarray:
    del allow_synthetic_fixture
    if states.shape != actions.shape or states.ndim != 2 or states.shape[1] != ACTION_DIM:
        raise ValueError(f"expected matching [frames, {ACTION_DIM}] state/action arrays")
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    indices = np.minimum(
        np.arange(len(actions))[:, None] + np.arange(action_horizon)[None, :],
        len(actions) - 1,
    )
    chunks = actions[indices].astype(np.float64)
    chunks[..., DELTA_ACTION_DIMS] -= states[:, None, DELTA_ACTION_DIMS]
    return chunks


def _statistics(values: np.ndarray) -> dict[str, list[float]]:
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("statistics require at least two vectors")
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("normalization input contains non-finite values")
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _fixed_size_column(path: Path, name: str) -> np.ndarray:
    column = pq.read_table(path, columns=[name]).column(name).combine_chunks()
    values = np.asarray(column.values.to_numpy(zero_copy_only=False))
    width = column.type.list_size
    result = values.reshape(len(column), width)
    if result.shape[1] != ACTION_DIM:
        raise ValueError(f"{path}: {name} has width {result.shape[1]}")
    return result


def _scalar_column(path: Path, name: str) -> np.ndarray:
    column = pq.read_table(path, columns=[name]).column(name).combine_chunks()
    return np.asarray(column.to_numpy(zero_copy_only=False))


def _sha256_inventory(paths: Iterable[Path], root: Path) -> tuple[list[dict[str, object]], str]:
    inventory = []
    aggregate = hashlib.sha256()
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        inventory.append({"path": relative, "size": size, "sha256": digest})
        aggregate.update(f"{digest}\t{size}\t{relative}\n".encode())
    return inventory, aggregate.hexdigest()


def compute_norm_stats(
    dataset_root: Path,
    action_horizon: int,
    *,
    allow_synthetic_fixture: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    parquet_paths = sorted(dataset_root.glob("data/chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"no parquet episodes under {dataset_root}")
    if not allow_synthetic_fixture and len(parquet_paths) != 50:
        raise ValueError(f"expected 50 real episodes, found {len(parquet_paths)}")

    all_states = []
    all_action_chunks = []
    episode_lengths = []
    for expected_episode, path in enumerate(parquet_paths):
        states = _fixed_size_column(path, "observation.state")
        actions = _fixed_size_column(path, "action")
        episode_indices = _scalar_column(path, "episode_index")
        frame_indices = _scalar_column(path, "frame_index")
        if not np.all(episode_indices == expected_episode):
            raise ValueError(f"{path}: episode_index mismatch")
        if not np.array_equal(frame_indices, np.arange(len(states))):
            raise ValueError(f"{path}: frame_index is not contiguous")
        all_states.append(states.astype(np.float64))
        all_action_chunks.append(
            _episode_action_chunks(
                states,
                actions,
                action_horizon,
                allow_synthetic_fixture=allow_synthetic_fixture,
            ).reshape(-1, ACTION_DIM)
        )
        episode_lengths.append(len(states))

    states = np.concatenate(all_states)
    action_chunks = np.concatenate(all_action_chunks)
    norm_stats = {
        "norm_stats": {
            "state": _statistics(states),
            "actions": _statistics(action_chunks),
        }
    }
    inventory, inventory_sha256 = _sha256_inventory(parquet_paths, dataset_root)
    receipt = {
        "status": "passed",
        "implementation": "pytorch_arrow_full_frame_v1",
        "dataset_root": str(dataset_root.resolve()),
        "episode_count": len(parquet_paths),
        "episode_lengths": episode_lengths,
        "frame_count": len(states),
        "action_horizon": action_horizon,
        "action_vector_count": len(action_chunks),
        "action_dim": ACTION_DIM,
        "delta_action_dims": list(DELTA_ACTION_DIMS),
        "episode_end_padding": "repeat_last_action",
        "parquet_inventory_sha256": inventory_sha256,
        "parquet_inventory": inventory,
    }
    return norm_stats, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--action-horizon", type=int, default=50)
    args = parser.parse_args()

    norm_stats, receipt = compute_norm_stats(args.dataset_root, args.action_horizon)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "norm_stats.json"
    output_text = json.dumps(norm_stats, indent=2, sort_keys=True) + "\n"
    output_path.write_text(output_text)
    receipt["dataset_revision"] = args.dataset_revision
    receipt["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt["norm_stats_sha256"] = hashlib.sha256(output_text.encode()).hexdigest()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output_path}")
    print(f"wrote {args.receipt}")


if __name__ == "__main__":
    main()
