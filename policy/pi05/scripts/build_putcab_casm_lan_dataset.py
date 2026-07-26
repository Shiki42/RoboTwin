#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.models import casm_language


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path, *, excluded: set[str] | None = None) -> list[dict]:
    excluded = excluded or set()
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return rows


def _inventory_digest(rows: list[dict]) -> str:
    payload = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _clean_producer_commit() -> str:
    repository = Path(__file__).resolve().parents[3]
    status = subprocess.run(
        ["git", "-C", repository, "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("dataset conversion requires a clean committed worktree")
    return subprocess.run(
        ["git", "-C", repository, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _copy_with_hardlinks(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"target already exists: {target}")
    shutil.copytree(source, target, copy_function=os.link)


def _validated_object_arm(annotation_path: Path) -> str:
    annotation = json.loads(annotation_path.read_text())
    if not annotation.get("accepted") or not annotation.get("quality", {}).get("acceptable"):
        raise ValueError(f"annotation is not accepted: {annotation_path}")
    repair = annotation.get("gt_repair", {})
    if repair.get("warnings"):
        raise ValueError(f"annotation has unresolved GT repair warnings: {annotation_path}")
    object_arm = repair.get("object_arm")
    drawer_arm = repair.get("drawer_arm")
    if {object_arm, drawer_arm} != {"left", "right"}:
        raise ValueError(f"invalid object/drawer roles in {annotation_path}")
    return object_arm


def _phase_ids(table: pa.Table) -> np.ndarray:
    phase = np.asarray(table["observation.phase_one_hot"].to_pylist(), dtype=np.float32)
    if phase.ndim != 2 or phase.shape[1] != 2:
        raise ValueError(f"phase feature must have shape [frames, 2], got {phase.shape}")
    if not np.all(np.isin(phase, (0.0, 1.0))) or not np.all(phase.sum(axis=1) == 1.0):
        raise ValueError("phase feature must be binary one-hot")
    phase_ids = np.argmax(phase, axis=1).astype(np.int64)
    transitions = np.flatnonzero(np.diff(phase_ids) != 0)
    if phase_ids[0] != 1 or phase_ids[-1] != 0 or len(transitions) != 1:
        raise ValueError("CASM-LAN expects one monotonic ASYNC-to-SYNC transition")
    return phase_ids


def _semantic_stage_ids(table: pa.Table, phase_ids: np.ndarray, object_arm: str) -> np.ndarray:
    activity = np.asarray(table["observation.arm_active_mask"].to_pylist(), dtype=np.float32)
    if activity.shape != (len(phase_ids), 2):
        raise ValueError(f"arm activity must have shape [frames, 2], got {activity.shape}")
    if not np.all(np.isin(activity, (0.0, 1.0))):
        raise ValueError("arm activity must be binary")
    object_index = 0 if object_arm == "left" else 1
    drawer_index = 1 - object_index
    stage_ids = []
    object_acquisition_started = False
    for frame_index, (phase_id, arms) in enumerate(zip(phase_ids, activity, strict=True)):
        active = tuple(int(value) for value in arms)
        if phase_id == 1:
            stage_id = _async_stage_id(
                active,
                object_index,
                drawer_index,
                object_acquisition_started=object_acquisition_started,
                frame_index=frame_index,
            )
            object_acquisition_started |= stage_id == 2
        else:
            stage_id = _sync_stage_id(active, object_index, drawer_index, frame_index)
        stage_ids.append(stage_id)
    stage_ids = np.asarray(stage_ids, dtype=np.int64)
    if np.any(np.diff(stage_ids) < 0):
        raise ValueError("semantic stages must be monotonic within an episode")
    return stage_ids


def _async_stage_id(
    active: tuple[int, int],
    object_index: int,
    drawer_index: int,
    *,
    object_acquisition_started: bool,
    frame_index: int,
) -> int:
    if active == (0, 0):
        return 4 if object_acquisition_started else 0
    drawer_only = tuple(int(index == drawer_index) for index in range(2))
    if active == drawer_only:
        return 3 if object_acquisition_started else 1
    object_only = tuple(int(index == object_index) for index in range(2))
    if active in {(1, 1), object_only}:
        return 2
    raise ValueError(f"invalid async arm activity {active} at frame {frame_index}")


def _sync_stage_id(active: tuple[int, int], object_index: int, drawer_index: int, frame_index: int) -> int:
    del drawer_index
    if active == (0, 0):
        return 4
    object_only = tuple(int(index == object_index) for index in range(2))
    if active in {(1, 1), object_only}:
        return 5
    raise ValueError(f"invalid sync arm activity {active} at frame {frame_index}")


def _scalar_stats(values: np.ndarray) -> dict[str, list[float | int]]:
    values = np.asarray(values)
    return {
        "min": [values.min().item()],
        "max": [values.max().item()],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [int(values.size)],
    }


def _replace_columns(table: pa.Table, semantic_ids: np.ndarray, task_indices: np.ndarray) -> pa.Table:
    task_column = table.schema.get_field_index("task_index")
    table = table.set_column(task_column, "task_index", pa.array(task_indices, type=pa.int64()))
    return table.append_column(
        "observation.semantic_subtask_id",
        pa.array(semantic_ids, type=pa.int64()),
    )


def _episode_path(root: Path, episode_index: int) -> Path:
    info = json.loads((root / "meta/info.json").read_text())
    relative = info["data_path"].format(
        episode_chunk=episode_index // int(info["chunks_size"]),
        episode_index=episode_index,
    )
    return root / relative


def _build_episode(
    staging: Path,
    source: Path,
    annotation_dir: Path,
    tasks_by_index: dict[int, str],
    episode_index: int,
) -> tuple[list[str], dict[int, int], str]:
    object_arm = _validated_object_arm(annotation_dir / f"episode{episode_index}.json")
    source_path = _episode_path(source, episode_index)
    target_path = _episode_path(staging, episode_index)
    table = pq.read_table(source_path)
    phase_ids = _phase_ids(table)
    stage_ids = _semantic_stage_ids(table, phase_ids, object_arm)
    source_task_indices = np.asarray(table["task_index"].to_numpy(), dtype=np.int64)
    if np.unique(source_task_indices).size != 1:
        raise ValueError(f"episode {episode_index} must have exactly one overall task")
    overall_task = tasks_by_index[int(source_task_indices[0])]
    semantic_ids = np.asarray(
        [casm_language.semantic_subtask_id(object_arm, stage_id=int(stage)) for stage in stage_ids],
        dtype=np.int64,
    )
    task_indices = episode_index * casm_language.SEMANTIC_STAGE_COUNT + stage_ids
    prompts = [
        casm_language.format_action_prompt(
            overall_task,
            casm_language.semantic_subtask_id(object_arm, stage_id=stage_id),
        )
        for stage_id in range(casm_language.SEMANTIC_STAGE_COUNT)
    ]
    updated = _replace_columns(table, semantic_ids, task_indices)
    temporary = target_path.with_suffix(".parquet.part")
    target_path.unlink()
    pq.write_table(updated, temporary)
    temporary.replace(target_path)
    semantic_ids_unique, semantic_id_counts = np.unique(semantic_ids, return_counts=True)
    counts = {
        int(key): int(value)
        for key, value in zip(semantic_ids_unique, semantic_id_counts, strict=True)
    }
    return prompts, counts, object_arm


def _update_metadata(
    staging: Path,
    tasks: list[dict],
    episodes: list[dict],
    episode_stats: list[dict],
) -> None:
    _atomic_jsonl(staging / "meta/tasks.jsonl", tasks)
    _atomic_jsonl(staging / "meta/episodes.jsonl", episodes)
    _atomic_jsonl(staging / "meta/episodes_stats.jsonl", episode_stats)
    info_path = staging / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["total_tasks"] = len(tasks)
    info["features"]["observation.semantic_subtask_id"] = {
        "dtype": "int64",
        "shape": [1],
        "names": ["dual_arm_semantic_subtask"],
    }
    _atomic_json(info_path, info)


def build_dataset(
    source: Path,
    annotation_dir: Path,
    output: Path,
    *,
    producer_commit: str = "allow_synthetic_fixture=true",
) -> dict:
    source = source.resolve()
    annotation_dir = annotation_dir.resolve()
    output = output.resolve()
    staging = output.with_name(output.name + ".part")
    if output.exists() or staging.exists():
        raise FileExistsError(f"output or staging path already exists: {output}")
    _copy_with_hardlinks(source, staging)

    source_tasks = _read_jsonl(source / "meta/tasks.jsonl")
    tasks_by_index = {int(row["task_index"]): row["task"] for row in source_tasks}
    source_episodes = _read_jsonl(source / "meta/episodes.jsonl")
    episode_stats = _read_jsonl(source / "meta/episodes_stats.jsonl")
    if len(source_episodes) != len(episode_stats):
        raise ValueError("episode metadata and stats length mismatch")

    generated_tasks = []
    semantic_counts = {index: 0 for index in range(casm_language.SEMANTIC_SUBTASK_COUNT)}
    role_counts = {"left": 0, "right": 0}
    updated_episodes = []
    updated_stats = []
    for episode_index, (episode, stats) in enumerate(zip(source_episodes, episode_stats, strict=True)):
        if int(episode["episode_index"]) != episode_index or int(stats["episode_index"]) != episode_index:
            raise ValueError("episode metadata must be contiguous and ordered")
        prompts, counts, object_arm = _build_episode(
            staging,
            source,
            annotation_dir,
            tasks_by_index,
            episode_index,
        )
        role_counts[object_arm] += 1
        for semantic_id, count in counts.items():
            semantic_counts[semantic_id] += count
        generated_tasks.extend(
            {
                "task_index": episode_index * casm_language.SEMANTIC_STAGE_COUNT + offset,
                "task": prompt,
            }
            for offset, prompt in enumerate(prompts)
        )
        updated_episodes.append({**episode, "tasks": prompts})
        table = pq.read_table(
            _episode_path(staging, episode_index),
            columns=["task_index", "observation.semantic_subtask_id"],
        )
        task_values = np.asarray(table["task_index"].to_numpy(), dtype=np.int64)
        semantic_values = np.asarray(table["observation.semantic_subtask_id"].to_numpy(), dtype=np.int64)
        stats["stats"]["task_index"] = _scalar_stats(task_values)
        stats["stats"]["observation.semantic_subtask_id"] = _scalar_stats(semantic_values)
        updated_stats.append(stats)

    if len(source_episodes) == 50 and role_counts != {"left": 25, "right": 25}:
        raise ValueError(f"object-arm roles must be balanced 25/25, got {role_counts}")
    _update_metadata(staging, generated_tasks, updated_episodes, updated_stats)

    receipt_name = "CASM_LAN_DATASET_RECEIPT.json"
    manifest_name = "CASM_LAN_SHA256SUMS.json"
    inventory = _inventory(staging, excluded={receipt_name, manifest_name})
    source_inventory = _inventory(source)
    annotation_inventory = _inventory(annotation_dir)
    manifest = {
        "files": inventory,
        "file_count": len(inventory),
        "total_bytes": sum(row["bytes"] for row in inventory),
    }
    _atomic_json(staging / manifest_name, manifest)
    receipt = {
        "format": "robotwin_put_object_cabinet_casm_lan_v2",
        "source": str(source),
        "source_inventory_sha256": _inventory_digest(source_inventory),
        "annotation_dir": str(annotation_dir),
        "annotation_inventory_sha256": _inventory_digest(annotation_inventory),
        "producer_code_commit": producer_commit,
        "episodes": len(source_episodes),
        "frames": sum(semantic_counts.values()),
        "semantic_subtask_count": casm_language.SEMANTIC_SUBTASK_COUNT,
        "semantic_counts": semantic_counts,
        "object_arm_episode_counts": role_counts,
        "mllm_then_gt_contract": (
            "accepted Qwen2.5-VL stage/role annotation repaired by GT phase and proprioception"
        ),
        "manifest_sha256": _sha256(staging / manifest_name),
    }
    _atomic_json(staging / receipt_name, receipt)
    staging.replace(output)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_dataset(
        args.source,
        args.annotation_dir,
        args.output,
        producer_commit=_clean_producer_commit(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
