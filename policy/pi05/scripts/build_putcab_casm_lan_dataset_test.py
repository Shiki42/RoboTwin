import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.models import casm_language
from scripts import build_putcab_casm_lan_dataset as builder


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    annotation_dir = tmp_path / "annotations"
    (source / "data/chunk-000").mkdir(parents=True)
    annotation_dir.mkdir()
    info = {
        "total_episodes": 2,
        "total_frames": 12,
        "total_tasks": 2,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "observation.phase_one_hot": {
                "dtype": "float32",
                "shape": [2],
                "names": ["sync", "async"],
            },
            "observation.arm_active_mask": {
                "dtype": "float32",
                "shape": [2],
                "names": ["left", "right"],
            },
        },
    }
    (source / "meta").mkdir()
    (source / "meta/info.json").write_text(json.dumps(info))
    _write_jsonl(
        source / "meta/tasks.jsonl",
        [{"task_index": index, "task": f"overall task {index}"} for index in range(2)],
    )
    _write_jsonl(
        source / "meta/episodes.jsonl",
        [
            {"episode_index": index, "tasks": [f"overall task {index}"], "length": 6}
            for index in range(2)
        ],
    )
    _write_jsonl(
        source / "meta/episodes_stats.jsonl",
        [
            {
                "episode_index": index,
                "stats": {
                    "task_index": {
                        "min": [index],
                        "max": [index],
                        "mean": [index],
                        "std": [0.0],
                        "count": [6],
                    }
                },
            }
            for index in range(2)
        ],
    )
    phases = [[0.0, 1.0]] * 4 + [[1.0, 0.0]] * 2
    for episode_index, object_arm in enumerate(("left", "right")):
        activity = (
            [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]]
            if object_arm == "left"
            else [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]]
        )
        table = pa.table(
            {
                "observation.phase_one_hot": pa.array(phases, type=pa.list_(pa.float32(), 2)),
                "observation.arm_active_mask": pa.array(activity, type=pa.list_(pa.float32(), 2)),
                "task_index": pa.array([episode_index] * 6, type=pa.int64()),
            }
        )
        pq.write_table(table, source / f"data/chunk-000/episode_{episode_index:06d}.parquet")
        (annotation_dir / f"episode{episode_index}.json").write_text(
            json.dumps(
                {
                    "accepted": True,
                    "quality": {"acceptable": True},
                    "gt_repair": {
                        "object_arm": object_arm,
                        "drawer_arm": "right" if object_arm == "left" else "left",
                        "warnings": [],
                    },
                }
            )
        )
    return source, annotation_dir


def test_semantic_mapping_and_prompts_are_explicit_for_both_arms():
    assert casm_language.semantic_subtask_id("left", stage_id=0) == 0
    assert casm_language.semantic_subtask_id("left", stage_id=5) == 5
    assert casm_language.semantic_subtask_id("right", stage_id=0) == 6
    assert casm_language.semantic_subtask_id("right", stage_id=5) == 11
    waiting_prompt = casm_language.format_action_prompt("put object away", 1)
    assert "Left arm: wait without moving while the drawer is opened" in waiting_prompt
    assert "Right arm: reach for the drawer handle" in waiting_prompt
    prompt = casm_language.format_action_prompt("put object away", 5)
    assert "Overall task: put object away" in prompt
    assert "Left arm: carry and place" in prompt
    assert "Right arm: hold the drawer open and wait" in prompt
    predicted = casm_language.semantic_id_from_prediction(
        0.9,
        0.8,
        np.array([0.01, 0.04, 0.90, 0.02, 0.01, 0.02]),
    )
    assert predicted == 8


def test_build_dataset_rewrites_tasks_and_adds_semantic_target(tmp_path):
    source, annotation_dir = _fixture(tmp_path)
    output = tmp_path / "casm_lan"
    receipt = builder.build_dataset(source, annotation_dir, output)

    assert receipt["semantic_counts"] == {index: 1 for index in range(12)}
    assert receipt["object_arm_episode_counts"] == {"left": 1, "right": 1}
    tasks = [json.loads(line) for line in (output / "meta/tasks.jsonl").read_text().splitlines()]
    assert len(tasks) == 12
    assert all("Left arm:" in row["task"] and "Right arm:" in row["task"] for row in tasks)
    first = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    second = pq.read_table(output / "data/chunk-000/episode_000001.parquet")
    assert first["observation.semantic_subtask_id"].to_pylist() == list(range(6))
    assert second["observation.semantic_subtask_id"].to_pylist() == list(range(6, 12))
    assert first["task_index"].to_pylist() == list(range(6))
    assert second["task_index"].to_pylist() == list(range(6, 12))
    assert (output / "CASM_LAN_DATASET_RECEIPT.json").is_file()
    assert (output / "CASM_LAN_SHA256SUMS.json").is_file()


def test_build_rejects_non_monotonic_phase(tmp_path):
    source, annotation_dir = _fixture(tmp_path)
    path = source / "data/chunk-000/episode_000000.parquet"
    table = pa.table(
        {
            "observation.phase_one_hot": pa.array(
                [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
                type=pa.list_(pa.float32(), 2),
            ),
            "observation.arm_active_mask": pa.array(
                [[0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]],
                type=pa.list_(pa.float32(), 2),
            ),
            "task_index": pa.array([0] * 6, type=pa.int64()),
        }
    )
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="one monotonic ASYNC-to-SYNC transition"):
        builder.build_dataset(source, annotation_dir, tmp_path / "bad")
