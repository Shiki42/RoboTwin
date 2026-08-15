from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import build_putcab_subtask_aux_dataset as builder


def _write_fixture(tmp_path):
    source = tmp_path / "source"
    parquet_path = source / "data/chunk-000/episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([0, 0, 0], type=pa.int64()),
                "frame_index": pa.array([0, 1, 2], type=pa.int64()),
            }
        ),
        parquet_path,
    )
    (source / "meta").mkdir()
    (source / "meta/info.json").write_text(json.dumps({"features": {}}))
    (source / "meta/episodes_stats.jsonl").write_text(json.dumps({"episode_index": 0, "stats": {}}) + "\n")
    inventory = [
        {
            "path": str(path.relative_to(source)),
            "bytes": path.stat().st_size,
            "sha256": builder.sha256(path),
        }
        for path in sorted(source.rglob("*"))
        if path.is_file()
    ]
    (source / "parallelvla_dataset_receipt.json").write_text(json.dumps({"inventory": inventory}))

    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "episode0.json").write_text(
        json.dumps(
            {
                "episode": 0,
                "frame_count": 4,
                "segments": [
                    {
                        "start": 0,
                        "end": 2,
                        "stage_id": 6,
                        "subtask_text": "first",
                        "arm_semantic_keys": {"left": "reach_grasp_object", "right": "wait"},
                    },
                    {
                        "start": 2,
                        "end": 4,
                        "stage_id": 9,
                        "subtask_text": "second",
                        "arm_semantic_keys": {"left": "reach_grasp_object", "right": "reach_grasp_object"},
                    },
                ],
            }
        )
    )
    return source, annotations, parquet_path


def test_build_adds_dense_joint_subtask_target_without_mutating_source(tmp_path):
    source, annotations, source_parquet = _write_fixture(tmp_path)
    output = tmp_path / "derived"

    receipt = builder.build(
        source,
        annotations,
        output,
        source_revision="source-revision",
        annotation_revision="annotation-revision",
        producer_commit="allow_synthetic_fixture=true",
        allow_synthetic_fixture=True,
    )

    target = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    unchanged = pq.read_table(source_parquet)
    assert target["observation.semantic_subtask_id"].to_pylist() == [0, 0, 1]
    action_masks = np.asarray(target["observation.action_loss_mask"].to_pylist(), dtype=np.float32)
    assert action_masks.shape == (3, 50, 32)
    assert np.all(action_masks[0, :3, :7] == 1)
    assert np.all(action_masks[0, :, 7:] == 0)
    assert np.all(action_masks[2, 0, :14] == 1)
    assert np.all(action_masks[2, 1:] == 0)
    assert "observation.action_loss_mask" not in unchanged.column_names
    assert receipt["schema"] == "parallelvla.putcab_subtask_aux_dataset.v2"
    assert "observation.semantic_subtask_id" not in unchanged.column_names
    assert receipt["class_counts"]["0"] == 2
    assert receipt["class_counts"]["1"] == 1
    assert receipt["allow_synthetic_fixture"] is True
    assert json.loads((output / "parallelvla_dataset_receipt.json").read_text()) == receipt


def test_action_mask_excludes_artificial_delay():
    annotation = {
        "frame_count": 4,
        "rewritten_relative_delay": {
            "delayed_arm": "left",
            "start": 1,
            "end": 3,
            "physical_semantic_key": "wait",
            "supervision_semantic_key": "reach_grasp_object",
        },
        "segments": [
            {
                "start": 0,
                "end": 4,
                "stage_id": 6,
                "subtask_text": "first",
                "arm_semantic_keys": {"left": "reach_grasp_object", "right": "wait"},
            }
        ],
    }

    _, _, masks = builder.annotation_targets(annotation, lerobot_frame_count=3)

    assert np.all(masks[0, :2] == 0)
    assert np.all(masks[0, 2, :7] == 1)
    assert np.all(masks[0, 2, 7:] == 0)


def test_annotation_targets_rejects_gaps():
    annotation = {
        "frame_count": 3,
        "segments": [{"start": 0, "end": 1, "stage_id": 6, "subtask_text": "first"}],
    }

    with pytest.raises(ValueError, match="cover every frame"):
        builder.annotation_targets(annotation, lerobot_frame_count=2)
