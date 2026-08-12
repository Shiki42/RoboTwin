import hashlib
import json
from pathlib import Path

import pytest
import torch

from openpi.training.subtask_dataset import AnnotatedSubtaskDataset
from openpi.training.subtask_dataset import SubtaskAnnotationIndex

TEXT_FORMAT = "Left arm: <semantic>; Right arm: <semantic>."
SCHEMA = "parallelvla.putcab_pi05_subtask_supervision.v1"


def episode_record(
    episode: int,
    object_arm: str = "left",
    *,
    split: str = "train",
) -> dict:
    return {
        "episode": episode,
        "source_episode": episode,
        "split": split,
        "frame_count": 5,
        "object_arm": object_arm,
        "drawer_arm": "right" if object_arm == "left" else "left",
        "relative_delay_target": "physical",
        "rewritten_relative_delay": None,
        "segments": [
            {
                "start": 0,
                "end": 2,
                "stage_id": 7,
                "stage_key": "left_reach_grasp_object__right_reach_open_drawer",
                "arm_semantic_keys": {
                    "left": "reach_grasp_object",
                    "right": "reach_open_drawer",
                },
                "subtask_text": ("Left arm: reach and grasp object; " "Right arm: reach and open drawer."),
            },
            {
                "start": 2,
                "end": 5,
                "stage_id": 23,
                "stage_key": ("left_insert_place_object__right_wait_hold_drawer_open"),
                "arm_semantic_keys": {
                    "left": "insert_place_object",
                    "right": "wait_hold_drawer_open",
                },
                "subtask_text": ("Left arm: insert and place object; " "Right arm: wait while holding drawer open."),
            },
        ],
    }


def write_annotations(path: Path) -> str:
    records = [
        episode_record(0),
        episode_record(1, object_arm="right", split="heldout"),
    ]
    path.mkdir()
    manifest = {
        "schema_version": SCHEMA,
        "task": "put_object_cabinet",
        "text_format": TEXT_FORMAT,
        "supervision_contract": "fixture",
        "source_annotation_revision": "f" * 64,
        "sources": [],
        "episode_counts": {"train": 1, "heldout": 1},
        "frame_count": 10,
        "paired_text_frame_counts": {},
        "episodes": records,
    }
    encoded = json.dumps(
        manifest,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    revision = hashlib.sha256(encoded).hexdigest()
    manifest["data_revision"] = revision
    (path / "manifest.json").write_text(json.dumps(manifest))
    for record in records:
        (path / f"episode{record['episode']}.json").write_text(json.dumps(record))
    return revision


class FixtureDataset:
    def __init__(self):
        self.samples = [
            {"episode_index": 0, "frame_index": 0},
            {"episode_index": 0, "frame_index": 2},
        ]
        self.episode_data_index = {
            "from": torch.tensor([0, 4]),
            "to": torch.tensor([4, 8]),
        }

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


def test_annotation_index_looks_up_and_dataset_adds_fields(tmp_path: Path):
    root = tmp_path / "annotations"
    revision = write_annotations(root)
    index = SubtaskAnnotationIndex(
        root,
        expected_revision=revision,
        expected_text_format=TEXT_FORMAT,
    )
    dataset = FixtureDataset()
    index.validate_dataset_coverage(dataset, [0], expected_split="train")
    index.validate_dataset_coverage(dataset, [1], expected_split="heldout")
    annotated = AnnotatedSubtaskDataset(dataset, index)

    assert annotated[0]["subtask_stage_id"] == 7
    assert annotated[1]["subtask_stage_id"] == 23
    assert annotated[1]["subtask"].startswith("Left arm: insert")
    assert len(annotated) == 2


def test_annotation_index_requires_exact_episodes_and_lengths(tmp_path: Path):
    root = tmp_path / "annotations"
    revision = write_annotations(root)
    index = SubtaskAnnotationIndex(
        root,
        expected_revision=revision,
        expected_text_format=TEXT_FORMAT,
    )
    dataset = FixtureDataset()

    with pytest.raises(ValueError, match="exactly match"):
        index.validate_dataset_coverage(dataset, [0, 1], expected_split="train")
    dataset.episode_data_index["to"][0] = 5
    with pytest.raises(ValueError, match="native annotation length minus one"):
        index.validate_dataset_coverage(dataset, [0], expected_split="train")


def test_annotation_index_rejects_revision_and_manifest_drift(tmp_path: Path):
    root = tmp_path / "annotations"
    revision = write_annotations(root)
    with pytest.raises(ValueError, match="revision"):
        SubtaskAnnotationIndex(
            root,
            expected_revision="4" * 64,
            expected_text_format=TEXT_FORMAT,
        )

    episode_path = root / "episode0.json"
    record = json.loads(episode_path.read_text())
    record["frame_count"] = 6
    episode_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="differs from manifest"):
        SubtaskAnnotationIndex(
            root,
            expected_revision=revision,
            expected_text_format=TEXT_FORMAT,
        )


def test_annotation_index_requires_train_and_heldout_splits(tmp_path: Path):
    root = tmp_path / "annotations"
    revision = write_annotations(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["episode_counts"] = {"train": 2}
    manifest.pop("data_revision")
    revision = hashlib.sha256(
        json.dumps(
            manifest,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifest["data_revision"] = revision
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="train and heldout"):
        SubtaskAnnotationIndex(
            root,
            expected_revision=revision,
            expected_text_format=TEXT_FORMAT,
        )


def test_annotation_lookup_rejects_native_last_frame(tmp_path: Path):
    root = tmp_path / "annotations"
    revision = write_annotations(root)
    index = SubtaskAnnotationIndex(
        root,
        expected_revision=revision,
        expected_text_format=TEXT_FORMAT,
    )

    with pytest.raises(IndexError, match="outside annotation coverage"):
        index.lookup(0, 5)
    with pytest.raises(KeyError, match="not annotated"):
        index.lookup(99, 0)
