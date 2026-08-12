"""Strict per-frame subtask annotations for PI0.5 training."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Protocol, SupportsIndex

SCHEMA_VERSION = "parallelvla.putcab_pi05_subtask_supervision.v1"
TASK = "put_object_cabinet"


class RandomAccessDataset(Protocol):
    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]: ...
    def __len__(self) -> int: ...


@dataclasses.dataclass(frozen=True)
class FrameSubtask:
    stage_id: int
    stage_key: str
    text: str
    object_arm: str


@dataclasses.dataclass(frozen=True)
class EpisodeAnnotations:
    episode: int
    split: Literal["train", "heldout"]
    frame_count: int
    object_arm: str
    segments: tuple[dict[str, Any], ...]

    def lookup(self, frame: int) -> FrameSubtask:
        if not 0 <= frame < self.frame_count:
            raise IndexError(f"episode {self.episode} frame {frame} is outside annotation coverage")
        for segment in self.segments:
            if int(segment["start"]) <= frame < int(segment["end"]):
                return FrameSubtask(
                    stage_id=int(segment["stage_id"]),
                    stage_key=str(segment["stage_key"]),
                    text=str(segment["subtask_text"]),
                    object_arm=self.object_arm,
                )
        raise ValueError(f"episode {self.episode} frame {frame} has no subtask annotation")


class SubtaskAnnotationIndex:
    def __init__(
        self,
        annotation_dir: str | Path,
        *,
        expected_revision: str,
        expected_text_format: str,
    ):
        self.annotation_dir = Path(annotation_dir)
        manifest_path = self.annotation_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        self._validate_manifest(
            manifest,
            expected_revision=expected_revision,
            expected_text_format=expected_text_format,
        )
        self.revision = str(manifest["data_revision"])
        self._manifest_episode_counts = manifest["episode_counts"]
        self._episodes = self._load_episodes(manifest["episodes"])

    @staticmethod
    def _validate_manifest(
        manifest: dict[str, Any],
        *,
        expected_revision: str,
        expected_text_format: str,
    ) -> None:
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("subtask annotation schema does not match")
        if manifest.get("task") != TASK:
            raise ValueError("subtask annotation task does not match")
        recorded_revision = manifest.get("data_revision")
        if recorded_revision != expected_revision:
            raise ValueError("subtask annotation revision does not match")
        revision_payload = dict(manifest)
        revision_payload.pop("data_revision")
        encoded = json.dumps(
            revision_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        if hashlib.sha256(encoded).hexdigest() != recorded_revision:
            raise ValueError("subtask annotation manifest content hash does not match")
        if manifest.get("text_format") != expected_text_format:
            raise ValueError("subtask annotation text format does not match")
        counts = manifest.get("episode_counts")
        if not isinstance(counts, dict) or set(counts) != {"train", "heldout"}:
            raise ValueError("subtask annotation manifest must record train and heldout counts")
        if any(not isinstance(value, int) or value < 1 for value in counts.values()):
            raise ValueError("subtask annotation split counts must be positive")

    def _load_episodes(
        self,
        manifest_records: list[dict[str, Any]],
    ) -> dict[int, EpisodeAnnotations]:
        episodes = {}
        for manifest_record in manifest_records:
            episode = int(manifest_record["episode"])
            path = self.annotation_dir / f"episode{episode}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            record = json.loads(path.read_text())
            if record != manifest_record:
                raise ValueError(f"episode {episode} annotation differs from manifest")
            split = str(record["split"])
            if split not in {"train", "heldout"}:
                raise ValueError(f"episode {episode} has an invalid annotation split")
            episodes[episode] = EpisodeAnnotations(
                episode=episode,
                split=split,
                frame_count=int(record["frame_count"]),
                object_arm=str(record["object_arm"]),
                segments=tuple(record["segments"]),
            )
        if len(episodes) != len(manifest_records):
            raise ValueError("duplicate episode in subtask annotation manifest")
        expected_counts = {
            split: sum(record.split == split for record in episodes.values()) for split in ("train", "heldout")
        }
        manifest_counts = {split: int(count) for split, count in self._manifest_episode_counts.items()}
        if manifest_counts != expected_counts:
            raise ValueError("subtask annotation episode counts do not match records")
        return episodes

    def validate_dataset_coverage(
        self,
        dataset: Any,
        episode_indices: Sequence[int],
        *,
        expected_split: Literal["train", "heldout"],
    ) -> None:
        selected = tuple(int(episode) for episode in episode_indices)
        annotated = {episode for episode, record in self._episodes.items() if record.split == expected_split}
        if set(selected) != annotated:
            raise ValueError(f"LeRobot {expected_split} episodes do not exactly match subtask annotations")
        index = dataset.episode_data_index
        if set(index) != {"from", "to"}:
            raise ValueError("unexpected LeRobot episode data index keys")
        for episode in selected:
            start = int(index["from"][episode])
            end = int(index["to"][episode])
            expected = self._episodes[episode].frame_count - 1
            if end - start != expected:
                raise ValueError(
                    f"episode {episode} LeRobot length {end - start} != "
                    f"native annotation length minus one ({expected})"
                )

    def lookup(self, episode: int, frame: int) -> FrameSubtask:
        try:
            annotation = self._episodes[int(episode)]
        except KeyError as error:
            raise KeyError(f"episode {episode} is not annotated") from error
        return annotation.lookup(int(frame))


class AnnotatedSubtaskDataset:
    def __init__(
        self,
        dataset: RandomAccessDataset,
        annotations: SubtaskAnnotationIndex,
    ):
        self._dataset = dataset
        self._annotations = annotations

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        sample = self._dataset[index]
        subtask = self._annotations.lookup(
            int(sample["episode_index"]),
            int(sample["frame_index"]),
        )
        return {
            **sample,
            "subtask": subtask.text,
            "subtask_stage_id": subtask.stage_id,
            "subtask_stage_key": subtask.stage_key,
            "subtask_object_arm": subtask.object_arm,
        }

    def __len__(self) -> int:
        return len(self._dataset)
