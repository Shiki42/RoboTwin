#!/usr/bin/env python3
"""Add closed-set joint-subtask targets to a real put-cabinet LeRobot dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

STAGE_IDS = (6, 9, 11, 15, 17, 18, 19, 20, 29, 31, 32, 34)
STAGE_TO_CLASS = {stage_id: index for index, stage_id in enumerate(STAGE_IDS)}
FEATURE = {"dtype": "int64", "names": ["joint_subtask_class"], "shape": [1]}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def annotation_targets(annotation: dict, lerobot_frame_count: int) -> tuple[list[int], dict[int, str]]:
    native_frame_count = int(annotation.get("frame_count", -1))
    if native_frame_count != lerobot_frame_count + 1:
        raise ValueError(
            f"annotation/native alignment requires {lerobot_frame_count + 1} frames, got {native_frame_count}"
        )
    targets = [-1] * native_frame_count
    texts: dict[int, str] = {}
    for segment in annotation["segments"]:
        start = int(segment["start"])
        end = int(segment["end"])
        stage_id = int(segment["stage_id"])
        if stage_id not in STAGE_TO_CLASS:
            raise ValueError(f"unregistered subtask stage id: {stage_id}")
        if not 0 <= start < end <= native_frame_count:
            raise ValueError(f"invalid subtask segment [{start}, {end})")
        if any(target != -1 for target in targets[start:end]):
            raise ValueError("subtask segments overlap")
        class_id = STAGE_TO_CLASS[stage_id]
        targets[start:end] = [class_id] * (end - start)
        text = str(segment["subtask_text"])
        if stage_id in texts and texts[stage_id] != text:
            raise ValueError(f"stage {stage_id} has inconsistent text")
        texts[stage_id] = text
    if any(target == -1 for target in targets):
        raise ValueError("subtask segments do not cover every frame")
    return targets[:lerobot_frame_count], texts


def update_huggingface_metadata(table: pa.Table) -> pa.Table:
    metadata = dict(table.schema.metadata or {})
    key = b"huggingface"
    if key in metadata:
        payload = json.loads(metadata[key])
        payload.setdefault("info", {}).setdefault("features", {})["observation.semantic_subtask_id"] = {
            "dtype": "int64",
            "_type": "Value",
        }
        metadata[key] = json.dumps(payload, sort_keys=True).encode()
    return table.replace_schema_metadata(metadata)


def scalar_stats(values: list[int]) -> dict[str, list[float | int]]:
    if not values:
        raise ValueError("cannot summarize an empty subtask target")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "min": [min(values)],
        "max": [max(values)],
        "mean": [mean],
        "std": [variance**0.5],
        "count": [len(values)],
    }


def update_episode_stats(output: Path, targets_by_episode: dict[int, list[int]]) -> Path:
    path = output / "meta/episodes_stats.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen = set()
    for row in rows:
        episode = int(row["episode_index"])
        if episode not in targets_by_episode:
            raise ValueError(f"episode stats references unknown episode {episode}")
        row.setdefault("stats", {})["observation.semantic_subtask_id"] = scalar_stats(targets_by_episode[episode])
        seen.add(episode)
    if seen != set(targets_by_episode):
        raise ValueError("episode stats do not cover every derived episode")
    part = path.with_suffix(".jsonl.part")
    part.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    part.replace(path)
    return path


def derived_inventory(
    source_receipt: dict,
    output: Path,
    changed: dict[str, str],
) -> list[dict[str, str | int]]:
    source_inventory = {row["path"]: row for row in source_receipt["inventory"]}
    output_paths = {
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file() and path.name != "parallelvla_dataset_receipt.json" and not path.name.endswith(".part")
    }
    if output_paths != set(source_inventory):
        missing = sorted(set(source_inventory) - output_paths)
        extra = sorted(output_paths - set(source_inventory))
        raise ValueError(f"derived inventory path mismatch: missing={missing[:5]} extra={extra[:5]}")
    inventory = []
    for relative, source_row in sorted(source_inventory.items()):
        path = output / relative
        size = path.stat().st_size
        if relative not in changed and size != int(source_row["bytes"]):
            raise ValueError(f"unchanged hardlink size mismatch: {relative}")
        inventory.append(
            {
                "path": relative,
                "bytes": size,
                "sha256": changed.get(relative, source_row["sha256"]),
            }
        )
    return inventory


def build(
    source: Path,
    annotations: Path,
    output: Path,
    *,
    source_revision: str,
    annotation_revision: str,
    producer_commit: str,
    require_all_classes: bool = False,
    allow_synthetic_fixture: bool = False,
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    source_receipt = source / "parallelvla_dataset_receipt.json"
    if not source_receipt.is_file() and not allow_synthetic_fixture:
        raise FileNotFoundError("real dataset receipt is required")
    source_manifest = json.loads(source_receipt.read_text()) if source_receipt.is_file() else {}
    if not allow_synthetic_fixture:
        if len(producer_commit) != 40 or any(character not in "0123456789abcdef" for character in producer_commit):
            raise ValueError("producer commit must be a full lowercase Git SHA")
        if source_manifest.get("data_revision") != source_revision:
            raise ValueError("source dataset revision does not match its receipt")
        if source_manifest.get("source_supervision_revision") != annotation_revision:
            raise ValueError("annotation revision does not match the source dataset receipt")
    parquet_files = sorted((source / "data").rglob("episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("source has no episode parquet files")
    shutil.copytree(source, output, copy_function=os.link)
    text_by_stage: dict[int, str] = {}
    class_counts = dict.fromkeys(range(len(STAGE_IDS)), 0)
    targets_by_episode: dict[int, list[int]] = {}
    changed: dict[str, str] = {}
    modified: list[dict] = []
    total_rows = 0
    try:
        for source_parquet in parquet_files:
            relative = source_parquet.relative_to(source)
            target_parquet = output / relative
            table = pq.read_table(source_parquet)
            episodes = set(table.column("episode_index").to_pylist())
            if len(episodes) != 1:
                raise ValueError(f"{relative} does not contain exactly one episode")
            episode = int(episodes.pop())
            annotation_path = annotations / f"episode{episode}.json"
            annotation = json.loads(annotation_path.read_text())
            if int(annotation["episode"]) != episode:
                raise ValueError("annotation episode does not match parquet")
            frames = table.column("frame_index").to_pylist()
            if frames != list(range(table.num_rows)):
                raise ValueError(f"{relative} frame indices are not contiguous")
            targets, texts = annotation_targets(annotation, table.num_rows)
            if episode in targets_by_episode:
                raise ValueError(f"duplicate episode index {episode}")
            targets_by_episode[episode] = targets
            for stage_id, text in texts.items():
                if stage_id in text_by_stage and text_by_stage[stage_id] != text:
                    raise ValueError(f"stage {stage_id} text changed across episodes")
                text_by_stage[stage_id] = text
            for class_id in targets:
                class_counts[class_id] += 1
            table = table.append_column("observation.semantic_subtask_id", pa.array(targets, type=pa.int64()))
            table = update_huggingface_metadata(table)
            part = target_parquet.with_suffix(".parquet.part")
            pq.write_table(table, part)
            part.replace(target_parquet)
            relative_string = str(relative)
            digest = sha256(target_parquet)
            changed[relative_string] = digest
            modified.append(
                {
                    "path": relative_string,
                    "rows": table.num_rows,
                    "sha256": digest,
                }
            )
            total_rows += table.num_rows
        if require_all_classes:
            missing_classes = [
                stage
                for stage, class_id in STAGE_TO_CLASS.items()
                if stage not in text_by_stage or class_counts[class_id] == 0
            ]
            if missing_classes:
                raise ValueError(f"real dataset is missing joint subtask stages: {missing_classes}")
        stats_path = update_episode_stats(output, targets_by_episode)
        stats_relative = str(stats_path.relative_to(output))
        changed[stats_relative] = sha256(stats_path)
        info_path = output / "meta/info.json"
        info = json.loads(info_path.read_text())
        info.setdefault("features", {})["observation.semantic_subtask_id"] = FEATURE
        info_part = info_path.with_suffix(".json.part")
        info_part.write_text(json.dumps(info, indent=4, sort_keys=True) + "\n")
        info_part.replace(info_path)
        info_relative = str(info_path.relative_to(output))
        changed[info_relative] = sha256(info_path)
        inventory = derived_inventory(source_manifest, output, changed)
        inventory_payload = json.dumps(inventory, separators=(",", ":"), sort_keys=True).encode()
        data_revision = hashlib.sha256(inventory_payload).hexdigest()
        manifest = {
            "schema": "parallelvla.putcab_subtask_aux_dataset.v1",
            "data_revision": data_revision,
            "producer_commit": producer_commit,
            "require_all_classes": require_all_classes,
            "source_dataset": str(source),
            "source_revision": source_revision,
            "source_receipt_sha256": sha256(source_receipt) if source_receipt.is_file() else None,
            "annotation_root": str(annotations),
            "annotation_revision": annotation_revision,
            "output_dataset": str(output),
            "episodes": len(parquet_files),
            "rows": total_rows,
            "stage_to_class": {str(stage): class_id for stage, class_id in STAGE_TO_CLASS.items()},
            "class_to_text": {str(STAGE_TO_CLASS[stage]): text_by_stage.get(stage) for stage in STAGE_IDS},
            "class_counts": {str(key): value for key, value in class_counts.items()},
            "modified_parquet": modified,
            "inventory": inventory,
            "file_count": len(inventory),
            "total_bytes": sum(int(row["bytes"]) for row in inventory),
            "info_sha256": changed[info_relative],
            "episodes_stats_sha256": changed[stats_relative],
            "allow_synthetic_fixture": allow_synthetic_fixture,
            "alignment": "lerobot_row_state_t_with_native_action_t_plus_1; terminal_native_target_excluded",
        }
        receipt_part = output / "parallelvla_dataset_receipt.json.part"
        receipt_part.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        receipt_part.replace(output / "parallelvla_dataset_receipt.json")
        return manifest
    except BaseException:
        shutil.rmtree(output)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--annotation-revision", required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--require-all-classes", action="store_true")
    parser.add_argument("--allow-synthetic-fixture", action="store_true")
    args = parser.parse_args()
    manifest = build(
        args.source,
        args.annotations,
        args.output,
        source_revision=args.source_revision,
        annotation_revision=args.annotation_revision,
        producer_commit=args.producer_commit,
        require_all_classes=args.require_all_classes,
        allow_synthetic_fixture=args.allow_synthetic_fixture,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
