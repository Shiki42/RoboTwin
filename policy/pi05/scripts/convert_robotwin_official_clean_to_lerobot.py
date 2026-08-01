"""Convert immutable RoboTwin 2.0 official clean data to LeRobot v2.1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

CAMERA_MAP = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}
MOTORS = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]
EPISODE_PATTERN = re.compile(r"episode(\d+)\.hdf5")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_index(path: Path) -> int:
    match = EPISODE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"invalid RoboTwin episode filename: {path.name}")
    return int(match.group(1))


def discover_episodes(
    raw_dir: Path,
    *,
    expected_episodes: int,
    allow_synthetic_fixture: bool,
) -> list[tuple[int, Path, Path]]:
    if expected_episodes != 50 and not allow_synthetic_fixture:
        raise ValueError("non-official episode count requires allow_synthetic_fixture=true")
    data_dir = raw_dir / "data"
    instruction_dir = raw_dir / "instructions"
    if not data_dir.is_dir() or not instruction_dir.is_dir():
        raise FileNotFoundError(f"missing data or instructions directory under {raw_dir}")

    hdf5_by_index = {episode_index(path): path for path in data_dir.glob("episode*.hdf5")}
    expected_indices = set(range(expected_episodes))
    if set(hdf5_by_index) != expected_indices:
        raise ValueError(f"episode indices do not match 0..{expected_episodes - 1}: " f"{sorted(hdf5_by_index)}")

    episodes = []
    for index in range(expected_episodes):
        instruction_path = instruction_dir / f"episode{index}.json"
        if not instruction_path.is_file():
            raise FileNotFoundError(f"missing instruction file: {instruction_path}")
        episodes.append((index, hdf5_by_index[index], instruction_path))
    return episodes


def select_seen_instruction(instruction_path: Path, *, seed: int, index: int) -> tuple[str, int]:
    payload = json.loads(instruction_path.read_text())
    if set(payload) != {"seen", "unseen"}:
        raise ValueError(f"unexpected instruction splits in {instruction_path}: {sorted(payload)}")
    seen = payload["seen"]
    unseen = payload["unseen"]
    if len(seen) != 100 or len(unseen) != 100:
        raise ValueError(f"expected 100 seen and 100 unseen prompts in {instruction_path}")
    if not all(isinstance(prompt, str) and prompt.strip() for prompt in seen + unseen):
        raise ValueError(f"empty or non-string prompt in {instruction_path}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    prompt_index = int(rng.integers(len(seen)))
    return seen[prompt_index].strip(), prompt_index


def decode_source_image(encoded: np.bytes_) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode source JPEG")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"invalid decoded image: shape={image.shape}, dtype={image.dtype}")
    # RoboTwin encoded SAPIEN RGB arrays with cv2.imencode. cv2.imdecode recovers
    # that original array ordering; applying BGR->RGB here would swap it.
    return image


def canonical_joint_vector(episode: h5py.File) -> np.ndarray:
    left_arm = np.asarray(episode["joint_action/left_arm"], dtype=np.float32)
    left_gripper = np.asarray(episode["joint_action/left_gripper"], dtype=np.float32)
    right_arm = np.asarray(episode["joint_action/right_arm"], dtype=np.float32)
    right_gripper = np.asarray(episode["joint_action/right_gripper"], dtype=np.float32)
    vector = np.asarray(episode["joint_action/vector"], dtype=np.float32)
    reconstructed = np.concatenate(
        [
            left_arm,
            left_gripper[:, None],
            right_arm,
            right_gripper[:, None],
        ],
        axis=1,
    )
    if reconstructed.shape != vector.shape or not np.allclose(reconstructed, vector, atol=1e-6):
        raise ValueError("joint_action/vector does not match left-gripper-right ordering")
    if vector.ndim != 2 or vector.shape[1] != 14 or vector.shape[0] < 2:
        raise ValueError(f"invalid joint vector shape: {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError("joint vector contains non-finite values")
    return vector


def inspect_episode(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as episode:
        vector = canonical_joint_vector(episode)
        frame_count = vector.shape[0]
        image_shapes = {}
        for source_camera in CAMERA_MAP.values():
            frames = episode[f"observation/{source_camera}/rgb"]
            if len(frames) != frame_count:
                raise ValueError(f"{path}: {source_camera} has {len(frames)} frames, expected {frame_count}")
            first_image = decode_source_image(frames[0])
            image_shapes[source_camera] = list(first_image.shape)
    return {
        "raw_frames": frame_count,
        "training_frames": frame_count - 1,
        "image_shapes_hwc": image_shapes,
        "joint_min": vector.min(axis=0).tolist(),
        "joint_max": vector.max(axis=0).tolist(),
    }


def create_dataset(repo_id: str, target: Path, *, image_shape: tuple[int, int, int]) -> LeRobotDataset:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite dataset target: {target}")
    height, width, channels = image_shape
    if channels != 3:
        raise ValueError(f"expected three-channel images, got {image_shape}")
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": [MOTORS],
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": [MOTORS],
        },
    }
    for destination_camera in CAMERA_MAP:
        features[f"observation.images.{destination_camera}"] = {
            "dtype": "video",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=target,
        fps=50,
        robot_type="aloha",
        features=features,
        use_videos=True,
        tolerance_s=1e-4,
        image_writer_processes=4,
        image_writer_threads=4,
        video_backend="pyav",
    )


def add_episode(
    dataset: LeRobotDataset,
    path: Path,
    *,
    prompt: str,
    expected_image_shape: tuple[int, int, int],
) -> int:
    with h5py.File(path, "r") as episode:
        vector = canonical_joint_vector(episode)
        source_cameras = {
            destination: episode[f"observation/{source}/rgb"] for destination, source in CAMERA_MAP.items()
        }
        training_frames = vector.shape[0] - 1
        for frame_index in range(training_frames):
            images = {
                destination: decode_source_image(frames[frame_index]) for destination, frames in source_cameras.items()
            }
            shapes = {image.shape for image in images.values()}
            if shapes != {expected_image_shape}:
                raise ValueError(f"{path}: camera shape mismatch at frame {frame_index}: {shapes}")
            dataset.add_frame(
                {
                    "observation.state": vector[frame_index],
                    "action": vector[frame_index + 1],
                    "observation.images.cam_high": images["cam_high"],
                    "observation.images.cam_left_wrist": images["cam_left_wrist"],
                    "observation.images.cam_right_wrist": images["cam_right_wrist"],
                    "task": prompt,
                }
            )
    dataset.save_episode()
    return training_frames


def inventory_tree(root: Path) -> tuple[list[dict[str, Any]], str]:
    inventory = []
    aggregate = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative_path = path.relative_to(root).as_posix()
        digest = sha256_path(path)
        row = {"path": relative_path, "bytes": path.stat().st_size, "sha256": digest}
        inventory.append(row)
        aggregate.update(f"{relative_path}\t{row['bytes']}\t{digest}\n".encode())
    return inventory, aggregate.hexdigest()


def convert(
    raw_dir: Path,
    output_root: Path,
    repo_id: str,
    receipt_path: Path,
    source_revision: str,
    source_archive_sha256: str,
    *,
    seed: int = 42,
    expected_episodes: int = 50,
    allow_synthetic_fixture: bool = False,
) -> None:
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {receipt_path}")
    if len(source_revision) != 40:
        raise ValueError("source_revision must be an immutable 40-character revision")
    if not re.fullmatch(r"[0-9a-f]{64}", source_archive_sha256):
        raise ValueError("source_archive_sha256 must be a lowercase SHA-256")

    episodes = discover_episodes(
        raw_dir,
        expected_episodes=expected_episodes,
        allow_synthetic_fixture=allow_synthetic_fixture,
    )
    inspections = [inspect_episode(path) for _, path, _ in episodes]
    image_shapes = {tuple(shape) for inspection in inspections for shape in inspection["image_shapes_hwc"].values()}
    if len(image_shapes) != 1:
        raise ValueError(f"source camera shapes are inconsistent: {sorted(image_shapes)}")
    image_shape = next(iter(image_shapes))

    target = output_root / repo_id
    dataset = create_dataset(repo_id, target, image_shape=image_shape)
    episode_receipts = []
    for (index, hdf5_path, instruction_path), inspection in zip(episodes, inspections, strict=True):
        prompt, prompt_index = select_seen_instruction(instruction_path, seed=seed, index=index)
        training_frames = add_episode(
            dataset,
            hdf5_path,
            prompt=prompt,
            expected_image_shape=image_shape,
        )
        if training_frames != inspection["training_frames"]:
            raise RuntimeError(f"frame count changed during conversion for episode {index}")
        episode_receipts.append(
            {
                "episode_index": index,
                "source_hdf5": hdf5_path.relative_to(raw_dir).as_posix(),
                "source_hdf5_sha256": sha256_path(hdf5_path),
                "instruction_file": instruction_path.relative_to(raw_dir).as_posix(),
                "instruction_sha256": sha256_path(instruction_path),
                "selected_split": "seen",
                "selected_prompt_index": prompt_index,
                "selected_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                **inspection,
            }
        )

    output_inventory, output_inventory_sha256 = inventory_tree(target)
    receipt = {
        "format": "robotwin2_official_clean_to_lerobot_v1",
        "source": {
            "dataset": "TianxingChen/RoboTwin2.0",
            "revision": source_revision,
            "subpath": "dataset/put_object_cabinet/aloha-agilex_clean_50.zip",
            "archive_sha256": source_archive_sha256,
        },
        "conversion": {
            "seed": seed,
            "prompt_policy": "one deterministic seen prompt per episode",
            "alignment": "state[t], native_rgb[t] -> joint_position_action[t+1]",
            "camera_map": CAMERA_MAP,
            "fps": 50,
            "expected_episodes": expected_episodes,
            "allow_synthetic_fixture": allow_synthetic_fixture,
            "source_image_encoding": "SAPIEN RGB encoded and decoded through OpenCV without channel swap",
        },
        "output": {
            "repo_id": repo_id,
            "root": str(target.resolve()),
            "episodes": expected_episodes,
            "frames": sum(row["training_frames"] for row in episode_receipts),
            "files": len(output_inventory),
            "bytes": sum(row["bytes"] for row in output_inventory),
            "inventory_sha256": output_inventory_sha256,
            "inventory": output_inventory,
        },
        "episodes": episode_receipts,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    tyro.cli(convert)
