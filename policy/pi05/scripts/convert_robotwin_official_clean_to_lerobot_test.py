import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from scripts import convert_robotwin_official_clean_to_lerobot as converter


def write_synthetic_episode(root: Path, index: int, *, frames: int = 3) -> None:
    data_dir = root / "data"
    instruction_dir = root / "instructions"
    data_dir.mkdir(parents=True, exist_ok=True)
    instruction_dir.mkdir(parents=True, exist_ok=True)
    joint = np.arange(frames * 14, dtype=np.float32).reshape(frames, 14) / 10
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[..., 0] = 11
    image[..., 1] = 22
    image[..., 2] = 33
    success, encoded = cv2.imencode(".jpg", image)
    assert success

    with h5py.File(data_dir / f"episode{index}.hdf5", "w") as episode:
        action = episode.create_group("joint_action")
        action.create_dataset("left_arm", data=joint[:, :6])
        action.create_dataset("left_gripper", data=joint[:, 6])
        action.create_dataset("right_arm", data=joint[:, 7:13])
        action.create_dataset("right_gripper", data=joint[:, 13])
        action.create_dataset("vector", data=joint)
        observation = episode.create_group("observation")
        for camera in converter.CAMERA_MAP.values():
            camera_group = observation.create_group(camera)
            camera_group.create_dataset(
                "rgb",
                data=[encoded.tobytes()] * frames,
                dtype=f"S{len(encoded)}",
            )

    payload = {
        "seen": [f"seen prompt {prompt_index}" for prompt_index in range(100)],
        "unseen": [f"unseen prompt {prompt_index}" for prompt_index in range(100)],
    }
    (instruction_dir / f"episode{index}.json").write_text(json.dumps(payload))


def test_discover_episodes_requires_explicit_synthetic_fixture(tmp_path):
    write_synthetic_episode(tmp_path, 0)
    write_synthetic_episode(tmp_path, 1)

    with pytest.raises(ValueError, match="allow_synthetic_fixture=true"):
        converter.discover_episodes(
            tmp_path,
            expected_episodes=2,
            allow_synthetic_fixture=False,
        )

    episodes = converter.discover_episodes(
        tmp_path,
        expected_episodes=2,
        allow_synthetic_fixture=True,
    )
    assert [index for index, _, _ in episodes] == [0, 1]


def test_select_seen_instruction_is_seeded_and_never_uses_unseen(tmp_path):
    write_synthetic_episode(tmp_path, 0)
    instruction_path = tmp_path / "instructions" / "episode0.json"

    first = converter.select_seen_instruction(instruction_path, seed=42, index=0)
    second = converter.select_seen_instruction(instruction_path, seed=42, index=0)

    assert first == second
    assert first[0].startswith("seen prompt ")


def test_joint_alignment_and_image_decode(tmp_path):
    write_synthetic_episode(tmp_path, 0)
    hdf5_path = tmp_path / "data" / "episode0.hdf5"

    inspection = converter.inspect_episode(hdf5_path)
    assert inspection["raw_frames"] == 3
    assert inspection["training_frames"] == 2
    assert set(map(tuple, inspection["image_shapes_hwc"].values())) == {(8, 12, 3)}

    with h5py.File(hdf5_path) as episode:
        joint = converter.canonical_joint_vector(episode)
        image = converter.decode_source_image(episode["observation/head_camera/rgb"][0])
    assert joint.shape == (3, 14)
    assert np.allclose(joint[1] - joint[0], 1.4)
    assert np.allclose(image.mean(axis=(0, 1)), [11, 22, 33], atol=2)


def test_joint_vector_order_mismatch_is_rejected(tmp_path):
    write_synthetic_episode(tmp_path, 0)
    hdf5_path = tmp_path / "data" / "episode0.hdf5"
    with h5py.File(hdf5_path, "r+") as episode:
        episode["joint_action/vector"][0, 0] = -999

    with h5py.File(hdf5_path) as episode, pytest.raises(ValueError, match="does not match"):
        converter.canonical_joint_vector(episode)
