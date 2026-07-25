from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.process_data import load_hdf5, native_dynamic_camera_indices


def write_source(path: Path, frames: int = 4, *, internal_metadata: bool = False):
    with h5py.File(path, "w") as root:
        joint = root.create_group("joint_action")
        joint.create_dataset("left_gripper", data=np.zeros(frames))
        joint.create_dataset("left_arm", data=np.zeros((frames, 6)))
        joint.create_dataset("right_gripper", data=np.zeros(frames))
        joint.create_dataset("right_arm", data=np.zeros((frames, 6)))
        observation = root.create_group("observation")
        camera = observation.create_group("head_camera")
        camera.create_dataset("rgb", data=np.zeros((frames, 1), dtype=np.uint8))
        if internal_metadata:
            metadata = root.create_group("parallelvla")
            metadata.create_dataset("phase_type_id", data=np.arange(frames) % 2)
            metadata.create_dataset("arm_active_mask", data=np.ones((frames, 2)))


def test_load_hdf5_accepts_external_phase_metadata(tmp_path: Path):
    source = tmp_path / "episode.hdf5"
    metadata = tmp_path / "episode.npz"
    write_source(source)
    np.savez(
        metadata,
        phase_type_id=np.array([0, 1, 1, 0], dtype=np.int8),
        arm_active_mask=np.array([[1, 1], [1, 0], [0, 1], [1, 1]], dtype=np.uint8),
    )

    *_, phase, mask = load_hdf5(source, metadata)

    assert phase.tolist() == [0, 1, 1, 0]
    assert mask.tolist() == [[1, 1], [1, 0], [0, 1], [1, 1]]


def test_load_hdf5_rejects_ambiguous_internal_and_external_metadata(tmp_path: Path):
    source = tmp_path / "episode.hdf5"
    metadata = tmp_path / "episode.npz"
    write_source(source, internal_metadata=True)
    np.savez(metadata, phase_type_id=np.zeros(4), arm_active_mask=np.ones((4, 2)))

    with pytest.raises(ValueError, match="both internally and externally"):
        load_hdf5(source, metadata)


def test_native_dynamic_camera_indices_preserve_every_frame():
    indices = native_dynamic_camera_indices(4)

    assert indices.tolist() == [0, 1, 2, 3]


def test_native_dynamic_camera_indices_reject_empty_episode():
    with pytest.raises(ValueError, match="positive"):
        native_dynamic_camera_indices(0)
