from pathlib import Path
from typing import ClassVar

import h5py
import numpy as np

from policy.pi05.episode_capture import RolloutEpisodeRecorder
from policy.pi05.episode_capture import phase_spans


class FakeWriter:
    instances: ClassVar[list["FakeWriter"]] = []

    def __init__(self, path: Path, frame: np.ndarray, fps: int) -> None:
        self.path = path
        self.frames = [np.asarray(frame).copy()]
        self.fps = fps
        FakeWriter.instances.append(self)

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(np.asarray(frame).copy())

    def close(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"fake mp4")


def observation(value: int) -> dict:
    frame = np.full((4, 6, 3), value, dtype=np.uint8)
    return {
        "joint_action": {"vector": np.full(14, value, dtype=np.float32)},
        "observation": {
            "head_camera": {"rgb": frame},
            "left_camera": {"rgb": frame},
            "right_camera": {"rgb": frame},
        },
    }


def test_phase_spans_preserve_monotonic_gate_transition() -> None:
    assert phase_spans(np.asarray([1, 1, 0, 0], dtype=np.uint8)) == [
        {"phase": "async", "start_step": 0, "end_step": 2},
        {"phase": "sync", "start_step": 2, "end_step": 4},
    ]


def test_rollout_episode_recorder_writes_frontend_dataset(tmp_path: Path) -> None:
    FakeWriter.instances.clear()
    recorder = RolloutEpisodeRecorder(
        tmp_path,
        episode_index=8,
        seed=100048,
        policy_seed=100048,
        instruction="put the object in the cabinet",
        writer_type=FakeWriter,
    )
    recorder.record(
        observation(1),
        np.ones(14, dtype=np.float32),
        async_phase=True,
        async_probability=0.8,
        inference_index=0,
    )
    recorder.record(
        observation(2),
        np.full(14, 2, dtype=np.float32),
        async_phase=False,
        async_probability=0.1,
        inference_index=1,
    )

    receipt = recorder.close(success=False, collision=True)

    assert receipt["sync_transition_step"] == 1
    assert receipt["frames"] == 2
    assert len(FakeWriter.instances) == 3
    assert all(len(writer.frames) == 2 for writer in FakeWriter.instances)
    with h5py.File(tmp_path / "data/episode8.hdf5", "r") as root:
        np.testing.assert_array_equal(root["observation/phase_type_id"][:], [1, 0])
        np.testing.assert_allclose(root["observation/async_probability"][:], [0.8, 0.1])
        assert root["joint_action/vector"].shape == (2, 14)
        assert root.attrs["parallelvla_schema"].endswith(".v1")
