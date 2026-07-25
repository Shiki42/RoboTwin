from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np

SCHEMA = "parallelvla.robotwin_policy_rollout_episode.v1"
CAMERAS = {
    "head": "head_camera",
    "left_wrist": "left_camera",
    "right_wrist": "right_camera",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phase_spans(phase_ids: np.ndarray) -> list[dict[str, int | str]]:
    values = np.asarray(phase_ids, dtype=np.uint8)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("phase IDs must be a non-empty vector")
    if not np.all(np.isin(values, (0, 1))):
        raise ValueError("phase IDs must contain only sync=0 and async=1")
    spans = []
    start = 0
    for index in range(1, len(values) + 1):
        if index < len(values) and values[index] == values[start]:
            continue
        spans.append(
            {
                "phase": "async" if values[start] else "sync",
                "start_step": start,
                "end_step": index,
            }
        )
        start = index
    return spans


class FrameVideoWriter:
    def __init__(self, path: Path, frame: np.ndarray, fps: int) -> None:
        image = self._validate_frame(frame)
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_path = path.with_name(f".{path.stem}.{os.getpid()}.partial{path.suffix}")
        height, width = image.shape[:2]
        self.command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.temporary_path),
        ]
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if self.process.stdin is None or self.process.stderr is None:
            raise RuntimeError("ffmpeg pipes were not created")
        self.shape = image.shape
        self.write(image)

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> np.ndarray:
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"expected uint8 HWC RGB frame, got {image.shape} {image.dtype}")
        return np.ascontiguousarray(image)

    def write(self, frame: np.ndarray) -> None:
        image = self._validate_frame(frame)
        if image.shape != self.shape:
            raise ValueError(f"video frame shape changed from {self.shape} to {image.shape}")
        self.process.stdin.write(image.tobytes())

    def close(self) -> None:
        self.process.stdin.close()
        stderr = self.process.stderr.read()
        return_code = self.process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, self.command, stderr=stderr)
        self.temporary_path.replace(self.path)


class RolloutEpisodeRecorder:
    def __init__(
        self,
        root: str | Path,
        *,
        episode_index: int,
        seed: int,
        policy_seed: int,
        instruction: str,
        fps: int = 10,
        writer_type=FrameVideoWriter,
    ) -> None:
        if min(episode_index, seed, policy_seed) < 0 or fps <= 0:
            raise ValueError("episode identifiers and FPS must be non-negative")
        self.root = Path(root).resolve()
        self.episode_index = int(episode_index)
        self.seed = int(seed)
        self.policy_seed = int(policy_seed)
        self.instruction = str(instruction)
        self.fps = int(fps)
        self.writer_type = writer_type
        self.writers: dict[str, FrameVideoWriter] = {}
        self.qpos: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.phase_ids: list[int] = []
        self.async_probabilities: list[float] = []
        self.inference_indices: list[int] = []

    def record(
        self,
        observation: dict,
        action: np.ndarray,
        *,
        async_phase: bool,
        async_probability: float | None,
        inference_index: int,
    ) -> None:
        qpos = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        commanded = np.asarray(action, dtype=np.float32)
        if qpos.shape != (14,) or commanded.shape != (14,):
            raise ValueError(f"expected 14-DoF qpos/action, got {qpos.shape}/{commanded.shape}")
        probability = np.nan if async_probability is None else float(async_probability)
        if not np.isnan(probability) and not 0.0 <= probability <= 1.0:
            raise ValueError(f"invalid async probability: {probability}")
        for camera, observation_key in CAMERAS.items():
            frame = observation["observation"][observation_key]["rgb"]
            writer = self.writers.get(camera)
            path = self.root / "videos" / camera / f"episode{self.episode_index}.mp4"
            if writer is None:
                self.writers[camera] = self.writer_type(path, frame, self.fps)
            else:
                writer.write(frame)
        self.qpos.append(qpos.copy())
        self.actions.append(commanded.copy())
        self.phase_ids.append(1 if async_phase else 0)
        self.async_probabilities.append(probability)
        self.inference_indices.append(int(inference_index))

    def close(self, *, success: bool, collision: bool) -> dict:
        if not self.actions:
            raise ValueError("cannot close an empty rollout Episode")
        for writer in self.writers.values():
            writer.close()
        phase_ids = np.asarray(self.phase_ids, dtype=np.uint8)
        spans = phase_spans(phase_ids)
        data_path = self.root / "data" / f"episode{self.episode_index}.hdf5"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = data_path.with_suffix(".building.hdf5")
        with h5py.File(temporary_path, "w") as root:
            root.attrs["parallelvla_schema"] = SCHEMA
            root.attrs["parallelvla_episode_index"] = self.episode_index
            root.attrs["parallelvla_scene_seed"] = self.seed
            root.attrs["parallelvla_policy_seed"] = self.policy_seed
            root.attrs["parallelvla_instruction"] = self.instruction
            root.attrs["parallelvla_success"] = bool(success)
            root.attrs["parallelvla_collision"] = bool(collision)
            root.attrs["parallelvla_fps"] = self.fps
            root.attrs["parallelvla_phase_spans"] = json.dumps(spans, separators=(",", ":"))
            root.create_dataset("joint_action/vector", data=np.stack(self.qpos))
            root.create_dataset("policy_action/vector", data=np.stack(self.actions))
            root.create_dataset("observation/phase_type_id", data=phase_ids)
            root.create_dataset(
                "observation/async_probability",
                data=np.asarray(self.async_probabilities, dtype=np.float32),
            )
            root.create_dataset(
                "observation/policy_inference_index",
                data=np.asarray(self.inference_indices, dtype=np.int32),
            )
        temporary_path.replace(data_path)
        video_paths = {camera: self.root / "videos" / camera / f"episode{self.episode_index}.mp4" for camera in CAMERAS}
        receipt = {
            "schema": SCHEMA,
            "episode_index": self.episode_index,
            "seed": self.seed,
            "policy_seed": self.policy_seed,
            "frames": len(self.actions),
            "fps": self.fps,
            "success": bool(success),
            "collision": bool(collision),
            "phase_spans": spans,
            "sync_transition_step": next(
                (index for index, value in enumerate(phase_ids) if value == 0),
                None,
            ),
            "sha256": {
                "hdf5": _sha256(data_path),
                "videos": {camera: _sha256(path) for camera, path in video_paths.items()},
            },
        }
        receipt_path = self.root / "episodes" / f"episode{self.episode_index}.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt
