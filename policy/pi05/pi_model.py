#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from openpi.policies.episode_rng import EPISODE_SEED_KEY
from openpi.policies.episode_rng import INFERENCE_INDEX_KEY
from openpi.training.robotwin_routing import EpisodeStartSceneContextRouter
from openpi.training.robotwin_routing import LearnedAsyncToSyncRouter
from openpi.training.robotwin_routing import RolloutActivityMetrics

_CLIENT_SRC = Path(__file__).parent / "packages" / "openpi-client" / "src"


def _client_policy(host, port):
    sys.path.insert(0, str(_CLIENT_SRC))
    from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: PLC0415

    return WebsocketClientPolicy(host=host, port=port)


def _array_receipt(value):
    array = np.ascontiguousarray(value)
    hasher = hashlib.sha256()
    hasher.update(array.dtype.str.encode("ascii"))
    hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    hasher.update(array.tobytes())
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hasher.hexdigest(),
    }


def _observation_receipt(observation):
    return {
        "state": _array_receipt(observation["state"]),
        "cam_high": _array_receipt(observation["images"]["cam_high"]),
        "cam_left_wrist": _array_receipt(observation["images"]["cam_left_wrist"]),
        "cam_right_wrist": _array_receipt(observation["images"]["cam_right_wrist"]),
        "prompt_sha256": hashlib.sha256(observation["prompt"].encode("utf-8")).hexdigest(),
    }


class PI0:
    def __init__(
        self,
        policy_server_host,
        policy_server_port,
        pi0_step,
        async_scene_context_steps=0,
        boundary_observation=False,  # noqa: FBT002
        *,
        visual_phase_gate=False,
        sync_action_chunk_steps=10,
        gate_sync_threshold=0.5,
        gate_sync_confirmations=2,
    ):
        self.policy = _client_policy(policy_server_host, int(policy_server_port))
        print("connected to policy server")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step
        self.boundary_observation = bool(boundary_observation)
        self.sync_action_chunk_steps = sync_action_chunk_steps
        self.learned_phase_routing = bool(visual_phase_gate)
        if self.learned_phase_routing:
            self.main_camera_router = LearnedAsyncToSyncRouter(
                sync_threshold=gate_sync_threshold,
                sync_confirmations=gate_sync_confirmations,
            )
        else:
            self.main_camera_router = EpisodeStartSceneContextRouter(async_scene_context_steps)
        self.episode_seed = None
        self.inference_index = 0
        self._episode_action_hasher = None
        self.audit_trace_path = os.environ.get("PARALLELVLA_PI05_AUDIT_TRACE")
        self.activity_metrics = RolloutActivityMetrics()
        self._reset_profile()

    def _reset_profile(self):
        self.profile_inference_s = 0.0
        self.profile_inference_calls = 0
        self.profile_action_s = 0.0
        self.profile_action_calls = 0

    def set_img_size(self, img_size):
        self.img_size = img_size

    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    def execution_steps(self):
        return self.main_camera_router.execution_steps(
            self.pi0_step,
            self.sync_action_chunk_steps,
        )

    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left = img_arr
        if not self.learned_phase_routing:
            img_front = self.main_camera_router.route(img_front)
        self.observation_window = {
            "state": state,
            "phase_id": np.asarray(
                [1 if self.main_camera_router.async_phase else 0],
                dtype=np.int32,
            ),
            "images": {
                "cam_high": np.transpose(img_front, (2, 0, 1)),
                "cam_left_wrist": np.transpose(img_left, (2, 0, 1)),
                "cam_right_wrist": np.transpose(img_right, (2, 0, 1)),
            },
            "prompt": self.instruction,
        }

    def advance_after_action(self):
        if not self.learned_phase_routing:
            self.main_camera_router.advance()

    def set_episode_seed(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("episode seed must be an integer")
        if seed < 0:
            raise ValueError("episode seed must be non-negative")
        self.episode_seed = int(seed)
        self.inference_index = 0
        self._episode_action_hasher = hashlib.sha256()

    def get_action(self):
        if self.observation_window is None:
            raise RuntimeError("update observation_window before inference")
        observation = dict(self.observation_window)
        if self.episode_seed is not None:
            observation[EPISODE_SEED_KEY] = self.episode_seed
            observation[INFERENCE_INDEX_KEY] = self.inference_index
        observation_receipt = _observation_receipt(observation)
        inference_started = time.perf_counter()
        response = self.policy.infer(observation)
        self.profile_inference_s += time.perf_counter() - inference_started
        self.profile_inference_calls += 1
        if self.learned_phase_routing:
            if "async_probability" not in response:
                raise ValueError("visual phase gate inference must return async_probability")
            self.main_camera_router.update(response["async_probability"])
        actions = response["actions"]
        action_receipt = _array_receipt(actions)
        if self.episode_seed is not None:
            contiguous = np.ascontiguousarray(actions)
            self._episode_action_hasher.update(contiguous.dtype.str.encode("ascii"))
            self._episode_action_hasher.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            self._episode_action_hasher.update(contiguous.tobytes())
            self.inference_index += 1
        if self.audit_trace_path:
            self._append_audit_trace(observation_receipt, action_receipt)
        return actions

    def _append_audit_trace(self, observation_receipt, action_receipt):
        trace_path = Path(self.audit_trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "episode_seed": self.episode_seed,
            "inference_index": self.inference_index - 1,
            "observation": observation_receipt,
            "actions": action_receipt,
        }
        with trace_path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(row, sort_keys=True) + "\n")
            trace_file.flush()
            os.fsync(trace_file.fileno())

    def record_chunk(self, length):
        self.activity_metrics.record_chunk(
            async_phase=self.main_camera_router.async_phase,
            length=length,
        )

    def record_action(self, action, state):
        self.activity_metrics.record_action(
            action,
            state,
            async_phase=self.main_camera_router.async_phase,
        )

    def episode_action_sha256(self):
        if self._episode_action_hasher is None:
            return None
        return self._episode_action_hasher.hexdigest()

    def rollout_metrics(self):
        metrics = self.activity_metrics.summary()
        if self.learned_phase_routing:
            metrics["phase_router"] = self.main_camera_router.summary()
        return metrics

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        self.main_camera_router.reset()
        self.activity_metrics.reset()
        self._reset_profile()
        print("successfully unset obs and language intruction")
