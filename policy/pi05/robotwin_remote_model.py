from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys

import numpy as np

from robotwin_image_transport import ROBOTWIN_POLICY_PROTOCOL
from robotwin_image_transport import encode_images

_CLIENT_SRC = Path(__file__).parent / "packages" / "openpi-client" / "src"


def _client_policy(host: str, port: int):
    sys.path.insert(0, str(_CLIENT_SRC))
    client_module = importlib.import_module(
        "openpi_client.websocket_client_policy"
    )

    return client_module.WebsocketClientPolicy(host=host, port=port)


class RobotwinRemoteModel:
    """RoboTwin-side proxy for a PI0 model hosted in its Python 3.11 environment."""

    def __init__(self, host: str, port: int):
        self._client = _client_policy(host, port)
        metadata = self._client.get_server_metadata()
        if metadata.get("protocol") != ROBOTWIN_POLICY_PROTOCOL:
            raise ValueError(
                "policy server protocol mismatch: "
                f"{metadata.get('protocol')!r} != {ROBOTWIN_POLICY_PROTOCOL!r}"
            )
        self.observation_window = None
        self.base_instruction = None
        self._pending_action = None
        self._pending_state = None
        self._execution_steps = 0
        self.inference_index = 0
        self._episode_action_hasher = None

    def set_language(self, instruction):
        self.base_instruction = instruction

    def update_observation_window(self, img_arr, state, *, action_executed=False):
        observation = {
            "images": encode_images(img_arr),
            "state": np.asarray(state),
        }
        if action_executed:
            if self._pending_action is None or self._pending_state is None:
                raise RuntimeError("record_action must precede an executed observation update")
            self._client.infer(
                {
                    "command": "observe",
                    "action": self._pending_action,
                    "previous_state": self._pending_state,
                }
            )
            self._pending_action = None
            self._pending_state = None
        self.observation_window = observation

    def get_action(self):
        if self.observation_window is None or self.base_instruction is None:
            raise RuntimeError("language and observation must be set before inference")
        response = self._client.infer(
            {
                "command": "infer",
                **self.observation_window,
                "instruction": self.base_instruction,
            }
        )
        actions = np.asarray(response["actions"])
        self._execution_steps = len(actions)
        if self._episode_action_hasher is not None:
            contiguous = np.ascontiguousarray(actions)
            self._episode_action_hasher.update(contiguous.dtype.str.encode("ascii"))
            self._episode_action_hasher.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            self._episode_action_hasher.update(contiguous.tobytes())
            self.inference_index += 1
        return actions

    def execution_steps(self):
        return self._execution_steps

    def record_chunk(self, length):
        if length != self._execution_steps:
            raise ValueError(f"client executed {length} actions, server returned {self._execution_steps}")

    def record_action(self, action, state):
        self._pending_action = np.asarray(action)
        self._pending_state = np.asarray(state)

    def rollout_metrics(self):
        return self._client.infer({"command": "metrics"})["metrics"]

    def episode_action_sha256(self):
        if self._episode_action_hasher is None:
            return None
        return self._episode_action_hasher.hexdigest()

    def set_episode_seed(self, seed):
        self._client.infer({"command": "seed", "seed": int(seed)})
        self.inference_index = 0
        self._episode_action_hasher = hashlib.sha256()

    def reset_obsrvationwindows(self):
        self._client.infer({"command": "reset"})
        self.observation_window = None
        self.base_instruction = None
        self._pending_action = None
        self._pending_state = None
        self._execution_steps = 0
