from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
import sys

import numpy as np
import tyro

from openpi.serving import websocket_policy_server

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from pi_model import PI0  # noqa: E402


class RobotwinPolicyService:
    """Stateful RPC adapter around PI0 for native RoboTwin rollouts."""

    def __init__(self, model: PI0):
        self._model = model

    def infer(self, request):
        command = request["command"]
        if command == "reset":
            self._model.reset_obsrvationwindows()
            return {"ok": True}
        if command == "metrics":
            return {"metrics": self._model.rollout_metrics()}
        if command == "observe":
            self._model.record_action(request["action"], request["previous_state"])
            self._update_observation(request, action_executed=True)
            return {"ok": True}
        if command == "infer":
            if self._model.observation_window is None:
                self._model.set_language(request["instruction"])
            self._update_observation(request, action_executed=False)
            actions = self._model.get_action()[: self._model.execution_steps()]
            self._model.record_chunk(len(actions))
            return {"actions": np.asarray(actions)}
        raise ValueError(f"unknown command: {command}")

    def _update_observation(self, request, *, action_executed: bool):
        self._model.update_observation_window(
            request["images"],
            request["state"],
            action_executed=action_executed,
        )


@dataclasses.dataclass
class Args:
    train_config_name: str
    model_name: str
    checkpoint_id: int
    async_scene_context_steps: int
    port: int = 8000
    pi0_step: int = 50
    sync_action_chunk_steps: int = 10
    phase_prompt_conditioning: bool = True
    boundary_context_steps: int = 20


def main(args: Args):
    model = PI0(
        args.train_config_name,
        args.model_name,
        args.checkpoint_id,
        args.pi0_step,
        async_scene_context_steps=args.async_scene_context_steps,
        sync_action_chunk_steps=args.sync_action_chunk_steps,
        phase_prompt_conditioning=args.phase_prompt_conditioning,
        boundary_context_steps=args.boundary_context_steps,
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=RobotwinPolicyService(model),
        host="0.0.0.0",
        port=args.port,
        metadata={"protocol": "robotwin_pi0_v1"},
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
