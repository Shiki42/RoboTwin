#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

from pathlib import Path

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training.robotwin_routing import EpisodePhaseRouter
from openpi.training.robotwin_routing import RolloutActivityMetrics


def checkpoint_asset_id(assets_dir: str | Path) -> str:
    assets_path = Path(assets_dir)
    norm_stats_paths = sorted(assets_path.rglob("norm_stats.json"))
    if len(norm_stats_paths) != 1:
        raise ValueError(f"expected exactly one norm_stats.json under {assets_path}, " f"found {len(norm_stats_paths)}")
    return norm_stats_paths[0].parent.relative_to(assets_path).as_posix()


class PI0:
    def __init__(
        self,
        train_config_name,
        checkpoint_dir,
        pi0_step,
        *,
        async_phase_steps=0,
        sync_action_chunk_steps=10,
        phase_prompt_conditioning=True,
        boundary_phase_steps=20,
    ):
        self.train_config_name = train_config_name
        checkpoint_dir = Path(checkpoint_dir).resolve()
        assets_id = checkpoint_asset_id(checkpoint_dir / "assets")

        config = _config.get_config(self.train_config_name)
        self.policy = _policy_config.create_trained_policy(
            config,
            checkpoint_dir,
            robotwin_repo_id=assets_id,
        )
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step
        self.sync_action_chunk_steps = sync_action_chunk_steps
        self.phase_prompt_conditioning = phase_prompt_conditioning
        self.phase_router = EpisodePhaseRouter(
            async_phase_steps,
            boundary_phase_steps,
        )
        self.base_instruction = None
        self.activity_metrics = RolloutActivityMetrics()

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.base_instruction = instruction
        print(f"successfully set instruction:{self.phase_conditioned_instruction}")

    @property
    def phase_conditioned_instruction(self):
        if self.base_instruction is None:
            return None
        if not self.phase_prompt_conditioning:
            return self.base_instruction
        return self.phase_router.phase_conditioned_prompt(self.base_instruction)

    def execution_steps(self):
        return self.phase_router.execution_steps(self.pi0_step, self.sync_action_chunk_steps)

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state, *, action_executed=False):
        img_front, img_right, img_left = img_arr
        if action_executed:
            self.phase_router.advance()
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "phase_id": np.asarray(
                [1 if self.phase_router.async_phase else 0],
                dtype=np.int32,
            ),
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.phase_conditioned_instruction,
        }

    def record_chunk(self, length):
        self.activity_metrics.record_chunk(
            async_phase=self.phase_router.async_phase,
            length=length,
        )

    def record_action(self, action, state):
        self.activity_metrics.record_action(
            action,
            state,
            async_phase=self.phase_router.async_phase,
        )

    def rollout_metrics(self):
        return self.activity_metrics.summary()

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def set_episode_seed(self, seed):
        self.policy.set_episode_seed(seed)

    def reset_obsrvationwindows(self):
        self.base_instruction = None
        self.observation_window = None
        self.phase_router.reset()
        self.activity_metrics.reset()
        print("successfully unset obs and language intruction")
