#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

from pathlib import Path

import numpy as np

from openpi.models import casm
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training.robotwin_routing import EpisodeStartSceneContextRouter
from openpi.training.robotwin_routing import LearnedAsyncToSyncRouter
from openpi.training.robotwin_routing import RolloutActivityMetrics


def checkpoint_asset_id(assets_dir: str | Path) -> str:
    assets_path = Path(assets_dir)
    norm_stats_paths = sorted(assets_path.rglob("norm_stats.json"))
    if len(norm_stats_paths) != 1:
        raise ValueError(f"expected exactly one norm_stats.json under {assets_path}, " f"found {len(norm_stats_paths)}")
    return norm_stats_paths[0].parent.relative_to(assets_path).as_posix()


def uses_learned_phase_router(casm_mode: str) -> bool:
    return casm_mode in casm.VISUAL_PHASE_GATE_MODES


class PI0:
    def __init__(
        self,
        train_config_name,
        checkpoint_dir,
        pi0_step,
        *,
        async_scene_context_steps=0,
        sync_action_chunk_steps=10,
        boundary_context_steps=20,
        gate_sync_threshold=0.5,
        gate_sync_confirmations=2,
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
        self.learned_phase_routing = uses_learned_phase_router(config.model.casm_mode)
        self.semantic_subtask_prediction = config.model.semantic_subtask_prediction
        if self.learned_phase_routing:
            self.main_camera_router = LearnedAsyncToSyncRouter(
                sync_threshold=gate_sync_threshold,
                sync_confirmations=gate_sync_confirmations,
            )
        else:
            self.main_camera_router = EpisodeStartSceneContextRouter(
                async_scene_context_steps,
                boundary_context_steps,
            )
        self.base_instruction = None
        self.activity_metrics = RolloutActivityMetrics()
        self.semantic_subtask_history = []

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.base_instruction = instruction
        print(f"successfully set instruction:{self.base_instruction}")

    def execution_steps(self):
        return self.main_camera_router.execution_steps(self.pi0_step, self.sync_action_chunk_steps)

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state, *, action_executed=False):
        img_front, img_right, img_left = img_arr
        if action_executed and not self.learned_phase_routing:
            self.main_camera_router.advance()
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "phase_id": np.asarray(
                [1 if self.main_camera_router.async_phase else 0],
                dtype=np.int32,
            ),
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.base_instruction,
        }

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

    def rollout_metrics(self):
        metrics = self.activity_metrics.summary()
        if self.learned_phase_routing:
            metrics["phase_router"] = self.main_camera_router.summary()
        if self.semantic_subtask_prediction:
            metrics["semantic_subtask_history"] = self.semantic_subtask_history.copy()
        return metrics

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        outputs = self.policy.infer(self.observation_window)
        if self.semantic_subtask_prediction:
            self.semantic_subtask_history.append(
                {
                    "semantic_subtask_id": int(outputs["semantic_subtask_id"]),
                    "semantic_subtask_prompt": outputs["semantic_subtask_prompt"],
                    "object_arm_right_probability": float(outputs["semantic_object_arm_right_probability"]),
                    "stage_probabilities": outputs["semantic_stage_probabilities"],
                    "confirmed_phase_id": int(self.observation_window["phase_id"][0]),
                }
            )
        if self.learned_phase_routing:
            if "async_probability" not in outputs:
                raise ValueError("visual phase gate inference must return async_probability")
            self.main_camera_router.update(outputs["async_probability"])
        return outputs["actions"]

    def set_episode_seed(self, seed):
        self.policy.set_episode_seed(seed)

    def reset_obsrvationwindows(self):
        self.base_instruction = None
        self.observation_window = None
        self.main_camera_router.reset()
        self.activity_metrics.reset()
        self.semantic_subtask_history = []
        print("successfully unset obs and language intruction")
