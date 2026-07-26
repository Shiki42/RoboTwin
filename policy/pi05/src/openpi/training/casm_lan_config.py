from __future__ import annotations

import dataclasses

from openpi import transforms


def create(base_config, *, name: str, project_name: str):
    """Create CASM-LAN from an anchored visual-gate CASM config."""
    repack = transforms.Group(
        inputs=[
            transforms.RepackTransform(
                {
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "action_mask": "observation.arm_active_mask",
                    "action_phase": "observation.phase_one_hot",
                    "semantic_subtask_id": "observation.semantic_subtask_id",
                    "prompt": "prompt",
                }
            )
        ]
    )
    data = dataclasses.replace(base_config.data, repack_transforms=repack)
    model = dataclasses.replace(base_config.model, semantic_subtask_prediction=True)
    weight_loader = dataclasses.replace(
        base_config.weight_loader,
        missing_regex=".*semantic_(role|stage)_head.*",
    )
    return dataclasses.replace(
        base_config,
        name=name,
        project_name=project_name,
        model=model,
        data=data,
        weight_loader=weight_loader,
        num_train_steps=500,
        save_interval=500,
        keep_period=500,
    )
