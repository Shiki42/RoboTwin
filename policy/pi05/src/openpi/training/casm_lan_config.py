from __future__ import annotations

import dataclasses

import flax.nnx as nnx

from openpi import transforms
from openpi.shared import nnx_utils


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
                    "action_is_pad": "action_is_pad",
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


def create_heads_only(base_config, *, name: str, project_name: str):
    """Load a trained CASM-LAN checkpoint and update only its semantic decision heads."""
    config = create(base_config, name=name, project_name=project_name)
    model = dataclasses.replace(
        config.model,
        semantic_role_loss_weight=0.5,
        semantic_stage_loss_weight=1.0,
    )
    weight_loader = dataclasses.replace(config.weight_loader, missing_regex=r"(?!x)x")
    trainable_heads = nnx_utils.PathRegex(r".*(phase_gate|semantic_(role|stage)_head).*")
    return dataclasses.replace(
        config,
        model=model,
        weight_loader=weight_loader,
        freeze_filter=nnx.Not(trainable_heads),
        project_name=project_name,
    )


def create_variants(base_config):
    return (
        create(
            base_config,
            name="pi05_putcab_casm_lan_anchor_adapt_lora",
            project_name="parallelvla-casm-lan",
        ),
        create_heads_only(
            base_config,
            name="pi05_putcab_casm_lan_semantic_heads_only",
            project_name="parallelvla-casm-lan-heads",
        ),
    )
