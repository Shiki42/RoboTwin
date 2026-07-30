from __future__ import annotations

import dataclasses
import os

import flax.nnx as nnx

from openpi.shared import nnx_utils

ACCEPTED_NORMALIZER_ASSET_ID = "Shiki42/robotwin_put_obj_cabinet_50_dynFcam_nFov_dynamicMain_lerobot"


def _create(base_config, *, name: str, mode: str, trainable_pattern: str):
    model = dataclasses.replace(base_config.model, casm_mode=mode)
    assets = dataclasses.replace(
        base_config.data.assets,
        assets_dir=os.environ.get("PARALLELVLA_NORM_ASSETS_DIR", base_config.data.assets.assets_dir),
        asset_id=ACCEPTED_NORMALIZER_ASSET_ID,
    )
    data = dataclasses.replace(base_config.data, inactive_action_weight=1.0, assets=assets)
    missing_regex = ".*(cross_output_adapter|phase_gate).*" if mode == "cross_output_shared_head" else ".*phase_gate.*"
    weight_loader = dataclasses.replace(base_config.weight_loader, missing_regex=missing_regex)
    trainable = nnx_utils.PathRegex(trainable_pattern)
    return dataclasses.replace(
        base_config,
        name=name,
        project_name="parallelvla-cross-output",
        model=model,
        data=data,
        weight_loader=weight_loader,
        freeze_filter=nnx.Not(trainable),
        seed=87431,
        batch_size=1,
        gradient_accumulation_steps=16,
        num_workers=0,
        num_train_steps=500,
        log_interval=1,
        save_interval=100,
        keep_period=500,
        params_only_checkpoint=False,
        ema_decay=None,
        wandb_enabled=False,
    )


def create_variants(base_config):
    return (
        _create(
            base_config,
            name="pi05_putcab_casm_joint_head_frozen_trunk",
            mode="visual_phase_gate",
            trainable_pattern=r".*(action_out_proj|phase_gate).*",
        ),
        _create(
            base_config,
            name="pi05_putcab_casm_cross_output_shared_head",
            mode="cross_output_shared_head",
            trainable_pattern=r".*(action_out_proj|phase_gate|cross_output_adapter).*",
        ),
    )
