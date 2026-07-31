from __future__ import annotations

import dataclasses
import os

import flax.nnx as nnx

from openpi.shared import nnx_utils

ACCEPTED_NORMALIZER_ASSET_ID = "Shiki42/robotwin_put_obj_cabinet_50_dynFcam_nFov_dynamicMain_lerobot"


def create_variant(base_config):
    model = dataclasses.replace(
        base_config.model,
        casm_mode="skillvla_per_arm_gated",
        cross_attention_dim=32,
    )
    assets = dataclasses.replace(
        base_config.data.assets,
        assets_dir=os.environ.get("PARALLELVLA_NORM_ASSETS_DIR", base_config.data.assets.assets_dir),
        asset_id=ACCEPTED_NORMALIZER_ASSET_ID,
    )
    data = dataclasses.replace(base_config.data, inactive_action_weight=1.0, assets=assets)
    weight_loader = dataclasses.replace(
        base_config.weight_loader,
        missing_regex=r".*(phase_gate|cross_attention|per_arm_adapter).*",
    )
    trainable = nnx_utils.PathRegex(r".*(action_out_proj|phase_gate|cross_attention|per_arm_adapter).*")
    return dataclasses.replace(
        base_config,
        name="pi05_putcab_skillvla_per_arm_gated_probe",
        project_name="parallelvla-skillvla-probe",
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
