"""Strict π0.5 full-finetuning recipe for RoboTwin official clean data."""

import os

import openpi.models.pi0_config as pi0_config
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as transforms


def create_data_config(repo_id: str):
    from openpi.training.config import DataConfig
    from openpi.training.config import LeRobotAlohaDataConfig

    return LeRobotAlohaDataConfig(
        repo_id=repo_id,
        adapt_to_pi=False,
        repack_transforms=transforms.Group(
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
                        "action_is_pad": "action_is_pad",
                        "prompt": "prompt",
                    }
                )
            ]
        ),
        base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        action_sequence_keys=("action",),
    )


def create_config():
    from openpi.training.config import TrainConfig

    dataset_repo = os.environ.get(
        "PARALLELVLA_DATASET_REPO",
        "Shiki42/robotwin_put_object_cabinet_official_clean50",
    )
    return TrainConfig(
        name="pi05_putcab_official_clean50_full",
        project_name="parallelvla-pi05-official-clean50",
        model=pi0_config.Pi0Config(
            pi05=True,
            casm_mode="none",
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
        ),
        data=create_data_config(dataset_repo),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.environ.get("PI05_BASE_CHECKPOINT", "gs://openpi-assets/checkpoints/pi05_base/params"),
            missing_regex=r"(?!x)x",
        ),
        pytorch_training_precision="float32",
        batch_size=1,
        gradient_accumulation_steps=32,
        num_workers=0,
        num_train_steps=30_000,
        save_interval=5_000,
        keep_period=30_000,
        params_only_checkpoint=False,
        ema_decay=0.99,
        seed=42,
        wandb_enabled=True,
        fsdp_devices=1,
    )
