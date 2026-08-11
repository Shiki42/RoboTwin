"""Matched PyTorch recipes for RoboTwin dual-arm PI0.5 data."""

from __future__ import annotations

import os
import random

import openpi.models.pi0_config as pi0_config
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


def _episode_split(seed: int = 42) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    order = list(range(100))
    random.Random(seed).shuffle(order)
    train = tuple(sorted(order[:50]))
    validation = tuple(sorted(order[50:53]))
    unused = tuple(sorted(order[53:]))
    return train, validation, unused


def _create_config(*, lora: bool):
    from openpi.training.config import TrainConfig

    dataset_repo = os.environ.get("PARALLELVLA_DATASET_REPO", "pi05_scan_object_retime_100")
    suffix = "lora" if lora else "full"
    train_episodes, validation_episodes, _ = _episode_split()
    model = pi0_config.Pi0Config(
        pi05=True,
        casm_mode="none",
        paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
        action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
    )
    return TrainConfig(
        name=f"pi05_robotwin_parallel100_pytorch_{suffix}",
        project_name=f"parallelvla-pi05-robotwin-{suffix}",
        model=model,
        data=create_data_config(dataset_repo),
        pytorch_weight_path=os.environ.get("PI05_PYTORCH_BASE"),
        pytorch_training_precision="bfloat16",
        pytorch_trainable_scope="lora" if lora else "all",
        train_episodes=train_episodes if not lora else None,
        validation_episodes=validation_episodes if not lora else (),
        validation_interval=2_000 if not lora else 0,
        validation_num_workers=2 if not lora else 0,
        batch_size=32 if lora else 16,
        gradient_accumulation_steps=1,
        num_workers=4 if not lora else 2,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        num_train_steps=30_000,
        save_interval=10_000,
        keep_period=10_000,
        params_only_checkpoint=False,
        ema_decay=0.99,
        seed=42,
        wandb_enabled=False,
        pytorch_compile_mode="default",
        pytorch_attention_implementation="sdpa",
        pytorch_gradient_checkpointing=False,
        pytorch_fused_optimizer=False,
        fsdp_devices=1,
    )


def create_config():
    return _create_config(lora=True)


def create_full_config():
    return _create_config(lora=False)
