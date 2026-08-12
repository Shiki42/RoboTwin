"""PI0.5 full-finetune recipe for joint textual-subtask and action learning."""

from __future__ import annotations

import os

import openpi.models.pi0_config as pi0_config
import openpi.transforms as transforms


def _episode_indices(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.environ.get(name)
    if value is None:
        return default
    episodes = tuple(int(item) for item in value.split(",") if item)
    if not episodes:
        raise ValueError(f"{name} must contain at least one episode")
    return episodes


def _data_config(repo_id: str, annotation_dir: str, annotation_revision: str):
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
                        "subtask": "subtask",
                    }
                )
            ]
        ),
        base_config=DataConfig(
            prompt_from_task=True,
            video_backend="pyav",
            subtask_annotation_dir=annotation_dir,
            subtask_annotation_revision=annotation_revision,
        ),
        action_sequence_keys=("action",),
    )


def create_config():
    from openpi.training.config import TrainConfig

    repo_id = os.environ.get(
        "PARALLELVLA_DATASET_REPO",
        "Shiki42/robotwin_put_obj_cabinet_parallel_50",
    )
    annotation_dir = os.environ.get(
        "PARALLELVLA_SUBTASK_ANNOTATION_DIR",
        "/home/coder/share/datasets/putcab_pi05_subtasks_missing",
    )
    annotation_revision = os.environ.get(
        "PARALLELVLA_SUBTASK_ANNOTATION_REVISION",
        "0" * 64,
    )
    model = pi0_config.Pi0Config(
        pi05=True,
        casm_mode="none",
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        online_subtask_prediction=True,
        lambda_subtask=float(os.environ.get("PARALLELVLA_LAMBDA_SUBTASK", "0.1")),
        subtask_max_token_len=200,
        subtask_text_format="Left arm: <semantic>; Right arm: <semantic>.",
        subtask_action_prompt_format="{task}\nCurrent subtask: {subtask}",
        subtask_annotation_revision=annotation_revision,
    )
    return TrainConfig(
        name="pi05_putcab_online_subtask_pytorch_full",
        project_name="parallelvla-pi05-putcab-online-subtask-full",
        model=model,
        data=_data_config(repo_id, annotation_dir, annotation_revision),
        pytorch_weight_path=os.environ.get("PI05_PYTORCH_BASE"),
        pytorch_training_precision="bfloat16",
        pytorch_trainable_scope="all",
        train_episodes=_episode_indices(
            "PARALLELVLA_TRAIN_EPISODES",
            tuple(range(50)),
        ),
        validation_episodes=_episode_indices(
            "PARALLELVLA_VALIDATION_EPISODES",
            (50, 51, 52),
        ),
        validation_interval=int(os.environ.get("PARALLELVLA_VALIDATION_INTERVAL", "1_000")),
        validation_num_workers=2,
        batch_size=16,
        gradient_accumulation_steps=1,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        num_train_steps=5_000,
        save_interval=1_000,
        keep_period=5_000,
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
