from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np

from openpi import transforms
from openpi.training import optimizer

SUBTASK_CLASSES = 12
JOINT_SUBTASK_TEXTS = (
    "Left arm: reach and grasp object; Right arm: wait.",
    "Left arm: reach and grasp object; Right arm: reach and open drawer.",
    "Left arm: reach and grasp object; Right arm: wait while holding drawer open.",
    "Left arm: wait while holding object; Right arm: reach and open drawer.",
    "Left arm: wait while holding object; Right arm: wait while holding drawer open.",
    "Left arm: reach and open drawer; Right arm: wait.",
    "Left arm: reach and open drawer; Right arm: reach and grasp object.",
    "Left arm: reach and open drawer; Right arm: wait while holding object.",
    "Left arm: insert and place object; Right arm: wait while holding drawer open.",
    "Left arm: wait while holding drawer open; Right arm: reach and grasp object.",
    "Left arm: wait while holding drawer open; Right arm: wait while holding object.",
    "Left arm: wait while holding drawer open; Right arm: insert and place object.",
)
# Mean-one normalized inverse square roots of train revision 85c010cd class counts
# (10, 4674, 427, 1422, 108, 40, 4005, 1169, 5008, 448, 92, 4242).
SQRT_BALANCED_CLASS_WEIGHTS = (
    4.28230671,
    0.19807671,
    0.65533571,
    0.35911039,
    1.30306443,
    2.14115335,
    0.21398164,
    0.39606869,
    0.19135755,
    0.63979194,
    1.41183471,
    0.20791817,
)


@dataclasses.dataclass(frozen=True)
class TeacherForcedSubtaskPrompt(transforms.DataTransformFn):
    """Reproduce the factorized checkpoint's action-prompt contract."""

    def __call__(self, data: dict) -> dict:
        target = np.asarray(data["semantic_subtask_id"])
        if target.size != 1:
            raise ValueError("semantic subtask target must contain one class id")
        class_id = int(target.reshape(-1)[0])
        if not 0 <= class_id < len(JOINT_SUBTASK_TEXTS):
            raise ValueError(f"semantic subtask class is invalid: {class_id}")
        prompt = data["prompt"]
        if isinstance(prompt, np.ndarray):
            if prompt.size != 1:
                raise ValueError("action prompt must contain one string")
            prompt = prompt.item()
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("action prompt must be a non-empty string")
        return {
            **data,
            "prompt": f"{prompt.strip()}\nCurrent subtask: {JOINT_SUBTASK_TEXTS[class_id]}",
        }


def create(
    base_config,
    *,
    name: str = "pi05_putcab_factorized_anchor_subtask_head_pytorch",
    class_weights: tuple[float, ...] | None = None,
    stop_gradient: bool = True,
    trainable_scope: Literal[
        "subtask_head",
        "subtask_head_and_projector",
        "subtask_head_and_action_policy",
    ] = "subtask_head",
    loss_weight: float = 1.0,
    peak_lr: float = 3e-4,
    decay_lr: float = 3e-5,
    action_loss_mode: Literal["unmasked", "factorized", "full"] = "unmasked",
    teacher_forced_action_prompt: bool = False,
):
    """Add joint-subtask supervision while preserving the native PI0.5 action contract."""
    repack_structure = {
        "images": {
            "cam_high": "observation.images.cam_high",
            "cam_left_wrist": "observation.images.cam_left_wrist",
            "cam_right_wrist": "observation.images.cam_right_wrist",
        },
        "state": "observation.state",
        "actions": "action",
        "semantic_subtask_id": "observation.semantic_subtask_id",
        "prompt": "prompt",
    }
    action_sequence_keys = ("action",)
    if action_loss_mode == "factorized":
        repack_structure.update(
            {
                "action_loss_mask": "observation.action_loss_mask",
                "action_is_pad": "action_is_pad",
            }
        )
    elif action_loss_mode == "full":
        repack_structure["action_is_pad"] = "action_is_pad"
    elif action_loss_mode != "unmasked":
        raise ValueError(f"unsupported action loss mode: {action_loss_mode}")
    repack_inputs = [transforms.RepackTransform(repack_structure)]
    if teacher_forced_action_prompt:
        repack_inputs.append(TeacherForcedSubtaskPrompt())
    repack = transforms.Group(inputs=repack_inputs)
    data = dataclasses.replace(
        base_config.data,
        repack_transforms=repack,
        action_sequence_keys=action_sequence_keys,
    )
    model = dataclasses.replace(
        base_config.model,
        pytorch_aux_subtask_classes=SUBTASK_CLASSES,
        pytorch_aux_subtask_hidden_dim=512,
        pytorch_aux_subtask_state_dim=14,
        pytorch_aux_subtask_loss_weight=loss_weight,
        pytorch_aux_subtask_class_weights=class_weights,
        pytorch_aux_subtask_stop_gradient=stop_gradient,
    )
    return dataclasses.replace(
        base_config,
        name=name,
        project_name="parallelvla-putcab-subtask-aux",
        model=model,
        data=data,
        pytorch_trainable_scope=trainable_scope,
        pytorch_gradient_checkpointing=False,
        pytorch_compile_mode="default",
        pytorch_action_prompt_mode=("teacher_forced_joint_subtask" if teacher_forced_action_prompt else "task_only"),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=peak_lr,
            decay_steps=2_000,
            decay_lr=decay_lr,
        ),
        ema_decay=None,
        batch_size=16,
        gradient_accumulation_steps=1,
        num_train_steps=2_000,
        save_interval=2_000,
        keep_period=2_000,
    )
