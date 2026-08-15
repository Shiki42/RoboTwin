from __future__ import annotations

import dataclasses
from typing import Literal

from openpi import transforms
from openpi.training import optimizer

SUBTASK_CLASSES = 12
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


def create(
    base_config,
    *,
    name: str = "pi05_putcab_factorized_anchor_subtask_head_pytorch",
    class_weights: tuple[float, ...] | None = None,
    stop_gradient: bool = True,
    trainable_scope: Literal["subtask_head", "subtask_head_and_projector"] = "subtask_head",
    loss_weight: float = 1.0,
    peak_lr: float = 3e-4,
    decay_lr: float = 3e-5,
    factorized_action_loss: bool = False,
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
    if factorized_action_loss:
        repack_structure.update(
            {
                "action_loss_mask": "observation.action_loss_mask",
                "action_is_pad": "action_is_pad",
            }
        )
    repack = transforms.Group(
        inputs=[
            transforms.RepackTransform(repack_structure)
        ]
    )
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
