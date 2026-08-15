from __future__ import annotations

import dataclasses

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
):
    """Add a detached joint-subtask head to an unchanged PI0.5 action policy."""
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
                    "semantic_subtask_id": "observation.semantic_subtask_id",
                    "prompt": "prompt",
                }
            )
        ]
    )
    data = dataclasses.replace(
        base_config.data,
        repack_transforms=repack,
        action_sequence_keys=("action",),
    )
    model = dataclasses.replace(
        base_config.model,
        pytorch_aux_subtask_classes=SUBTASK_CLASSES,
        pytorch_aux_subtask_hidden_dim=512,
        pytorch_aux_subtask_state_dim=14,
        pytorch_aux_subtask_loss_weight=1.0,
        pytorch_aux_subtask_class_weights=class_weights,
        pytorch_aux_subtask_stop_gradient=True,
    )
    return dataclasses.replace(
        base_config,
        name=name,
        project_name="parallelvla-putcab-subtask-aux",
        model=model,
        data=data,
        pytorch_trainable_scope="subtask_head",
        pytorch_gradient_checkpointing=False,
        pytorch_compile_mode="default",
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=3e-4,
            decay_steps=2_000,
            decay_lr=3e-5,
        ),
        ema_decay=None,
        batch_size=16,
        gradient_accumulation_steps=1,
        num_train_steps=2_000,
        save_interval=2_000,
        keep_period=2_000,
    )
