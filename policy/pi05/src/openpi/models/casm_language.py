from __future__ import annotations

import numpy as np

SEMANTIC_STAGE_COUNT = 6
SEMANTIC_SUBTASK_COUNT = 12

_SUBTASK_TEXT = (
    "Left arm: hold its initial pose and wait without moving. "
    "Right arm: hold its initial pose and wait without moving.",
    "Left arm: wait without moving while the drawer is opened. "
    "Right arm: reach for the drawer handle and pull the drawer open.",
    "Left arm: reach for and grasp the target object. "
    "Right arm: continue pulling the drawer open, then hold it open.",
    "Left arm: hold the grasped object still and wait. "
    "Right arm: continue pulling the drawer open.",
    "Left arm: hold the grasped object still and wait. "
    "Right arm: hold the drawer open and wait.",
    "Left arm: carry and place the target object inside the open drawer. "
    "Right arm: hold the drawer open and wait.",
    "Left arm: hold its initial pose and wait without moving. "
    "Right arm: hold its initial pose and wait without moving.",
    "Left arm: reach for the drawer handle and pull the drawer open. "
    "Right arm: wait without moving while the drawer is opened.",
    "Left arm: continue pulling the drawer open, then hold it open. "
    "Right arm: reach for and grasp the target object.",
    "Left arm: continue pulling the drawer open. "
    "Right arm: hold the grasped object still and wait.",
    "Left arm: hold the drawer open and wait. "
    "Right arm: hold the grasped object still and wait.",
    "Left arm: hold the drawer open and wait. "
    "Right arm: carry and place the target object inside the open drawer.",
)


def semantic_subtask_id(object_arm: str, *, stage_id: int) -> int:
    if object_arm not in {"left", "right"}:
        raise ValueError(f"object arm must be left or right, got {object_arm!r}")
    _validate_stage_id(stage_id)
    role_offset = 0 if object_arm == "left" else SEMANTIC_STAGE_COUNT
    return role_offset + stage_id


def object_arm_from_subtask(subtask_id: int) -> str:
    _validate_subtask_id(subtask_id)
    return "left" if subtask_id < SEMANTIC_STAGE_COUNT else "right"


def stage_id_from_subtask(subtask_id: int) -> int:
    _validate_subtask_id(subtask_id)
    return int(subtask_id) % SEMANTIC_STAGE_COUNT


def is_async_subtask(subtask_id: int) -> bool:
    return stage_id_from_subtask(subtask_id) < 4


def subtask_text(subtask_id: int) -> str:
    _validate_subtask_id(subtask_id)
    return _SUBTASK_TEXT[subtask_id]


def format_action_prompt(overall_task: str, subtask_id: int) -> str:
    task = overall_task.strip()
    if not task:
        raise ValueError("overall task prompt must be non-empty")
    return (
        f"Overall task: {task} "
        f"Current dual-arm semantic subtask: {subtask_text(subtask_id)}"
    )


def semantic_id_from_prediction(
    object_arm_right_probability: float | np.ndarray,
    async_probability: float | np.ndarray,
    stage_probabilities: np.ndarray,
) -> int:
    role_probability = _validated_probability(object_arm_right_probability, name="role")
    phase_probability = _validated_probability(async_probability, name="async phase")
    probabilities = np.asarray(stage_probabilities, dtype=np.float32)
    if probabilities.shape != (SEMANTIC_STAGE_COUNT,):
        raise ValueError(f"stage probabilities must have shape (6,), got {probabilities.shape}")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("stage probabilities must be finite and non-negative")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-4):
        raise ValueError(f"stage probabilities must sum to one, got {probabilities.sum()}")
    stage_id = (
        int(np.argmax(probabilities[:4]))
        if phase_probability >= 0.5
        else 4 + int(np.argmax(probabilities[4:]))
    )
    object_arm = "right" if role_probability >= 0.5 else "left"
    return semantic_subtask_id(object_arm, stage_id=stage_id)


def _validated_probability(value: float | np.ndarray, *, name: str) -> float:
    probability = float(np.asarray(value).reshape(()))
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} probability must be finite and in [0, 1], got {probability}")
    return probability


def _validate_stage_id(stage_id: int) -> None:
    if not isinstance(stage_id, int | np.integer):
        raise TypeError(f"semantic stage id must be an integer, got {type(stage_id)}")
    if not 0 <= int(stage_id) < SEMANTIC_STAGE_COUNT:
        raise ValueError(f"semantic stage id must be in [0, 5], got {stage_id}")


def _validate_subtask_id(subtask_id: int) -> None:
    if not isinstance(subtask_id, int | np.integer):
        raise TypeError(f"semantic subtask id must be an integer, got {type(subtask_id)}")
    if not 0 <= int(subtask_id) < SEMANTIC_SUBTASK_COUNT:
        raise ValueError(f"semantic subtask id must be in [0, 11], got {subtask_id}")
