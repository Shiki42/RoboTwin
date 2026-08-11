from __future__ import annotations

import numpy as np

SYNC_PHASE_ID = 0
ASYNC_PHASE_ID = 1
SYNC_PHASE_TAG = "[PHASE=SYNC]"
ASYNC_PHASE_TAG = "[PHASE=ASYNC]"
PHASE_TAGS = (SYNC_PHASE_TAG, ASYNC_PHASE_TAG)


def phase_conditioned_prompt(instruction: str, *, async_phase: bool) -> str:
    prompt = instruction.strip()
    for tag in PHASE_TAGS:
        if prompt.endswith(tag):
            prompt = prompt[: -len(tag)].rstrip()
            break
    phase_tag = ASYNC_PHASE_TAG if async_phase else SYNC_PHASE_TAG
    return f"{prompt} {phase_tag}"


class EpisodePhaseRouter:
    def __init__(
        self,
        async_phase_steps: int,
        boundary_phase_steps: int = 0,
    ) -> None:
        if async_phase_steps < 0 or boundary_phase_steps < 0:
            raise ValueError("phase routing steps must be non-negative")
        self.async_phase_steps = async_phase_steps
        self.boundary_phase_steps = boundary_phase_steps
        self.reset()

    @property
    def step(self) -> int:
        return self._step

    @property
    def async_phase(self) -> bool:
        return self._step < self.async_phase_steps

    @property
    def remaining_async_steps(self) -> int:
        return max(0, self.async_phase_steps - self._step)

    @property
    def boundary_phase(self) -> bool:
        return (
            self.async_phase
            and self.boundary_phase_steps > 0
            and self.remaining_async_steps <= self.boundary_phase_steps
        )

    def phase_conditioned_prompt(self, instruction: str) -> str:
        return phase_conditioned_prompt(instruction, async_phase=self.async_phase)

    def execution_steps(self, requested_steps: int, sync_steps: int) -> int:
        if requested_steps <= 0 or sync_steps <= 0:
            raise ValueError("execution steps must be positive")
        if self.async_phase:
            limit = min(requested_steps, self.remaining_async_steps)
            return min(limit, sync_steps) if self.boundary_phase else limit
        return min(requested_steps, sync_steps)

    def advance(self, steps: int = 1) -> None:
        if steps < 0:
            raise ValueError("advance steps must be non-negative")
        self._step += steps

    def reset(self) -> None:
        self._step = 0


class RolloutActivityMetrics:
    def __init__(self, motion_threshold: float = 0.01) -> None:
        if motion_threshold <= 0:
            raise ValueError("motion threshold must be positive")
        self.motion_threshold = motion_threshold
        self.reset()

    def record_chunk(self, *, async_phase: bool, length: int) -> None:
        if length <= 0:
            raise ValueError("chunk length must be positive")
        self.chunk_records.append({"async": bool(async_phase), "length": int(length)})

    def record_action(
        self,
        action: np.ndarray,
        state: np.ndarray,
        *,
        async_phase: bool,
    ) -> None:
        action = np.asarray(action, dtype=np.float64)
        state = np.asarray(state, dtype=np.float64)
        if action.shape != (14,) or state.shape != (14,):
            raise ValueError(f"expected 14-DoF action/state, got {action.shape} and {state.shape}")
        delta = action - state
        left_active = float(np.linalg.norm(delta[:7])) > self.motion_threshold
        right_active = float(np.linalg.norm(delta[7:])) > self.motion_threshold
        self.action_records.append(
            {
                "async": bool(async_phase),
                "left_active": left_active,
                "right_active": right_active,
            }
        )

    @staticmethod
    def _activity_bounds(records: list[dict], key: str) -> tuple[int | None, int | None]:
        active = [index for index, record in enumerate(records) if record[key]]
        if not active:
            return None, None
        return active[0], active[-1]

    def summary(self) -> dict:
        async_records = [record for record in self.action_records if record["async"]]
        concurrent = sum(
            record["left_active"] and record["right_active"]
            for record in async_records
        )
        left_start, left_end = self._activity_bounds(async_records, "left_active")
        right_start, right_end = self._activity_bounds(async_records, "right_active")
        denominator = max(1, len(async_records))
        return {
            "async_action_steps": len(async_records),
            "sync_action_steps": len(self.action_records) - len(async_records),
            "async_concurrent_steps": concurrent,
            "async_concurrency_ratio": concurrent / denominator,
            "async_left_start_step": left_start,
            "async_left_end_step": left_end,
            "async_right_start_step": right_start,
            "async_right_end_step": right_end,
            "obi_left": bool(
                left_start is not None
                and right_end is not None
                and left_start > right_end
            ),
            "obi_right": bool(
                right_start is not None
                and left_end is not None
                and right_start > left_end
            ),
            "executed_chunk_lengths": [
                record["length"] for record in self.chunk_records
            ],
            "async_chunk_count": sum(record["async"] for record in self.chunk_records),
            "sync_chunk_count": sum(not record["async"] for record in self.chunk_records),
        }

    def reset(self) -> None:
        self.chunk_records: list[dict] = []
        self.action_records: list[dict] = []
