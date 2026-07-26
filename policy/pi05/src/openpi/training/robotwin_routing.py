from __future__ import annotations

import numpy as np

SYNC_PHASE_ID = 0
ASYNC_PHASE_ID = 1
SYNC_PHASE_TAG = "[PHASE=SYNC]"
ASYNC_PHASE_TAG = "[PHASE=ASYNC]"
PHASE_TAGS = (SYNC_PHASE_TAG, ASYNC_PHASE_TAG)


def phase_neutral_prompt(instruction: str) -> str:
    prompt = instruction.strip()
    for tag in PHASE_TAGS:
        if prompt.endswith(tag):
            return prompt[: -len(tag)].rstrip()
    return prompt


def phase_conditioned_prompt(instruction: str, *, async_phase: bool) -> str:
    prompt = phase_neutral_prompt(instruction)
    phase_tag = ASYNC_PHASE_TAG if async_phase else SYNC_PHASE_TAG
    return f"{prompt} {phase_tag}"


class EpisodeStartSceneContextRouter:
    def __init__(self, async_steps: int, boundary_context_steps: int = 0) -> None:
        if async_steps < 0 or boundary_context_steps < 0:
            raise ValueError("phase routing steps must be non-negative")
        self.async_steps = async_steps
        self.boundary_context_steps = boundary_context_steps
        self.reset()

    @property
    def step(self) -> int:
        return self._step

    @property
    def async_phase(self) -> bool:
        return self._step < self.async_steps

    @property
    def remaining_async_steps(self) -> int:
        return max(0, self.async_steps - self._step)

    @property
    def boundary_context(self) -> bool:
        return (
            self.async_phase
            and self.boundary_context_steps > 0
            and self.remaining_async_steps <= self.boundary_context_steps
        )

    def phase_conditioned_prompt(self, instruction: str) -> str:
        return phase_conditioned_prompt(instruction, async_phase=self.async_phase)

    def execution_steps(self, requested_steps: int, sync_steps: int) -> int:
        if requested_steps <= 0 or sync_steps <= 0:
            raise ValueError("execution steps must be positive")
        if self.async_phase:
            limit = min(requested_steps, self.remaining_async_steps)
            return min(limit, sync_steps) if self.boundary_context else limit
        return min(requested_steps, sync_steps)

    def route(self, main_frame: np.ndarray) -> np.ndarray:
        return np.asarray(main_frame)

    def advance(self, steps: int = 1) -> None:
        if steps < 0:
            raise ValueError("advance steps must be non-negative")
        self._step += steps

    def reset(self) -> None:
        self._step = 0


class LearnedAsyncToSyncRouter:
    """Monotonic deployment router for tasks with one async-to-sync boundary."""

    def __init__(self, sync_threshold: float = 0.5, sync_confirmations: int = 2) -> None:
        if not 0.0 < sync_threshold < 1.0:
            raise ValueError("sync threshold must be in (0, 1)")
        if sync_confirmations <= 0:
            raise ValueError("sync confirmations must be positive")
        self.sync_threshold = sync_threshold
        self.sync_confirmations = sync_confirmations
        self.reset()

    @property
    def async_phase(self) -> bool:
        return self._async_phase

    @property
    def latest_async_probability(self) -> float | None:
        return self._latest_async_probability

    def update(self, async_probability: float | np.ndarray) -> None:
        probability = float(np.asarray(async_probability).reshape(()))
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"async probability must be finite and in [0, 1], got {probability}")
        self._latest_async_probability = probability
        self._probability_history.append(probability)
        if not self._async_phase:
            return
        if probability >= self.sync_threshold:
            self._sync_evidence = 0
            return
        self._sync_evidence += 1
        if self._sync_evidence >= self.sync_confirmations:
            self._async_phase = False
            self._transition_observation = len(self._probability_history)

    def execution_steps(self, requested_steps: int, sync_steps: int) -> int:
        if requested_steps <= 0 or sync_steps <= 0:
            raise ValueError("execution steps must be positive")
        return requested_steps if self._async_phase else min(requested_steps, sync_steps)

    def route(self, main_frame: np.ndarray) -> np.ndarray:
        return np.asarray(main_frame)

    def summary(self) -> dict:
        return {
            "router": "learned_async_to_sync",
            "sync_threshold": self.sync_threshold,
            "sync_confirmations": self.sync_confirmations,
            "async_phase": self._async_phase,
            "latest_async_probability": self._latest_async_probability,
            "transition_observation": self._transition_observation,
            "probability_history": self._probability_history.copy(),
        }

    def reset(self) -> None:
        self._async_phase = True
        self._sync_evidence = 0
        self._latest_async_probability: float | None = None
        self._transition_observation: int | None = None
        self._probability_history: list[float] = []


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
