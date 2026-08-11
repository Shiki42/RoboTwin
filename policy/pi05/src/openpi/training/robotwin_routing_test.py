import numpy as np
import pytest

from openpi.training import robotwin_routing


def test_phase_router_tracks_async_and_sync_without_camera_state():
    router = robotwin_routing.EpisodePhaseRouter(async_phase_steps=2)

    assert router.async_phase
    router.advance(2)
    assert not router.async_phase
    router.reset()
    assert router.step == 0
    assert not hasattr(router, "route")


def test_action_chunk_is_truncated_at_async_sync_boundary():
    router = robotwin_routing.EpisodePhaseRouter(async_phase_steps=110)
    router.advance(100)

    assert router.execution_steps(requested_steps=50, sync_steps=10) == 10
    router.advance(10)
    assert not router.async_phase
    assert router.execution_steps(requested_steps=50, sync_steps=10) == 10


def test_action_chunk_keeps_full_length_away_from_boundary():
    router = robotwin_routing.EpisodePhaseRouter(async_phase_steps=110)

    assert router.execution_steps(requested_steps=50, sync_steps=10) == 50


@pytest.mark.parametrize(
    ("requested_steps", "sync_steps"),
    [(0, 10), (50, 0), (-1, 10), (50, -1)],
)
def test_execution_steps_rejects_non_positive_lengths(requested_steps, sync_steps):
    router = robotwin_routing.EpisodePhaseRouter(async_phase_steps=110)

    with pytest.raises(ValueError, match="positive"):
        router.execution_steps(requested_steps, sync_steps)


def test_phase_prompt_switches_without_accumulating_tags():
    router = robotwin_routing.EpisodePhaseRouter(async_phase_steps=2)

    async_prompt = router.phase_conditioned_prompt("put the object away")
    assert async_prompt == "put the object away [PHASE=ASYNC]"
    router.advance(2)
    assert router.phase_conditioned_prompt(async_prompt) == "put the object away [PHASE=SYNC]"


def test_phase_router_uses_short_chunks_in_boundary_phase():
    router = robotwin_routing.EpisodePhaseRouter(
        async_phase_steps=110,
        boundary_phase_steps=20,
    )
    router.advance(89)

    assert not router.boundary_phase
    router.advance()
    assert router.boundary_phase
    assert router.execution_steps(requested_steps=50, sync_steps=10) == 10


def test_phase_router_rejects_negative_phase_window():
    with pytest.raises(ValueError, match="non-negative"):
        robotwin_routing.EpisodePhaseRouter(
            async_phase_steps=1,
            boundary_phase_steps=-1,
        )


def test_rollout_activity_metrics_reports_concurrency_and_obi():
    metrics = robotwin_routing.RolloutActivityMetrics(motion_threshold=0.01)
    state = np.zeros(14)
    left = state.copy()
    left[0] = 0.1
    both = left.copy()
    both[7] = 0.1
    right = state.copy()
    right[7] = 0.1

    metrics.record_chunk(async_phase=True, length=3)
    metrics.record_action(left, state, async_phase=True)
    metrics.record_action(both, state, async_phase=True)
    metrics.record_action(right, state, async_phase=True)
    metrics.record_chunk(async_phase=False, length=1)
    metrics.record_action(state, state, async_phase=False)

    summary = metrics.summary()
    assert summary["async_action_steps"] == 3
    assert summary["sync_action_steps"] == 1
    assert summary["async_concurrent_steps"] == 1
    assert summary["async_concurrency_ratio"] == pytest.approx(1 / 3)
    assert summary["async_left_start_step"] == 0
    assert summary["async_left_end_step"] == 1
    assert summary["async_right_start_step"] == 1
    assert summary["async_right_end_step"] == 2
    assert not summary["obi_left"]
    assert not summary["obi_right"]
    assert summary["executed_chunk_lengths"] == [3, 1]


def test_rollout_activity_metrics_detects_serial_right_bias():
    metrics = robotwin_routing.RolloutActivityMetrics()
    state = np.zeros(14)
    left = state.copy()
    left[0] = 0.1
    right = state.copy()
    right[7] = 0.1

    metrics.record_action(left, state, async_phase=True)
    metrics.record_action(state, state, async_phase=True)
    metrics.record_action(right, state, async_phase=True)

    assert metrics.summary()["obi_right"]


def test_rollout_activity_metrics_reset_clears_records():
    metrics = robotwin_routing.RolloutActivityMetrics()
    metrics.record_chunk(async_phase=True, length=1)
    metrics.record_action(np.zeros(14), np.zeros(14), async_phase=True)
    metrics.reset()

    assert metrics.summary()["async_action_steps"] == 0
    assert metrics.summary()["executed_chunk_lengths"] == []
