import numpy as np
import pytest

from openpi.training import robotwin_routing


def test_scene_context_is_frozen_per_async_segment():
    phase_ids = np.array([0, 0, 1, 1, 2, 0, 1, 1], dtype=np.int8)

    indices = robotwin_routing.scene_context_indices(phase_ids)

    assert indices.tolist() == [0, 1, 2, 2, 2, 5, 6, 6]


def test_route_main_camera_preserves_sync_and_freezes_async():
    frames = np.arange(8 * 2).reshape(8, 2)
    phase_ids = np.array([0, 0, 1, 1, 2, 0, 1, 1], dtype=np.int8)

    routed = robotwin_routing.route_main_camera(frames, phase_ids)

    assert routed.tolist() == frames[[0, 1, 2, 2, 2, 5, 6, 6]].tolist()


def test_route_main_camera_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        robotwin_routing.route_main_camera(
            np.zeros((3, 2)),
            np.zeros(2, dtype=np.int8),
        )


def test_online_router_freezes_episode_start_then_switches_to_dynamic():
    router = robotwin_routing.EpisodeStartSceneContextRouter(async_steps=2)
    first = np.full((2, 2), 1)
    second = np.full((2, 2), 2)
    third = np.full((2, 2), 3)

    assert np.array_equal(router.route(first), first)
    router.advance()
    assert np.array_equal(router.route(second), first)
    router.advance()
    assert np.array_equal(router.route(third), third)
    assert not router.async_phase


def test_online_router_zero_async_steps_keeps_dynamic_main_camera():
    router = robotwin_routing.EpisodeStartSceneContextRouter(async_steps=0)

    assert np.array_equal(router.route(np.array([1])), np.array([1]))
    assert np.array_equal(router.route(np.array([2])), np.array([2]))


def test_online_router_reset_starts_new_scene_context():
    router = robotwin_routing.EpisodeStartSceneContextRouter(async_steps=3)
    router.route(np.array([1]))
    router.advance(2)
    router.reset()

    assert router.step == 0
    assert np.array_equal(router.route(np.array([9])), np.array([9]))


def test_action_chunk_is_truncated_at_async_sync_boundary():
    router = robotwin_routing.EpisodeStartSceneContextRouter(async_steps=110)
    router.advance(100)

    assert router.execution_steps(requested_steps=50, sync_steps=10) == 10
    router.advance(10)
    assert not router.async_phase
    assert router.execution_steps(requested_steps=50, sync_steps=10) == 10


def test_action_chunk_keeps_full_length_away_from_boundary():
    router = robotwin_routing.EpisodeStartSceneContextRouter(async_steps=110)

    assert router.execution_steps(requested_steps=50, sync_steps=10) == 50


@pytest.mark.parametrize(
    ("requested_steps", "sync_steps"),
    [(0, 10), (50, 0), (-1, 10), (50, -1)],
)
def test_execution_steps_rejects_non_positive_lengths(requested_steps, sync_steps):
    router = robotwin_routing.EpisodeStartSceneContextRouter(async_steps=110)

    with pytest.raises(ValueError, match="positive"):
        router.execution_steps(requested_steps, sync_steps)


def test_phase_prompt_switches_without_accumulating_tags():
    router = robotwin_routing.EpisodeStartSceneContextRouter(async_steps=2)

    async_prompt = router.phase_conditioned_prompt("put the object away")
    assert async_prompt == "put the object away [PHASE=ASYNC]"
    router.advance(2)
    assert router.phase_conditioned_prompt(async_prompt) == "put the object away [PHASE=SYNC]"


def test_phase_neutral_prompt_removes_ground_truth_phase_tag():
    assert robotwin_routing.phase_neutral_prompt("put the object away [PHASE=ASYNC]") == "put the object away"
    assert robotwin_routing.phase_neutral_prompt("put the object away [PHASE=SYNC]") == "put the object away"
    assert robotwin_routing.phase_neutral_prompt("  put the object away  ") == "put the object away"


def test_scene_context_restores_dynamic_frames_near_boundaries():
    phase_ids = np.array([0, 0, 1, 1, 2, 0, 1, 1], dtype=np.int8)

    indices = robotwin_routing.scene_context_indices(
        phase_ids,
        boundary_context_steps=1,
    )

    assert indices.tolist() == [0, 1, 2, 2, 4, 5, 6, 7]


def test_online_router_uses_short_dynamic_chunks_in_boundary_context():
    router = robotwin_routing.EpisodeStartSceneContextRouter(
        async_steps=110,
        boundary_context_steps=20,
    )
    first = np.array([1])
    router.route(first)
    router.advance(89)

    assert not router.boundary_context
    assert np.array_equal(router.route(np.array([2])), first)
    router.advance()
    assert router.boundary_context
    assert np.array_equal(router.route(np.array([3])), np.array([3]))
    assert router.execution_steps(requested_steps=50, sync_steps=10) == 10


def test_scene_context_rejects_negative_boundary_window():
    with pytest.raises(ValueError, match="non-negative"):
        robotwin_routing.scene_context_indices(
            np.zeros(2, dtype=np.int8),
            boundary_context_steps=-1,
        )


def test_learned_router_uses_confirmed_monotonic_async_to_sync_transition():
    router = robotwin_routing.LearnedAsyncToSyncRouter(sync_threshold=0.5, sync_confirmations=2)
    initial = np.zeros((2, 2, 3), dtype=np.uint8)
    later = np.ones((2, 2, 3), dtype=np.uint8)

    assert router.async_phase
    assert np.array_equal(router.route(initial), initial)
    assert np.array_equal(router.route(later), initial)
    assert router.execution_steps(requested_steps=50, sync_steps=10) == 50

    router.update(0.8)
    router.update(np.array(0.4))
    assert router.async_phase
    router.update(0.3)
    assert not router.async_phase
    assert np.array_equal(router.route(later), later)
    assert router.execution_steps(requested_steps=50, sync_steps=10) == 10

    router.update(0.9)
    assert not router.async_phase
    assert router.summary()["transition_observation"] == 3


@pytest.mark.parametrize("probability", [-0.1, 1.1, np.nan])
def test_learned_router_rejects_invalid_probability(probability):
    router = robotwin_routing.LearnedAsyncToSyncRouter()

    with pytest.raises(ValueError, match="probability"):
        router.update(probability)


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
