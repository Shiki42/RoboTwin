import numpy as np
import pytest

from scripts import compute_official_clean_norm_stats as norm_stats


def test_episode_samples_use_only_valid_targets_and_delta_joints():
    states = np.asarray(
        [
            np.arange(14),
            np.arange(14) + 10,
            np.arange(14) + 20,
        ],
        dtype=np.float64,
    )
    actions = states + 2

    state_samples, action_samples = norm_stats.episode_normalization_samples(
        states,
        actions,
        action_horizon=3,
        allow_synthetic_fixture=True,
    )

    assert np.array_equal(state_samples, states)
    assert action_samples.shape == (6, 14)
    expected_first = actions[0].copy()
    expected_first[norm_stats.JOINT_DELTA_MASK] -= states[0, norm_stats.JOINT_DELTA_MASK]
    assert np.array_equal(action_samples[0], expected_first)
    assert np.array_equal(action_samples[:3, 6], actions[:3, 6])
    assert np.array_equal(action_samples[:3, 13], actions[:3, 13])


def test_short_real_episode_is_rejected_without_fixture_flag():
    values = np.zeros((3, 14))

    with pytest.raises(ValueError, match="allow_synthetic_fixture=true"):
        norm_stats.episode_normalization_samples(
            values,
            values,
            action_horizon=50,
            allow_synthetic_fixture=False,
        )


def test_exact_stats_are_computed_over_concatenated_values():
    values = np.arange(140, dtype=np.float64).reshape(10, 14)

    stats, count = norm_stats.exact_stats(values)

    assert count == 10
    assert np.array_equal(stats.mean, np.mean(values, axis=0))
    assert np.array_equal(stats.std, np.std(values, axis=0))
    assert np.array_equal(stats.q01, np.quantile(values, 0.01, axis=0))
    assert np.array_equal(stats.q99, np.quantile(values, 0.99, axis=0))
