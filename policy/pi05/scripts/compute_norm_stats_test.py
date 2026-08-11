import numpy as np
import pytest

from scripts import compute_norm_stats


def test_exact_statistics_matches_direct_concat_mean_std_and_quantiles():
    values = [np.array([[1.0, 4.0], [3.0, 2.0]]), np.array([[5.0, 0.0]])]
    combined = np.concatenate(values, axis=0)

    stats = compute_norm_stats.exact_statistics(values)

    assert np.allclose(stats.mean, combined.mean(axis=0))
    assert np.allclose(stats.std, combined.std(axis=0))
    assert np.allclose(stats.q01, np.quantile(combined, 0.01, axis=0))
    assert np.allclose(stats.q99, np.quantile(combined, 0.99, axis=0))


def test_valid_action_values_excludes_temporally_padded_targets():
    actions = np.array([[[1.0, 2.0], [99.0, 99.0], [3.0, 4.0]]])
    action_mask = np.array([[[1, 1], [0, 0], [1, 1]]], dtype=np.bool_)

    valid = compute_norm_stats.valid_action_values({"actions": actions, "action_mask": action_mask})

    assert np.array_equal(valid, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_valid_action_values_rejects_dimension_specific_masks():
    actions = np.array([[[1.0, 2.0]]])
    action_mask = np.array([[[1, 0]]], dtype=np.bool_)

    with pytest.raises(ValueError, match="same valid dimensions"):
        compute_norm_stats.valid_action_values({"actions": actions, "action_mask": action_mask})


def test_valid_action_values_requires_pad_derived_mask():
    with pytest.raises(ValueError, match="action_is_pad"):
        compute_norm_stats.valid_action_values({"actions": np.ones((1, 1, 2))})
