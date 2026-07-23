import numpy as np

from compute_putcab_norm_stats_pytorch import _episode_action_chunks, _statistics


def test_episode_action_chunks_repeat_last_and_delta_joints() -> None:
    states = np.array(
        [
            np.arange(14, dtype=np.float32),
            np.arange(14, dtype=np.float32) + 10,
        ]
    )
    actions = states + 1

    chunks = _episode_action_chunks(
        states,
        actions,
        action_horizon=3,
        allow_synthetic_fixture=True,
    )

    assert chunks.shape == (2, 3, 14)
    np.testing.assert_allclose(chunks[0, 0, :6], 1)
    np.testing.assert_allclose(chunks[0, 1, :6], 11)
    np.testing.assert_allclose(chunks[0, 2], chunks[0, 1])
    np.testing.assert_allclose(chunks[1, :, :6], 1)
    np.testing.assert_allclose(chunks[0, :, 6], [7, 17, 17])
    np.testing.assert_allclose(chunks[1, :, 13], [24, 24, 24])


def test_statistics_are_full_population_and_exact_quantiles() -> None:
    values = np.array([[0.0, 10.0], [2.0, 14.0], [4.0, 18.0]])

    result = _statistics(values)

    np.testing.assert_allclose(result["mean"], [2.0, 14.0])
    np.testing.assert_allclose(result["std"], np.std(values, axis=0))
    np.testing.assert_allclose(result["q01"], np.quantile(values, 0.01, axis=0))
    np.testing.assert_allclose(result["q99"], np.quantile(values, 0.99, axis=0))
