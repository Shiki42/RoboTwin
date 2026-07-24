import jax
import numpy as np
import pytest

from openpi.policies.episode_rng import episode_addressable_rng


def test_episode_addressable_rng_is_stable_and_indexed():
    first = episode_addressable_rng(123, 4)
    repeated = episode_addressable_rng(123, 4)
    next_index = episode_addressable_rng(123, 5)

    np.testing.assert_array_equal(jax.random.key_data(first), jax.random.key_data(repeated))
    assert not np.array_equal(jax.random.key_data(first), jax.random.key_data(next_index))


@pytest.mark.parametrize("seed,index", [(-1, 0), (0, -1)])
def test_episode_addressable_rng_rejects_negative_components(seed, index):
    with pytest.raises(ValueError, match="non-negative"):
        episode_addressable_rng(seed, index)


def test_episode_addressable_noise_is_stable_and_indexed():
    from openpi.policies.episode_rng import episode_addressable_noise

    first = episode_addressable_noise(123, 4, (50, 32))
    repeated = episode_addressable_noise(123, 4, (50, 32))
    next_index = episode_addressable_noise(123, 5, (50, 32))

    np.testing.assert_array_equal(first, repeated)
    assert first.dtype == np.float32
    assert not np.array_equal(first, next_index)
