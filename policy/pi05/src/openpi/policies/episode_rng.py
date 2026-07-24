from __future__ import annotations

import numpy as np

EPISODE_SEED_KEY = "_openpi_episode_seed"
INFERENCE_INDEX_KEY = "_openpi_inference_index"


def episode_addressable_rng(episode_seed: int, inference_index: int):
    import jax  # noqa: PLC0415

    if isinstance(episode_seed, bool) or not isinstance(episode_seed, (int, np.integer)):
        raise TypeError("episode_seed must be an integer")
    if isinstance(inference_index, bool) or not isinstance(inference_index, (int, np.integer)):
        raise TypeError("inference_index must be an integer")
    if episode_seed < 0 or inference_index < 0:
        raise ValueError("episode_seed and inference_index must be non-negative")
    return jax.random.fold_in(jax.random.key(int(episode_seed)), int(inference_index))


def episode_addressable_noise(
    episode_seed: int,
    inference_index: int,
    shape: tuple[int, ...],
) -> np.ndarray:
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("noise shape must contain positive dimensions")
    episode_addressable_rng(episode_seed, inference_index)
    generator = np.random.default_rng(np.random.SeedSequence([int(episode_seed), int(inference_index)]))
    return generator.standard_normal(shape, dtype=np.float32)
