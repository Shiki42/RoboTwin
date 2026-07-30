import jax.numpy as jnp
import optax
import pytest

from openpi.training import optimizer


def test_multi_step_optimizer_updates_only_after_accumulation_window():
    schedule = optimizer.CosineDecaySchedule(warmup_steps=0, peak_lr=0.1, decay_steps=10, decay_lr=0.1)
    tx = optimizer.create_optimizer(
        optimizer.AdamW(weight_decay=0.0, clip_gradient_norm=100.0),
        schedule,
        gradient_accumulation_steps=2,
    )
    params = {"weight": jnp.array(1.0)}
    state = tx.init(params)

    first_updates, state = tx.update({"weight": jnp.array(2.0)}, state, params)
    first_params = optax.apply_updates(params, first_updates)
    assert first_params["weight"] == params["weight"]

    second_updates, state = tx.update({"weight": jnp.array(4.0)}, state, first_params)
    second_params = optax.apply_updates(first_params, second_updates)
    assert second_params["weight"] < first_params["weight"]


def test_optimizer_rejects_non_positive_accumulation():
    with pytest.raises(ValueError, match="gradient accumulation steps must be positive"):
        optimizer.create_optimizer(
            optimizer.AdamW(),
            optimizer.CosineDecaySchedule(),
            gradient_accumulation_steps=0,
        )
