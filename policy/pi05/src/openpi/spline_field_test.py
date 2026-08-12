import jax.numpy as jnp
import numpy as np
import pytest

from openpi import spline_field
from openpi import transforms
from openpi.models import pi0


def test_weighted_spline_ignores_padding_and_weights_supported_coefficients() -> None:
    horizon = 50
    points = 16
    times = np.linspace(0.0, 1.0, horizon, dtype=np.float32)
    actions = np.stack([times, times**2], axis=-1)
    valid = np.zeros(horizon, dtype=np.float32)
    valid[:23] = 1.0
    padded = actions.copy()
    padded[23:] = 999.0

    coefficients = spline_field.encode_actions(
        padded,
        control_horizon=horizon,
        control_points=points,
        valid_timestep_mask=valid,
    )
    decoded = spline_field.decode_actions(coefficients, control_horizon=horizon, control_points=points)
    weights = spline_field.coefficient_supervision_weights(
        np.ones_like(actions) * valid[:, None],
        control_horizon=horizon,
        control_points=points,
    )

    assert np.sqrt(np.mean(np.square(decoded[:23] - actions[:23]))) < 1e-3
    assert weights.shape == (points, 2)
    assert weights[0, 0] == pytest.approx(1.0)
    assert weights[-1, 0] == pytest.approx(0.0)


def test_full_valid_mask_matches_unmasked_encoder() -> None:
    rng = np.random.default_rng(7)
    actions = rng.normal(size=(50, 14)).astype(np.float32)
    expected = spline_field.encode_actions(actions, control_horizon=50, control_points=16)
    actual = spline_field.encode_actions(
        actions,
        control_horizon=50,
        control_points=16,
        valid_timestep_mask=np.ones(50, dtype=np.float32),
    )
    np.testing.assert_array_equal(actual, expected)


def test_padding_mask_is_required_and_loss_uses_total_weight() -> None:
    actions = np.zeros((50, 2), dtype=np.float32)
    transform = transforms.RequireActionPaddingMask(50)
    with pytest.raises(ValueError, match="action_is_pad"):
        transform({"actions": actions})
    with pytest.raises(ValueError, match="at least one valid"):
        spline_field.encode_actions(
            actions,
            control_horizon=50,
            control_points=16,
            valid_timestep_mask=np.zeros(50, dtype=np.float32),
        )

    mask = jnp.asarray([[[1.0], [0.25]]])
    loss = pi0.reduce_action_loss(jnp.ones((1, 2, 1)), mask)
    assert loss.shape == (1, 2)
    assert float(jnp.mean(loss)) == pytest.approx(1.0)
