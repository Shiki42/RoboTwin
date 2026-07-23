import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi.models import casm
from openpi.models import pi0
from openpi.models_pytorch import casm_pytorch


def test_action_mask_reduction_matches_pytorch():
    squared_error = np.asarray(
        [
            [[1.0, 9.0, 25.0], [4.0, 16.0, 36.0]],
            [[9.0, 4.0, 1.0], [16.0, 4.0, 1.0]],
        ],
        dtype=np.float32,
    )
    action_mask = np.asarray(
        [
            [[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
        ],
        dtype=np.float32,
    )

    jax_loss = pi0.reduce_action_loss(jnp.asarray(squared_error), jnp.asarray(action_mask))
    torch_loss = casm_pytorch.reduce_action_loss(
        torch.from_numpy(squared_error),
        torch.from_numpy(action_mask),
    )

    np.testing.assert_allclose(np.asarray(jax_loss), torch_loss.numpy(), rtol=0, atol=0)


def test_visual_phase_gate_targets_and_weighted_bce_match_pytorch():
    phase_id = np.asarray([[0], [1], [2], [0]], dtype=np.int32)
    logits = np.asarray([-2.0, -0.25, 0.5, 3.0], dtype=np.float32)
    positive_weight = 2.5

    jax_target = casm.async_target(jnp.asarray(phase_id))
    torch_target = casm_pytorch.async_target(torch.from_numpy(phase_id))
    jax_loss = casm.binary_cross_entropy_with_logits(
        jnp.asarray(logits),
        jax_target,
        positive_weight,
    )
    torch_loss = casm_pytorch.binary_cross_entropy_with_logits(
        torch.from_numpy(logits),
        torch_target,
        positive_weight,
    )

    np.testing.assert_array_equal(np.asarray(jax_target), torch_target.numpy())
    np.testing.assert_allclose(np.asarray(jax_loss), torch_loss.numpy(), rtol=1e-6, atol=1e-6)


def test_visual_phase_gate_total_and_metrics_match_pytorch():
    action_loss = np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    logits = np.zeros(2, dtype=np.float32)
    phase_id = np.asarray([[1], [0]], dtype=np.int32)
    gate_loss_weight = 0.2
    positive_weight = 3.0

    target = casm.async_target(jnp.asarray(phase_id))
    gate_loss = casm.binary_cross_entropy_with_logits(
        jnp.asarray(logits),
        target,
        positive_weight,
    )
    jax_total = jnp.asarray(action_loss) + gate_loss_weight * gate_loss[..., None]
    torch_result = casm_pytorch.visual_phase_gate_loss(
        torch.from_numpy(action_loss),
        torch.from_numpy(logits),
        torch.from_numpy(phase_id),
        gate_loss_weight=gate_loss_weight,
        gate_positive_weight=positive_weight,
    )

    np.testing.assert_allclose(np.asarray(jax_total), torch_result.total.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(
            [
                jnp.mean(action_loss),
                jnp.mean(gate_loss),
                jnp.mean((jax.nn.sigmoid(logits) >= 0.5) == (target >= 0.5)),
            ]
        ),
        np.asarray(
            [
                torch_result.metrics["action_loss"],
                torch_result.metrics["gate_loss"],
                torch_result.metrics["gate_accuracy"],
            ]
        ),
        rtol=1e-6,
        atol=1e-6,
    )
