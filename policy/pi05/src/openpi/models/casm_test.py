from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import casm
from openpi.models import model


def observation(phase_ids=(1, 0)) -> model.Observation:
    batch = len(phase_ids)
    masks = {
        "base_0_rgb": jnp.ones(batch, dtype=jnp.bool_),
        "left_wrist_0_rgb": jnp.ones(batch, dtype=jnp.bool_),
        "right_wrist_0_rgb": jnp.ones(batch, dtype=jnp.bool_),
    }
    return model.Observation(
        images={key: jnp.zeros((batch, 2, 2, 3), dtype=jnp.float32) for key in masks},
        image_masks=masks,
        state=jnp.zeros((batch, 32), dtype=jnp.float32),
        phase_id=jnp.asarray(phase_ids, dtype=jnp.int32)[:, None],
    )


def test_coordination_target_and_usefulness_target():
    phase = jnp.array([[0], [1], [2]], dtype=jnp.int32)
    assert casm.coordination_target(phase).tolist() == [1.0, 0.0, 0.0]
    target = casm.usefulness_target(jnp.array([2.0, 1.0]), jnp.array([1.0, 2.0]), 0.5)
    assert target[0] > 0.5
    assert target[1] < 0.5


def test_soft_mixture_gate_endpoints_and_gradient():
    factorized = jnp.zeros((2, 3, 4))
    joint = jnp.ones((2, 3, 4))
    assert np.array_equal(casm.mix_vector_fields(factorized, joint, jnp.zeros(2)), factorized)
    assert np.array_equal(casm.mix_vector_fields(factorized, joint, jnp.ones(2)), joint)
    gradient = jax.grad(lambda gate: jnp.sum(casm.mix_vector_fields(factorized, joint, gate)))(jnp.full(2, 0.5))
    assert np.all(np.asarray(gradient) > 0)


def test_hard_mask_keeps_joint_context_only_for_sync_samples():
    obs = observation()
    left = casm.hard_mask_arm_observation(obs, "left")
    assert left.image_masks["right_wrist_0_rgb"].tolist() == [False, True]
    isolated = casm.isolate_arm_observation(obs, "left")
    assert isolated.image_masks["right_wrist_0_rgb"].tolist() == [False, False]


def test_stream_action_routing_and_merge():
    actions = jnp.arange(2 * 1 * 32, dtype=jnp.float32).reshape(2, 1, 32)
    routed = casm.hard_mask_stream_action_inputs(actions, jnp.array([[1], [0]]))
    left, right = np.asarray(routed[:2]), np.asarray(routed[2:])
    assert np.all(left[0, :, 7:] == 0)
    assert np.all(right[0, :, :7] == 0)
    assert np.array_equal(left[1], np.asarray(actions[1]))
    streams = jnp.concatenate([jnp.ones((2, 1, 32)), jnp.full((2, 1, 32), 2.0)])
    merged = np.asarray(casm.merge_stream_vector_fields(streams, 2))
    assert np.all(merged[..., :7] == 1)
    assert np.all(merged[..., 7:14] == 2)
    assert np.all(merged[..., 14:] == 0)


def test_gate_and_cross_attention_shapes_and_zero_gate_identity():
    gate_model = casm.CooperationGate(32, 8, rngs=nnx.Rngs(0))
    probability = gate_model(jnp.zeros((2, 32)))
    assert probability.shape == (2,)
    assert np.allclose(probability, 0.5)

    module = casm.GatedBidirectionalCrossAttention(16, 4, rngs=nnx.Rngs(1))
    left = jax.random.normal(jax.random.key(2), (2, 5, 16))
    right = jax.random.normal(jax.random.key(3), (2, 5, 16))
    left_off, right_off = module(left, right, jnp.zeros(2))
    assert np.array_equal(left_off, left)
    assert np.array_equal(right_off, right)
    left_on, right_on = module(left, right, jnp.ones(2))
    assert not np.array_equal(left_on, left)
    assert not np.array_equal(right_on, right)
