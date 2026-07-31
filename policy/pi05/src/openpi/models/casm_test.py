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
    assert casm.async_target(phase).tolist() == [0.0, 1.0, 1.0]
    target = casm.usefulness_target(jnp.array([2.0, 1.0]), jnp.array([1.0, 2.0]), 0.5)
    assert target[0] > 0.5
    assert target[1] < 0.5


def test_hard_gate_route_threshold():
    selected = casm.select_joint_route(jnp.array([0.49, 0.5]))
    assert selected.tolist() == [False, True]


def test_weighted_bce_with_logits_is_finite_and_weights_async_examples():
    logits = jnp.zeros(2)
    target = jnp.array([0.0, 1.0])
    unweighted = casm.binary_cross_entropy_with_logits(logits, target, 1.0)
    weighted = casm.binary_cross_entropy_with_logits(logits, target, 3.0)
    assert np.isfinite(weighted).all()
    assert weighted[0] == unweighted[0]
    assert weighted[1] == 3 * unweighted[1]


def test_hard_mask_keeps_joint_context_only_for_sync_samples():
    obs = observation()
    left = casm.hard_mask_arm_observation(obs, "left")
    assert left.image_masks["right_wrist_0_rgb"].tolist() == [False, True]
    isolated = casm.isolate_arm_observation(obs, "left")
    assert isolated.image_masks["right_wrist_0_rgb"].tolist() == [False, False]


def test_skill_observation_keeps_only_native_arm_state_and_wrist():
    obs = observation().replace(state=jnp.arange(64, dtype=jnp.float32).reshape(2, 32))
    left = casm.isolate_arm_skill_observation(obs, "left")
    right = casm.isolate_arm_skill_observation(obs, "right")

    assert np.array_equal(left.state[..., :7], obs.state[..., :7])
    assert np.all(np.asarray(left.state[..., 7:]) == 0)
    assert np.all(np.asarray(right.state[..., :7]) == 0)
    assert np.array_equal(right.state[..., 7:14], obs.state[..., 7:14])
    assert np.all(np.asarray(right.state[..., 14:]) == 0)


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


def test_native_per_arm_projection_uses_both_pretrained_output_bases():
    left_hidden = jnp.ones((1, 1, 2), dtype=jnp.float32)
    right_hidden = jnp.full((1, 1, 2), 2.0, dtype=jnp.float32)
    kernel = jnp.zeros((2, 32), dtype=jnp.float32)
    kernel = kernel.at[:, :7].set(3.0)
    kernel = kernel.at[:, 7:14].set(5.0)
    bias = jnp.arange(32, dtype=jnp.float32)

    projected = casm.native_per_arm_projection(left_hidden, right_hidden, kernel, bias, action_dim=32)
    assert np.allclose(projected[..., :7], 6.0 + bias[:7])
    assert np.allclose(projected[..., 7:14], 20.0 + bias[7:14])
    assert np.all(np.asarray(projected[..., 14:]) == 0)

    def loss(candidate_kernel):
        return jnp.sum(casm.native_per_arm_projection(left_hidden, right_hidden, candidate_kernel, bias, 32))

    gradient = jax.grad(loss)(kernel)
    assert np.all(np.asarray(gradient[:, :7]) == 1.0)
    assert np.all(np.asarray(gradient[:, 7:14]) == 2.0)
    assert np.all(np.asarray(gradient[:, 14:]) == 0.0)


def test_shared_single_arm_head_uses_canonical_columns_and_routes_both_arms():
    hidden = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4)
    kernel = jnp.arange(4 * 32, dtype=jnp.float32).reshape(4, 32)
    bias = jnp.arange(32, dtype=jnp.float32)
    projected = casm.shared_single_arm_projection(hidden, kernel, bias)
    expected = jnp.einsum("...d,df->...f", hidden, kernel[:, :7]) + bias[:7]
    assert np.array_equal(projected, expected)

    left = projected
    right = projected + 1000
    routed = casm.merge_shared_arm_vector_fields(left, right, action_dim=32)
    assert np.array_equal(routed[..., :7], left)
    assert np.array_equal(routed[..., 7:14], right)
    assert np.all(np.asarray(routed[..., 14:]) == 0)


def test_shared_head_accumulates_both_arm_gradients_into_canonical_channels():
    left_hidden = jnp.ones((1, 1, 4), dtype=jnp.float32)
    right_hidden = jnp.full((1, 1, 4), 2.0, dtype=jnp.float32)
    kernel = jnp.zeros((4, 32), dtype=jnp.float32)
    bias = jnp.zeros(32, dtype=jnp.float32)

    def loss(candidate_kernel):
        left = casm.shared_single_arm_projection(left_hidden, candidate_kernel, bias)
        right = casm.shared_single_arm_projection(right_hidden, candidate_kernel, bias)
        return jnp.sum(casm.merge_shared_arm_vector_fields(left, right, 32))

    gradient = jax.grad(loss)(kernel)

    assert np.all(np.asarray(gradient[:, :7]) == 3.0)
    assert np.all(np.asarray(gradient[:, 7:]) == 0.0)


def test_low_rank_cross_residual_gate_and_initialization_contract():
    module = casm.LowRankBidirectionalCrossResidual(4, 2, rngs=nnx.Rngs(4))
    left = jnp.ones((2, 3, 4))
    right = jnp.full((2, 3, 4), 2.0)

    initial_left, initial_right = module(left, right, jnp.ones(2))
    assert np.array_equal(initial_left, left)
    assert np.array_equal(initial_right, right)

    module.down.kernel.value = jnp.ones_like(module.down.kernel.value)
    module.up.kernel.value = jnp.ones_like(module.up.kernel.value)
    module.up.bias.value = jnp.zeros_like(module.up.bias.value)
    off_left, off_right = module(left, right, jnp.zeros(2))
    assert np.array_equal(off_left, left)
    assert np.array_equal(off_right, right)

    on_left, on_right = module(left, right, jnp.ones(2))
    assert np.all(np.asarray(on_left) > np.asarray(left))
    assert np.all(np.asarray(on_right) > np.asarray(right))
    assert np.all(np.asarray(on_left - left) > np.asarray(on_right - right))


def test_per_arm_residuals_are_zero_initialized_and_independent():
    module = casm.PerArmLowRankResidual(4, 2, rngs=nnx.Rngs(5))
    left = jnp.ones((1, 3, 4))
    right = jnp.full((1, 3, 4), 2.0)

    initial_left, initial_right = module(left, right)
    assert np.array_equal(initial_left, left)
    assert np.array_equal(initial_right, right)

    module.left_down.kernel.value = jnp.ones_like(module.left_down.kernel.value)
    module.right_down.kernel.value = jnp.ones_like(module.right_down.kernel.value)
    module.left_up.kernel.value = jnp.ones_like(module.left_up.kernel.value)
    module.right_up.kernel.value = jnp.full_like(module.right_up.kernel.value, 2.0)

    first_left, first_right = module(left, right)
    second_left, second_right = module(3.0 * left, right)
    assert not np.array_equal(first_left, second_left)
    assert np.array_equal(first_right, second_right)
    assert np.all(np.asarray(first_left) > np.asarray(left))
    assert np.all(np.asarray(first_right) > np.asarray(right))


def test_shared_arm_head_rejects_invalid_shapes():
    with np.testing.assert_raises(ValueError):
        casm.shared_single_arm_projection(jnp.ones((1, 4)), jnp.ones((4, 6)), jnp.ones(6))
    with np.testing.assert_raises(ValueError):
        casm.merge_shared_arm_vector_fields(jnp.ones((1, 6)), jnp.ones((1, 6)), 32)
    with np.testing.assert_raises(ValueError):
        casm.merge_shared_arm_vector_fields(jnp.ones((1, 7)), jnp.ones((1, 7)), 13)


def test_gate_and_cross_attention_shapes_and_zero_gate_identity():
    gate_model = casm.CooperationGate(32, 8, rngs=nnx.Rngs(0))
    probability = gate_model(jnp.zeros((2, 32)))
    assert probability.shape == (2,)
    assert np.allclose(probability, 0.5)

    module = casm.GatedBidirectionalCrossAttention(16, 4, rngs=nnx.Rngs(1))
    second_module = casm.GatedBidirectionalCrossAttention(16, 4, rngs=nnx.Rngs(2))
    assert module.left_output.kernel_init is second_module.left_output.kernel_init
    assert module.right_output.kernel_init is second_module.right_output.kernel_init
    state = nnx.state(module).to_pure_dict()
    for name in ("left_query", "right_query", "left_key", "right_key", "left_value", "right_value"):
        assert set(state[name]) == {"kernel"}

    left = jax.random.normal(jax.random.key(2), (2, 5, 16))
    right = jax.random.normal(jax.random.key(3), (2, 5, 16))
    left_off, right_off = module(left, right, jnp.zeros(2))
    assert np.array_equal(left_off, left)
    assert np.array_equal(right_off, right)
    left_on, right_on = module(left, right, jnp.ones(2))
    assert not np.array_equal(left_on, left)
    assert not np.array_equal(right_on, right)


def test_visual_proprioception_gate_shapes_and_detaches_inputs():
    gate = casm.VisualProprioceptionGate(16, 32, 8, rngs=nnx.Rngs(3))
    visual = jnp.ones((2, 16))
    state = jnp.ones((2, 32))
    logits = gate(visual, state)
    assert logits.shape == (2,)
    assert np.allclose(logits, 0)

    visual_grad, state_grad = jax.grad(
        lambda visual_value, state_value: gate(visual_value, state_value).sum(),
        argnums=(0, 1),
    )(visual, state)
    assert np.allclose(visual_grad, 0)
    assert np.allclose(state_grad, 0)
