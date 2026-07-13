from __future__ import annotations

from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.shared import array_typing as at

CasmMode = Literal[
    "none",
    "hard_mask",
    "hard_gate",
    "gated_cross_attention",
    "usefulness_gate",
]
LEARNED_GATE_MODES = frozenset({"hard_gate", "gated_cross_attention", "usefulness_gate"})
CROSS_ATTENTION_MODES = frozenset({"gated_cross_attention", "usefulness_gate"})
VALID_CASM_MODES = frozenset({"none", "hard_mask", *LEARNED_GATE_MODES})
CROSS_ATTENTION_OUTPUT_INIT = jax.nn.initializers.normal(1e-3)


def coordination_target(phase_id: at.Int[at.Array, "*b p"]) -> at.Float[at.Array, "*b"]:
    return (phase_id[..., 0] == 0).astype(jnp.float32)


def binary_cross_entropy(
    probability: at.Float[at.Array, "*b"],
    target: at.Float[at.Array, "*b"],
) -> at.Float[at.Array, "*b"]:
    probability = jnp.clip(probability, 1e-6, 1 - 1e-6)
    return -(target * jnp.log(probability) + (1 - target) * jnp.log1p(-probability))


def usefulness_target(
    communication_off_error: at.Float[at.Array, "*b"],
    communication_on_error: at.Float[at.Array, "*b"],
    temperature: float,
) -> at.Float[at.Array, "*b"]:
    if temperature <= 0:
        raise ValueError("usefulness temperature must be positive")
    advantage = (communication_off_error - communication_on_error) / temperature
    return jax.lax.stop_gradient(jax.nn.sigmoid(advantage))


def select_joint_route(gate: at.Float[at.Array, "*b"]) -> at.Bool[at.Array, "*b"]:
    return gate >= 0.5


def isolate_arm_observation(
    observation: _model.Observation,
    arm: str,
) -> _model.Observation:
    if arm not in ("left", "right"):
        raise ValueError(f"unknown arm: {arm}")
    masked_wrist = "right_wrist_0_rgb" if arm == "left" else "left_wrist_0_rgb"
    image_masks = dict(observation.image_masks)
    image_masks[masked_wrist] = jnp.zeros_like(image_masks[masked_wrist])
    return observation.replace(image_masks=image_masks)


def hard_mask_arm_observation(
    observation: _model.Observation,
    arm: str,
) -> _model.Observation:
    if observation.phase_id is None:
        raise ValueError("hard-mask CASM requires phase_id")
    isolated = isolate_arm_observation(observation, arm)
    sync_phase = observation.phase_id[..., 0] == 0
    masked_wrist = "right_wrist_0_rgb" if arm == "left" else "left_wrist_0_rgb"
    image_masks = dict(isolated.image_masks)
    image_masks[masked_wrist] = jnp.where(
        sync_phase,
        observation.image_masks[masked_wrist],
        isolated.image_masks[masked_wrist],
    )
    return isolated.replace(image_masks=image_masks)


def concatenate_observations(
    left: _model.Observation,
    right: _model.Observation,
) -> _model.Observation:
    return jax.tree.map(
        lambda left_value, right_value: jnp.concatenate([left_value, right_value], axis=0),
        left,
        right,
    )


def isolated_stream_action_inputs(noisy_actions: _model.Actions) -> _model.Actions:
    indices = jnp.arange(noisy_actions.shape[-1])
    left = noisy_actions * (indices < 7)
    right = noisy_actions * ((indices >= 7) & (indices < 14))
    return jnp.concatenate([left, right], axis=0)


def hard_mask_stream_action_inputs(
    noisy_actions: _model.Actions,
    phase_id: at.Int[at.Array, "*b p"],
) -> _model.Actions:
    isolated = isolated_stream_action_inputs(noisy_actions)
    batch_size = noisy_actions.shape[0]
    sync_phase = (phase_id[..., :1] == 0)[..., None]
    left = jnp.where(sync_phase, noisy_actions, isolated[:batch_size])
    right = jnp.where(sync_phase, noisy_actions, isolated[batch_size:])
    return jnp.concatenate([left, right], axis=0)


def merge_stream_vector_fields(
    streams: _model.Actions,
    batch_size: int,
) -> _model.Actions:
    left = streams[:batch_size]
    right = streams[batch_size:]
    indices = jnp.arange(streams.shape[-1])
    return jnp.where(indices < 7, left, 0.0) + jnp.where(
        (indices >= 7) & (indices < 14),
        right,
        0.0,
    )


class CooperationGate(nnx.Module):
    def __init__(self, state_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        self.input = nnx.Linear(state_dim, hidden_dim, rngs=rngs)
        self.output = nnx.Linear(
            hidden_dim,
            1,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, state: at.Float[at.Array, "*b d"]) -> at.Float[at.Array, "*b"]:
        return jax.nn.sigmoid(self.output(nnx.swish(self.input(state)))[..., 0])


class GatedBidirectionalCrossAttention(nnx.Module):
    def __init__(self, width: int, attention_dim: int, *, rngs: nnx.Rngs):
        if attention_dim < 1:
            raise ValueError("cross-attention dimension must be positive")
        self.attention_dim = attention_dim
        self.left_query = nnx.Linear(width, attention_dim, use_bias=False, rngs=rngs)
        self.right_query = nnx.Linear(width, attention_dim, use_bias=False, rngs=rngs)
        self.left_key = nnx.Linear(width, attention_dim, use_bias=False, rngs=rngs)
        self.right_key = nnx.Linear(width, attention_dim, use_bias=False, rngs=rngs)
        self.left_value = nnx.Linear(width, attention_dim, use_bias=False, rngs=rngs)
        self.right_value = nnx.Linear(width, attention_dim, use_bias=False, rngs=rngs)
        self.left_output = nnx.Linear(attention_dim, width, kernel_init=CROSS_ATTENTION_OUTPUT_INIT, rngs=rngs)
        self.right_output = nnx.Linear(attention_dim, width, kernel_init=CROSS_ATTENTION_OUTPUT_INIT, rngs=rngs)

    def _message(self, query, key, value):
        scores = jnp.einsum("bhd,bkd->bhk", query, key) / self.attention_dim**0.5
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(value.dtype)
        return jnp.einsum("bhk,bkd->bhd", weights, value)

    def __call__(self, left, right, gate):
        right_to_left = self._message(
            self.left_query(left),
            self.right_key(right),
            self.right_value(right),
        )
        left_to_right = self._message(
            self.right_query(right),
            self.left_key(left),
            self.left_value(left),
        )
        weight = gate[..., None, None].astype(left.dtype)
        left = left + weight * self.left_output(right_to_left)
        right = right + weight * self.right_output(left_to_right)
        return left, right
