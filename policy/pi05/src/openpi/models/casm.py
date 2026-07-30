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
    "visual_phase_gate",
    "cross_output_shared_head",
]
VISUAL_PHASE_GATE_MODES = frozenset({"visual_phase_gate", "cross_output_shared_head"})
LEARNED_GATE_MODES = frozenset({"hard_gate", "gated_cross_attention", "usefulness_gate", *VISUAL_PHASE_GATE_MODES})
CROSS_ATTENTION_MODES = frozenset({"gated_cross_attention", "usefulness_gate"})
VALID_CASM_MODES = frozenset({"none", "hard_mask", *LEARNED_GATE_MODES})
CROSS_ATTENTION_OUTPUT_INIT = jax.nn.initializers.normal(1e-3)
ROBOT_ARM_DIM = 7


def coordination_target(phase_id: at.Int[at.Array, "*b p"]) -> at.Float[at.Array, "*b"]:
    return (phase_id[..., 0] == 0).astype(jnp.float32)


def async_target(phase_id: at.Int[at.Array, "*b p"]) -> at.Float[at.Array, "*b"]:
    return (phase_id[..., 0] != 0).astype(jnp.float32)


def binary_cross_entropy(
    probability: at.Float[at.Array, "*b"],
    target: at.Float[at.Array, "*b"],
) -> at.Float[at.Array, "*b"]:
    probability = jnp.clip(probability, 1e-6, 1 - 1e-6)
    return -(target * jnp.log(probability) + (1 - target) * jnp.log1p(-probability))


def binary_cross_entropy_with_logits(
    logits: at.Float[at.Array, "*b"],
    target: at.Float[at.Array, "*b"],
    positive_weight: float,
) -> at.Float[at.Array, "*b"]:
    if positive_weight <= 0:
        raise ValueError("positive weight must be positive")
    return (1 - target) * jax.nn.softplus(logits) + target * positive_weight * jax.nn.softplus(-logits)


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


def shared_single_arm_projection(hidden, kernel, bias):
    """Project either arm with the canonical first seven PI0.5 output channels."""
    if kernel.ndim != 2 or kernel.shape[1] < ROBOT_ARM_DIM:
        raise ValueError(f"invalid PI0.5 output kernel shape: {kernel.shape}")
    if bias.ndim != 1 or bias.shape[0] < ROBOT_ARM_DIM:
        raise ValueError(f"invalid PI0.5 output bias shape: {bias.shape}")
    return (
        jnp.einsum(
            "...d,df->...f",
            hidden,
            kernel[:, :ROBOT_ARM_DIM],
            precision=jax.lax.Precision.DEFAULT,
        )
        + bias[:ROBOT_ARM_DIM]
    )


def merge_shared_arm_vector_fields(left, right, action_dim: int):
    """Route shared 7D head outputs into the fixed ALOHA left/right slots."""
    expected = (*left.shape[:-1], ROBOT_ARM_DIM)
    if left.shape != expected or right.shape != expected:
        raise ValueError(f"shared arm output shape mismatch: {left.shape}, {right.shape}")
    if action_dim < 2 * ROBOT_ARM_DIM:
        raise ValueError(f"action dimension {action_dim} cannot hold two robot arms")
    merged = jnp.zeros((*left.shape[:-1], action_dim), dtype=left.dtype)
    merged = merged.at[..., :ROBOT_ARM_DIM].set(left)
    return merged.at[..., ROBOT_ARM_DIM : 2 * ROBOT_ARM_DIM].set(right)


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


class VisualProprioceptionGate(nnx.Module):
    """Predicts labels from pooled visual tokens and continuous robot state."""

    def __init__(
        self,
        visual_dim: int,
        state_dim: int,
        hidden_dim: int,
        *,
        output_dim: int = 1,
        rngs: nnx.Rngs,
    ):
        if output_dim < 1:
            raise ValueError("visual-proprioception output dimension must be positive")
        self.output_dim = output_dim
        self.visual_norm = nnx.LayerNorm(visual_dim, rngs=rngs)
        self.state_norm = nnx.LayerNorm(state_dim, rngs=rngs)
        self.visual_proj = nnx.Linear(visual_dim, hidden_dim, rngs=rngs)
        self.state_proj = nnx.Linear(state_dim, hidden_dim, rngs=rngs)
        self.fusion = nnx.Linear(2 * hidden_dim, hidden_dim, rngs=rngs)
        self.output = nnx.Linear(
            hidden_dim,
            output_dim,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )

    def __call__(
        self,
        visual_features: at.Float[at.Array, "*b v"],
        state: at.Float[at.Array, "*b d"],
    ) -> at.Float[at.Array, "*b"]:
        # Semantic objectives train only compact heads. This keeps 50-demo
        # supervision from perturbing the pretrained visual representation.
        visual_features = jax.lax.stop_gradient(visual_features)
        state = jax.lax.stop_gradient(state)
        visual = jax.nn.gelu(self.visual_proj(self.visual_norm(visual_features)))
        proprioception = jax.nn.gelu(self.state_proj(self.state_norm(state)))
        fused = jnp.concatenate([visual, proprioception], axis=-1)
        logits = self.output(jax.nn.gelu(self.fusion(fused)))
        if self.output_dim == 1:
            return logits[..., 0]
        return logits


class BiasFreeLinear(nnx.Module):
    """Linear projection whose parameter tree contains only the kernel."""

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        self.kernel = nnx.Param(
            jax.nn.initializers.lecun_normal()(
                rngs.params(),
                (in_features, out_features),
                jnp.float32,
            )
        )

    def __call__(self, inputs: at.Float[at.Array, "... d"]) -> at.Float[at.Array, "... f"]:
        return jnp.einsum(
            "...d,df->...f",
            inputs,
            self.kernel.value,
            precision=jax.lax.Precision.DEFAULT,
        )


class LowRankBidirectionalCrossResidual(nnx.Module):
    """A symmetric low-rank cross-arm residual applied at matching chunk steps."""

    def __init__(self, width: int, rank: int, *, rngs: nnx.Rngs):
        if rank < 1:
            raise ValueError("cross-output rank must be positive")
        self.down = BiasFreeLinear(width, rank, rngs=rngs)
        self.up = nnx.Linear(
            rank,
            width,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, left, right, sync_probability):
        weight = sync_probability[..., None, None].astype(left.dtype)
        right_message = self.up(nnx.gelu(self.down(right)))
        left_message = self.up(nnx.gelu(self.down(left)))
        return left + weight * right_message, right + weight * left_message


class GatedBidirectionalCrossAttention(nnx.Module):
    def __init__(self, width: int, attention_dim: int, *, rngs: nnx.Rngs):
        if attention_dim < 1:
            raise ValueError("cross-attention dimension must be positive")
        self.attention_dim = attention_dim
        self.left_query = BiasFreeLinear(width, attention_dim, rngs=rngs)
        self.right_query = BiasFreeLinear(width, attention_dim, rngs=rngs)
        self.left_key = BiasFreeLinear(width, attention_dim, rngs=rngs)
        self.right_key = BiasFreeLinear(width, attention_dim, rngs=rngs)
        self.left_value = BiasFreeLinear(width, attention_dim, rngs=rngs)
        self.right_value = BiasFreeLinear(width, attention_dim, rngs=rngs)
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
