import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import casm
from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def reduce_action_loss(
    squared_error: at.Float[at.Array, "*b ah ad"],
    action_mask: at.Float[at.Array, "*b ah ad"] | None,
) -> at.Float[at.Array, "*b ah"]:
    if action_mask is None:
        return jnp.mean(squared_error, axis=-1)
    mask = jnp.asarray(action_mask, dtype=squared_error.dtype)
    if mask.shape != squared_error.shape:
        raise ValueError(f"action mask shape mismatch: {mask.shape} != {squared_error.shape}")
    denominator = jnp.maximum(jnp.sum(mask, axis=-1), 1.0)
    return jnp.sum(squared_error * mask, axis=-1) / denominator


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.casm_mode = config.casm_mode
        self.gate_loss_weight = config.gate_loss_weight
        self.usefulness_loss_weight = config.usefulness_loss_weight
        self.phase_prior_loss_weight = config.phase_prior_loss_weight
        self.usefulness_temperature = config.usefulness_temperature
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        if self.casm_mode in casm.LEARNED_GATE_MODES:
            self.cooperation_gate = casm.CooperationGate(
                config.action_dim,
                config.coordination_gate_hidden_dim,
                rngs=rngs,
            )
        if self.casm_mode in casm.CROSS_ATTENTION_MODES:
            self.cross_attention = casm.GatedBidirectionalCrossAttention(
                action_expert_config.width,
                config.cross_attention_dim,
                rngs=rngs,
            )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _hidden_field(
        self,
        observation: _model.Observation,
        noisy_actions: _model.Actions,
        time: at.Float[at.Array, " b"],
    ):
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation,
            noisy_actions,
            time,
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        return suffix_out[:, -self.action_horizon :]

    def _vector_field(
        self,
        observation: _model.Observation,
        noisy_actions: _model.Actions,
        time: at.Float[at.Array, " b"],
    ) -> _model.Actions:
        return self.action_out_proj(self._hidden_field(observation, noisy_actions, time))

    def _cooperation_probability(self, observation: _model.Observation):
        return self.cooperation_gate(observation.state)

    def _stream_hidden_fields(
        self,
        observation: _model.Observation,
        noisy_actions: _model.Actions,
        time: at.Float[at.Array, " b"],
        *,
        hard_mask: bool,
    ):
        if hard_mask:
            if observation.phase_id is None:
                raise ValueError("hard-mask CASM requires phase_id")
            left_observation = casm.hard_mask_arm_observation(observation, "left")
            right_observation = casm.hard_mask_arm_observation(observation, "right")
            stream_actions = casm.hard_mask_stream_action_inputs(noisy_actions, observation.phase_id)
        else:
            left_observation = casm.isolate_arm_observation(observation, "left")
            right_observation = casm.isolate_arm_observation(observation, "right")
            stream_actions = casm.isolated_stream_action_inputs(noisy_actions)
        stream_observation = casm.concatenate_observations(left_observation, right_observation)
        stream_time = jnp.concatenate([time, time], axis=0)
        stream_hidden = self._hidden_field(stream_observation, stream_actions, stream_time)
        batch_size = noisy_actions.shape[0]
        return stream_hidden[:batch_size], stream_hidden[batch_size:]

    def _project_stream_hidden(self, left_hidden, right_hidden, batch_size: int):
        streams = self.action_out_proj(jnp.concatenate([left_hidden, right_hidden], axis=0))
        return casm.merge_stream_vector_fields(streams, batch_size)

    def _factorized_vector_field(
        self,
        observation: _model.Observation,
        noisy_actions: _model.Actions,
        time: at.Float[at.Array, " b"],
        *,
        hard_mask: bool,
    ) -> _model.Actions:
        left_hidden, right_hidden = self._stream_hidden_fields(
            observation,
            noisy_actions,
            time,
            hard_mask=hard_mask,
        )
        return self._project_stream_hidden(left_hidden, right_hidden, noisy_actions.shape[0])

    def _cross_attention_fields(self, left_hidden, right_hidden, gate, batch_size: int):
        left_fused, right_fused = self.cross_attention(left_hidden, right_hidden, gate)
        return self._project_stream_hidden(left_fused, right_fused, batch_size)

    def _routed_vector_fields(
        self,
        observation: _model.Observation,
        noisy_actions: _model.Actions,
        time: at.Float[at.Array, " b"],
    ):
        if self.casm_mode == "none":
            return self._vector_field(observation, noisy_actions, time), None, None
        if self.casm_mode == "hard_mask":
            factorized = self._factorized_vector_field(
                observation,
                noisy_actions,
                time,
                hard_mask=True,
            )
            return factorized, None, None

        gate = self._cooperation_probability(observation)
        if self.casm_mode == "hard_gate":
            if observation.phase_id is None:
                raise ValueError("hard-gate CASM requires phase_id during training")
            if noisy_actions.shape[0] != 1:
                raise ValueError("hard-gate CASM requires batch size one during training")
            use_joint = casm.coordination_target(observation.phase_id)[0] >= 0.5
            predicted = jax.lax.cond(
                use_joint,
                lambda _: self._vector_field(observation, noisy_actions, time),
                lambda _: self._factorized_vector_field(
                    observation,
                    noisy_actions,
                    time,
                    hard_mask=False,
                ),
                operand=None,
            )
            return predicted, gate, None

        left_hidden, right_hidden = self._stream_hidden_fields(
            observation,
            noisy_actions,
            time,
            hard_mask=False,
        )
        factorized = self._project_stream_hidden(
            left_hidden,
            right_hidden,
            noisy_actions.shape[0],
        )
        predicted = self._cross_attention_fields(
            left_hidden,
            right_hidden,
            gate,
            noisy_actions.shape[0],
        )
        if self.casm_mode == "gated_cross_attention":
            return predicted, gate, None
        if self.casm_mode == "usefulness_gate":
            zeros = jnp.zeros_like(gate)
            ones = jnp.ones_like(gate)
            communication_off = self._cross_attention_fields(
                left_hidden,
                right_hidden,
                zeros,
                noisy_actions.shape[0],
            )
            communication_on = self._cross_attention_fields(
                left_hidden,
                right_hidden,
                ones,
                noisy_actions.shape[0],
            )
            return predicted, gate, (communication_off, communication_on)
        raise ValueError(f"unknown CASM mode: {self.casm_mode}")

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        v_t, gate, alternatives = self._routed_vector_fields(observation, x_t, time)
        action_loss = reduce_action_loss(jnp.square(v_t - u_t), observation.action_mask)
        if gate is None:
            return action_loss
        if observation.phase_id is None:
            raise ValueError("learned CASM gates require phase_id during training")

        phase_target = casm.coordination_target(observation.phase_id)
        phase_loss = casm.binary_cross_entropy(gate, phase_target)[..., None]
        if self.casm_mode != "usefulness_gate":
            return action_loss + self.gate_loss_weight * phase_loss
        if alternatives is None:
            raise ValueError("usefulness gate requires communication on/off predictions")
        communication_off, communication_on = alternatives
        off_error = jnp.mean(
            reduce_action_loss(jnp.square(communication_off - u_t), observation.action_mask),
            axis=-1,
        )
        on_error = jnp.mean(
            reduce_action_loss(jnp.square(communication_on - u_t), observation.action_mask),
            axis=-1,
        )
        target = casm.usefulness_target(off_error, on_error, self.usefulness_temperature)
        usefulness_loss = casm.binary_cross_entropy(gate, target)[..., None]
        return action_loss + self.usefulness_loss_weight * usefulness_loss + self.phase_prior_loss_weight * phase_loss

    def _prepare_prefix(self, observation: _model.Observation):
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )
        return prefix_tokens, prefix_mask, kv_cache

    def _cached_hidden_field(
        self,
        observation: _model.Observation,
        noisy_actions: _model.Actions,
        time: at.Float[at.Array, " b"],
        prefix_tokens,
        prefix_mask,
        kv_cache,
    ):
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation,
            noisy_actions,
            time,
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        return suffix_out[:, -self.action_horizon :]

    def _cached_vector_field(
        self,
        observation: _model.Observation,
        noisy_actions: _model.Actions,
        time: at.Float[at.Array, " b"],
        *prefix,
    ) -> _model.Actions:
        hidden = self._cached_hidden_field(observation, noisy_actions, time, *prefix)
        return self.action_out_proj(hidden)

    def _integrate_actions(self, noise, num_steps, vector_field) -> _model.Actions:
        dt = -1.0 / num_steps
        batch_size = noise.shape[0]

        def step(carry):
            x_t, time = carry
            batch_time = jnp.broadcast_to(time, batch_size)
            v_t = vector_field(x_t, batch_time)
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def _cached_factorized_vector_field(
        self,
        stream_observation: _model.Observation,
        x_t: _model.Actions,
        batch_time: at.Float[at.Array, " b"],
        stream_prefix,
    ) -> _model.Actions:
        batch_size = x_t.shape[0]
        stream_actions = casm.isolated_stream_action_inputs(x_t)
        stream_time = jnp.concatenate([batch_time, batch_time], axis=0)
        stream_hidden = self._cached_hidden_field(
            stream_observation,
            stream_actions,
            stream_time,
            *stream_prefix,
        )
        left_hidden = stream_hidden[:batch_size]
        right_hidden = stream_hidden[batch_size:]
        return self._project_stream_hidden(left_hidden, right_hidden, batch_size)

    def _sample_hard_gate_actions(
        self,
        observation: _model.Observation,
        noise: _model.Actions,
        num_steps: int | at.Int[at.Array, ""],
    ) -> _model.Actions:
        if observation.state.shape[0] != 1:
            raise ValueError("hard-gate CASM requires batch size one during sampling")
        gate = self._cooperation_probability(observation)

        def joint_route(_):
            joint_prefix = self._prepare_prefix(observation)
            return self._integrate_actions(
                noise,
                num_steps,
                lambda x_t, batch_time: self._cached_vector_field(
                    observation,
                    x_t,
                    batch_time,
                    *joint_prefix,
                ),
            )

        def factorized_route(_):
            stream_observation = casm.concatenate_observations(
                casm.isolate_arm_observation(observation, "left"),
                casm.isolate_arm_observation(observation, "right"),
            )
            stream_prefix = self._prepare_prefix(stream_observation)
            return self._integrate_actions(
                noise,
                num_steps,
                lambda x_t, batch_time: self._cached_factorized_vector_field(
                    stream_observation,
                    x_t,
                    batch_time,
                    stream_prefix,
                ),
            )

        return jax.lax.cond(casm.select_joint_route(gate)[0], joint_route, factorized_route, operand=None)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        if self.casm_mode == "hard_mask" and observation.phase_id is None:
            raise ValueError("hard-mask CASM requires phase_id")
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        if self.casm_mode == "hard_gate":
            return self._sample_hard_gate_actions(observation, noise, num_steps)

        joint_prefix = None
        stream_prefix = None
        stream_observation = None
        gate = None
        if self.casm_mode in casm.CROSS_ATTENTION_MODES:
            gate = self._cooperation_probability(observation)
        if self.casm_mode in {"hard_mask", *casm.CROSS_ATTENTION_MODES}:
            observation_fn = (
                casm.hard_mask_arm_observation if self.casm_mode == "hard_mask" else casm.isolate_arm_observation
            )
            stream_observation = casm.concatenate_observations(
                observation_fn(observation, "left"),
                observation_fn(observation, "right"),
            )
            stream_prefix = self._prepare_prefix(stream_observation)
        if self.casm_mode == "none":
            joint_prefix = self._prepare_prefix(observation)

        def step(carry):
            x_t, time = carry
            batch_time = jnp.broadcast_to(time, batch_size)
            if self.casm_mode == "none":
                v_t = self._cached_vector_field(
                    observation,
                    x_t,
                    batch_time,
                    *joint_prefix,
                )
                return x_t + dt * v_t, time + dt

            if self.casm_mode == "hard_mask":
                stream_actions = casm.hard_mask_stream_action_inputs(x_t, observation.phase_id)
            else:
                stream_actions = casm.isolated_stream_action_inputs(x_t)
            stream_time = jnp.concatenate([batch_time, batch_time], axis=0)
            stream_hidden = self._cached_hidden_field(
                stream_observation,
                stream_actions,
                stream_time,
                *stream_prefix,
            )
            left_hidden = stream_hidden[:batch_size]
            right_hidden = stream_hidden[batch_size:]
            factorized = self._project_stream_hidden(left_hidden, right_hidden, batch_size)
            if self.casm_mode == "hard_mask":
                v_t = factorized
            else:
                v_t = self._cross_attention_fields(
                    left_hidden,
                    right_hidden,
                    gate,
                    batch_size,
                )
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
