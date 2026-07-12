import functools

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def _casm_training_shape(config: _pi0_config.Pi0Config):
    pi0 = config.create(jax.random.key(0))
    observation = config.fake_obs(2).replace(
        phase_id=jnp.array([[1], [0]], dtype=jnp.int32),
        action_mask=jnp.ones((2, config.action_horizon, config.action_dim)),
    )
    return pi0.compute_loss(jax.random.key(1), observation, config.fake_act(2))


def _casm_sampling_shape(config: _pi0_config.Pi0Config):
    pi0 = config.create(jax.random.key(0))
    observation = config.fake_obs(1).replace(
        phase_id=jnp.array([[1]], dtype=jnp.int32),
    )
    return pi0.sample_actions(
        jax.random.key(1),
        observation,
        num_steps=2,
        noise=config.fake_act(1),
    )


def test_casm_modes_support_training_and_sampling_shapes():
    modes = ("hard_mask", "soft_mixture", "gated_cross_attention", "usefulness_gate")
    for mode in modes:
        config = _pi0_config.Pi0Config(pi05=True, casm_mode=mode)
        training = nnx.eval_shape(functools.partial(_casm_training_shape, config))
        sampling = nnx.eval_shape(functools.partial(_casm_sampling_shape, config))
        assert training.shape == (2, config.action_horizon)
        assert sampling.shape == (1, config.action_horizon, config.action_dim)
