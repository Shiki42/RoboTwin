import functools

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

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
    batch_size = 1 if config.casm_mode == "hard_gate" else 2
    pi0 = config.create(jax.random.key(0))
    observation = config.fake_obs(batch_size).replace(
        phase_id=jnp.array([[1]], dtype=jnp.int32) if batch_size == 1 else jnp.array([[1], [0]], dtype=jnp.int32),
        action_mask=jnp.ones((batch_size, config.action_horizon, config.action_dim)),
    )
    return pi0.compute_loss(jax.random.key(1), observation, config.fake_act(batch_size))


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
    modes = (
        "hard_mask",
        "hard_gate",
        "gated_cross_attention",
        "usefulness_gate",
        "visual_phase_gate",
        "cross_output_shared_head",
        "skillvla_per_arm_gated",
    )
    for mode in modes:
        config = _pi0_config.Pi0Config(pi05=True, casm_mode=mode)
        training = nnx.eval_shape(functools.partial(_casm_training_shape, config))
        sampling = nnx.eval_shape(functools.partial(_casm_sampling_shape, config))
        expected_batch_size = 1 if mode == "hard_gate" else 2
        assert training.shape == (expected_batch_size, config.action_horizon)
        assert sampling.shape == (1, config.action_horizon, config.action_dim)


def test_cross_output_rank_must_be_positive():
    with pytest.raises(ValueError, match="cross-output rank must be positive"):
        _pi0_config.Pi0Config(pi05=True, casm_mode="cross_output_shared_head", cross_output_rank=0)


def test_skill_adapter_rank_must_be_positive():
    with pytest.raises(ValueError, match="skill adapter rank must be positive"):
        _pi0_config.Pi0Config(pi05=True, casm_mode="skillvla_per_arm_gated", skill_adapter_rank=0)


def _visual_phase_gate_aux_shape():
    config = _pi0_config.Pi0Config(pi05=True, casm_mode="visual_phase_gate")
    model = config.create(jax.random.key(0))
    observation = config.fake_obs(2).replace(
        phase_id=jnp.array([[1], [0]], dtype=jnp.int32),
        action_mask=jnp.ones((2, config.action_horizon, config.action_dim)),
    )
    return model.compute_loss(
        jax.random.key(1),
        observation,
        config.fake_act(2),
        train=True,
        return_aux=True,
    )


def test_visual_phase_gate_returns_separate_training_metrics():
    loss, aux = nnx.eval_shape(_visual_phase_gate_aux_shape)
    assert loss.shape == (2, 50)
    assert set(aux) == {
        "action_loss",
        "left_action_loss",
        "right_action_loss",
        "gate_loss",
        "gate_accuracy",
        "gate_async_probability",
        "gate_predicted_async_rate",
        "gate_target_async_rate",
    }
    assert all(value.shape == () for value in aux.values())


def test_visual_phase_gate_prediction_shape():
    config = _pi0_config.Pi0Config(pi05=True, casm_mode="visual_phase_gate")
    model = config.create(jax.random.key(0))
    observation = config.fake_obs(3)
    probability = nnx.eval_shape(model.predict_async_probability, observation)
    assert probability.shape == (3,)


def _casm_lan_aux_shape():
    config = _pi0_config.Pi0Config(
        pi05=True,
        casm_mode="visual_phase_gate",
        semantic_subtask_prediction=True,
    )
    model = config.create(jax.random.key(0))
    observation = config.fake_obs(2).replace(
        phase_id=jnp.array([[1], [0]], dtype=jnp.int32),
        semantic_subtask_id=jnp.array([[1], [11]], dtype=jnp.int32),
        action_mask=jnp.ones((2, config.action_horizon, config.action_dim)),
    )
    return model.compute_loss(
        jax.random.key(1),
        observation,
        config.fake_act(2),
        train=True,
        return_aux=True,
    )


def test_casm_lan_returns_semantic_metrics_and_prediction_shape():
    loss, aux = nnx.eval_shape(_casm_lan_aux_shape)
    assert loss.shape == (2, 50)
    assert {
        "semantic_role_loss",
        "semantic_role_accuracy",
        "semantic_stage_loss",
        "semantic_stage_accuracy",
        "semantic_subtask_accuracy",
        "semantic_object_arm_right_probability",
    }.issubset(aux)
    config = _pi0_config.Pi0Config(
        pi05=True,
        casm_mode="visual_phase_gate",
        semantic_subtask_prediction=True,
    )
    model = config.create(jax.random.key(0))
    prediction = nnx.eval_shape(model.predict_casm_language, config.fake_obs(3))
    assert prediction.shape == (3, 9)
