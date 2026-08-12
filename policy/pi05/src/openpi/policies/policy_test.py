import jax.numpy as jnp
import numpy as np
from openpi_client import action_chunk_broker
import pytest
import torch

from openpi import transforms
from openpi.policies import aloha_policy
from openpi.policies import policy as policy_module
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)


class _SemanticTransform(transforms.DataTransformFn):
    def __call__(self, data):
        prompt = data.pop("prompt")
        semantic = "Current dual-arm semantic subtask" in prompt
        return {
            "image": data["image"],
            "image_mask": data["image_mask"],
            "state": data["state"],
            "phase_id": data["phase_id"],
            "tokenized_prompt": np.array([int(semantic)], dtype=np.int32),
            "tokenized_prompt_mask": np.array([True]),
        }


class _FakeCrossOutputModel:
    casm_mode = "cross_output_shared_head"
    semantic_subtask_prediction = False

    def predict_async_probability(self, observation):
        del observation
        return jnp.array([0.25], dtype=jnp.float32)

    def sample_actions(self, rng, observation, **kwargs):
        del rng, observation, kwargs
        return jnp.zeros((1, 2, 14), dtype=jnp.float32)


class _FakeCasmLanModel:
    casm_mode = "visual_phase_gate"
    semantic_subtask_prediction = True

    def predict_casm_language(self, observation):
        assert int(observation.tokenized_prompt[0, 0]) == 0
        return jnp.array([[0.2, 0.2, 0.01, 0.90, 0.02, 0.02, 0.02, 0.03, 0.95]], dtype=jnp.float32)

    def sample_actions(self, rng, observation, **kwargs):
        del rng, kwargs
        assert int(observation.tokenized_prompt[0, 0]) == 1
        return jnp.zeros((1, 2, 14), dtype=jnp.float32)


def test_cross_output_policy_emits_async_probability(monkeypatch):
    monkeypatch.setattr(policy_module.nnx_utils, "module_jit", lambda function: function)
    policy = policy_module.Policy(_FakeCrossOutputModel(), transforms=(_SemanticTransform(),))
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    outputs = policy.infer(
        {
            "image": {"base_0_rgb": image},
            "image_mask": {"base_0_rgb": np.True_},
            "state": np.zeros(14, dtype=np.float32),
            "phase_id": np.array([1], dtype=np.int32),
            "prompt": "put the target object in the drawer",
        }
    )

    assert outputs["async_probability"] == pytest.approx(0.25)


def test_casm_lan_predicts_semantics_before_action_prompt(monkeypatch):
    monkeypatch.setattr(policy_module.nnx_utils, "module_jit", lambda function: function)
    policy = policy_module.Policy(
        _FakeCasmLanModel(),
        transforms=(_SemanticTransform(),),
    )
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    outputs = policy.infer(
        {
            "image": {"base_0_rgb": image},
            "image_mask": {"base_0_rgb": np.True_},
            "state": np.zeros(14, dtype=np.float32),
            "phase_id": np.array([1], dtype=np.int32),
            "prompt": "put the target object in the drawer",
        }
    )
    assert outputs["semantic_subtask_id"] == 5
    assert "Left arm: carry and place the target object" in outputs["semantic_subtask_prompt"]
    assert "Right arm: hold the drawer open and wait" in outputs["semantic_subtask_prompt"]
    assert outputs["semantic_stage_probabilities"] == pytest.approx([0.01, 0.90, 0.02, 0.02, 0.02, 0.03])
    assert outputs["async_probability"] == pytest.approx(0.2)
    assert outputs["phase_gate_async_probability"] == pytest.approx(0.2)
    assert outputs["semantic_phase_is_async"] is False
    assert outputs["semantic_stage_async_probability"] == pytest.approx(0.95)


def test_casm_lan_latched_sync_phase_cannot_revert_prompt(monkeypatch):
    monkeypatch.setattr(policy_module.nnx_utils, "module_jit", lambda function: function)
    model = _FakeCasmLanModel()
    model.predict_casm_language = lambda observation: jnp.array(
        [[0.8, 0.2, 0.01, 0.90, 0.02, 0.02, 0.02, 0.03, 0.95]],
        dtype=jnp.float32,
    )
    policy = policy_module.Policy(model, transforms=(_SemanticTransform(),))
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    outputs = policy.infer(
        {
            "image": {"base_0_rgb": image},
            "image_mask": {"base_0_rgb": np.True_},
            "state": np.zeros(14, dtype=np.float32),
            "phase_id": np.array([0], dtype=np.int32),
            "prompt": "put the target object in the drawer",
        }
    )

    assert outputs["semantic_phase_is_async"] is False
    assert outputs["semantic_subtask_id"] == 5
    assert "Left arm: carry and place the target object" in outputs["semantic_subtask_prompt"]


class _OnlineSubtaskTransform(transforms.DataTransformFn):
    def __call__(self, data):
        data.pop("prompt")
        return {
            "image": data["image"],
            "image_mask": data["image_mask"],
            "state": data["state"],
            "tokenized_prompt": np.array([1], dtype=np.int32),
            "tokenized_prompt_mask": np.array([True]),
        }


class _FakeOnlineSubtaskModel(torch.nn.Module):
    online_subtask_prediction = True

    def sample_actions(self, device, observation, **kwargs):
        raise AssertionError("standard sample_actions must not run for feature-on inference")

    def sample_actions_with_subtask(self, observation, task, **kwargs):
        del kwargs
        assert task == "put the target object in the drawer"
        assert observation.state.shape == (1, 14)
        actions = torch.zeros((1, 2, 14), device=observation.state.device)
        subtask = "Left arm: hold object; Right arm: open drawer."
        return actions, subtask


def test_online_subtask_policy_predicts_before_each_action_chunk():
    policy = policy_module.Policy(
        _FakeOnlineSubtaskModel(),
        transforms=(_OnlineSubtaskTransform(),),
        is_pytorch=True,
    )
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    outputs = policy.infer(
        {
            "image": {"base_0_rgb": image},
            "image_mask": {"base_0_rgb": np.True_},
            "state": np.zeros(14, dtype=np.float32),
            "prompt": "put the target object in the drawer",
        }
    )

    assert outputs["actions"].shape == (2, 14)
    assert outputs["subtask"] == "Left arm: hold object; Right arm: open drawer."
