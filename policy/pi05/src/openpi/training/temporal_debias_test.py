from flax import nnx
import jax.numpy as jnp
import numpy as np
import pytest

from openpi import transforms
from openpi.models import model
from openpi.models.casm import hard_mask_arm_observation
from openpi.models.casm import hard_mask_stream_action_inputs
from openpi.models.casm import merge_stream_vector_fields
from openpi.models.pi0 import reduce_action_loss
from openpi.policies.aloha_policy import AlohaInputs
from openpi.policies.aloha_policy import AlohaOutputs
from openpi.training import config as _config


def test_aloha_inputs_expands_arm_mask_to_action_dimensions():
    image = np.zeros((3, 8, 8), dtype=np.uint8)
    output = AlohaInputs(adapt_to_pi=False)(
        {
            "images": {"cam_high": image},
            "state": np.zeros(14, dtype=np.float32),
            "actions": np.zeros((2, 14), dtype=np.float32),
            "action_mask": np.array([[1, 0], [0, 1]], dtype=np.float32),
            "action_is_pad": np.zeros(2, dtype=np.bool_),
        }
    )

    assert output["action_mask"].shape == (2, 14)
    assert output["action_mask"][0].tolist() == [1] * 7 + [0] * 7
    assert output["action_mask"][1].tolist() == [0] * 7 + [1] * 7


def test_aloha_inputs_masks_temporal_padding_and_requires_pad_receipt():
    image = np.zeros((3, 8, 8), dtype=np.uint8)
    data = {
        "images": {"cam_high": image},
        "state": np.zeros(14, dtype=np.float32),
        "actions": np.zeros((3, 14), dtype=np.float32),
        "action_mask": np.ones((3, 2), dtype=np.float32),
        "action_is_pad": np.array([False, True, False]),
    }
    output = AlohaInputs(adapt_to_pi=False)(data)
    assert np.all(output["action_mask"][[0, 2]] == 1)
    assert np.all(output["action_mask"][1] == 0)

    del data["action_is_pad"]
    with pytest.raises(ValueError, match="requires action_is_pad"):
        AlohaInputs(adapt_to_pi=False)(data)


def test_aloha_outputs_preserves_async_probability():
    output = AlohaOutputs(adapt_to_pi=False)(
        {
            "actions": np.zeros((2, 14), dtype=np.float32),
            "async_probability": np.array(0.75, dtype=np.float32),
        }
    )
    assert output["async_probability"] == pytest.approx(0.75)


def test_soft_arm_mask_supervises_cross_phase_labels():
    image = np.zeros((3, 8, 8), dtype=np.uint8)
    output = AlohaInputs(adapt_to_pi=False, inactive_action_weight=0.1)(
        {
            "images": {"cam_high": image},
            "state": np.zeros(14, dtype=np.float32),
            "actions": np.zeros((2, 14), dtype=np.float32),
            "action_mask": np.array([[1, 0], [0, 1]], dtype=np.float32),
            "action_is_pad": np.zeros(2, dtype=np.bool_),
            "action_phase": np.array([[0, 1], [1, 0]], dtype=np.float32),
        }
    )

    assert np.allclose(output["action_mask"][0], [1] * 7 + [0.1] * 7)
    assert np.allclose(output["action_mask"][1], [0.1] * 7 + [1] * 7)


def test_soft_arm_mask_rejects_invalid_weight():
    with pytest.raises(ValueError, match="inactive action weight"):
        AlohaInputs(inactive_action_weight=1.1)


def test_pad_states_and_actions_pads_supervision_mask():
    data = {
        "state": np.zeros(14, dtype=np.float32),
        "actions": np.zeros((2, 14), dtype=np.float32),
        "action_mask": np.ones((2, 14), dtype=np.float32),
    }

    output = transforms.PadStatesAndActions(32)(data)

    assert output["action_mask"].shape == (2, 32)
    assert np.all(output["action_mask"][:, :14] == 1)
    assert np.all(output["action_mask"][:, 14:] == 0)


def test_reduce_action_loss_ignores_inactive_arm_dimensions():
    squared_error = jnp.array([[[1.0, 9.0], [4.0, 16.0]]])
    action_mask = jnp.array([[[1.0, 0.0], [0.0, 1.0]]])

    loss = reduce_action_loss(squared_error, action_mask)

    assert np.asarray(loss).tolist() == [[1.0, 16.0]]


def test_reduce_action_loss_normalizes_only_over_valid_temporal_dimensions():
    squared_error = jnp.array([[[1.0, 1.0], [100.0, 100.0]]])
    action_mask = jnp.array([[[1.0, 1.0], [0.0, 0.0]]])

    loss = reduce_action_loss(squared_error, action_mask)

    assert np.isclose(np.mean(loss), 1.0)


def test_reduce_action_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        reduce_action_loss(jnp.ones((1, 2, 3)), jnp.ones((1, 2, 2)))


def test_observation_round_trip_preserves_action_mask():
    action_mask = np.ones((2, 32), dtype=np.float32)
    observation = model.Observation.from_dict(
        {
            "image": {},
            "image_mask": {},
            "state": np.zeros(32, dtype=np.float32),
            "action_mask": action_mask,
        }
    )

    assert observation.action_mask is action_mask
    assert np.array_equal(observation.to_dict()["action_mask"], action_mask)


def test_action_mask_supervises_complete_horizon_across_phase_boundary():
    image = np.zeros((3, 8, 8), dtype=np.uint8)
    output = AlohaInputs(adapt_to_pi=False)(
        {
            "images": {"cam_high": image},
            "state": np.zeros(14, dtype=np.float32),
            "actions": np.zeros((4, 14), dtype=np.float32),
            "action_mask": np.ones((4, 2), dtype=np.float32),
            "action_is_pad": np.zeros(4, dtype=np.bool_),
            "action_phase": np.array([[0, 1], [0, 1], [1, 0], [1, 0]], dtype=np.float32),
        }
    )

    assert output["phase_id"].tolist() == [1]
    assert np.all(output["action_mask"] == 1)


def test_action_phase_sequence_accepts_one_hot_feature_chunks():
    image = np.zeros((3, 8, 8), dtype=np.uint8)
    output = AlohaInputs(adapt_to_pi=False)(
        {
            "images": {"cam_high": image},
            "state": np.zeros(14, dtype=np.float32),
            "actions": np.zeros((3, 14), dtype=np.float32),
            "action_mask": np.ones((3, 2), dtype=np.float32),
            "action_is_pad": np.zeros(3, dtype=np.bool_),
            "action_phase": np.array([[0, 1], [0, 1], [1, 0]], dtype=np.float32),
        }
    )

    assert output["phase_id"].tolist() == [1]
    assert np.all(output["action_mask"] == 1)


def test_wait_and_async_are_one_canonical_phase():
    image = np.zeros((3, 8, 8), dtype=np.uint8)
    output = AlohaInputs(adapt_to_pi=False)(
        {
            "images": {"cam_high": image},
            "state": np.zeros(14, dtype=np.float32),
            "actions": np.zeros((3, 14), dtype=np.float32),
            "action_mask": np.ones((3, 2), dtype=np.float32),
            "action_is_pad": np.zeros(3, dtype=np.bool_),
            "action_phase": np.array([[0, 1], [0, 1], [1, 0]], dtype=np.float32),
        }
    )

    assert np.all(output["action_mask"] == 1)


def test_isolate_arm_observation_masks_only_opposite_wrist():
    masks = {
        "base_0_rgb": np.array(1, dtype=np.bool_),
        "left_wrist_0_rgb": np.array(1, dtype=np.bool_),
        "right_wrist_0_rgb": np.array(1, dtype=np.bool_),
    }
    observation = model.Observation(
        images={key: np.zeros((2, 2, 3), dtype=np.float32) for key in masks},
        image_masks=masks,
        state=np.zeros(32, dtype=np.float32),
        phase_id=np.array([1], dtype=np.int32),
    )

    left = hard_mask_arm_observation(observation, "left")
    right = hard_mask_arm_observation(observation, "right")

    assert bool(left.image_masks["left_wrist_0_rgb"])
    assert not bool(left.image_masks["right_wrist_0_rgb"])
    assert bool(right.image_masks["right_wrist_0_rgb"])
    assert not bool(right.image_masks["left_wrist_0_rgb"])
    assert bool(left.image_masks["base_0_rgb"])
    assert bool(right.image_masks["base_0_rgb"])


def test_casm_stream_inputs_gate_cross_arm_actions_by_phase():
    actions = jnp.arange(2 * 1 * 32, dtype=jnp.float32).reshape(2, 1, 32)
    phase_id = jnp.array([[1], [0]], dtype=jnp.int32)

    streams = np.asarray(hard_mask_stream_action_inputs(actions, phase_id))
    left, right = streams[:2], streams[2:]

    assert np.all(left[0, 0, 7:] == 0)
    assert np.all(right[0, 0, :7] == 0)
    assert np.all(right[0, 0, 14:] == 0)
    assert np.array_equal(left[1], np.asarray(actions[1]))
    assert np.array_equal(right[1], np.asarray(actions[1]))


def test_merge_casm_stream_vector_fields_selects_arm_outputs():
    streams = jnp.concatenate(
        [
            jnp.full((2, 1, 32), 1.0),
            jnp.full((2, 1, 32), 2.0),
        ],
        axis=0,
    )

    output = np.asarray(merge_stream_vector_fields(streams, batch_size=2))

    assert output[:, 0, :7].tolist() == [[1.0] * 7, [1.0] * 7]
    assert output[:, 0, 7:14].tolist() == [[2.0] * 7, [2.0] * 7]
    assert output[:, 0, 14:].tolist() == [[0.0] * 18, [0.0] * 18]


@pytest.mark.parametrize(
    "name",
    [
        "pi05_putcab_casm_hard_gate_lora",
        "pi05_putcab_casm_gated_cross_attention_lora",
        "pi05_putcab_casm_usefulness_gate_lora",
    ],
)
def test_casm_configs_use_public_pi05_base_weights(name):
    config = _config.get_config(name)

    assert config.data.base_config.video_backend == "pyav"
    assert config.batch_size == 1
    assert config.weight_loader.params_path == "gs://openpi-assets/checkpoints/pi05_base/params"
    assert config.data.repo_id == ("Shiki42/parallelvla_putcab_official_clean50_native_path_retimed_paired_v2")


def test_visual_phase_gate_config_anchors_and_adapts_lora_actions():
    config = _config.get_config("pi05_putcab_casm_visual_phase_gate_pi05_anchor_adapt_lora")

    assert config.model.casm_mode == "visual_phase_gate"
    assert config.model.paligemma_variant == "gemma_2b_lora"
    assert config.model.action_expert_variant == "gemma_300m_lora"
    assert config.project_name == "parallelvla-casm"
    assert config.batch_size == 16
    assert config.num_train_steps == 2_000
    assert config.save_interval == 500
    assert config.params_only_checkpoint is False
    assert config.ema_decay is None
    assert config.wandb_enabled is True
    assert config.data.repo_id == "Shiki42/robotwin_put_obj_cabinet_50_dynFcam_nFov"
    assert config.weight_loader.missing_regex == ".*phase_gate.*"
    trainable = nnx.filterlib.to_predicate(config.trainable_filter)
    parameter = nnx.Param(0.0)
    assert trainable(("phase_gate", "kernel"), parameter)
    assert trainable(("PaliGemma", "llm", "lora_a"), parameter)
    assert trainable(("PaliGemma", "img", "kernel"), parameter)
    assert trainable(("action_in_proj", "kernel"), parameter)
    assert not trainable(("PaliGemma", "llm", "kernel"), parameter)


def test_matched_pi05_config_uses_same_anchor_adaptation_budget_without_gate():
    config = _config.get_config("pi05_putcab_pi05_anchor_adapt_matched_lora")

    assert config.model.casm_mode == "none"
    assert config.model.paligemma_variant == "gemma_2b_lora"
    assert config.model.action_expert_variant == "gemma_300m_lora"
    assert config.project_name == "parallelvla-pi05-matched"
    assert config.batch_size == 16
    assert config.num_train_steps == 2_000
    assert config.save_interval == 500
    assert config.params_only_checkpoint is False
    assert config.ema_decay is None
    assert config.wandb_enabled is True
    assert config.data.repo_id == "Shiki42/robotwin_put_obj_cabinet_50_dynFcam_nFov"
    assert config.weight_loader.missing_regex == r"(?!x)x"
    trainable = nnx.filterlib.to_predicate(config.trainable_filter)
    parameter = nnx.Param(0.0)
    assert trainable(("PaliGemma", "llm", "lora_a"), parameter)
    assert trainable(("action_in_proj", "kernel"), parameter)
    assert not trainable(("PaliGemma", "llm", "kernel"), parameter)


def test_vision_frozen_pi05_config_only_trains_lora_and_action_heads():
    config = _config.get_config("pi05_putcab_pi05_anchor_adapt_vision_frozen_lora")

    assert config.model.casm_mode == "none"
    assert config.project_name == "parallelvla-pi05-vision-frozen"
    assert config.weight_loader.missing_regex == r"(?!x)x"
    trainable = nnx.filterlib.to_predicate(config.trainable_filter)
    parameter = nnx.Param(0.0)
    assert trainable(("PaliGemma", "llm", "lora_a"), parameter)
    assert trainable(("PaliGemma", "llm_1", "lora_a"), parameter)
    assert not trainable(("PaliGemma", "img", "kernel"), parameter)
    assert trainable(("action_in_proj", "kernel"), parameter)
    assert trainable(("action_out_proj", "kernel"), parameter)
    assert not trainable(("PaliGemma", "llm", "kernel"), parameter)


def test_gate_only_casm_config_freezes_everything_except_phase_gate():
    config = _config.get_config("pi05_putcab_casm_visual_phase_gate_pi05_anchor_gate_only")

    assert config.model.casm_mode == "visual_phase_gate"
    assert config.project_name == "parallelvla-casm-gate-only"
    assert config.weight_loader.missing_regex == ".*phase_gate.*"
    trainable = nnx.filterlib.to_predicate(config.trainable_filter)
    parameter = nnx.Param(0.0)
    assert trainable(("phase_gate", "kernel"), parameter)
    assert not trainable(("PaliGemma", "llm", "lora_a"), parameter)
    assert not trainable(("PaliGemma", "img", "kernel"), parameter)
    assert not trainable(("action_in_proj", "kernel"), parameter)


@pytest.mark.parametrize(
    ("name", "mode"),
    [
        ("pi05_putcab_pytorch_matched_full", "none"),
        ("pi05_putcab_casm_visual_phase_gate_pytorch_full", "visual_phase_gate"),
    ],
)
def test_putcab_pytorch_configs_are_matched_full_finetunes(name, mode):
    config = _config.get_config(name)

    assert config.model.casm_mode == mode
    assert config.model.paligemma_variant == "gemma_2b"
    assert config.model.action_expert_variant == "gemma_300m"
    assert config.batch_size == 16
    assert config.num_workers == 2
    assert config.prefetch_factor == 2
    assert config.persistent_workers is True
    assert config.pin_memory is True
    assert config.num_train_steps == 20_000
    assert config.save_interval == 2_000
    assert config.params_only_checkpoint is False
    assert config.ema_decay is None
    assert config.wandb_enabled is True
    assert config.data.repo_id == "Shiki42/robotwin_put_obj_cabinet_50_dynFcam_nFov_lerobot"
    assert config.data.action_sequence_keys == (
        "action",
        "observation.arm_active_mask",
        "observation.phase_one_hot",
    )
