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
from openpi.training import config as _config


def test_aloha_inputs_expands_arm_mask_to_action_dimensions():
    image = np.zeros((3, 8, 8), dtype=np.uint8)
    output = AlohaInputs(adapt_to_pi=False)(
        {
            "images": {"cam_high": image},
            "state": np.zeros(14, dtype=np.float32),
            "actions": np.zeros((2, 14), dtype=np.float32),
            "action_mask": np.array([[1, 0], [0, 1]], dtype=np.float32),
        }
    )

    assert output["action_mask"].shape == (2, 14)
    assert output["action_mask"][0].tolist() == [1] * 7 + [0] * 7
    assert output["action_mask"][1].tolist() == [0] * 7 + [1] * 7


def test_soft_arm_mask_keeps_boundary_labels_zero():
    image = np.zeros((3, 8, 8), dtype=np.uint8)
    output = AlohaInputs(adapt_to_pi=False, inactive_action_weight=0.1)(
        {
            "images": {"cam_high": image},
            "state": np.zeros(14, dtype=np.float32),
            "actions": np.zeros((2, 14), dtype=np.float32),
            "action_mask": np.array([[1, 0], [0, 1]], dtype=np.float32),
            "action_phase": np.array([[0, 1], [1, 0]], dtype=np.float32),
        }
    )

    assert np.allclose(output["action_mask"][0], [1] * 7 + [0.1] * 7)
    assert np.all(output["action_mask"][1] == 0)


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


def test_action_mask_stops_supervision_at_phase_boundary():
    image = np.zeros((3, 8, 8), dtype=np.uint8)
    output = AlohaInputs(adapt_to_pi=False)(
        {
            "images": {"cam_high": image},
            "state": np.zeros(14, dtype=np.float32),
            "actions": np.zeros((4, 14), dtype=np.float32),
            "action_mask": np.ones((4, 2), dtype=np.float32),
            "action_phase": np.array([[0, 1], [0, 1], [1, 0], [1, 0]], dtype=np.float32),
        }
    )

    assert output["phase_id"].tolist() == [1]
    assert np.all(output["action_mask"][:2] == 1)
    assert np.all(output["action_mask"][2:] == 0)


def test_action_phase_sequence_accepts_one_hot_feature_chunks():
    image = np.zeros((3, 8, 8), dtype=np.uint8)
    output = AlohaInputs(adapt_to_pi=False)(
        {
            "images": {"cam_high": image},
            "state": np.zeros(14, dtype=np.float32),
            "actions": np.zeros((3, 14), dtype=np.float32),
            "action_mask": np.ones((3, 2), dtype=np.float32),
            "action_phase": np.array([[0, 1], [0, 1], [1, 0]], dtype=np.float32),
        }
    )

    assert output["phase_id"].tolist() == [1]
    assert np.all(output["action_mask"][:2] == 1)
    assert np.all(output["action_mask"][2:] == 0)


def test_wait_and_async_are_one_canonical_phase():
    image = np.zeros((3, 8, 8), dtype=np.uint8)
    output = AlohaInputs(adapt_to_pi=False)(
        {
            "images": {"cam_high": image},
            "state": np.zeros(14, dtype=np.float32),
            "actions": np.zeros((3, 14), dtype=np.float32),
            "action_mask": np.ones((3, 2), dtype=np.float32),
            "action_phase": np.array([[0, 1], [0, 1], [1, 0]], dtype=np.float32),
        }
    )

    assert np.all(output["action_mask"][:2] == 1)
    assert np.all(output["action_mask"][2:] == 0)


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
        "pi05_putcab_casm_soft_mixture_lora",
        "pi05_putcab_casm_gated_cross_attention_lora",
        "pi05_putcab_casm_usefulness_gate_lora",
    ],
)
def test_casm_configs_use_public_pi05_base_weights(name):
    config = _config.get_config(name)

    assert config.weight_loader.params_path == "gs://openpi-assets/checkpoints/pi05_base/params"
    assert config.data.repo_id == (
        "Shiki42/parallelvla_putcab_official_clean50_native_path_retimed_paired_v2"
    )
