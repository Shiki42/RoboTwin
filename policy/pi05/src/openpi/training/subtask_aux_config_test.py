import pytest

from openpi.training import config
from openpi.training import subtask_aux_config


@pytest.mark.parametrize(
    "config_name",
    [
        "pi05_putcab_factorized_anchor_subtask_head_pytorch",
        "pi05_putcab_factorized_anchor_subtask_head_sqrt_balanced_pytorch",
    ],
)
def test_subtask_aux_config_preserves_pi05_action_contract(config_name):
    train_config = config.get_config(config_name)

    assert train_config.model.pi05
    assert train_config.model.casm_mode == "none"
    assert train_config.model.action_horizon == 50
    assert train_config.model.action_dim == 32
    assert train_config.pytorch_trainable_scope == "subtask_head"
    assert train_config.model.pytorch_aux_subtask_classes == 12
    assert train_config.model.pytorch_aux_subtask_state_dim == 14
    assert train_config.model.pytorch_aux_subtask_stop_gradient
    assert train_config.model.pytorch_aux_subtask_loss_weight == 1.0
    assert train_config.num_train_steps == 2_000
    assert train_config.batch_size == 16
    assert train_config.gradient_accumulation_steps == 1
    assert train_config.ema_decay is None


def test_shared_projector_config_preserves_native_action_contract():
    train_config = config.get_config("pi05_putcab_factorized_anchor_subtask_shared_projector_pytorch")
    repack = train_config.data.repack_transforms.inputs[0].structure

    assert train_config.model.pi05
    assert train_config.model.casm_mode == "none"
    assert train_config.model.action_horizon == 50
    assert train_config.model.action_dim == 32
    assert train_config.pytorch_trainable_scope == "subtask_head_and_projector"
    assert train_config.model.pytorch_aux_subtask_classes == 12
    assert train_config.model.pytorch_aux_subtask_state_dim == 14
    assert train_config.model.pytorch_aux_subtask_stop_gradient is False
    assert train_config.model.pytorch_aux_subtask_loss_weight == 0.01
    assert train_config.model.pytorch_aux_subtask_class_weights is None
    assert train_config.lr_schedule.peak_lr == 1e-4
    assert train_config.lr_schedule.decay_lr == 1e-5
    assert train_config.num_train_steps == 2_000
    assert train_config.batch_size == 16
    assert train_config.gradient_accumulation_steps == 1
    assert train_config.ema_decay is None
    assert repack["actions"] == "action"
    assert repack["semantic_subtask_id"] == "observation.semantic_subtask_id"
    assert repack["action_mask"] == "observation.arm_active_mask"
    assert repack["action_is_pad"] == "action_is_pad"
    assert "action_phase" not in repack
    assert train_config.data.action_sequence_keys == ("action", "observation.arm_active_mask")


def test_subtask_aux_configs_record_expected_class_weights():
    unweighted = config.get_config("pi05_putcab_factorized_anchor_subtask_head_pytorch")
    balanced = config.get_config("pi05_putcab_factorized_anchor_subtask_head_sqrt_balanced_pytorch")

    assert unweighted.model.pytorch_aux_subtask_class_weights is None
    assert balanced.model.pytorch_aux_subtask_class_weights == subtask_aux_config.SQRT_BALANCED_CLASS_WEIGHTS
    assert len(balanced.model.pytorch_aux_subtask_class_weights) == 12
    assert sum(balanced.model.pytorch_aux_subtask_class_weights) / 12 == pytest.approx(1.0)


@pytest.mark.parametrize(
    "config_name",
    [
        "pi05_putcab_factorized_anchor_subtask_head_pytorch",
        "pi05_putcab_factorized_anchor_subtask_head_sqrt_balanced_pytorch",
    ],
)
def test_subtask_aux_config_keeps_native_action_chunk_and_adds_current_frame_target(config_name):
    train_config = config.get_config(config_name)
    repack = train_config.data.repack_transforms.inputs[0].structure

    assert repack["actions"] == "action"
    assert "action_mask" not in repack
    assert "action_phase" not in repack
    assert repack["semantic_subtask_id"] == "observation.semantic_subtask_id"
    assert train_config.data.action_sequence_keys == ("action",)
