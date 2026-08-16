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
    assert train_config.pytorch_action_prompt_mode == "task_only"
    assert len(train_config.data.repack_transforms.inputs) == 1


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
    assert train_config.pytorch_action_prompt_mode == "teacher_forced_joint_subtask"
    assert len(train_config.data.repack_transforms.inputs) == 2
    assert isinstance(train_config.data.repack_transforms.inputs[1], subtask_aux_config.TeacherForcedSubtaskPrompt)
    assert repack["actions"] == "action"
    assert repack["semantic_subtask_id"] == "observation.semantic_subtask_id"
    assert "action_mask" not in repack
    assert repack["action_loss_mask"] == "observation.action_loss_mask"
    assert repack["action_is_pad"] == "action_is_pad"
    assert "action_phase" not in repack
    assert train_config.data.action_sequence_keys == ("action",)
    assert train_config.data.use_delta_joint_actions is False


def test_joint_action_policy_config_preserves_native_action_contract():
    train_config = config.get_config("pi05_putcab_factorized_anchor_subtask_joint_action_policy_pytorch")
    repack = train_config.data.repack_transforms.inputs[0].structure

    assert train_config.model.pi05
    assert train_config.model.casm_mode == "none"
    assert train_config.model.action_horizon == 50
    assert train_config.model.action_dim == 32
    assert train_config.pytorch_trainable_scope == "subtask_head_and_action_policy"
    assert train_config.model.pytorch_aux_subtask_classes == 12
    assert train_config.model.pytorch_aux_subtask_state_dim == 14
    assert train_config.model.pytorch_aux_subtask_stop_gradient is True
    assert train_config.model.pytorch_aux_subtask_loss_weight == 0.01
    assert train_config.model.pytorch_aux_subtask_class_weights is None
    assert train_config.lr_schedule.warmup_steps == 100
    assert train_config.lr_schedule.peak_lr == 1e-5
    assert train_config.lr_schedule.decay_steps == 2_000
    assert train_config.lr_schedule.decay_lr == 1e-6
    assert train_config.num_train_steps == 2_000
    assert train_config.batch_size == 16
    assert train_config.gradient_accumulation_steps == 1
    assert train_config.ema_decay is None
    assert train_config.pytorch_action_prompt_mode == "teacher_forced_joint_subtask"
    assert len(train_config.data.repack_transforms.inputs) == 2
    assert isinstance(train_config.data.repack_transforms.inputs[1], subtask_aux_config.TeacherForcedSubtaskPrompt)
    assert repack["actions"] == "action"
    assert repack["semantic_subtask_id"] == "observation.semantic_subtask_id"
    assert "action_mask" not in repack
    assert repack["action_loss_mask"] == "observation.action_loss_mask"
    assert repack["action_is_pad"] == "action_is_pad"
    assert "action_phase" not in repack
    assert train_config.data.action_sequence_keys == ("action",)
    assert train_config.data.use_delta_joint_actions is False


def test_task_only_joint_action_policy_config_is_matched_control():
    train_config = config.get_config("pi05_putcab_factorized_anchor_subtask_joint_action_policy_task_only_pytorch")
    repack = train_config.data.repack_transforms.inputs[0].structure

    assert train_config.model.pi05
    assert train_config.model.casm_mode == "none"
    assert train_config.model.action_horizon == 50
    assert train_config.model.action_dim == 32
    assert train_config.pytorch_trainable_scope == "subtask_head_and_action_policy"
    assert train_config.model.pytorch_aux_subtask_classes == 12
    assert train_config.model.pytorch_aux_subtask_state_dim == 14
    assert train_config.model.pytorch_aux_subtask_stop_gradient is True
    assert train_config.model.pytorch_aux_subtask_loss_weight == 0.01
    assert train_config.model.pytorch_aux_subtask_class_weights is None
    assert train_config.lr_schedule.warmup_steps == 100
    assert train_config.lr_schedule.peak_lr == 1e-5
    assert train_config.lr_schedule.decay_steps == 2_000
    assert train_config.lr_schedule.decay_lr == 1e-6
    assert train_config.num_train_steps == 2_000
    assert train_config.batch_size == 16
    assert train_config.gradient_accumulation_steps == 1
    assert train_config.ema_decay is None
    assert train_config.pytorch_action_prompt_mode == "task_only"
    assert len(train_config.data.repack_transforms.inputs) == 1
    assert repack["actions"] == "action"
    assert repack["semantic_subtask_id"] == "observation.semantic_subtask_id"
    assert "action_mask" not in repack
    assert repack["action_loss_mask"] == "observation.action_loss_mask"
    assert repack["action_is_pad"] == "action_is_pad"
    assert "action_phase" not in repack
    assert train_config.data.action_sequence_keys == ("action",)
    assert train_config.data.use_delta_joint_actions is False


def test_task_only_full_action_config_supervises_both_arms_and_masks_padding():
    train_config = config.get_config(
        "pi05_putcab_factorized_anchor_subtask_joint_action_policy_task_only_full_action_pytorch"
    )
    repack = train_config.data.repack_transforms.inputs[0].structure

    assert train_config.model.pi05
    assert train_config.model.action_horizon == 50
    assert train_config.model.action_dim == 32
    assert train_config.pytorch_trainable_scope == "subtask_head_and_action_policy"
    assert train_config.model.pytorch_aux_subtask_loss_weight == 0.01
    assert train_config.lr_schedule.warmup_steps == 100
    assert train_config.lr_schedule.peak_lr == 1e-5
    assert train_config.lr_schedule.decay_steps == 2_000
    assert train_config.lr_schedule.decay_lr == 1e-6
    assert train_config.num_train_steps == 2_000
    assert train_config.batch_size == 16
    assert train_config.pytorch_action_prompt_mode == "task_only"
    assert len(train_config.data.repack_transforms.inputs) == 1
    assert repack["actions"] == "action"
    assert repack["semantic_subtask_id"] == "observation.semantic_subtask_id"
    assert repack["action_is_pad"] == "action_is_pad"
    assert "action_loss_mask" not in repack
    assert "action_mask" not in repack
    assert train_config.data.action_sequence_keys == ("action",)
    assert train_config.data.use_delta_joint_actions is False


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
    assert train_config.data.use_delta_joint_actions is False


def test_teacher_forced_prompt_matches_factorized_checkpoint_contract():
    transform = subtask_aux_config.TeacherForcedSubtaskPrompt()

    output = transform(
        {
            "prompt": "  put the object in the cabinet  ",
            "semantic_subtask_id": 11,
        }
    )

    assert output["prompt"] == (
        "put the object in the cabinet\n"
        "Current subtask: Left arm: wait while holding drawer open; Right arm: insert and place object."
    )


def test_self_conditioned_full_action_config():
    train_config = config.get_config(
        "pi05_putcab_factorized_anchor_subtask_self_conditioned_full_action_pytorch"
    )
    repack = train_config.data.repack_transforms.inputs[0].structure

    assert train_config.model.pytorch_aux_subtask_prompt_variants
    assert train_config.pytorch_action_prompt_mode == "self_conditioned_predicted_text"
    assert train_config.pytorch_subtask_conditioning_warmup_steps == 500
    assert train_config.pytorch_subtask_conditioning_max_prob == 0.9
    assert train_config.pytorch_trainable_scope == "subtask_head_and_action_policy"
    assert repack["actions"] == "action"
    assert repack["action_is_pad"] == "action_is_pad"
    assert "action_loss_mask" not in repack
    assert train_config.data.action_sequence_keys == ("action",)
    assert train_config.data.use_delta_joint_actions is False
