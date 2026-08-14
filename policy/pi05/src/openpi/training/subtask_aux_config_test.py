from openpi.training import config


def test_subtask_aux_config_preserves_pi05_action_contract():
    train_config = config.get_config("pi05_putcab_factorized_anchor_subtask_head_pytorch")

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


def test_subtask_aux_config_keeps_native_action_chunk_and_adds_current_frame_target():
    train_config = config.get_config("pi05_putcab_factorized_anchor_subtask_head_pytorch")
    repack = train_config.data.repack_transforms.inputs[0].structure

    assert repack["actions"] == "action"
    assert "action_mask" not in repack
    assert "action_phase" not in repack
    assert repack["semantic_subtask_id"] == "observation.semantic_subtask_id"
    assert train_config.data.action_sequence_keys == ("action",)
