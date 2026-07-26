from openpi.training import config


def test_casm_lan_config_has_semantic_target_and_strict_head_load():
    train_config = config.get_config("pi05_putcab_casm_lan_anchor_adapt_lora")
    assert train_config.model.semantic_subtask_prediction
    assert train_config.model.casm_mode == "visual_phase_gate"
    assert train_config.num_train_steps == 500
    assert train_config.weight_loader.missing_regex == ".*semantic_(role|stage)_head.*"
    repack = train_config.data.repack_transforms.inputs[0].structure
    assert repack["semantic_subtask_id"] == "observation.semantic_subtask_id"
