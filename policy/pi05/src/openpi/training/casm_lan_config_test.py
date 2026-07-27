import flax.nnx as nnx

from openpi.training import config


def test_casm_lan_config_has_semantic_target_and_strict_head_load():
    train_config = config.get_config("pi05_putcab_casm_lan_anchor_adapt_lora")
    assert train_config.model.semantic_subtask_prediction
    assert train_config.model.casm_mode == "visual_phase_gate"
    assert train_config.num_train_steps == 500
    assert train_config.weight_loader.missing_regex == ".*semantic_(role|stage)_head.*"
    repack = train_config.data.repack_transforms.inputs[0].structure
    assert repack["semantic_subtask_id"] == "observation.semantic_subtask_id"


def test_casm_lan_heads_only_loads_trained_heads_and_freezes_action_model():
    train_config = config.get_config("pi05_putcab_casm_lan_semantic_heads_only")

    assert train_config.weight_loader.missing_regex == r"(?!x)x"
    assert train_config.model.semantic_role_loss_weight == 0.5
    assert train_config.model.semantic_stage_loss_weight == 1.0
    trainable = nnx.filterlib.to_predicate(train_config.trainable_filter)
    parameter = nnx.Param(0.0)
    for path in (
        ("phase_gate", "kernel"),
        ("semantic_role_head", "kernel"),
        ("semantic_stage_head", "kernel"),
    ):
        assert trainable(path, parameter)
    for path in (
        ("PaliGemma", "img", "kernel"),
        ("PaliGemma", "llm", "lora_a"),
        ("action_in_proj", "kernel"),
        ("action_out_proj", "kernel"),
    ):
        assert not trainable(path, parameter)
