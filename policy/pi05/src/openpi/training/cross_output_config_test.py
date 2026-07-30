import flax.nnx as nnx

from openpi.training import config


def _is_trainable(train_config, path):
    predicate = nnx.filterlib.to_predicate(train_config.trainable_filter)
    return predicate(path, nnx.Param(0.0))


def test_cross_output_variants_share_control_signature():
    baseline = config.get_config("pi05_putcab_casm_joint_head_frozen_trunk")
    cross_output = config.get_config("pi05_putcab_casm_cross_output_shared_head")

    for train_config in (baseline, cross_output):
        assert train_config.seed == 87431
        assert train_config.batch_size == 1
        assert train_config.gradient_accumulation_steps == 16
        assert train_config.batch_size * train_config.gradient_accumulation_steps == 16
        assert train_config.num_train_steps == 500
        assert train_config.data.repo_id == baseline.data.repo_id
        assert train_config.data.inactive_action_weight == 1.0
        repack = train_config.data.repack_transforms.inputs[0].structure
        assert repack["action_is_pad"] == "action_is_pad"
        assert train_config.data.assets.asset_id == (
            "Shiki42/robotwin_put_obj_cabinet_50_dynFcam_nFov_dynamicMain_lerobot"
        )
        assert train_config.params_only_checkpoint is False
        assert train_config.ema_decay is None

    assert baseline.model.casm_mode == "visual_phase_gate"
    assert cross_output.model.casm_mode == "cross_output_shared_head"
    assert baseline.weight_loader.params_path == cross_output.weight_loader.params_path


def test_cross_output_variants_have_strict_and_minimal_trainable_scopes():
    baseline = config.get_config("pi05_putcab_casm_joint_head_frozen_trunk")
    cross_output = config.get_config("pi05_putcab_casm_cross_output_shared_head")

    assert baseline.weight_loader.missing_regex == ".*phase_gate.*"
    assert cross_output.weight_loader.missing_regex == ".*(cross_output_adapter|phase_gate).*"

    for path in (
        ("PaliGemma", "img", "kernel"),
        ("PaliGemma", "llm", "lora_a"),
        ("action_in_proj", "kernel"),
    ):
        assert not _is_trainable(baseline, path)
        assert not _is_trainable(cross_output, path)

    for path in (("action_out_proj", "kernel"), ("phase_gate", "output", "kernel")):
        assert _is_trainable(baseline, path)
        assert _is_trainable(cross_output, path)

    adapter_path = ("cross_output_adapter", "down", "kernel")
    assert not _is_trainable(baseline, adapter_path)
    assert _is_trainable(cross_output, adapter_path)
