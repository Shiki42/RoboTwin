import flax.nnx as nnx

from openpi.training import config


def _is_trainable(train_config, path):
    predicate = nnx.filterlib.to_predicate(train_config.trainable_filter)
    return predicate(path, nnx.Param(0.0))


def test_skillvla_probe_matches_control_data_and_budget():
    baseline = config.get_config("pi05_putcab_casm_joint_head_frozen_trunk")
    skillvla = config.get_config("pi05_putcab_skillvla_per_arm_gated_probe")

    for train_config in (baseline, skillvla):
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

    assert skillvla.model.casm_mode == "skillvla_per_arm_gated"
    assert skillvla.model.cross_attention_dim == 32
    assert skillvla.weight_loader.params_path == baseline.weight_loader.params_path
    assert skillvla.weight_loader.missing_regex == r".*(phase_gate|cross_attention|per_arm_adapter).*"


def test_skillvla_adaptation_probe_has_explicit_short_schedule():
    adaptation = config.get_config("pi05_putcab_skillvla_per_arm_gated_adaptation_probe")

    assert adaptation.model.casm_mode == "skillvla_per_arm_gated"
    assert adaptation.num_train_steps == 100
    assert adaptation.lr_schedule.warmup_steps == 20
    assert adaptation.lr_schedule.decay_steps == 100
    assert adaptation.lr_schedule.peak_lr == 2.5e-5
    assert adaptation.lr_schedule.decay_lr == 2.5e-6


def test_skillvla_probe_has_strict_trainable_scope():
    skillvla = config.get_config("pi05_putcab_skillvla_per_arm_gated_probe")

    for path in (
        ("PaliGemma", "img", "kernel"),
        ("PaliGemma", "llm", "lora_a"),
        ("action_in_proj", "kernel"),
    ):
        assert not _is_trainable(skillvla, path)

    for path in (
        ("action_out_proj", "kernel"),
        ("phase_gate", "output", "kernel"),
        ("cross_attention", "left_query", "kernel"),
        ("per_arm_adapter", "right_up", "kernel"),
    ):
        assert _is_trainable(skillvla, path)
