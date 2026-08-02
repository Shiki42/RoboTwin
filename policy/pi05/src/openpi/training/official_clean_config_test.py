import flax.nnx as nnx

from openpi.training import config
from openpi.training import optimizer


def test_official_clean_config_matches_priorvla_pi05_recipe():
    train_config = config.get_config("pi05_putcab_official_clean50_full")

    assert train_config.model.pi05 is True
    assert train_config.model.casm_mode == "none"
    assert train_config.model.action_horizon == 50
    assert train_config.model.paligemma_variant == "gemma_2b"
    assert train_config.model.action_expert_variant == "gemma_300m"
    assert isinstance(train_config.freeze_filter, nnx.Nothing)
    assert train_config.pytorch_training_precision == "float32"
    assert train_config.batch_size == 1
    assert train_config.gradient_accumulation_steps == 32
    assert train_config.batch_size * train_config.gradient_accumulation_steps == 32
    assert train_config.num_train_steps == 30_000
    assert train_config.seed == 42
    assert train_config.ema_decay == 0.99
    assert train_config.params_only_checkpoint is False

    assert train_config.lr_schedule == optimizer.CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=2.5e-5,
        decay_steps=30_000,
        decay_lr=2.5e-6,
    )


def test_official_clean_config_has_no_phase_or_gate_inputs():
    train_config = config.get_config("pi05_putcab_official_clean50_full")
    repack = train_config.data.repack_transforms.inputs[0].structure

    assert repack["action_is_pad"] == "action_is_pad"
    assert "action_mask" not in repack
    assert "action_phase" not in repack
    assert train_config.data.action_sequence_keys == ("action",)
    assert train_config.data.adapt_to_pi is False
    assert train_config.data.use_delta_joint_actions is True


def test_official_clean_pytorch_config_preserves_global_recipe():
    train_config = config.get_config("pi05_putcab_official_clean50_pytorch_full")

    assert train_config.model.pi05 is True
    assert train_config.model.casm_mode == "none"
    assert train_config.pytorch_training_precision == "bfloat16"
    assert train_config.batch_size == 16
    assert train_config.gradient_accumulation_steps == 2
    assert train_config.batch_size * train_config.gradient_accumulation_steps == 32
    assert train_config.num_train_steps == 30_000
    assert train_config.seed == 42
    assert train_config.ema_decay == 0.99
    assert train_config.pytorch_compile_mode == "default"
    assert train_config.pytorch_attention_implementation == "sdpa"
    assert train_config.pytorch_gradient_checkpointing is False
    assert train_config.num_workers == 2
    assert train_config.persistent_workers is True
    assert train_config.pin_memory is True

    repack = train_config.data.repack_transforms.inputs[0].structure
    assert repack["action_is_pad"] == "action_is_pad"
    assert "action_mask" not in repack
    assert "action_phase" not in repack
