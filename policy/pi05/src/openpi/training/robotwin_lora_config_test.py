from openpi.training import optimizer
from openpi.training import robotwin_lora_config


def test_robotwin_lora_config_matches_requested_smoke_and_formal_recipe(monkeypatch):
    monkeypatch.setenv("PARALLELVLA_DATASET_REPO", "pi05_place_dual_shoes_retime_100")

    train_config = robotwin_lora_config.create_config()

    assert train_config.data.repo_id == "pi05_place_dual_shoes_retime_100"
    assert train_config.model.pi05 is True
    assert train_config.model.paligemma_variant == "gemma_2b_lora"
    assert train_config.model.action_expert_variant == "gemma_300m_lora"
    assert train_config.pytorch_training_precision == "bfloat16"
    assert train_config.pytorch_trainable_scope == "lora"
    assert train_config.batch_size == 32
    assert train_config.gradient_accumulation_steps == 1
    assert train_config.num_train_steps == 30_000
    assert train_config.save_interval == 10_000
    assert train_config.ema_decay == 0.99
    assert train_config.pytorch_compile_mode == "default"
    assert train_config.pytorch_attention_implementation == "sdpa"
    assert train_config.pytorch_gradient_checkpointing is False
    assert train_config.lr_schedule == optimizer.CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=2.5e-5,
        decay_steps=30_000,
        decay_lr=2.5e-6,
    )


def test_robotwin_lora_config_routes_pad_mask_into_action_supervision():
    train_config = robotwin_lora_config.create_config()
    repack = train_config.data.repack_transforms.inputs[0].structure

    assert repack["action_is_pad"] == "action_is_pad"
    assert train_config.data.action_sequence_keys == ("action",)
    assert train_config.data.adapt_to_pi is False
