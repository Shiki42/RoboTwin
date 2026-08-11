import dataclasses

import pytest

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


def test_robotwin_full_config_uses_fixed_episode_split_and_fast_bf16_recipe(monkeypatch):
    monkeypatch.setenv("PARALLELVLA_DATASET_REPO", "pi05_scan_object_retime_100")

    train, validation, unused = robotwin_lora_config._episode_split()  # noqa: SLF001
    train_config = robotwin_lora_config.create_full_config()

    assert len(train) == 50
    assert len(validation) == 3
    assert len(unused) == 47
    assert sorted((*train, *validation, *unused)) == list(range(100))
    assert train_config.name == "pi05_robotwin_parallel100_pytorch_full"
    assert train_config.data.repo_id == "pi05_scan_object_retime_100"
    assert train_config.model.paligemma_variant == "gemma_2b"
    assert train_config.model.action_expert_variant == "gemma_300m"
    assert train_config.pytorch_training_precision == "bfloat16"
    assert train_config.pytorch_trainable_scope == "all"
    assert train_config.train_episodes == train
    assert train_config.validation_episodes == validation
    assert train_config.validation_interval == 2_000
    assert train_config.validation_num_workers == 2
    assert train_config.batch_size == 16
    assert train_config.gradient_accumulation_steps == 1
    assert train_config.num_workers == 4
    assert train_config.save_interval == 10_000
    assert train_config.pytorch_compile_mode == "default"
    assert train_config.pytorch_gradient_checkpointing is False


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"validation_interval": -1}, "non-negative"),
        ({"validation_num_workers": -1}, "workers"),
        ({"validation_episodes": (2,)}, "positive validation interval"),
        ({"validation_interval": 1}, "requires validation episodes"),
        (
            {"train_episodes": None, "validation_episodes": (2,), "validation_interval": 1},
            "explicit training episode split",
        ),
        (
            {"train_episodes": (1, 1)},
            "duplicate",
        ),
        (
            {"train_episodes": (1,), "validation_episodes": (1,), "validation_interval": 1},
            "disjoint",
        ),
    ],
)
def test_train_config_rejects_invalid_episode_validation_controls(updates, error):
    config = robotwin_lora_config.create_config()

    with pytest.raises(ValueError, match=error):
        dataclasses.replace(config, **updates)
