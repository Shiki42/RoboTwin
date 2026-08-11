from __future__ import annotations

import dataclasses
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from openpi.models_pytorch import lora_pytorch
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
from scripts import train_pytorch


class TinyPolicy(nn.Module):
    casm_mode = "none"

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, observation, actions):
        del observation
        return (self.weight - actions) ** 2


class TinyScopedPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = nn.Linear(2, 2)
        self.paligemma_with_expert.gemma_expert = nn.Linear(2, 2)
        self.action_in_proj = nn.Linear(2, 2)
        self.action_out_proj = nn.Linear(2, 2)
        self.time_mlp_in = nn.Linear(2, 2)
        self.time_mlp_out = nn.Linear(2, 2)
        self.phase_gate = nn.Linear(2, 1)


class TinyLoRAScopedPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = nn.Module()
        self.paligemma_with_expert.paligemma.model = nn.Module()
        self.paligemma_with_expert.paligemma.model.language_model = lora_pytorch.LoRALinear(
            nn.Linear(2, 2), rank=1, alpha=1.0
        )
        self.paligemma_with_expert.paligemma.model.vision_tower = nn.Linear(2, 2)
        self.paligemma_with_expert.gemma_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert.model = lora_pytorch.LoRALinear(nn.Linear(2, 2), rank=1, alpha=1.0)
        self.action_in_proj = nn.Linear(2, 2)
        self.action_out_proj = nn.Linear(2, 2)
        self.time_mlp_in = nn.Linear(2, 2)
        self.time_mlp_out = nn.Linear(2, 2)


def _tiny_config():
    return SimpleNamespace(
        lr_schedule=SimpleNamespace(warmup_steps=0, peak_lr=0.1, decay_steps=10, decay_lr=0.01),
        optimizer=SimpleNamespace(clip_gradient_norm=1.0),
        gradient_accumulation_steps=1,
        ema_decay=None,
        pytorch_trainable_scope="all",
    )


def test_train_step_updates_torch_model_and_reports_finite_metrics():
    model = TinyPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

    metrics = train_pytorch.train_step(
        model,
        optimizer,
        batches=[(None, torch.ones(2, 3))],
        config=_tiny_config(),
        global_step=0,
    )

    assert model.weight.detach() > 0
    assert metrics["loss"] == pytest.approx(1.0)
    assert np.isfinite(metrics["gradient_norm"])


def test_train_step_accumulation_matches_one_global_batch():
    full_batch_model = TinyPolicy()
    accumulated_model = TinyPolicy()
    full_batch_optimizer = torch.optim.SGD(full_batch_model.parameters(), lr=0.1)
    accumulated_optimizer = torch.optim.SGD(accumulated_model.parameters(), lr=0.1)
    first = torch.tensor([[1.0, 2.0]])
    second = torch.tensor([[3.0, 4.0]])
    full_config = _tiny_config()
    full_config.optimizer = SimpleNamespace(clip_gradient_norm=100.0)
    accumulated_config = _tiny_config()
    accumulated_config.optimizer = SimpleNamespace(clip_gradient_norm=100.0)
    accumulated_config.gradient_accumulation_steps = 2

    full_metrics = train_pytorch.train_step(
        full_batch_model,
        full_batch_optimizer,
        [(None, torch.cat((first, second), dim=0))],
        full_config,
        global_step=0,
    )
    accumulated_metrics = train_pytorch.train_step(
        accumulated_model,
        accumulated_optimizer,
        [(None, first), (None, second)],
        accumulated_config,
        global_step=0,
    )

    assert torch.allclose(full_batch_model.weight, accumulated_model.weight)
    assert accumulated_metrics["loss"] == pytest.approx(full_metrics["loss"])
    assert accumulated_metrics["gradient_norm"] == pytest.approx(full_metrics["gradient_norm"])


def test_action_expert_scope_freezes_only_pretrained_paligemma():
    model = TinyScopedPolicy()

    trainable_names = train_pytorch.configure_trainable_parameters(
        model,
        "action_expert_and_gate",
    )

    assert trainable_names
    assert not any(name.startswith("paligemma_with_expert.paligemma.") for name in trainable_names)
    assert all(parameter.requires_grad == (name in trainable_names) for name, parameter in model.named_parameters())
    for prefix in (
        "paligemma_with_expert.gemma_expert.",
        "action_in_proj.",
        "action_out_proj.",
        "time_mlp_in.",
        "time_mlp_out.",
        "phase_gate.",
    ):
        assert any(name.startswith(prefix) for name in trainable_names)


def test_lora_scope_freezes_gemma_base_but_keeps_adapters_vision_and_action_heads():
    model = TinyLoRAScopedPolicy()

    trainable_names = train_pytorch.configure_trainable_parameters(model, "lora")

    assert trainable_names
    assert any(name.endswith(".lora_a") for name in trainable_names)
    assert any(name.endswith(".lora_b") for name in trainable_names)
    assert any("vision_tower" in name for name in trainable_names)
    assert any(name.startswith("action_in_proj.") for name in trainable_names)
    assert not any(name.endswith(".weight") and "language_model" in name for name in trainable_names)
    assert not any(name.endswith(".weight") and "gemma_expert" in name for name in trainable_names)


def test_action_loss_uses_total_valid_dimension_weight():
    losses = torch.tensor([[1.0, 3.0]])
    action_mask = torch.tensor([[[1.0, 1.0], [1.0, 0.0]]])

    loss = train_pytorch._mean_action_loss(losses, action_mask)  # noqa: SLF001

    assert loss == pytest.approx(5.0 / 3.0)


def test_lora_signature_records_both_adapter_ranks(monkeypatch):
    monkeypatch.setenv("PARALLELVLA_CODE_COMMIT", "a" * 40)
    config = _config.get_config("pi05_robotwin_parallel100_pytorch_lora")

    signature = train_pytorch.config_signature(config)

    assert signature["model"]["lora"]["paligemma"] == {"rank": 16, "alpha": 16.0}
    assert signature["model"]["lora"]["action_expert"] == {"rank": 32, "alpha": 32.0}
    assert signature["action_loss_normalization"] == "valid_dimension_weighted_v1"


def test_resume_signature_allows_only_training_budget_extension(monkeypatch):
    monkeypatch.setenv("PARALLELVLA_DATASET_REVISION", "revision")
    monkeypatch.setenv("PI05_BASE_SHA256", "sha")
    monkeypatch.setenv("PARALLELVLA_CODE_COMMIT", "a" * 40)
    config = _config.get_config("pi05_putcab_pytorch_matched_full")
    extended = dataclasses.replace(config, num_train_steps=config.num_train_steps + 1)

    assert train_pytorch.config_signature(config) == train_pytorch.config_signature(extended)
    accumulated = dataclasses.replace(config, gradient_accumulation_steps=2)
    assert train_pytorch.config_signature(accumulated)["batch_size"] == 32
    assert train_pytorch.config_signature(accumulated)["microbatch_size"] == 16
    assert train_pytorch.config_signature(config) != train_pytorch.config_signature(accumulated)
    eager_attention = dataclasses.replace(config, pytorch_attention_implementation="eager")
    assert train_pytorch.config_signature(eager_attention)["pytorch_attention_implementation"] == "eager"
    assert train_pytorch.config_signature(config) != train_pytorch.config_signature(eager_attention)
    full_checkpointing = dataclasses.replace(config, pytorch_gradient_checkpointing_scope="full")
    assert train_pytorch.config_signature(full_checkpointing)["pytorch_gradient_checkpointing_scope"] == "full"
    assert train_pytorch.config_signature(config) != train_pytorch.config_signature(full_checkpointing)
    all_actions = dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, inactive_action_weight=1.0),
    )
    assert train_pytorch.config_signature(all_actions)["inactive_action_weight"] == 1.0
    assert train_pytorch.config_signature(config) != train_pytorch.config_signature(all_actions)
    short_schedule = dataclasses.replace(
        config,
        lr_schedule=dataclasses.replace(config.lr_schedule, decay_steps=4_000),
    )
    assert train_pytorch.config_signature(config) != train_pytorch.config_signature(short_schedule)
    expert_only = dataclasses.replace(
        config,
        pytorch_trainable_scope="action_expert_and_gate",
    )
    assert train_pytorch.config_signature(config) != train_pytorch.config_signature(expert_only)


@pytest.mark.parametrize("code_commit", ["", "abc123", "A" * 40, "g" * 40])
def test_config_signature_requires_full_lowercase_code_commit(monkeypatch, code_commit):
    monkeypatch.setenv("PARALLELVLA_CODE_COMMIT", code_commit)
    config = _config.get_config("pi05_putcab_pytorch_matched_full")

    with pytest.raises(ValueError, match="PARALLELVLA_CODE_COMMIT"):
        train_pytorch.config_signature(config)


def test_checkpoint_files_include_normalizer_and_manifest():
    stats = {
        "state": _normalize.NormStats(
            mean=np.zeros(2),
            std=np.ones(2),
            q01=np.zeros(2),
            q99=np.ones(2),
        )
    }
    data_config = _config.DataConfig(asset_id="putcab", norm_stats=stats)

    files = train_pytorch._checkpoint_files(data_config, {"run": "smoke"})  # noqa: SLF001

    assert pathlib.Path("run_manifest.json") in files
    norm_path = pathlib.Path("assets/putcab/norm_stats.json")
    assert _normalize.deserialize_json(files[norm_path])["state"].mean.tolist() == [0.0, 0.0]


def test_checkpoint_files_reject_missing_normalizer():
    with pytest.raises(ValueError, match="normalization stats"):
        train_pytorch._checkpoint_files(_config.DataConfig(asset_id="putcab"), {})  # noqa: SLF001
