from __future__ import annotations

import dataclasses
import json
import pathlib
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from openpi.models_pytorch import lora_pytorch
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
from openpi.training import pytorch_training
from openpi.training import robotwin_lora_config
from openpi.training import subtask_aux_config
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
        self.paligemma_with_expert.paligemma.model = nn.Module()
        self.paligemma_with_expert.paligemma.model.multi_modal_projector = nn.Linear(2, 2)
        self.paligemma_with_expert.gemma_expert = nn.Linear(2, 2)
        self.action_in_proj = nn.Linear(2, 2)
        self.action_out_proj = nn.Linear(2, 2)
        self.time_mlp_in = nn.Linear(2, 2)
        self.time_mlp_out = nn.Linear(2, 2)
        self.phase_gate = nn.Linear(2, 1)
        self.subtask_head = nn.Linear(2, 2)


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


class TinyPrecisionPolicy(nn.Module):
    casm_mode = "none"

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.last_output_dtype = None

    def forward(self, observation, actions):
        del observation
        output = self.linear(actions)
        self.last_output_dtype = output.dtype
        return output.square()


class TinyAuxPolicy(nn.Module):
    casm_mode = "none"
    aux_subtask_classes = 2
    aux_subtask_loss_weight = 0.5

    def __init__(self):
        super().__init__()
        self.subtask_head = nn.Parameter(torch.tensor(0.0))
        self.action_forward_called = False

    def forward(self, observation, actions=None, *, return_aux=False, subtask_only=False):
        del observation, actions
        auxiliary = {
            "subtask_loss": (self.subtask_head - 1.0).square(),
            "subtask_accuracy": torch.tensor(0.5),
        }
        if subtask_only:
            return auxiliary
        self.action_forward_called = True
        losses = torch.tensor([[1.0, 3.0]])
        return (losses, auxiliary) if return_aux else losses


def _tiny_config():
    return SimpleNamespace(
        lr_schedule=SimpleNamespace(warmup_steps=0, peak_lr=0.1, decay_steps=10, decay_lr=0.01),
        optimizer=SimpleNamespace(clip_gradient_norm=1.0),
        gradient_accumulation_steps=1,
        ema_decay=None,
        pytorch_trainable_scope="all",
        pytorch_compute_precision="float32",
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


def test_bfloat16_compute_keeps_fp32_master_parameters():
    model = TinyPrecisionPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    config = _tiny_config()
    config.pytorch_compute_precision = "bfloat16"

    metrics = train_pytorch.train_step(
        model,
        optimizer,
        batches=[(None, torch.ones(2, 4))],
        config=config,
        global_step=0,
    )

    assert model.last_output_dtype == torch.bfloat16
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert np.isfinite(metrics["loss"])


def test_build_model_rejects_bfloat16_master_parameters(monkeypatch):
    config = _config.get_config("pi05_putcab_pytorch_matched_full")
    config = dataclasses.replace(config, pytorch_training_precision="bfloat16")
    monkeypatch.setattr(train_pytorch.pi0_pytorch, "PI0Pytorch", TinyPolicy)

    with pytest.raises(ValueError, match="float32 master parameters"):
        train_pytorch.build_model(config, torch.device("cpu"))


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


def test_subtask_head_scope_freezes_every_other_parameter():
    model = TinyScopedPolicy()

    trainable_names = train_pytorch.configure_trainable_parameters(model, "subtask_head")

    assert trainable_names
    assert all(name.startswith("subtask_head.") for name in trainable_names)
    assert all(parameter.requires_grad == (name in trainable_names) for name, parameter in model.named_parameters())


def test_shared_projector_scope_keeps_action_head_frozen():
    model = TinyScopedPolicy()

    trainable_names = train_pytorch.configure_trainable_parameters(model, "subtask_head_and_projector")

    prefixes = (
        "subtask_head.",
        "paligemma_with_expert.paligemma.model.multi_modal_projector.",
    )
    assert trainable_names
    assert all(name.startswith(prefixes) for name in trainable_names)
    assert any(name.startswith("subtask_head.") for name in trainable_names)
    assert any("multi_modal_projector" in name for name in trainable_names)
    assert not any(name.startswith("action_out_proj.") for name in trainable_names)
    assert all(parameter.requires_grad == (name in trainable_names) for name, parameter in model.named_parameters())


def test_joint_action_policy_scope_freezes_pretrained_prefix():
    model = TinyScopedPolicy()

    trainable_names = train_pytorch.configure_trainable_parameters(
        model,
        "subtask_head_and_action_policy",
    )

    prefixes = (
        "subtask_head.",
        "paligemma_with_expert.gemma_expert.",
        "action_in_proj.",
        "action_out_proj.",
        "time_mlp_in.",
        "time_mlp_out.",
    )
    assert trainable_names
    assert all(name.startswith(prefixes) for name in trainable_names)
    for prefix in prefixes:
        assert any(name.startswith(prefix) for name in trainable_names)
    assert not any(name.startswith("paligemma_with_expert.paligemma.") for name in trainable_names)
    assert not any(name.startswith("phase_gate.") for name in trainable_names)
    assert all(parameter.requires_grad == (name in trainable_names) for name, parameter in model.named_parameters())


def test_subtask_head_training_skips_action_forward():
    model = TinyAuxPolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = _tiny_config()
    config.pytorch_trainable_scope = "subtask_head"
    observation = SimpleNamespace(
        action_mask=torch.tensor([[[1.0, 1.0], [1.0, 0.0]]]),
    )

    metrics = train_pytorch.train_step(
        model,
        optimizer,
        [(observation, torch.zeros(1, 2, 2))],
        config,
        global_step=0,
    )

    assert model.subtask_head.detach() > 0
    assert not model.action_forward_called
    assert "action_loss" not in metrics
    assert metrics["subtask_loss"] == pytest.approx(1.0)
    assert metrics["loss"] == pytest.approx(0.5)


def test_shared_projector_training_keeps_action_objective():
    model = TinyAuxPolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = _tiny_config()
    config.pytorch_trainable_scope = "subtask_head_and_projector"
    observation = SimpleNamespace(
        action_mask=torch.tensor([[[1.0, 1.0], [0.0, 0.0]]]),
    )

    metrics = train_pytorch.train_step(
        model,
        optimizer,
        [(observation, torch.zeros(1, 2, 2))],
        config,
        global_step=0,
    )

    assert model.action_forward_called
    assert model.subtask_head.detach() > 0
    assert metrics["action_loss"] == pytest.approx(1.0)
    assert metrics["subtask_loss"] == pytest.approx(1.0)
    assert metrics["loss"] == pytest.approx(1.5)


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


def test_subtask_signature_records_class_weights(monkeypatch):
    monkeypatch.setenv("PARALLELVLA_CODE_COMMIT", "a" * 40)
    config = _config.get_config("pi05_putcab_factorized_anchor_subtask_head_sqrt_balanced_pytorch")

    signature = train_pytorch.config_signature(config)

    assert signature["model"]["aux_subtask"]["class_weights"] == (subtask_aux_config.SQRT_BALANCED_CLASS_WEIGHTS)


def test_shared_subtask_signature_records_factorized_prompt_contract(monkeypatch):
    monkeypatch.setenv("PARALLELVLA_CODE_COMMIT", "a" * 40)
    config = _config.get_config("pi05_putcab_factorized_anchor_subtask_shared_projector_pytorch")

    signature = train_pytorch.config_signature(config)

    assert signature["pytorch_action_prompt_mode"] == "teacher_forced_joint_subtask"
    assert signature["training_objective"] == "teacher_forced_factorized_action_plus_subtask_shared_projector_v1"


def test_joint_action_policy_signature_records_factorized_prompt_contract(monkeypatch):
    monkeypatch.setenv("PARALLELVLA_CODE_COMMIT", "a" * 40)
    config = _config.get_config("pi05_putcab_factorized_anchor_subtask_joint_action_policy_pytorch")

    signature = train_pytorch.config_signature(config)

    assert signature["pytorch_trainable_scope"] == "subtask_head_and_action_policy"
    assert signature["pytorch_action_prompt_mode"] == "teacher_forced_joint_subtask"
    assert signature["training_objective"] == "teacher_forced_factorized_action_plus_subtask_action_policy_v1"


def test_task_only_joint_action_policy_signature_records_control_contract(monkeypatch):
    monkeypatch.setenv("PARALLELVLA_CODE_COMMIT", "a" * 40)
    config = _config.get_config("pi05_putcab_factorized_anchor_subtask_joint_action_policy_task_only_pytorch")

    signature = train_pytorch.config_signature(config)

    assert signature["pytorch_trainable_scope"] == "subtask_head_and_action_policy"
    assert signature["pytorch_action_prompt_mode"] == "task_only"
    assert signature["training_objective"] == "factorized_action_plus_subtask_action_policy_task_only_v1"


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
    fp32_compute = dataclasses.replace(config, pytorch_compute_precision="float32")
    assert train_pytorch.config_signature(fp32_compute)["pytorch_compute_precision"] == "float32"
    assert train_pytorch.config_signature(config) != train_pytorch.config_signature(fp32_compute)
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


def test_action_loss_total_reports_weighted_sum_and_weight():
    losses = torch.tensor([[1.0, 3.0]])
    action_mask = torch.tensor([[[1.0, 1.0], [1.0, 0.0]]])

    total, weight = train_pytorch._action_loss_total(losses, action_mask)  # noqa: SLF001

    assert total == pytest.approx(5.0)
    assert weight == pytest.approx(3.0)


def _write_split_receipt(path, config, *, validation_episodes=None):
    train, validation, unused = robotwin_lora_config._episode_split()  # noqa: SLF001
    payload = {
        "dataset_repo": config.data.repo_id,
        "dataset_revision": "dataset-revision",
        "total_episodes": 100,
        "seed": 42,
        "train_episodes": list(train),
        "validation_episodes": list(validation if validation_episodes is None else validation_episodes),
        "unused_episodes": list(unused),
    }
    path.write_text(json.dumps(payload))


def test_episode_split_signature_accepts_matching_complete_partition(tmp_path, monkeypatch):
    config = robotwin_lora_config.create_full_config()
    receipt = tmp_path / "split.json"
    _write_split_receipt(receipt, config)
    monkeypatch.setenv("PARALLELVLA_SPLIT_RECEIPT", str(receipt))
    monkeypatch.setenv("PARALLELVLA_DATASET_REVISION", "dataset-revision")

    signature = train_pytorch._episode_split_signature(config)  # noqa: SLF001

    assert signature["total_episodes"] == 100
    assert signature["train_episodes"] == list(config.train_episodes)
    assert signature["validation_episodes"] == list(config.validation_episodes)
    assert signature["validation_interval"] == 2_000
    assert len(signature["receipt_sha256"]) == 64


def test_episode_split_signature_rejects_mismatched_validation_split(tmp_path, monkeypatch):
    config = robotwin_lora_config.create_full_config()
    receipt = tmp_path / "split.json"
    _write_split_receipt(receipt, config, validation_episodes=(99,))
    monkeypatch.setenv("PARALLELVLA_SPLIT_RECEIPT", str(receipt))
    monkeypatch.setenv("PARALLELVLA_DATASET_REVISION", "dataset-revision")

    with pytest.raises(ValueError, match="validation episodes"):
        train_pytorch._episode_split_signature(config)  # noqa: SLF001


class ValidationPolicy(nn.Module):
    def forward(self, observation, actions):
        del observation
        _ = random.random(), np.random.random(), torch.rand(())
        return actions[..., 0]


def test_validation_loss_is_weighted_deterministic_and_restores_rng_state():
    model = ValidationPolicy()
    loader = [
        (
            SimpleNamespace(action_mask=torch.tensor([[[1.0, 1.0]]])),
            torch.tensor([[[1.0, 0.0]]]),
        ),
        (
            SimpleNamespace(action_mask=torch.tensor([[[1.0, 0.0]]])),
            torch.tensor([[[3.0, 0.0]]]),
        ),
    ]
    pytorch_training.seed_everything(123)
    rng_state = pytorch_training.capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(()))
    pytorch_training.restore_rng_state(rng_state)

    config = _tiny_config()
    first = train_pytorch.evaluate_validation_loss(
        model,
        loader,
        torch.device("cpu"),
        seed=999,
        config=config,
    )
    actual = (random.random(), np.random.random(), torch.rand(()))
    second = train_pytorch.evaluate_validation_loss(
        model,
        loader,
        torch.device("cpu"),
        seed=999,
        config=config,
    )

    assert first["validation_loss"] == pytest.approx(5.0 / 3.0)
    assert first["validation_loss"] == second["validation_loss"]
    assert first["validation_batches"] == 2
    assert first["validation_samples"] == 2
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert actual[2] == expected[2]
    assert model.training is True


def test_normalizer_signature_binds_train_split_inventory_and_exact_file(tmp_path, monkeypatch):
    config = dataclasses.replace(
        robotwin_lora_config.create_full_config(),
        assets_base_dir=str(tmp_path / "assets"),
    )
    split_receipt = tmp_path / "split.json"
    _write_split_receipt(split_receipt, config)
    monkeypatch.setenv("PARALLELVLA_SPLIT_RECEIPT", str(split_receipt))
    monkeypatch.setenv("PARALLELVLA_DATASET_REVISION", "dataset-revision")
    episode_split = train_pytorch._episode_split_signature(config)  # noqa: SLF001

    normalizer_path = config.assets_dirs / config.data.repo_id / "norm_stats.json"
    normalizer_path.parent.mkdir(parents=True)
    normalizer_path.write_text("{}")
    normalizer_receipt = tmp_path / "normalizer.json"
    normalizer_receipt.write_text(
        json.dumps(
            {
                "dataset_repo": config.data.repo_id,
                "dataset_revision": "dataset-revision",
                "train_episodes": list(config.train_episodes),
                "split_receipt_sha256": episode_split["receipt_sha256"],
                "statistics": "exact_concat_train_only_valid_action_steps_v1",
                "normalizer_sha256": train_pytorch._sha256(normalizer_path),  # noqa: SLF001
                "input_inventory_sha256": "b" * 64,
            }
        )
    )
    monkeypatch.setenv("PARALLELVLA_NORM_STATS_RECEIPT", str(normalizer_receipt))

    signature = train_pytorch._normalizer_signature(config, episode_split)  # noqa: SLF001

    assert signature["normalizer_sha256"] == train_pytorch._sha256(normalizer_path)  # noqa: SLF001
    assert signature["input_inventory_sha256"] == "b" * 64
    assert signature["statistics"] == "exact_concat_train_only_valid_action_steps_v1"
