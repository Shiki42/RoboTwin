from __future__ import annotations

import dataclasses
import pathlib
import random

import numpy as np
import pytest
import safetensors.torch
import torch
from torch import nn

from openpi.training import pytorch_training


@dataclasses.dataclass(frozen=True)
class ObservationFixture:
    images: dict[str, np.ndarray]
    state: np.ndarray
    optional: None = None


def test_move_to_device_preserves_dataclass_and_nested_structure():
    value = ObservationFixture(
        images={"main": np.ones((2, 3), dtype=np.float32)},
        state=np.zeros((2, 4), dtype=np.float32),
    )

    moved = pytorch_training.move_to_device(value, torch.device("cpu"), non_blocking=True)

    assert isinstance(moved, ObservationFixture)
    assert isinstance(moved.images["main"], torch.Tensor)
    assert isinstance(moved.state, torch.Tensor)
    assert moved.optional is None


def test_learning_rate_warmup_and_cosine_endpoints():
    kwargs = {
        "warmup_steps": 2,
        "peak_lr": 3e-4,
        "decay_steps": 10,
        "end_lr": 3e-5,
    }

    assert pytorch_training.learning_rate(0, **kwargs) == pytest.approx(1e-4)
    assert pytorch_training.learning_rate(2, **kwargs) == pytest.approx(3e-4)
    assert pytorch_training.learning_rate(10, **kwargs) == pytest.approx(3e-5)
    assert pytorch_training.learning_rate(20, **kwargs) == pytest.approx(3e-5)


def test_load_pretrained_allows_only_explicit_new_head(tmp_path):
    source = nn.Linear(3, 2)
    checkpoint = tmp_path / "base"
    checkpoint.mkdir()
    safetensors.torch.save_model(source, checkpoint / "model.safetensors")
    target = nn.ModuleDict({"base": nn.Linear(3, 2), "phase_gate": nn.Linear(2, 1)})

    with pytest.raises(ValueError, match="pretrained state mismatch"):
        pytorch_training.load_pretrained(target, checkpoint)

    compatible = nn.ModuleDict({"base": nn.Linear(3, 2), "phase_gate": nn.Linear(2, 1)})
    base_only = nn.ModuleDict({"base": compatible["base"]})
    safetensors.torch.save_model(base_only, checkpoint / "model.safetensors")
    missing, unexpected = pytorch_training.load_pretrained(
        compatible,
        checkpoint,
        allowed_missing_prefixes=("phase_gate.",),
    )

    assert set(missing) == {"phase_gate.weight", "phase_gate.bias"}
    assert unexpected == []


def test_checkpoint_round_trip_restores_model_optimizer_step_and_rng(tmp_path):
    pytorch_training.seed_everything(17)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(4, 3)
    model(inputs).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    checkpoint = pytorch_training.save_checkpoint(
        model,
        optimizer,
        global_step=20,
        checkpoint_root=tmp_path,
        metadata={"run": "smoke"},
        extra_files={pathlib.Path("assets/task/norm_stats.json"): "{}"},
    )
    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(3)

    restored_model = nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=9e-4)
    step, metadata, ema_state = pytorch_training.load_checkpoint(
        restored_model,
        restored_optimizer,
        checkpoint,
        device=torch.device("cpu"),
    )

    assert step == 20
    assert metadata == {"run": "smoke"}
    assert ema_state is None
    assert (checkpoint / "assets/task/norm_stats.json").read_text() == "{}"
    for expected, actual in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert torch.equal(expected, actual)
    assert restored_optimizer.state_dict()["state"]
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)


def test_ema_checkpoint_preserves_training_and_inference_weights(tmp_path):
    model = nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema_state = pytorch_training.initialize_ema(model)
    initial = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(2.0)
    pytorch_training.update_ema(ema_state, model, decay=0.5)
    expected_training = model.weight.detach().clone()
    expected_ema = initial + 1.0

    checkpoint = pytorch_training.save_checkpoint(
        model,
        optimizer,
        global_step=3,
        checkpoint_root=tmp_path,
        metadata={"run": "ema"},
        ema_state=ema_state,
    )

    assert torch.equal(model.weight, expected_training)
    assert (checkpoint / "model.safetensors").is_file()
    assert (checkpoint / "training_model.safetensors").is_file()
    assert (checkpoint / "ema_state.safetensors").is_file()
    inference_model = nn.Linear(2, 1, bias=False)
    safetensors.torch.load_model(inference_model, checkpoint / "model.safetensors", strict=True)
    assert torch.equal(inference_model.weight, expected_ema)

    restored_model = nn.Linear(2, 1, bias=False)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    step, metadata, restored_ema = pytorch_training.load_checkpoint(
        restored_model,
        restored_optimizer,
        checkpoint,
        device=torch.device("cpu"),
        load_ema=True,
    )

    assert step == 3
    assert metadata == {"run": "ema"}
    assert torch.equal(restored_model.weight, expected_training)
    assert restored_ema is not None
    assert torch.equal(restored_ema["weight"], expected_ema)


def _adapter_model() -> nn.ModuleDict:
    model = nn.ModuleDict(
        {
            "frozen": nn.Linear(2, 2, bias=False),
            "adapter": nn.Linear(2, 1, bias=False),
        }
    )
    model["frozen"].weight.requires_grad = False
    return model


def test_ema_tracks_only_trainable_parameters_and_keeps_frozen_weights(tmp_path):
    model = _adapter_model()
    optimizer = torch.optim.AdamW(model["adapter"].parameters(), lr=1e-3)
    ema_state = pytorch_training.initialize_ema(model)
    initial_adapter = model["adapter"].weight.detach().clone()
    with torch.no_grad():
        model["frozen"].weight.add_(3.0)
        model["adapter"].weight.add_(2.0)
    expected_frozen = model["frozen"].weight.detach().clone()
    expected_training_adapter = model["adapter"].weight.detach().clone()
    pytorch_training.update_ema(ema_state, model, decay=0.5)

    assert set(ema_state) == {"adapter.weight"}
    checkpoint = pytorch_training.save_checkpoint(
        model,
        optimizer,
        global_step=4,
        checkpoint_root=tmp_path,
        metadata={"run": "adapter-ema"},
        ema_state=ema_state,
    )

    inference_model = _adapter_model()
    safetensors.torch.load_model(inference_model, checkpoint / "model.safetensors", strict=True)
    assert torch.equal(inference_model["frozen"].weight, expected_frozen)
    torch.testing.assert_close(inference_model["adapter"].weight, initial_adapter + 1.0)
    assert torch.equal(model["adapter"].weight, expected_training_adapter)

    restored_model = _adapter_model()
    restored_optimizer = torch.optim.AdamW(restored_model["adapter"].parameters(), lr=1e-3)
    step, metadata, restored_ema = pytorch_training.load_checkpoint(
        restored_model,
        restored_optimizer,
        checkpoint,
        device=torch.device("cpu"),
        load_ema=True,
    )

    assert step == 4
    assert metadata == {"run": "adapter-ema"}
    assert restored_ema is not None
    assert set(restored_ema) == {"adapter.weight"}


def test_save_checkpoint_is_atomic_and_rejects_duplicate_step(tmp_path):
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())

    pytorch_training.save_checkpoint(
        model,
        optimizer,
        global_step=1,
        checkpoint_root=tmp_path,
        metadata={},
    )

    with pytest.raises(FileExistsError):
        pytorch_training.save_checkpoint(
            model,
            optimizer,
            global_step=1,
            checkpoint_root=tmp_path,
            metadata={},
        )
    assert not list(tmp_path.glob(".tmp-*"))


def test_checkpoint_extra_files_reject_path_traversal(tmp_path):
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())

    with pytest.raises(ValueError, match="must be relative"):
        pytorch_training.save_checkpoint(
            model,
            optimizer,
            global_step=1,
            checkpoint_root=tmp_path,
            metadata={},
            extra_files={pathlib.Path("../outside"): "forbidden"},
        )
    assert not (tmp_path.parent / "outside").exists()


def test_prune_checkpoints_keeps_current_and_periodic_steps(tmp_path):
    for step in (2_000, 4_000, 20_000):
        (tmp_path / str(step)).mkdir()
    pytorch_training.prune_checkpoints(tmp_path, current_step=4_000, keep_period=20_000)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["20000", "4000"]
