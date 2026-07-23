from __future__ import annotations

import dataclasses
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

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


def _tiny_config():
    return SimpleNamespace(
        lr_schedule=SimpleNamespace(warmup_steps=0, peak_lr=0.1, decay_steps=10, decay_lr=0.01),
        optimizer=SimpleNamespace(clip_gradient_norm=1.0),
    )


def test_train_step_updates_torch_model_and_reports_finite_metrics():
    model = TinyPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

    metrics = train_pytorch._train_step(  # noqa: SLF001
        model,
        optimizer,
        observation=None,
        actions=torch.ones(2, 3),
        config=_tiny_config(),
        global_step=0,
    )

    assert model.weight.detach() > 0
    assert metrics["loss"] == pytest.approx(1.0)
    assert np.isfinite(metrics["gradient_norm"])


def test_resume_signature_allows_only_training_budget_extension(monkeypatch):
    monkeypatch.setenv("PARALLELVLA_DATASET_REVISION", "revision")
    monkeypatch.setenv("PI05_BASE_SHA256", "sha")
    monkeypatch.setenv("PARALLELVLA_CODE_COMMIT", "a" * 40)
    config = _config.get_config("pi05_putcab_pytorch_matched_full")
    extended = dataclasses.replace(config, num_train_steps=config.num_train_steps + 1)

    assert train_pytorch._config_signature(config) == train_pytorch._config_signature(extended)  # noqa: SLF001


@pytest.mark.parametrize("code_commit", ["", "abc123", "A" * 40, "g" * 40])
def test_config_signature_requires_full_lowercase_code_commit(monkeypatch, code_commit):
    monkeypatch.setenv("PARALLELVLA_CODE_COMMIT", code_commit)
    config = _config.get_config("pi05_putcab_pytorch_matched_full")

    with pytest.raises(ValueError, match="PARALLELVLA_CODE_COMMIT"):
        train_pytorch._config_signature(config)  # noqa: SLF001


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
