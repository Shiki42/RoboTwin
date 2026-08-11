from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


def load_dp3_wrapper(monkeypatch):
    hydra = ModuleType("hydra")
    hydra.main = lambda **kwargs: lambda function: function
    omegaconf = ModuleType("omegaconf")

    class OmegaConf:
        @staticmethod
        def register_new_resolver(*args, **kwargs):
            return None

    omegaconf.OmegaConf = OmegaConf
    train_dp3 = ModuleType("train_dp3")
    train_dp3.TrainDP3Workspace = object
    monkeypatch.setitem(sys.modules, "hydra", hydra)
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf)
    monkeypatch.setitem(sys.modules, "train_dp3", train_dp3)

    module_path = (
        Path(__file__).parents[1]
        / "policy/DP3/3D-Diffusion-Policy/dp3_policy.py"
    )
    spec = importlib.util.spec_from_file_location("dp3_episode_seed_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dp3_episode_seed_reproduces_sampling_stream(monkeypatch) -> None:
    module = load_dp3_wrapper(monkeypatch)
    model = object.__new__(module.DP3)
    original_state = torch.random.get_rng_state()

    try:
        model.set_episode_seed(100113)
        first = torch.rand(8)
        model.set_episode_seed(100113)
        repeated = torch.rand(8)
        model.set_episode_seed(100114)
        changed = torch.rand(8)
    finally:
        torch.random.set_rng_state(original_state)

    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed)


@pytest.mark.parametrize("seed", (True, 1.5, "100113"))
def test_dp3_episode_seed_requires_integer(monkeypatch, seed) -> None:
    module = load_dp3_wrapper(monkeypatch)
    model = object.__new__(module.DP3)
    with pytest.raises(TypeError):
        model.set_episode_seed(seed)


def test_dp3_episode_seed_requires_non_negative_value(monkeypatch) -> None:
    module = load_dp3_wrapper(monkeypatch)
    model = object.__new__(module.DP3)
    with pytest.raises(ValueError):
        model.set_episode_seed(-1)
