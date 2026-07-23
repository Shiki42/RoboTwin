import dataclasses

import jax
import pytest
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_pytorch_loader_does_not_call_jax_tree_or_process_api(monkeypatch):
    dataset = [{"nested": {"value": index}} for index in range(4)]

    def fail(*args, **kwargs):
        raise AssertionError("PyTorch loader called a JAX API")

    monkeypatch.setattr(_data_loader.jax, "process_count", fail)
    monkeypatch.setattr(_data_loader.jax.tree, "map", fail)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=2,
        num_batches=1,
        framework="pytorch",
    )
    batch = next(iter(loader))

    assert torch.equal(batch["nested"]["value"], torch.tensor([0, 1]))


def test_resumable_random_sampler_restores_exact_next_index():
    dataset = list(range(10))
    sampler = _data_loader.ResumableRandomSampler(dataset, seed=7)
    iterator = iter(sampler)
    consumed = [next(iterator) for _ in range(4)]
    state = sampler.state_dict()
    expected = [next(iterator) for _ in range(6)]

    restored = _data_loader.ResumableRandomSampler(dataset, seed=7)
    restored.load_state_dict(state)
    actual = list(iter(restored))

    assert len(set(consumed + expected)) == len(dataset)
    assert actual == expected


def test_pytorch_loader_state_restores_exact_next_batch():
    dataset = [{"value": index} for index in range(10)]
    sampler = _data_loader.ResumableRandomSampler(dataset, seed=11)
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=2,
        sampler=sampler,
        framework="pytorch",
    )
    iterator = iter(loader)
    next(iterator)
    next(iterator)
    state = loader.state_dict()
    expected = next(iterator)["value"]

    restored_sampler = _data_loader.ResumableRandomSampler(dataset, seed=11)
    restored_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=2,
        sampler=restored_sampler,
        framework="pytorch",
    )
    restored_loader.load_state_dict(state)
    actual = next(iter(restored_loader))["value"]

    assert torch.equal(actual, expected)


def test_exact_loader_checkpoint_rejects_worker_prefetch():
    dataset = [{"value": index} for index in range(4)]
    sampler = _data_loader.ResumableRandomSampler(dataset, seed=0)
    loader = _data_loader.TorchDataLoader(
        dataset, local_batch_size=2, sampler=sampler, num_workers=1, framework="pytorch"
    )

    with pytest.raises(ValueError, match="num_workers=0"):
        loader.state_dict()


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_create_torch_dataset_passes_explicit_video_backend(monkeypatch):
    model_config = pi0_config.Pi0Config(action_dim=24, action_horizon=2, max_token_len=48)
    captured = {}
    sentinel = object()

    class DatasetMetadata:
        fps = 10

    monkeypatch.setattr(
        _data_loader.lerobot_dataset,
        "LeRobotDatasetMetadata",
        lambda repo_id: DatasetMetadata(),
    )

    def create_dataset(repo_id, **kwargs):
        captured["repo_id"] = repo_id
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", create_dataset)
    data_config = _config.DataConfig(repo_id="owner/dataset", video_backend="pyav")

    dataset = _data_loader.create_torch_dataset(data_config, 2, model_config)

    assert dataset is sentinel
    assert captured["repo_id"] == "owner/dataset"
    assert captured["video_backend"] == "pyav"
