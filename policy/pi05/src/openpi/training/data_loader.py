from collections.abc import Iterator, Sequence
import dataclasses
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)
logger = logging.getLogger(__name__)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class DeterministicBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Derive each shuffled batch from seed and consumed batch count.

    Worker prefetch may request future batches, so iteration never mutates the
    checkpoint cursor. The loader advances the cursor only when a batch is
    returned to the training process.
    """

    def __init__(self, dataset: Dataset, *, batch_size: int, seed: int):
        if seed < 0:
            raise ValueError("batch-plan seed must be non-negative")
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        if len(dataset) < batch_size:
            raise ValueError(f"batch size ({batch_size}) is larger than dataset size ({len(dataset)})")
        self._dataset = dataset
        self._batch_size = batch_size
        self._seed = seed
        self._consumed_batches = 0

    @property
    def steps_per_epoch(self) -> int:
        return len(self._dataset) // self._batch_size

    def __iter__(self):
        start = self._consumed_batches
        epoch = start // self.steps_per_epoch
        batch_in_epoch = start % self.steps_per_epoch
        generator = torch.Generator().manual_seed(self._seed + epoch)
        permutation = torch.randperm(len(self._dataset), generator=generator)
        for batch_index in range(batch_in_epoch, self.steps_per_epoch):
            offset = batch_index * self._batch_size
            yield permutation[offset : offset + self._batch_size].tolist()

    def __len__(self) -> int:
        return self.steps_per_epoch - self._consumed_batches % self.steps_per_epoch

    def mark_consumed(self) -> None:
        self._consumed_batches += 1

    def state_dict(self) -> dict[str, int]:
        return {
            "seed": self._seed,
            "length": len(self._dataset),
            "batch_size": self._batch_size,
            "consumed_batches": self._consumed_batches,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        expected_keys = {"seed", "length", "batch_size", "consumed_batches"}
        if set(state) != expected_keys:
            raise ValueError(f"invalid batch-plan state keys: {sorted(state)}")
        expected = {"seed": self._seed, "length": len(self._dataset), "batch_size": self._batch_size}
        actual = {key: state[key] for key in expected}
        if actual != expected:
            raise ValueError(f"batch-plan signature mismatch: {actual} != {expected}")
        if state["consumed_batches"] < 0:
            raise ValueError(f"invalid consumed batch count: {state['consumed_batches']}")
        self._consumed_batches = state["consumed_batches"]


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    episodes: Sequence[int] | None = None,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        episodes=None if episodes is None else list(episodes),
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        video_backend=data_config.video_backend,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    split: Literal["train", "validation"] = "train",
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logger.info("data_config: %s", data_config)

    if split == "validation":
        if not config.validation_episodes:
            raise ValueError("validation split is empty")
        if shuffle:
            raise ValueError("validation data cannot be shuffled")
        episodes = config.validation_episodes
        single_epoch = True
        drop_last = False
    else:
        episodes = config.train_episodes
        single_epoch = False
        drop_last = True

    if data_config.rlds_data_dir is not None:
        if split != "train":
            raise NotImplementedError("RLDS validation splits are not supported")
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        episodes=episodes,
        single_epoch=single_epoch,
        drop_last=drop_last,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
        pin_memory=config.pin_memory,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    episodes: Sequence[int] | None = None,
    single_epoch: bool = False,
    drop_last: bool = True,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config, episodes=episodes)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    batch_sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=drop_last,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()
    if shuffle and sampler is None:
        batch_sampler = DeterministicBatchSampler(dataset, batch_size=local_batch_size, seed=seed)

    logger.info("local_batch_size: %s", local_batch_size)
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and batch_sampler is None and shuffle),
        sampler=sampler,
        batch_sampler=batch_sampler,
        single_epoch=single_epoch,
        drop_last=drop_last,
        num_batches=num_batches,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        batch_sampler: DeterministicBatchSampler | None = None,
        single_epoch: bool = False,
        drop_last: bool = True,
        num_batches: int | None = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if framework == "jax" and jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches
        self._batch_sampler = batch_sampler

        if prefetch_factor < 1:
            raise ValueError("prefetch factor must be positive")
        if batch_sampler is not None and (sampler is not None or shuffle):
            raise ValueError("batch sampler cannot be combined with sampler or shuffle")
        if single_epoch and num_batches is not None:
            raise ValueError("single-epoch loading cannot set num_batches")
        if batch_sampler is not None and not drop_last:
            raise ValueError("deterministic batch sampling requires drop_last")

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        loader_kwargs = {
            "dataset": typing.cast(torch.utils.data.Dataset, dataset),
            "num_workers": num_workers,
            "multiprocessing_context": mp_context,
            "persistent_workers": persistent_workers and num_workers > 0,
            "collate_fn": _collate_torch_fn if framework == "pytorch" else _collate_fn,
            "worker_init_fn": _worker_init_fn,
            "generator": generator,
            "pin_memory": pin_memory and framework == "pytorch",
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        if batch_sampler is None:
            loader_kwargs.update(
                batch_size=local_batch_size,
                shuffle=(sampler is None and shuffle),
                sampler=sampler,
                drop_last=drop_last,
            )
        else:
            loader_kwargs["batch_sampler"] = batch_sampler
        self._data_loader = torch.utils.data.DataLoader(**loader_kwargs)
        if single_epoch:
            self._num_batches = len(self._data_loader)

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def state_dict(self) -> dict[str, dict[str, int]]:
        if self._batch_sampler is None:
            raise ValueError("exact loader checkpointing requires DeterministicBatchSampler")
        return {"batch_plan": self._batch_sampler.state_dict()}

    def load_state_dict(self, state: dict[str, dict[str, int]]) -> None:
        if set(state) != {"batch_plan"}:
            raise ValueError(f"invalid loader state keys: {sorted(state)}")
        if self._batch_sampler is None:
            raise ValueError("exact loader checkpointing requires DeterministicBatchSampler")
        self._batch_sampler.load_state_dict(state["batch_plan"])

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                if self._batch_sampler is not None:
                    self._batch_sampler.mark_consumed()
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield _tree_map(torch.as_tensor, batch)


def _tree_map(function, value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        updates = {field.name: _tree_map(function, getattr(value, field.name)) for field in dataclasses.fields(value)}
        return dataclasses.replace(value, **updates)
    if isinstance(value, dict):
        return {key: _tree_map(function, item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_map(function, item) for item in value)
    if isinstance(value, list):
        return [_tree_map(function, item) for item in value]
    if value is None:
        return None
    return function(value)


def _tree_map_many(function, values):
    first = values[0]
    if first is None:
        if any(value is not None for value in values):
            raise ValueError("tree structure mismatch: mixed None and non-None values")
        return None
    if dataclasses.is_dataclass(first) and not isinstance(first, type):
        updates = {
            field.name: _tree_map_many(function, [getattr(value, field.name) for value in values])
            for field in dataclasses.fields(first)
        }
        return dataclasses.replace(first, **updates)
    if isinstance(first, dict):
        return {key: _tree_map_many(function, [value[key] for value in values]) for key in first}
    if isinstance(first, tuple):
        return tuple(_tree_map_many(function, [value[index] for value in values]) for index in range(len(first)))
    if isinstance(first, list):
        return [_tree_map_many(function, [value[index] for value in values]) for index in range(len(first))]
    return function(*values)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return _tree_map_many(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), items)


def _collate_torch_fn(items):
    """Prepare final numeric tensors before DataLoader moves them to pinned memory."""
    batch = _tree_map(torch.as_tensor, _collate_fn(items))
    if isinstance(batch, dict) and "image" in batch:
        _model.convert_uint8_images(batch["image"])
    return batch


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]

    def state_dict(self) -> dict[str, dict[str, int]]:
        return self._data_loader.state_dict()

    def load_state_dict(self, state: dict[str, dict[str, int]]) -> None:
        self._data_loader.load_state_dict(state)
