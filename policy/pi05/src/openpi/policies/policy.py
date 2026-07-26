from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import casm_language
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._predict_async_probability = None
        self._predict_casm_language = None

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)
            if getattr(model, "semantic_subtask_prediction", False):
                self._predict_casm_language = nnx_utils.module_jit(model.predict_casm_language)
            elif getattr(model, "casm_mode", "none") == "visual_phase_gate":
                self._predict_async_probability = nnx_utils.module_jit(model.predict_async_probability)

    def _transform_inputs(self, obs: dict):
        inputs = self._input_transform(jax.tree.map(lambda value: value, obs))
        if not self._is_pytorch_model:
            return jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        return jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...],
            inputs,
        )

    def _semantic_action_observation(self, obs: dict):
        high_level_inputs = self._transform_inputs(obs)
        high_level_observation = _model.Observation.from_dict(high_level_inputs)
        prediction = self._predict_casm_language(high_level_observation)
        async_probability, role_probability = np.asarray(prediction[0, :2])
        stage_probabilities = np.asarray(prediction[0, 2:])
        semantic_id = casm_language.semantic_id_from_prediction(
            role_probability,
            async_probability,
            stage_probabilities,
        )
        action_obs = jax.tree.map(lambda value: value, obs)
        action_obs["prompt"] = casm_language.format_action_prompt(obs["prompt"], semantic_id)
        metadata = {
            "semantic_subtask_id": semantic_id,
            "semantic_subtask_prompt": action_obs["prompt"],
            "semantic_object_arm_right_probability": float(role_probability),
            "semantic_stage_probabilities": stage_probabilities.tolist(),
        }
        return action_obs, prediction[:, 0], metadata

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        start_time = time.monotonic()
        semantic_metadata = {}
        semantic_async_probability = None
        action_obs = obs
        if self._predict_casm_language is not None:
            action_obs, semantic_async_probability, semantic_metadata = self._semantic_action_observation(obs)
        inputs = self._transform_inputs(action_obs)
        if self._is_pytorch_model:
            sample_rng_or_pytorch_device = self._pytorch_device
        else:
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        if semantic_async_probability is not None:
            outputs["async_probability"] = semantic_async_probability
        elif self._predict_async_probability is not None:
            outputs["async_probability"] = self._predict_async_probability(observation)
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs.update(semantic_metadata)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
