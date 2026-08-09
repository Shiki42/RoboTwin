import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import casm
from openpi.models import casm_language
from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    spline_field: bool = False
    control_horizon: int = 50
    spline_control_points: int = 16
    spline_degree: int = 3
    spline_regularization: float = 1e-6
    casm_mode: casm.CasmMode = "none"
    coordination_gate_hidden_dim: int = 64
    cross_attention_dim: int = 128
    gate_loss_weight: float = 0.2
    gate_positive_weight: float = 1.0
    usefulness_loss_weight: float = 0.2
    phase_prior_loss_weight: float = 0.1
    usefulness_temperature: float = 0.1
    semantic_subtask_prediction: bool = False
    semantic_role_loss_weight: float = 0.2
    semantic_stage_loss_weight: float = 0.2
    semantic_stage_class_weights: tuple[float, ...] = (13.81, 1.13, 0.93, 13.81, 3.78, 0.78)
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    def __post_init__(self):
        if self.spline_field:
            if not self.pi05:
                raise ValueError("spline action fields initially require pi05")
            if self.casm_mode != "none":
                raise ValueError("spline action fields initially require casm_mode none")
            if self.action_horizon != self.spline_control_points:
                raise ValueError("action_horizon must equal spline_control_points in spline field mode")
            if self.control_horizon <= self.spline_control_points:
                raise ValueError("control_horizon must be greater than spline_control_points")
            if self.spline_control_points <= self.spline_degree:
                raise ValueError("spline_control_points must be greater than spline_degree")
            if self.spline_regularization < 0:
                raise ValueError("spline_regularization must be non-negative")
        if self.casm_mode not in casm.VALID_CASM_MODES:
            raise ValueError(f"unknown CASM mode: {self.casm_mode}")
        if self.coordination_gate_hidden_dim < 1 or self.cross_attention_dim < 1:
            raise ValueError("CASM hidden dimensions must be positive")
        if min(self.gate_loss_weight, self.usefulness_loss_weight, self.phase_prior_loss_weight) < 0:
            raise ValueError("CASM loss weights must be non-negative")
        if self.gate_positive_weight <= 0:
            raise ValueError("gate positive weight must be positive")
        if self.usefulness_temperature <= 0:
            raise ValueError("usefulness temperature must be positive")
        if self.semantic_role_loss_weight < 0:
            raise ValueError("semantic role loss weight must be non-negative")
        if self.semantic_stage_loss_weight < 0:
            raise ValueError("semantic stage loss weight must be non-negative")
        if len(self.semantic_stage_class_weights) != casm_language.SEMANTIC_STAGE_COUNT:
            raise ValueError("semantic stage class weights must cover every stage")
        if any(weight <= 0 for weight in self.semantic_stage_class_weights):
            raise ValueError("semantic stage class weights must be positive")
        if self.semantic_subtask_prediction and self.casm_mode != "visual_phase_gate":
            raise ValueError("semantic subtask prediction requires visual-phase-gate CASM")
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
