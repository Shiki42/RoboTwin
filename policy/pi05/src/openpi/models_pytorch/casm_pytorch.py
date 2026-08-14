from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812


def reduce_action_loss(squared_error: Tensor, action_mask: Tensor | None) -> Tensor:
    """Reduce action-dimension error while excluding unsupervised dimensions."""
    if action_mask is None:
        return squared_error.mean(dim=-1)
    mask = action_mask.to(device=squared_error.device, dtype=squared_error.dtype)
    if mask.shape != squared_error.shape:
        raise ValueError(f"action mask shape mismatch: {mask.shape} != {squared_error.shape}")
    denominator = mask.sum(dim=-1).clamp_min(1.0)
    return (squared_error * mask).sum(dim=-1) / denominator


def async_target(phase_id: Tensor) -> Tensor:
    if phase_id.ndim < 2 or phase_id.shape[-1] < 1:
        raise ValueError(f"phase id must end in a non-empty phase dimension, got {phase_id.shape}")
    return (phase_id[..., 0] != 0).to(dtype=torch.float32)


def binary_cross_entropy_with_logits(logits: Tensor, target: Tensor, positive_weight: float) -> Tensor:
    if positive_weight <= 0:
        raise ValueError("positive weight must be positive")
    positive = torch.as_tensor(positive_weight, dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype), pos_weight=positive, reduction="none")


class VisualProprioceptionGate(nn.Module):
    """Predict async/sync from pooled visual tokens and continuous robot state."""

    def __init__(self, visual_dim: int, state_dim: int, hidden_dim: int):
        super().__init__()
        if min(visual_dim, state_dim, hidden_dim) < 1:
            raise ValueError("gate dimensions must be positive")
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.state_norm = nn.LayerNorm(state_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, visual_features: Tensor, state: Tensor) -> Tensor:
        visual_input = visual_features.detach().to(self.visual_norm.weight)
        state_input = state.detach().to(self.state_norm.weight)
        visual = F.gelu(self.visual_proj(self.visual_norm(visual_input)))
        proprioception = F.gelu(self.state_proj(self.state_norm(state_input)))
        fused = torch.cat([visual, proprioception], dim=-1)
        return self.output(F.gelu(self.fusion(fused)))[..., 0]


class VisualProprioceptionClassifier(nn.Module):
    """Predict a closed-set subtask without changing the action pathway."""

    def __init__(
        self,
        visual_dim: int,
        state_dim: int,
        hidden_dim: int,
        classes: int,
        *,
        stop_gradient: bool,
    ):
        super().__init__()
        if min(visual_dim, state_dim, hidden_dim, classes) < 1:
            raise ValueError("classifier dimensions must be positive")
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.state_norm = nn.LayerNorm(state_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, classes)
        self.stop_gradient = stop_gradient

    def forward(self, visual_features: Tensor, state: Tensor) -> Tensor:
        if self.stop_gradient:
            visual_features = visual_features.detach()
            state = state.detach()
        visual = F.gelu(self.visual_proj(self.visual_norm(visual_features.to(self.visual_norm.weight))))
        proprioception = F.gelu(self.state_proj(self.state_norm(state.to(self.state_norm.weight))))
        return self.output(F.gelu(self.fusion(torch.cat([visual, proprioception], dim=-1))))


@dataclass(frozen=True)
class SubtaskClassificationLoss:
    per_sample: Tensor
    metrics: dict[str, Tensor]


def subtask_classification_loss(logits: Tensor, target: Tensor) -> SubtaskClassificationLoss:
    target = target.to(device=logits.device, dtype=torch.long).reshape(-1)
    if logits.ndim != 2 or logits.shape[0] != target.shape[0]:
        raise ValueError(f"subtask logits/target shape mismatch: {logits.shape} != {target.shape}")
    if target.numel() and (target.min() < 0 or target.max() >= logits.shape[1]):
        raise ValueError(f"subtask target is outside [0, {logits.shape[1]})")
    loss = F.cross_entropy(logits.float(), target, reduction="none")
    return SubtaskClassificationLoss(
        per_sample=loss,
        metrics={"subtask_loss": loss.mean(), "subtask_accuracy": logits.argmax(-1).eq(target).float().mean()},
    )


@dataclass(frozen=True)
class VisualPhaseGateLoss:
    total: Tensor
    metrics: dict[str, Tensor]


def visual_phase_gate_loss(
    action_loss: Tensor,
    gate_logits: Tensor,
    phase_id: Tensor,
    *,
    gate_loss_weight: float,
    gate_positive_weight: float,
) -> VisualPhaseGateLoss:
    target = async_target(phase_id).to(device=gate_logits.device)
    gate_loss = binary_cross_entropy_with_logits(gate_logits, target, gate_positive_weight)
    probability = torch.sigmoid(gate_logits)
    metrics = {
        "action_loss": action_loss.mean(),
        "gate_loss": gate_loss.mean(),
        "gate_accuracy": ((probability >= 0.5) == (target >= 0.5)).to(torch.float32).mean(),
        "gate_async_probability": probability.mean(),
        "gate_predicted_async_rate": (probability >= 0.5).to(torch.float32).mean(),
        "gate_target_async_rate": target.mean(),
    }
    return VisualPhaseGateLoss(
        total=action_loss + gate_loss_weight * gate_loss[..., None],
        metrics=metrics,
    )
