"""LoRA layers matching the OpenPI Gemma adapter recipe."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

_TARGET_LINEAR_NAMES = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
)


class LoRALinear(nn.Linear):
    """Preserve a base Linear state-dict contract while adding a low-rank update."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, init_std: float = 0.01):
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if init_std <= 0:
            raise ValueError("LoRA init_std must be positive")
        nn.Module.__init__(self)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = base.weight
        self.bias = base.bias
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = float(alpha / rank)
        factory_kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, **factory_kwargs))
        self.lora_b = nn.Parameter(torch.empty(base.out_features, rank, **factory_kwargs))
        nn.init.normal_(self.lora_a, std=init_std)
        nn.init.normal_(self.lora_b, std=init_std)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = F.linear(inputs, self.weight, self.bias)
        adapter = F.linear(F.linear(inputs, self.lora_a), self.lora_b)
        return base + adapter * self.scaling


def inject_lora(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    target_names: Iterable[str] = _TARGET_LINEAR_NAMES,
) -> tuple[str, ...]:
    """Replace selected descendant Linear modules and return their relative paths."""
    targets = frozenset(target_names)
    replaced: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for name, child in tuple(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and name in targets:
                setattr(parent, name, LoRALinear(child, rank=rank, alpha=alpha))
                replaced.append(path)
            else:
                visit(child, path)

    visit(module, "")
    if not replaced:
        raise ValueError("LoRA injection selected no Linear modules")
    return tuple(replaced)


def lora_parameter_names(module: nn.Module) -> tuple[str, ...]:
    return tuple(name for name, _ in module.named_parameters() if name.endswith((".lora_a", ".lora_b")))
