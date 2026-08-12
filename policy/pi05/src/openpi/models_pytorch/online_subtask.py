from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F  # noqa: N812


class OnlineSubtaskLoss(NamedTuple):
    total: Tensor
    action: Tensor
    subtask_ce: Tensor
    action_elementwise: Tensor


def _validate_ce_shapes(
    logits: Tensor,
    tokens: Tensor,
    loss_mask: Tensor,
) -> None:
    if logits.ndim != 3 or tokens.ndim != 2 or loss_mask.ndim != 2:
        raise ValueError("subtask CE tensors have invalid ranks")
    if tokens.shape != loss_mask.shape:
        raise ValueError("subtask tokens and loss mask shapes differ")
    if logits.shape[:2] != (tokens.shape[0], tokens.shape[1] - 1):
        raise ValueError("subtask logits and tokens are not next-token aligned")


def next_token_cross_entropy(
    logits: Tensor,
    tokens: Tensor,
    loss_mask: Tensor,
) -> Tensor:
    _validate_ce_shapes(logits, tokens, loss_mask)
    target_mask = loss_mask[:, 1:].to(dtype=torch.bool)
    target_count = target_mask.sum()
    if torch.compiler.is_compiling():
        torch._assert_async(target_count > 0, "subtask target mask is empty")  # noqa: SLF001
    elif target_count.item() == 0:
        raise ValueError("subtask target mask is empty")
    token_loss = F.cross_entropy(
        logits.to(torch.float32).transpose(1, 2),
        tokens[:, 1:],
        reduction="none",
    )
    return (token_loss * target_mask).sum() / target_count
