from types import SimpleNamespace

import pytest
import torch

from openpi.models_pytorch import gemma_pytorch


@pytest.mark.parametrize(
    ("expert_enabled", "wrapper_enabled", "training", "expected"),
    [
        (False, False, True, False),
        (True, False, True, True),
        (False, True, True, True),
        (True, True, False, False),
    ],
)
def test_gradient_checkpointing_follows_explicit_flags(
    expert_enabled,
    wrapper_enabled,
    training,
    expected,
):
    assert (
        gemma_pytorch._gradient_checkpointing_enabled(  # noqa: SLF001
            expert_enabled=expert_enabled,
            wrapper_enabled=wrapper_enabled,
            training=training,
        )
        is expected
    )


def _run_attention(implementation):
    torch.manual_seed(0)
    query = torch.randn(2, 4, 6, 8, requires_grad=True)
    key = torch.randn(2, 2, 6, 8, requires_grad=True)
    value = torch.randn(2, 2, 6, 8, requires_grad=True)
    attention_mask = torch.full((6, 6), float("-inf")).triu(diagonal=1)[None, None]
    module = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation=implementation),
        num_key_value_groups=2,
        training=True,
    )
    output = gemma_pytorch._attention_forward(  # noqa: SLF001
        module,
        query,
        key,
        value,
        attention_mask,
        scaling=8**-0.5,
    )
    gradients = torch.autograd.grad(output.square().mean(), (query, key, value))
    return output, gradients


def test_sdpa_attention_matches_eager_forward_and_backward():
    eager_output, eager_gradients = _run_attention("eager")
    sdpa_output, sdpa_gradients = _run_attention("sdpa")

    torch.testing.assert_close(sdpa_output, eager_output, atol=1e-6, rtol=2e-5)
    for sdpa_gradient, eager_gradient in zip(sdpa_gradients, eager_gradients, strict=True):
        torch.testing.assert_close(sdpa_gradient, eager_gradient, atol=1e-7, rtol=2e-5)


def test_attention_rejects_unvalidated_implementation():
    with pytest.raises(ValueError, match="Unsupported Gemma attention implementation"):
        _run_attention("flash_attention_2")
