from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openpi.models_pytorch import pi0_pytorch


class _ImageEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma = SimpleNamespace(
            language_model=SimpleNamespace(gradient_checkpointing=False),
            vision_tower=SimpleNamespace(gradient_checkpointing=False),
        )
        self.gemma_expert = SimpleNamespace(model=SimpleNamespace(gradient_checkpointing=False))

    def embed_image(self, image):
        return image.square() + 1


def _make_model(*, gradient_checkpointing_enabled: bool) -> pi0_pytorch.PI0Pytorch:
    model = pi0_pytorch.PI0Pytorch.__new__(pi0_pytorch.PI0Pytorch)
    nn.Module.__init__(model)
    model.paligemma_with_expert = _ImageEmbedder()
    model.paligemma_with_expert.config = SimpleNamespace(_attn_implementation="sdpa")
    model.gradient_checkpointing_enabled = gradient_checkpointing_enabled
    model.train()
    return model


def test_selective_vision_checkpoint_preserves_output_and_gradient():
    checkpointed_input = torch.randn(2, 3, requires_grad=True)
    eager_input = checkpointed_input.detach().clone().requires_grad_()

    checkpointed_model = _make_model(gradient_checkpointing_enabled=True)
    eager_model = _make_model(gradient_checkpointing_enabled=False)
    checkpointed_output = checkpointed_model._embed_image(checkpointed_input)  # noqa: SLF001
    eager_output = eager_model._embed_image(eager_input)  # noqa: SLF001

    checkpointed_output.sum().backward()
    eager_output.sum().backward()

    torch.testing.assert_close(checkpointed_output, eager_output)
    torch.testing.assert_close(checkpointed_input.grad, eager_input.grad)


def test_attention_implementation_is_explicit_and_validated():
    model = _make_model(gradient_checkpointing_enabled=False)
    model.set_attention_implementation("eager")
    assert model.paligemma_with_expert.config._attn_implementation == "eager"  # noqa: SLF001

    with pytest.raises(ValueError, match="Unsupported attention implementation"):
        model.set_attention_implementation("flash_attention_2")


def test_gradient_checkpointing_scope_controls_transformer_recomputation():
    model = _make_model(gradient_checkpointing_enabled=False)
    model.gradient_checkpointing_enable(scope="vision")

    assert model.gradient_checkpointing_enabled is True
    assert model.paligemma_with_expert.paligemma.language_model.gradient_checkpointing is False
    assert model.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing is False
    assert model.paligemma_with_expert.gemma_expert.model.gradient_checkpointing is False

    model.gradient_checkpointing_enable(scope="full")
    assert model.paligemma_with_expert.paligemma.language_model.gradient_checkpointing is True
    assert model.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing is True
    assert model.paligemma_with_expert.gemma_expert.model.gradient_checkpointing is True


def test_gradient_checkpointing_rejects_unknown_scope():
    model = _make_model(gradient_checkpointing_enabled=False)
    with pytest.raises(ValueError, match="Unsupported gradient checkpointing scope"):
        model.gradient_checkpointing_enable(scope="unknown")  # type: ignore[arg-type]
