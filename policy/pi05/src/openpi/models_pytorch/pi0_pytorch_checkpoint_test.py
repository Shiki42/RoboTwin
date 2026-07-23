import torch
from torch import nn

from openpi.models_pytorch import pi0_pytorch


class _ImageEmbedder(nn.Module):
    def embed_image(self, image):
        return image.square() + 1


def _make_model(*, gradient_checkpointing_enabled: bool) -> pi0_pytorch.PI0Pytorch:
    model = pi0_pytorch.PI0Pytorch.__new__(pi0_pytorch.PI0Pytorch)
    nn.Module.__init__(model)
    model.paligemma_with_expert = _ImageEmbedder()
    model.gradient_checkpointing_enabled = gradient_checkpointing_enabled
    model.train()
    return model


def test_selective_vision_checkpoint_preserves_output_and_gradient():
    checkpointed_input = torch.randn(2, 3, requires_grad=True)
    eager_input = checkpointed_input.detach().clone().requires_grad_(True)

    checkpointed_model = _make_model(gradient_checkpointing_enabled=True)
    eager_model = _make_model(gradient_checkpointing_enabled=False)
    checkpointed_output = checkpointed_model._embed_image(checkpointed_input)  # noqa: SLF001
    eager_output = eager_model._embed_image(eager_input)  # noqa: SLF001

    checkpointed_output.sum().backward()
    eager_output.sum().backward()

    torch.testing.assert_close(checkpointed_output, eager_output)
    torch.testing.assert_close(checkpointed_input.grad, eager_input.grad)
