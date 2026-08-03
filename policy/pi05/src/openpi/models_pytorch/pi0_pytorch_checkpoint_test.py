from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipVisionModel

from openpi.models_pytorch import pi0_pytorch


class _VisionTower:
    def __init__(self):
        self.gradient_checkpointing = False
        self.gradient_checkpointing_kwargs = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        self.gradient_checkpointing = True
        self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        self.gradient_checkpointing_kwargs = None


class _ImageEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.paligemma = SimpleNamespace(
            language_model=SimpleNamespace(gradient_checkpointing=False),
            vision_tower=_VisionTower(),
        )
        self.gemma_expert = SimpleNamespace(model=SimpleNamespace(gradient_checkpointing=False))

    def embed_image(self, image):
        self.calls += 1
        return image.square() + 1


def _make_model(*, gradient_checkpointing_enabled: bool) -> pi0_pytorch.PI0Pytorch:
    model = pi0_pytorch.PI0Pytorch.__new__(pi0_pytorch.PI0Pytorch)
    nn.Module.__init__(model)
    model.paligemma_with_expert = _ImageEmbedder()
    model.paligemma_with_expert.config = SimpleNamespace(_attn_implementation="sdpa")
    model.gradient_checkpointing_enabled = gradient_checkpointing_enabled
    model.train()
    return model


def test_full_checkpointing_does_not_nest_whole_vision_recomputation():
    image = torch.randn(2, 3, requires_grad=True)
    model = _make_model(gradient_checkpointing_enabled=True)
    output = model._embed_image(image)  # noqa: SLF001
    output.sum().backward()

    assert model.paligemma_with_expert.calls == 1
    assert image.grad is not None


def test_sinusoidal_pos_embedding_matches_float32_jax_reference():
    timestep = torch.tensor([0.12345679, 0.9876543], dtype=torch.float32)
    actual = pi0_pytorch.create_sinusoidal_pos_embedding(
        timestep,
        dimension=8,
        min_period=4e-3,
        max_period=4.0,
        device=timestep.device,
    )
    expected = torch.tensor(
        [
            [-0.75344354, 0.51669806, 0.93288350, 0.19271228, 0.65751261, 0.85616767, -0.36017820, 0.98125529],
            [-0.51677901, -0.93287927, 0.19270824, 0.99981195, 0.85611880, -0.36018926, -0.98125613, 0.01939123],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


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
    assert model.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing is True
    assert model.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing_kwargs == {
        "use_reentrant": False,
        "preserve_rng_state": False,
    }
    assert model.paligemma_with_expert.gemma_expert.model.gradient_checkpointing is False

    model.gradient_checkpointing_enable(scope="full")
    assert model.paligemma_with_expert.paligemma.language_model.gradient_checkpointing is True
    assert model.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing is True
    assert model.paligemma_with_expert.gemma_expert.model.gradient_checkpointing is True

    model.gradient_checkpointing_disable()
    assert model.gradient_checkpointing_enabled is False
    assert model.paligemma_with_expert.paligemma.language_model.gradient_checkpointing is False
    assert model.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing is False
    assert model.paligemma_with_expert.gemma_expert.model.gradient_checkpointing is False


def test_vision_checkpointing_reaches_each_siglip_encoder_layer():
    vision_tower = SiglipVisionModel(
        SiglipVisionConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=4,
            patch_size=2,
        )
    )
    model = _make_model(gradient_checkpointing_enabled=False)
    model.paligemma_with_expert.paligemma.vision_tower = vision_tower

    model.gradient_checkpointing_enable(scope="vision")

    assert all(layer.gradient_checkpointing for layer in vision_tower.vision_model.encoder.layers)


def test_gradient_checkpointing_rejects_unknown_scope():
    model = _make_model(gradient_checkpointing_enabled=False)
    with pytest.raises(ValueError, match="Unsupported gradient checkpointing scope"):
        model.gradient_checkpointing_enable(scope="unknown")  # type: ignore[arg-type]
