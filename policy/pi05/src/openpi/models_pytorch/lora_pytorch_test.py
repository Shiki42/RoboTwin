import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch import lora_pytorch


def test_lora_linear_preserves_base_keys_and_applies_scaled_update():
    base = nn.Linear(4, 3, bias=False)
    layer = lora_pytorch.LoRALinear(base, rank=2, alpha=4.0)
    with torch.no_grad():
        layer.lora_a.copy_(torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10)
        layer.lora_b.copy_(torch.arange(6, dtype=torch.float32).reshape(3, 2) / 10)
    inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10

    expected = F.linear(inputs, base.weight) + 2.0 * F.linear(F.linear(inputs, layer.lora_a), layer.lora_b)

    torch.testing.assert_close(layer(inputs), expected)
    assert set(layer.state_dict()) == {"weight", "lora_a", "lora_b"}
    assert layer.weight is base.weight


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.down_proj = nn.Linear(4, 4)
        self.unrelated = nn.Linear(4, 4)


class _Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])


def test_inject_lora_replaces_only_openpi_gemma_targets():
    stack = _Stack()

    replaced = lora_pytorch.inject_lora(stack, rank=2, alpha=2.0)

    assert replaced == (
        "layers.0.q_proj",
        "layers.0.down_proj",
        "layers.1.q_proj",
        "layers.1.down_proj",
    )
    assert isinstance(stack.layers[0].q_proj, lora_pytorch.LoRALinear)
    assert isinstance(stack.layers[0].down_proj, lora_pytorch.LoRALinear)
    assert type(stack.layers[0].unrelated) is nn.Linear
    assert len(lora_pytorch.lora_parameter_names(stack)) == 8
