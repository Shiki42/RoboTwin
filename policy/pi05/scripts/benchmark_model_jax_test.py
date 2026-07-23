import dataclasses

import torch

from scripts import benchmark_model_jax


class _FakeRandom:
    def __init__(self, calls):
        self._calls = calls

    def key(self, seed):
        self._calls.append(("key", seed))
        return "startup-key"


class _FakeJax:
    def __init__(self):
        self.calls = []
        self.random = _FakeRandom(self.calls)

    def block_until_ready(self, value):
        self.calls.append(("ready", value))


def test_initialize_jax_runtime_blocks_on_a_startup_operation():
    fake_jax = _FakeJax()

    benchmark_model_jax._initialize_jax_runtime(fake_jax)  # noqa: SLF001

    assert fake_jax.calls == [("key", 0), ("ready", "startup-key")]


@dataclasses.dataclass(frozen=True)
class _Observation:
    images: dict[str, torch.Tensor]


def test_to_jax_model_layout_moves_only_the_image_channel_axis():
    image = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    observation = _Observation(images={"camera": image})
    actions = torch.tensor([1.0])

    converted_observation, converted_actions = benchmark_model_jax._to_jax_model_layout(  # noqa: SLF001
        (observation, actions)
    )

    assert converted_actions is actions
    assert converted_observation.images["camera"].shape == (2, 4, 5, 3)
    torch.testing.assert_close(converted_observation.images["camera"], image.permute(0, 2, 3, 1).contiguous())
