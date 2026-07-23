import pytest

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
