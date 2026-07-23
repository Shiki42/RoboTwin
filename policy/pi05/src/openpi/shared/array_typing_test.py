from flax import struct
import jax
import jax.numpy as jnp
from jaxtyping import TypeCheckError
import pytest

from openpi.shared import array_typing as at


@at.typecheck
@struct.dataclass
class _ScalarState:
    step: at.Int[at.ArrayLike, ""]


@jax.jit
def _increment(state: _ScalarState) -> _ScalarState:
    return _ScalarState(step=state.step + 1)


def test_lower_accepts_jax_arg_info_dataclass_leaves():
    state = _ScalarState(step=jnp.array(0))

    compiled_increment = _increment.lower(state).compile()
    result = compiled_increment(state)

    assert int(result.step) == 1


def test_user_dataclass_initialization_remains_typechecked():
    with pytest.raises(TypeCheckError, match="step"):
        _ScalarState(step="not-an-array")
