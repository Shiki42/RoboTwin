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
