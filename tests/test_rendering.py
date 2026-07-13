import pytest

from envs.rendering import resolve_ray_tracing_denoiser


def test_ray_tracing_denoiser_preserves_oidn_default():
    assert resolve_ray_tracing_denoiser(None) == "oidn"


@pytest.mark.parametrize("denoiser", ["none", "oidn", "optix"])
def test_ray_tracing_denoiser_accepts_supported_backends(denoiser):
    assert resolve_ray_tracing_denoiser(denoiser) == denoiser


def test_ray_tracing_denoiser_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unsupported ray tracing denoiser"):
        resolve_ray_tracing_denoiser("cuda")
