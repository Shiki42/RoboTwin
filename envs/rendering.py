SUPPORTED_RAY_TRACING_DENOISERS = ("none", "oidn", "optix")


def resolve_ray_tracing_denoiser(value: str | None) -> str:
    denoiser = "oidn" if value is None else value
    if denoiser not in SUPPORTED_RAY_TRACING_DENOISERS:
        raise ValueError(
            f"unsupported ray tracing denoiser {denoiser!r}; "
            f"expected one of {SUPPORTED_RAY_TRACING_DENOISERS}"
        )
    return denoiser
