import os
from pathlib import Path
import subprocess

SCRIPT = Path(__file__).with_name("bootstrap_blackwell_cuda128.sh")
REQUIREMENTS = (
    ("nvidia_cublas_cu12", "12.8.4.1"),
    ("nvidia_cuda_cupti_cu12", "12.8.90"),
    ("nvidia_cuda_nvcc_cu12", "12.8.93"),
    ("nvidia_cuda_nvrtc_cu12", "12.8.93"),
    ("nvidia_cuda_runtime_cu12", "12.8.90"),
    ("nvidia_cudnn_cu12", "9.8.0.87"),
    ("nvidia_cufft_cu12", "11.3.3.83"),
    ("nvidia_cusolver_cu12", "11.7.3.90"),
    ("nvidia_cusparse_cu12", "12.5.8.93"),
    ("nvidia_nccl_cu12", "2.26.5"),
    ("nvidia_nvjitlink_cu12", "12.8.93"),
)


def _wheelhouse(path: Path) -> None:
    for package, version in REQUIREMENTS:
        (path / f"{package}-{version}-py3-none-manylinux_x86_64.whl").touch()


def _run_check(wheelhouse: Path, overlay: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "BLACKWELL_WHEELHOUSE_CHECK_ONLY": "1"}
    return subprocess.run(
        [SCRIPT, wheelhouse, overlay],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_wheelhouse_check_accepts_exact_pinned_set(tmp_path):
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    _wheelhouse(wheelhouse)
    result = _run_check(wheelhouse, tmp_path / "overlay")
    assert result.returncode == 0, result.stderr
    assert "nvidia-cublas-cu12==12.8.4.1" in result.stdout


def test_wheelhouse_check_rejects_missing_wheel(tmp_path):
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    _wheelhouse(wheelhouse)
    next(wheelhouse.glob("nvidia_cudnn_cu12-*.whl")).unlink()
    result = _run_check(wheelhouse, tmp_path / "overlay")
    assert result.returncode != 0
    assert "nvidia-cudnn-cu12==9.8.0.87" in result.stderr
