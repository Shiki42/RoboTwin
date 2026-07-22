#!/usr/bin/env bash
set -Eeuo pipefail

WHEELHOUSE="${1:?usage: bootstrap_blackwell_cuda128.sh WHEELHOUSE OVERLAY_DIR}"
OVERLAY_DIR="${2:?usage: bootstrap_blackwell_cuda128.sh WHEELHOUSE OVERLAY_DIR}"
BASE_PYTHON="${BASE_PYTHON:-python}"

requirements=(
    nvidia-cublas-cu12==12.8.4.1
    nvidia-cuda-cupti-cu12==12.8.90
    nvidia-cuda-nvcc-cu12==12.8.93
    nvidia-cuda-nvrtc-cu12==12.8.93
    nvidia-cuda-runtime-cu12==12.8.90
    nvidia-cudnn-cu12==9.8.0.87
    nvidia-cufft-cu12==11.3.3.83
    nvidia-cusolver-cu12==11.7.3.90
    nvidia-cusparse-cu12==12.5.8.93
    nvidia-nccl-cu12==2.26.5
    nvidia-nvjitlink-cu12==12.8.93
)

require_wheel() {
    local requirement="$1"
    local package="${requirement%%==*}"
    local version="${requirement##*==}"
    local normalized="${package//-/_}"
    local matches=("${WHEELHOUSE}/${normalized}-${version}-"*.whl)
    if [[ "${#matches[@]}" -ne 1 || ! -f "${matches[0]}" ]]; then
        echo "expected one wheel for ${requirement}, found ${#matches[@]}" >&2
        exit 1
    fi
}

if [[ ! -d "${WHEELHOUSE}" ]]; then
    echo "wheelhouse does not exist: ${WHEELHOUSE}" >&2
    exit 1
fi
for requirement in "${requirements[@]}"; do
    require_wheel "${requirement}"
done

if [[ "${BLACKWELL_WHEELHOUSE_CHECK_ONLY:-0}" == 1 ]]; then
    printf '%s\n' "${requirements[@]}"
    exit 0
fi

if [[ -e "${OVERLAY_DIR}" ]]; then
    echo "overlay path already exists without a ready receipt: ${OVERLAY_DIR}" >&2
    exit 1
fi

"${BASE_PYTHON}" -m venv --system-site-packages "${OVERLAY_DIR}"
"${OVERLAY_DIR}/bin/python" -m pip install +    --no-index +    --find-links "${WHEELHOUSE}" +    --upgrade +    "${requirements[@]}"

"${OVERLAY_DIR}/bin/python" - "${OVERLAY_DIR}/cuda128-ready.json" <<'PY'
import importlib.metadata
import json
from pathlib import Path
import sys

expected = {
    "jax": "0.5.0",
    "jaxlib": "0.5.0",
    "nvidia-cublas-cu12": "12.8.4.1",
    "nvidia-cuda-runtime-cu12": "12.8.90",
    "nvidia-cudnn-cu12": "9.8.0.87",
}
actual = {name: importlib.metadata.version(name) for name in expected}
if actual != expected:
    raise RuntimeError(f"CUDA overlay version mismatch: {actual} != {expected}")
Path(sys.argv[1]).write_text(json.dumps({"status": "ready", "versions": actual}, indent=2) + "\n")
PY
