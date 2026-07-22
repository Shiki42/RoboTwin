#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG_NAME="pi05_putcab_casm_visual_phase_gate_lora"
DATASET_REPO="${DATASET_REPO:-Shiki42/robotwin_put_obj_cabinet_50_dynFcam_nFov}"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/casm-putcab}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${WORK_ROOT}/lerobot}"
DATASET_DIR="${DATASET_DIR:-${HF_LEROBOT_HOME}/${DATASET_REPO}}"
BASE_PARAMS="${BASE_PARAMS:-${WORK_ROOT}/base/pi05_base/params}"
DATASET_MANIFEST="${DATASET_MANIFEST:-${WORK_ROOT}/receipts/dataset.sha256}"
BASE_MANIFEST="${BASE_MANIFEST:-${WORK_ROOT}/receipts/pi05_base.sha256}"
AUDIT_RECEIPT="${AUDIT_RECEIPT:-${WORK_ROOT}/receipts/dataset_audit.json}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${WORK_ROOT}/checkpoints}"
ASSETS_ROOT="${ASSETS_ROOT:-${WORK_ROOT}/assets}"
LOG_ROOT="${LOG_ROOT:-${WORK_ROOT}/logs}"
PI05_RUNTIME="${PI05_RUNTIME:-/opt/pi05-training/activate.sh}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-87431}"
FORMAL_STEPS="${FORMAL_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
PROJECT_NAME="${PROJECT_NAME:-parallelvla-casm}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "required file missing: $1" >&2
        exit 1
    fi
}

require_dir() {
    if [[ ! -d "$1" ]]; then
        echo "required directory missing: $1" >&2
        exit 1
    fi
}

verify_manifest() {
    local root="$1"
    local manifest="$2"
    require_file "${manifest}"
    (cd "${root}" && sha256sum --strict --check "${manifest}")
}

run_train() {
    local exp_name="$1"
    local batch_size="$2"
    local steps="$3"
    local save_interval="$4"
    local wandb_flag="$5"
    local mode_flag="$6"
    local args=(
        "${PYTHON_BIN}"
        "${SOURCE_ROOT}/scripts/train.py"
        "${CONFIG_NAME}"
        --exp-name "${exp_name}"
        --project-name "${PROJECT_NAME}"
        --assets-base-dir "${ASSETS_ROOT}"
        --checkpoint-base-dir "${CHECKPOINT_ROOT}"
        --seed "${SEED}"
        --batch-size "${batch_size}"
        --num-workers 4
        --num-train-steps "${steps}"
        --log-interval 10
        --save-interval "${save_interval}"
        --keep-period "${steps}"
        --model.gate-positive-weight "${GATE_POSITIVE_WEIGHT}"
        "${wandb_flag}"
        "${mode_flag}"
    )
    "${args[@]}"
}

require_file "${PI05_RUNTIME}"
source "${PI05_RUNTIME}"
export PYTHONPATH="${SOURCE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_LEROBOT_HOME
export PARALLELVLA_DATASET_REPO="${DATASET_REPO}"
export PI05_BASE_CHECKPOINT="${BASE_PARAMS}"
export JAX_COMPILATION_CACHE_DIR="${WORK_ROOT}/jax-cache"
export WANDB_DIR="${WORK_ROOT}/wandb"
export WANDB_CACHE_DIR="${WORK_ROOT}/wandb-cache"
export WANDB_DATA_DIR="${WORK_ROOT}/wandb-data"
export WANDB_LOG_MODEL=false
export WANDB_DISABLE_CODE=true
export WANDB_SILENT=true
export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

require_dir /root/autodl-tmp
require_dir "${DATASET_DIR}"
require_dir "${BASE_PARAMS}"
runtime_dirs=(
    "${WORK_ROOT}/receipts"
    "${CHECKPOINT_ROOT}"
    "${ASSETS_ROOT}"
    "${LOG_ROOT}"
    "${JAX_COMPILATION_CACHE_DIR}"
    "${WANDB_DIR}"
    "${WANDB_CACHE_DIR}"
    "${WANDB_DATA_DIR}"
)
mkdir -p "${runtime_dirs[@]}"

verify_manifest "${DATASET_DIR}" "${DATASET_MANIFEST}"
verify_manifest "${BASE_PARAMS}" "${BASE_MANIFEST}"

"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/audit_putcab_casm_dataset.py" "${DATASET_DIR}" --expected-repo "${DATASET_REPO}" --expected-episodes 50 --receipt "${AUDIT_RECEIPT}"
GATE_POSITIVE_WEIGHT="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["gate_positive_weight"])' "${AUDIT_RECEIPT}")"
export WANDB_MODE=online

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${PYTHON_BIN}" - <<'PY'
import jax
import wandb

if len(jax.devices()) != 1:
    raise RuntimeError(f"expected exactly one JAX GPU device, got {jax.devices()}")
viewer = wandb.Api(timeout=20).viewer
if viewer is None:
    raise RuntimeError("W&B authentication is not available")
print(f"W&B user: {viewer.username}")
PY

NORM_STATS="${ASSETS_ROOT}/${CONFIG_NAME}/${DATASET_REPO}/norm_stats.json"
if [[ ! -f "${NORM_STATS}" ]]; then
    (
        cd "${WORK_ROOT}"
        "${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/compute_norm_stats.py" --config-name "${CONFIG_NAME}" --repo-id "${DATASET_REPO}"
    )
fi
require_file "${NORM_STATS}"

SMOKE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
SELECTED_BATCH=""
for candidate in 16 8; do
    exp_name="smoke-b${candidate}-${SMOKE_TAG}"
    log_path="${LOG_ROOT}/${exp_name}.log"
    set +e
    run_train "${exp_name}" "${candidate}" 20 20 --no-wandb-enabled --overwrite 2>&1 | tee "${log_path}"
    status=${PIPESTATUS[0]}
    set -e
    if [[ "${status}" -eq 0 ]]; then
        SELECTED_BATCH="${candidate}"
        SMOKE_EXP="${exp_name}"
        break
    fi
    if ! grep -Eiq "CUDA_ERROR_OUT_OF_MEMORY|RESOURCE_EXHAUSTED|out of memory" "${log_path}"; then
        echo "batch ${candidate} failed for a non-OOM reason; refusing fallback" >&2
        exit "${status}"
    fi
done

if [[ -z "${SELECTED_BATCH}" ]]; then
    echo "both batch sizes 16 and 8 exhausted GPU memory" >&2
    exit 1
fi

require_dir "${CHECKPOINT_ROOT}/${CONFIG_NAME}/${SMOKE_EXP}/20"
run_train "${SMOKE_EXP}" "${SELECTED_BATCH}" 21 21 --no-wandb-enabled --resume 2>&1 | tee "${LOG_ROOT}/${SMOKE_EXP}-reload.log"
require_dir "${CHECKPOINT_ROOT}/${CONFIG_NAME}/${SMOKE_EXP}/21"

FORMAL_EXP="putcab-visual-gate-b${SELECTED_BATCH}-s${SEED}-${SMOKE_TAG}"
run_train "${FORMAL_EXP}" "${SELECTED_BATCH}" "${FORMAL_STEPS}" "${SAVE_INTERVAL}" --wandb-enabled --overwrite 2>&1 | tee "${LOG_ROOT}/${FORMAL_EXP}.log"

require_dir "${CHECKPOINT_ROOT}/${CONFIG_NAME}/${FORMAL_EXP}/${FORMAL_STEPS}"
"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

receipt = {
    "status": "training_complete",
    "config": "${CONFIG_NAME}",
    "dataset_repo": "${DATASET_REPO}",
    "base_params": "${BASE_PARAMS}",
    "batch_size": int("${SELECTED_BATCH}"),
    "seed": int("${SEED}"),
    "optimizer_steps": int("${FORMAL_STEPS}"),
    "experiment": "${FORMAL_EXP}",
    "checkpoint": "${CHECKPOINT_ROOT}/${CONFIG_NAME}/${FORMAL_EXP}/${FORMAL_STEPS}",
    "wandb_project": "${PROJECT_NAME}",
}
path = Path("${WORK_ROOT}/receipts/training_complete.json")
path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
