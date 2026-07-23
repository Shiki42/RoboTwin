#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 OUTPUT_DIR [PYTHON]" >&2
  exit 2
fi

output_dir=$1
python_bin=${2:-python}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
requirements="$script_dir/jax_blackwell_cp311_requirements.txt"

[[ ! -e "$output_dir" ]] || {
  echo "output already exists: $output_dir" >&2
  exit 2
}
command -v "$python_bin" >/dev/null
[[ -f "$requirements" ]]

mkdir -p "$output_dir/wheels" "$output_dir/receipt"
cp "$requirements" "$output_dir/receipt/requirements.lock.txt"

"$python_bin" -m pip download \
  --only-binary=:all: \
  --dest "$output_dir/wheels" \
  --requirement "$requirements" \
  2>&1 | tee "$output_dir/receipt/pip_download.log"

"$python_bin" -m pip install \
  --dry-run \
  --ignore-installed \
  --no-index \
  --find-links "$output_dir/wheels" \
  --requirement "$requirements" \
  --report "$output_dir/receipt/pip_install_dry_run.json" \
  2>&1 | tee "$output_dir/receipt/pip_install_dry_run.log"

(
  cd "$output_dir/wheels"
  find . -maxdepth 1 -type f -name "*.whl" -print0 | sort -z | xargs -0 sha256sum
) >"$output_dir/receipt/SHA256SUMS"

find "$output_dir/wheels" -maxdepth 1 -type f -name "*.whl" -printf "%f\t%s\n" \
  | sort >"$output_dir/receipt/WHEEL_SIZES.tsv"

"$python_bin" - "$output_dir" <<'PY'
import json
import pathlib
import platform
import sys

root = pathlib.Path(sys.argv[1])
wheels = sorted((root / "wheels").glob("*.whl"))
receipt = {
    "format": 1,
    "python": sys.version,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "wheel_count": len(wheels),
    "wheel_bytes": sum(path.stat().st_size for path in wheels),
    "requirements": "receipt/requirements.lock.txt",
    "sha256_manifest": "receipt/SHA256SUMS",
}
(root / "receipt" / "SOURCE_RECEIPT.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(receipt, sort_keys=True))
PY
