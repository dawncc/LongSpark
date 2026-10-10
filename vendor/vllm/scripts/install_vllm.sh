#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3.12}"
env_dir="${VLLM_ENV_DIR:-$bundle_dir/.venv}"
cuda_variant="${VLLM_CUDA_VARIANT:-default}"
case "$cuda_variant" in
  default|cu129) ;;
  *)
    echo "VLLM_CUDA_VARIANT must be default or cu129; got: $cuda_variant" >&2
    exit 1
    ;;
esac
"$python_bin" -c 'import platform, sys; assert platform.system() == "Linux", "vLLM CUDA integration requires Linux"; assert (3, 10) <= sys.version_info[:2] < (3, 15), "Python 3.10-3.14 required"'
if [[ "$cuda_variant" == cu129 ]]; then
  "$python_bin" -c 'import platform; assert platform.machine() == "x86_64" and platform.python_implementation() == "CPython", "The pinned cu129 wheel requires Linux x86_64 CPython"'
  # Official 0.30.0 release asset; retain its checksum and select matching torch.
  vllm_wheel="https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl#sha256=e98cb69659bfcfc849cf11ce0781a7161d40b02b51a6c3636924a5909f2aabcc"
fi
if [[ -e "$env_dir" ]]; then
  echo "Environment already exists: $env_dir. Choose a new VLLM_ENV_DIR to preserve it." >&2
  exit 1
fi
"$python_bin" -m venv "$env_dir"
"$env_dir/bin/python" -m pip install --upgrade pip
if [[ "$cuda_variant" == cu129 ]]; then
  "$env_dir/bin/python" -m pip install uv
  "$env_dir/bin/python" -m uv pip install --python "$env_dir/bin/python" \
    --torch-backend=cu129 -r "$bundle_dir/requirements.txt" "$vllm_wheel"
else
  "$env_dir/bin/python" -m pip install -r "$bundle_dir/requirements.txt"
fi
"$env_dir/bin/python" -m pip check
"$env_dir/bin/python" - "$cuda_variant" <<'PY'
from importlib.metadata import version
import sys
import torch
import vllm

installed_vllm = version("vllm")
print("vLLM", installed_vllm, "PyTorch", torch.__version__, "CUDA", torch.version.cuda)
if sys.argv[1] == "cu129":
    if installed_vllm != "0.30.0+cu129" or torch.version.cuda != "12.9":
        raise RuntimeError("cu129 requires vLLM 0.30.0+cu129 and a CUDA 12.9 PyTorch build")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; inspect driver, GPU visibility and LD_LIBRARY_PATH")
PY
echo "Installed. Run: $env_dir/bin/python $bundle_dir/scripts/evaluate_native_v2_gsm8k.py --help"
