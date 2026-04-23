#!/bin/bash
# Wrapper to run TensorRT-Model-Optimizer PTQ and produce a single quantized HF checkpoint.
# Output: outputs/ckpts/saved_models_<model>_<quant>_kv_<kv> (then use rename_ckpts_to_convention.sh).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LLM_PTQ="${PROJECT_ROOT}/tools/TensorRT-Model-Optimizer/examples/llm_ptq"
ROOT_SAVE_PATH="${ROOT_SAVE_PATH:-${PROJECT_ROOT}/outputs/ckpts}"

export ROOT_SAVE_PATH
mkdir -p "$ROOT_SAVE_PATH"

if [ ! -d "$LLM_PTQ" ]; then
    echo "Error: llm_ptq not found at $LLM_PTQ (run ./scripts/setup.sh first)" >&2
    exit 1
fi

# Use project venv for Model Optimizer if present (so clone + setup.sh = runnable)
cd "$LLM_PTQ"
if [ -f "${PROJECT_ROOT}/venv_modelopt/bin/activate" ]; then
    source "${PROJECT_ROOT}/venv_modelopt/bin/activate"
fi
# Model Optimizer scripts call `python`; use real binary (not pyenv shim) so nohup works
PY3=""
[ -x "${PROJECT_ROOT}/venv_modelopt/bin/python3" ] && PY3="${PROJECT_ROOT}/venv_modelopt/bin/python3"
[ -z "$PY3" ] && [ -x "/usr/bin/python3" ] && PY3="/usr/bin/python3"
[ -z "$PY3" ] && PY3="$(command -v python3 2>/dev/null)" || true
if [ -n "$PY3" ]; then
    mkdir -p "${SCRIPT_DIR}/.bin"
    ln -sf "$PY3" "${SCRIPT_DIR}/.bin/python" 2>/dev/null || true
    export PATH="${SCRIPT_DIR}/.bin:$PATH"
    # Allow downstream scripts to use an explicit interpreter (avoid PATH surprises under nohup/pyenv)
    export PYTHON="$PY3"
fi
exec ./scripts/huggingface_example.sh "$@"
