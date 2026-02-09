#!/bin/bash
# Wrapper to run TensorRT-Model-Optimizer PTQ and produce a single quantized HF checkpoint.
# Output: outputs/ckpts/saved_models_<model>_<quant>_kv_<kv> (then use rename_ckpts_to_convention.sh).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LLM_PTQ="${PROJECT_ROOT}/tools/TensorRT-Model-Optimizer/examples/llm_ptq"
ROOT_SAVE_PATH="${ROOT_SAVE_PATH:-${PROJECT_ROOT}/outputs/ckpts}"

export ROOT_SAVE_PATH
mkdir -p "$ROOT_SAVE_PATH"

if [ ! -d "$LLM_PTQ" ]; then
    echo "Error: llm_ptq not found at $LLM_PTQ" >&2
    exit 1
fi

cd "$LLM_PTQ"
exec ./scripts/huggingface_example.sh "$@"
