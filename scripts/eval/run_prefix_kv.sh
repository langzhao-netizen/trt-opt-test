#!/bin/bash
# Quick run: prefix KV test on action_completion with llama-3.2-3b (smallest).
# Usage: ./scripts/run_prefix_kv_action_completion.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "${PROJECT_ROOT}/venv_trtllm1.1.0/bin/activate"
CKPT="${CKPT:-${PROJECT_ROOT}/outputs/ckpts/llama-3.2-3b-instruct-trtllm-ckpt-wq_fp8-kv_fp16}"

echo "Run prefix KV bench: ckpt=$CKPT (use --num-samples N --max-seq-len N to override)"
exec "${PROJECT_ROOT}/venv_trtllm1.1.0/bin/python3" \
    "${SCRIPT_DIR}/prefix_kv_bench.py" \
    --ckpt "$CKPT" \
    "$@"
