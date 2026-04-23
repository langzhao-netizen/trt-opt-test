#!/bin/bash
# Upload /home/langzhao/workspace/trt-opt-test/models to Hugging Face Hub.
# Progress [i/N], continues on failure. For long run: nohup ./scripts/upload_models_to_hf.sh > upload.log 2>&1 &
# Usage:
#   export REPO_ID="rungalileo/trt-models-test"
#   export HF_TOKEN="hf_xxx"
#   ./scripts/upload_models_to_hf.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_ROOT="${MODELS_ROOT:-${PROJECT_ROOT}/models}"

if [ ! -d "$MODELS_ROOT" ]; then
  echo "Models dir not found: $MODELS_ROOT (set MODEL_ROOT to override)"
  exit 1
fi

REPO_ID="${REPO_ID:?Set REPO_ID e.g. rungalileo/trt-models-test}"
[ -z "${HF_TOKEN:-}" ] && echo "WARN: HF_TOKEN not set"

source "${PROJECT_ROOT}/venv_trtllm1.1.0/bin/activate" 2>/dev/null || true
exec python3 "$SCRIPT_DIR/upload_to_hf.py" "$MODELS_ROOT"
