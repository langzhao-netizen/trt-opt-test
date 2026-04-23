#!/bin/bash
# Upload a folder of subdirs (ckpts or models) to Hugging Face Hub (one repo, one folder per subdir).
# Shows progress [i/N], continues on failure. For long run: nohup ./scripts/upload_ckpts_to_hf.sh ... & or use tmux/screen.
# Requires: pip install huggingface_hub (optional: pip install tqdm for progress bar).
# Usage:
#   export REPO_ID="your-username/trt-models-test"   # repo must exist
#   export HF_TOKEN="hf_xxx"                         # or: huggingface-cli login
#   ./scripts/upload_ckpts_to_hf.sh                 # upload outputs/ckpts
#   ./scripts/upload_ckpts_to_hf.sh 
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CKPT_ROOT="${1:-${PROJECT_ROOT}/outputs/ckpts}"
# 若传的是相对路径，则相对于项目根目录
[[ "$CKPT_ROOT" != /* ]] && CKPT_ROOT="${PROJECT_ROOT}/${CKPT_ROOT}"
REPO_ID="${REPO_ID:?Set REPO_ID e.g. your-username/trtllm-ckpts}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARN: HF_TOKEN not set. Use: export HF_TOKEN=... or huggingface-cli login"
fi

source "${PROJECT_ROOT}/venv_trtllm1.1.0/bin/activate" 2>/dev/null || true
python3 -c "import huggingface_hub" 2>/dev/null || { echo "Install: pip install huggingface_hub"; exit 1; }

exec python3 "$SCRIPT_DIR/upload_to_hf.py" "$CKPT_ROOT"
