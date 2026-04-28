#!/bin/bash
# Download models needed for benchmarking.
# Mistral-7B: public (no auth). Llama models: gated (requires HF login + license).
#
# Usage:
#   ./scripts/download_models.sh               # all models
#   ./scripts/download_models.sh --llama-only  # only Llama models
#   ./scripts/download_models.sh --mistral-only
#
# HF auth (required for Llama):
#   venv_modelopt/bin/huggingface-cli login
#   OR: export HF_TOKEN=hf_xxx

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

HF_CLI="venv_modelopt/bin/huggingface-cli"
if [ ! -x "$HF_CLI" ]; then
    echo "ERROR: $HF_CLI not found. Run ./scripts/setup.sh first."
    exit 1
fi

DOWNLOAD_LLAMA=true
DOWNLOAD_MISTRAL=true
for x in "$@"; do
    [ "$x" = "--llama-only"   ] && DOWNLOAD_MISTRAL=false
    [ "$x" = "--mistral-only" ] && DOWNLOAD_LLAMA=false
done

DL_ARGS="--quiet"

if [ "$DOWNLOAD_MISTRAL" = true ]; then
    echo "Downloading mistral-7b-instruct-v0.3 (public)..."
    "$HF_CLI" download mistralai/Mistral-7B-Instruct-v0.3 \
        --local-dir models/mistral-7b-instruct-v0.3 $DL_ARGS
    echo "  Done: models/mistral-7b-instruct-v0.3"
fi

if [ "$DOWNLOAD_LLAMA" = true ]; then
    echo "Downloading llama-3.1-8b-instruct (requires HF token + Meta license)..."
    "$HF_CLI" download meta-llama/Llama-3.1-8B-Instruct \
        --local-dir models/llama-3.1-8b-instruct $DL_ARGS
    echo "  Done: models/llama-3.1-8b-instruct"

    echo "Downloading llama-3.2-3b-instruct (requires HF token + Meta license)..."
    "$HF_CLI" download meta-llama/Llama-3.2-3B-Instruct \
        --local-dir models/llama-3.2-3b-instruct $DL_ARGS
    echo "  Done: models/llama-3.2-3b-instruct"
fi

echo ""
echo "All requested models downloaded to models/"
