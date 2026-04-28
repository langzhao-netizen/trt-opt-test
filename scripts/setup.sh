#!/bin/bash
# One-shot setup: clone deps, create venvs, create output dirs.
# Requires Python 3.12 (via pyenv) for TRT-LLM; Python 3.11 for modelopt.
# Usage: ./scripts/setup.sh [--no-trtllm-1.1.0] [--no-trtllm-1.2.0]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

SKIP_TRTLLM_110=false
SKIP_TRTLLM_120=false
for x in "$@"; do
    [ "$x" = "--no-trtllm-1.1.0" ] && SKIP_TRTLLM_110=true
    [ "$x" = "--no-trtllm-1.2.0" ] && SKIP_TRTLLM_120=true
done

# --- Resolve Python 3.12 (required for TRT-LLM wheels) ---
PYTHON312=""
if command -v python3.12 &>/dev/null; then
    PYTHON312=$(command -v python3.12)
elif [ -x "$HOME/.pyenv/versions/3.12.9/bin/python3.12" ]; then
    PYTHON312="$HOME/.pyenv/versions/3.12.9/bin/python3.12"
elif [ -x "$HOME/.pyenv/shims/python3.12" ]; then
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)" 2>/dev/null || true
    PYTHON312=$(command -v python3.12 2>/dev/null || echo "")
fi

if [ -z "$PYTHON312" ]; then
    echo "  Python 3.12 not found. Installing via pyenv..."
    if ! command -v pyenv &>/dev/null; then
        if [ ! -d "$HOME/.pyenv" ]; then
            curl -fsSL https://pyenv.run | bash
        fi
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init -)"
    fi
    sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
        libreadline-dev libsqlite3-dev wget llvm libncursesw5-dev xz-utils \
        tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev 2>/dev/null || true
    pyenv install 3.12.9
    PYTHON312="$HOME/.pyenv/versions/3.12.9/bin/python3.12"
fi
echo "  Using Python 3.12: $PYTHON312"

echo "========== 1. Clone dependency repos =========="
mkdir -p tools

if [ ! -d "tools/TensorRT-Model-Optimizer" ]; then
    git clone --depth 1 https://github.com/NVIDIA/TensorRT-Model-Optimizer.git tools/TensorRT-Model-Optimizer
else
    echo "  tools/TensorRT-Model-Optimizer already exists, skip clone"
fi

if [ "$SKIP_TRTLLM_110" = false ] && [ ! -d "TensorRT-LLM-1.1.0" ]; then
    git clone -b v1.1.0 --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git TensorRT-LLM-1.1.0
elif [ "$SKIP_TRTLLM_110" = true ]; then
    echo "  Skipping TensorRT-LLM-1.1.0 (--no-trtllm-1.1.0)"
else
    echo "  TensorRT-LLM-1.1.0 already exists, skip clone"
fi

if [ "$SKIP_TRTLLM_120" = false ] && [ ! -d "TensorRT-LLM-1.2.0" ]; then
    git clone -b v1.2.0 --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git TensorRT-LLM-1.2.0
elif [ "$SKIP_TRTLLM_120" = true ]; then
    echo "  Skipping TensorRT-LLM-1.2.0 (--no-trtllm-1.2.0)"
else
    echo "  TensorRT-LLM-1.2.0 already exists, skip clone"
fi

echo ""
echo "========== 2. Venv for Model Optimizer (PTQ) — Python 3.12 =========="
if [ ! -d "venv_modelopt" ]; then
    "$PYTHON312" -m venv venv_modelopt
    venv_modelopt/bin/pip install -U pip
    venv_modelopt/bin/pip install -U "nvidia-modelopt[hf]"
    venv_modelopt/bin/pip install torch transformers huggingface_hub
    venv_modelopt/bin/pip install -r tools/TensorRT-Model-Optimizer/examples/llm_ptq/requirements.txt || true
    echo "  Created venv_modelopt"
else
    echo "  venv_modelopt already exists, skip"
fi

echo ""
echo "========== 3. Venv for TensorRT-LLM 1.1.0 — Python 3.12 =========="
if [ "$SKIP_TRTLLM_110" = false ]; then
    if [ ! -d "venv_trtllm1.1.0" ]; then
        # TRT-LLM requires Python 3.12 (no 3.11 wheel) and OpenMPI
        sudo apt-get install -y libopenmpi-dev openmpi-bin 2>/dev/null || true
        "$PYTHON312" -m venv venv_trtllm1.1.0
        venv_trtllm1.1.0/bin/pip install -U pip
        venv_trtllm1.1.0/bin/pip install "tensorrt_llm==1.1.0"
        echo "  Created venv_trtllm1.1.0"
    else
        echo "  venv_trtllm1.1.0 already exists, skip"
    fi
fi

echo ""
echo "========== 4. Venv for TensorRT-LLM 1.2.0 — Python 3.12 =========="
if [ "$SKIP_TRTLLM_120" = false ]; then
    if [ ! -d "venv_trtllm1.2.0" ]; then
        sudo apt-get install -y libopenmpi-dev openmpi-bin 2>/dev/null || true
        "$PYTHON312" -m venv venv_trtllm1.2.0
        venv_trtllm1.2.0/bin/pip install -U pip
        venv_trtllm1.2.0/bin/pip install "tensorrt_llm==1.2.0"
        echo "  Created venv_trtllm1.2.0"
    else
        echo "  venv_trtllm1.2.0 already exists, skip"
    fi
fi

echo ""
echo "========== 5. Output and model dirs =========="
mkdir -p outputs/ckpts outputs/datasets outputs/results models
echo "  outputs/ckpts  outputs/datasets  outputs/results  models  ready"

echo ""
echo "========== Done =========="
echo "Next steps:"
echo "  - HF login:      venv_modelopt/bin/huggingface-cli login  (for gated Llama models)"
echo "  - Download models:"
echo "      venv_modelopt/bin/huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir models/llama-3.1-8b-instruct"
echo "      venv_modelopt/bin/huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir models/llama-3.2-3b-instruct"
echo "      venv_modelopt/bin/huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir models/mistral-7b-instruct-v0.3"
echo "  - PTQ (single):  ./scripts/ptq/run_ptq_single.sh --model <HF_MODEL> --quant fp8 --kv_cache_quant none --tasks quant"
echo "  - PTQ (batch):   ./scripts/ptq/run_ptq_all.sh"
echo "  - Bench:         ./scripts/bench/run_bench.sh  (or bench_daemon.sh for auto-resume)"
echo "  - Eval:          ./scripts/eval/run_prefix_kv.sh"
