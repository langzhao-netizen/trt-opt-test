#!/bin/bash
# One-shot setup: clone deps, create venvs, create output dirs.
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

echo "========== 1. Clone dependency repos =========="
mkdir -p tools

if [ ! -d "tools/TensorRT-Model-Optimizer" ]; then
    git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer.git tools/TensorRT-Model-Optimizer
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
echo "========== 2. Venv for Model Optimizer (PTQ) =========="
if [ ! -d "venv_modelopt" ]; then
    python3 -m venv venv_modelopt
    source venv_modelopt/bin/activate
    pip install -U pip
    pip install -U "nvidia-modelopt[hf]"
    pip install -r tools/TensorRT-Model-Optimizer/examples/llm_ptq/requirements.txt
    deactivate
    echo "  Created venv_modelopt"
else
    echo "  venv_modelopt already exists, skip"
fi

echo ""
echo "========== 3. Venv for TensorRT-LLM 1.1.0 =========="
if [ "$SKIP_TRTLLM_110" = false ]; then
    if [ ! -d "venv_trtllm1.1.0" ]; then
        python3 -m venv venv_trtllm1.1.0
        source venv_trtllm1.1.0/bin/activate
        pip install -U pip
        pip install "tensorrt_llm==1.1.0"
        deactivate
        echo "  Created venv_trtllm1.1.0"
    else
        echo "  venv_trtllm1.1.0 already exists, skip"
    fi
fi

echo ""
echo "========== 4. Venv for TensorRT-LLM 1.2.0 =========="
if [ "$SKIP_TRTLLM_120" = false ]; then
    if [ ! -d "venv_trtllm1.2.0" ]; then
        python3 -m venv venv_trtllm1.2.0
        source venv_trtllm1.2.0/bin/activate
        pip install -U pip
        pip install "tensorrt_llm==1.2.0"
        deactivate
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
echo "  - PTQ (single):  ./scripts/ptq/run_ptq_single.sh --model <HF_MODEL> --quant fp8 --kv_cache_quant none --tasks quant"
echo "  - PTQ (batch):   ./scripts/ptq/run_ptq_all.sh"
echo "  - Bench:         ./scripts/bench/run_bench.sh  (or bench_daemon.sh for auto-resume)"
echo "  - Eval:          ./scripts/eval/run_prefix_kv.sh"
echo "  - HF models:     put in models/ or use HF model id directly"
