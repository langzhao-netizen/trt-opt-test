#!/bin/bash
# One-shot setup: clone deps, create venvs, so another machine can run scripts after clone.
# Usage: ./scripts/setup.sh [--no-trtllm-1.1.0]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

SKIP_TRTLLM_110=false
for x in "$@"; do
    [ "$x" = "--no-trtllm-1.1.0" ] && SKIP_TRTLLM_110=true
done

echo "========== 1. Clone dependency repos =========="
mkdir -p tools

if [ ! -d "tools/TensorRT-Model-Optimizer" ]; then
    git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer.git tools/TensorRT-Model-Optimizer
else
    echo "  tools/TensorRT-Model-Optimizer already exists, skip clone"
fi

if [ ! -d "TensorRT-LLM" ]; then
    git clone -b v0.18.0 --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git
else
    echo "  TensorRT-LLM already exists, skip clone"
fi

if [ "$SKIP_TRTLLM_110" = false ] && [ ! -d "TensorRT-LLM-1.1.0" ]; then
    git clone -b v1.1.0 --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git TensorRT-LLM-1.1.0
elif [ "$SKIP_TRTLLM_110" = true ]; then
    echo "  Skipping TensorRT-LLM-1.1.0 (--no-trtllm-1.1.0)"
else
    echo "  TensorRT-LLM-1.1.0 already exists, skip clone"
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
echo "========== 3. Venv for TensorRT-LLM 0.18.0 (convert/build/serve) =========="
if [ ! -d "venv_trtllm0.18.0" ]; then
    python3 -m venv venv_trtllm0.18.0
    source venv_trtllm0.18.0/bin/activate
    pip install -U pip
    pip install "tensorrt_llm==0.18.0"
    pip install "onnx>=1.12,<1.20"
    deactivate
    echo "  Created venv_trtllm0.18.0"
else
    echo "  venv_trtllm0.18.0 already exists, skip"
fi

echo ""
echo "========== 4. Venv for TensorRT-LLM 1.1.0 (optional build/serve) =========="
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
echo "========== 5. Output dirs =========="
mkdir -p outputs/ckpts outputs/calib_data_tllm018
echo "  outputs/ckpts outputs/calib_data_tllm018 ready"

echo ""
echo "========== Done =========="
echo "Next steps:"
echo "  - PTQ (Model Optimizer):  source venv_modelopt/bin/activate && ./scripts/generate_quant_ckpt.sh --model <HF_MODEL> --quant int4_awq --kv_cache_quant fp8 --tasks quant"
echo "  - Or run without activating: ./scripts/generate_quant_ckpt.sh ...  (script will use venv_modelopt if present)"
echo "  - Batch PTQ:  ./scripts/run_all_ptq.sh  then  ./scripts/rename_ckpts_to_convention.sh"
echo "  - TensorRT-LLM build/serve:  source venv_trtllm0.18.0/bin/activate  or  venv_trtllm1.1.0/bin/activate"
echo "  - Download HF models to shared/ or use HF model id (e.g. meta-llama/Llama-3.2-3B-Instruct) when running PTQ"
