#!/bin/bash
# Batch PTQ: run generate_quant_ckpt.sh for all (model, quant, kv) combinations in the test matrix.
# Then rename saved_models_* -> convention names and verify.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GENERATE_SCRIPT="${SCRIPT_DIR}/generate_quant_ckpt.sh"
RENAME_SCRIPT="${SCRIPT_DIR}/rename_ckpts_to_convention.sh"
CHECK_SCRIPT="${SCRIPT_DIR}/check_ckpt_quant_config.py"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_ROOT}/outputs/ckpts}"

if [ ! -x "$GENERATE_SCRIPT" ]; then
    echo "Error: generate_quant_ckpt.sh not found or not executable: $GENERATE_SCRIPT" >&2
    exit 1
fi

# Models: HF path or name (adjust if using local paths)
MODELS=(
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
)

# Quant + KV combinations: fp8 and int4_awq, each with fp16 (no KV quant) and fp8 KV.
# Model Optimizer only supports --kv_cache_qformat none|fp8|nvfp4 (no "fp16").
# For KV fp16 we pass "none"; output dir will be saved_models_*_kv_none, renamed to *-kv_fp16.
QUANT_KV=(
    "fp8:fp16"
    "fp8:fp8"
    "int4_awq:fp16"
    "int4_awq:fp8"
)

TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:---trust_remote_code}"
TASKS="${TASKS:-quant}"
export ROOT_SAVE_PATH="$CKPT_ROOT"

for model in "${MODELS[@]}"; do
    for qkv in "${QUANT_KV[@]}"; do
        q="${qkv%%:*}"
        kv="${qkv##*:}"
        # Pass "none" to Model Optimizer when we want FP16 KV (no quantization).
        kv_arg="${kv}"
        if [ "$kv" = "fp16" ]; then
            kv_arg="none"
        fi
        echo "========== PTQ model=$model quant=$q kv_cache_quant=$kv (--kv_cache_qformat=$kv_arg) =========="
        "$GENERATE_SCRIPT" \
            --model "$model" \
            --quant "$q" \
            --kv_cache_quant "$kv_arg" \
            --tasks "$TASKS" \
            $TRUST_REMOTE_CODE
    done
done

echo "========== Step 2: Rename saved_models_* to convention names =========="
CKPT_ROOT="$CKPT_ROOT" "$RENAME_SCRIPT"

echo "========== Step 3: Verify ckpt configs =========="
if [ -f "$CHECK_SCRIPT" ]; then
    if python3 "$CHECK_SCRIPT" "$CKPT_ROOT" -v; then
        echo "Check passed: all ckpts match name vs hf_quant_config.json."
    else
        echo "Check failed: see above." >&2
        exit 1
    fi
fi
echo "========== Done. Ckpts in $CKPT_ROOT =========="
