#!/bin/bash
# Batch PTQ: run generate_quant_ckpt.sh for all (model, quant, kv) combinations in the test matrix.
# Output: outputs/ckpts/saved_models_* (then run rename_ckpts_to_convention.sh).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATE_SCRIPT="${SCRIPT_DIR}/generate_quant_ckpt.sh"

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

# Quant + KV combinations for 必测: fp8 and int4_awq, each with fp16 and fp8 KV
QUANT_KV=(
    "fp8:fp16"
    "fp8:fp8"
    "int4_awq:fp16"
    "int4_awq:fp8"
)

TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:---trust_remote_code}"
TASKS="${TASKS:-quant}"

for model in "${MODELS[@]}"; do
    for qkv in "${QUANT_KV[@]}"; do
        q="${qkv%%:*}"
        kv="${qkv##*:}"
        echo "========== PTQ model=$model quant=$q kv_cache_quant=$kv =========="
        "$GENERATE_SCRIPT" \
            --model "$model" \
            --quant "$q" \
            --kv_cache_quant "$kv" \
            --tasks "$TASKS" \
            $TRUST_REMOTE_CODE
    done
done

echo "========== All PTQ runs finished. Run rename_ckpts_to_convention.sh to unify dir names. =========="
