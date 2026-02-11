#!/bin/bash
# Regenerate only the *-kv_fp16 ckpts (they were wrongly generated with FP8 KV).
# 1) Back up existing *-kv_fp16 dirs  2) Run PTQ with --kv_cache_quant none  3) Rename saved_models_*_kv_none -> *-kv_fp16
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_ROOT}/outputs/ckpts}"
BACKUP_ROOT="${PROJECT_ROOT}/outputs/ckpts_backup_kv_fp16_wrong"
GENERATE_SCRIPT="${SCRIPT_DIR}/generate_quant_ckpt.sh"
RENAME_SCRIPT="${SCRIPT_DIR}/rename_ckpts_to_convention.sh"

if [ ! -x "$GENERATE_SCRIPT" ]; then
    echo "Error: generate_quant_ckpt.sh not found or not executable" >&2
    exit 1
fi

# Only regenerate kv_fp16: (model, fp8, none) and (model, int4_awq, none)
MODELS=(
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
)
# kv_fp16 -> pass "none" to Model Optimizer
QUANT_KV_FP16=(
    "fp8:none"
    "int4_awq:none"
)

TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:---trust_remote_code}"
TASKS="${TASKS:-quant}"

# ---------- 1) Back up existing *-kv_fp16 ----------
echo "========== Step 1: Back up existing *-kv_fp16 ckpts to $BACKUP_ROOT =========="
mkdir -p "$BACKUP_ROOT"
cd "$CKPT_ROOT"
for d in *-kv_fp16; do
    [ ! -d "$d" ] && continue
    echo "Backup: $d -> $BACKUP_ROOT/"
    rm -rf "$BACKUP_ROOT/$d"
    mv "$d" "$BACKUP_ROOT/"
done
cd - >/dev/null

# ---------- 2) Run PTQ for kv_fp16 only (--kv_cache_quant none) ----------
# Remove any existing saved_models_*_kv_none so PTQ creates fresh (no skip from existing config).
echo "========== Step 2: Run PTQ for kv_fp16 (output: saved_models_*_kv_none) =========="
for prefix in "Llama-3_2-3B-Instruct" "Llama-3_1-8B-Instruct" "Mistral-7B-Instruct-v0_3"; do
    for q in fp8 int4_awq; do
        d="$CKPT_ROOT/saved_models_${prefix}_${q}_kv_none"
        [ -d "$d" ] && echo "Remove (will regenerate): $d" && rm -rf "$d"
    done
done
export ROOT_SAVE_PATH="$CKPT_ROOT"
for model in "${MODELS[@]}"; do
    for qkv in "${QUANT_KV_FP16[@]}"; do
        q="${qkv%%:*}"
        kv_arg="${qkv##*:}"
        echo "========== PTQ model=$model quant=$q kv_cache_quant=none (-> kv_fp16) =========="
        "$GENERATE_SCRIPT" \
            --model "$model" \
            --quant "$q" \
            --kv_cache_quant "$kv_arg" \
            --tasks "$TASKS" \
            $TRUST_REMOTE_CODE
    done
done

# ---------- 3) Rename saved_models_*_kv_none -> *-kv_fp16 ----------
echo "========== Step 3: Rename saved_models_*_kv_none to *-kv_fp16 =========="
CKPT_ROOT="$CKPT_ROOT" "$RENAME_SCRIPT"

# ---------- 4) Verify: name vs hf_quant_config.json ----------
CHECK_SCRIPT="${SCRIPT_DIR}/check_ckpt_quant_config.py"
echo "========== Step 4: Verify ckpt configs =========="
if [ -x "$CHECK_SCRIPT" ] || [ -f "$CHECK_SCRIPT" ]; then
    if python3 "$CHECK_SCRIPT" "$CKPT_ROOT" -v; then
        echo "Check passed: all *-kv_fp16 / *-kv_fp8 ckpts match config."
    else
        echo "Check failed: see above. Fix or regenerate." >&2
        exit 1
    fi
else
    echo "Warning: $CHECK_SCRIPT not found; skip verification."
fi

echo "========== Done. Regenerated *-kv_fp16 ckpts; backup at $BACKUP_ROOT =========="
