#!/bin/bash
# 在已有 4 个 saved_models_*_kv_none 的前提下，只补跑缺失的 2 个（Mistral fp8、Mistral int4_awq），不删已有。
# 用法：先挪走残缺的 Mistral fp8（若有），再对本脚本 run；或先 run rename 得到 4 个 *-kv_fp16，再 run 本脚本补 2 个。
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_ROOT}/outputs/ckpts}"
GENERATE_SCRIPT="${SCRIPT_DIR}/run_ptq_single.sh"
RENAME_SCRIPT="${SCRIPT_DIR}/rename_ckpts.sh"
CHECK_SCRIPT="${SCRIPT_DIR}/check_ckpts.py"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:---trust_remote_code}"
TASKS="${TASKS:-quant}"
export ROOT_SAVE_PATH="$CKPT_ROOT"

# 确保 PTQ 用 venv（nohup/daemon 子进程常缺此环境）
[ -f "${PROJECT_ROOT}/venv_modelopt/bin/activate" ] && source "${PROJECT_ROOT}/venv_modelopt/bin/activate"

# 残缺的 Mistral fp8：只有 1 个 shard，挪走以便重新生成（不删）
MISTRAL_FP8_INCOMPLETE="$CKPT_ROOT/saved_models_Mistral-7B-Instruct-v0_3_fp8_kv_none"
if [ -d "$MISTRAL_FP8_INCOMPLETE" ]; then
    if [ ! -f "$MISTRAL_FP8_INCOMPLETE/model-00002-of-00002.safetensors" ] || [ ! -f "$MISTRAL_FP8_INCOMPLETE/tokenizer.json" ]; then
        echo "Move incomplete Mistral fp8 aside (will regenerate): $MISTRAL_FP8_INCOMPLETE -> ${MISTRAL_FP8_INCOMPLETE}.incomplete"
        mv "$MISTRAL_FP8_INCOMPLETE" "${MISTRAL_FP8_INCOMPLETE}.incomplete"
    fi
fi

echo "========== Rename existing saved_models_*_kv_none -> *-kv_fp16 =========="
CKPT_ROOT="$CKPT_ROOT" "$RENAME_SCRIPT"

echo "========== PTQ only Mistral 7B fp8 and int4_awq (kv_none) =========="
for qkv in "fp8:none" "int4_awq:none"; do
    q="${qkv%%:*}"
    kv_arg="${qkv##*:}"
    echo "========== PTQ model=mistralai/Mistral-7B-Instruct-v0.3 quant=$q kv_cache_quant=none =========="
    "$GENERATE_SCRIPT" \
        --model "mistralai/Mistral-7B-Instruct-v0.3" \
        --quant "$q" \
        --kv_cache_quant "$kv_arg" \
        --tasks "$TASKS" \
        $TRUST_REMOTE_CODE
done

echo "========== Rename new saved_models_*_kv_none -> *-kv_fp16 =========="
CKPT_ROOT="$CKPT_ROOT" "$RENAME_SCRIPT"

echo "========== Verify =========="
python3 "$CHECK_SCRIPT" "$CKPT_ROOT" -v
echo "Done. 6 *-kv_fp16 in $CKPT_ROOT"
