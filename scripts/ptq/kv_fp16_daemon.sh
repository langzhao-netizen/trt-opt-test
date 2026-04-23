#!/bin/bash
# 全自动：跑补全 6 个 *-kv_fp16、周期检查、失败自动拉起、完成后校验并写报告。无需人工参与。
# 用法: nohup ./scripts/auto_kv_fp16_daemon.sh >> outputs/auto_kv_fp16_daemon.log 2>&1 &
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_ROOT}/outputs/ckpts}"
LOG="${PROJECT_ROOT}/outputs/auto_kv_fp16_daemon.log"
REPORT="${PROJECT_ROOT}/outputs/ckpt_generation_done.txt"
CONTINUE_SCRIPT="${SCRIPT_DIR}/kv_fp16_continue.sh"
CHECK_SCRIPT="${SCRIPT_DIR}/check_ckpts.py"
INTERVAL=30
TARGET_KV_FP16=6
MAX_RESTARTS=99999

# 保证 nohup 子进程用上 venv，PTQ 才能找到 numpy 等
# 注意：venv 必须放在 PATH 最前，否则可能命中 pyenv shim / 系统 python 而缺包
export PATH="${HOME}/.pyenv/shims:${HOME}/.pyenv/bin:/usr/bin:${PATH}"
[ -d "${PROJECT_ROOT}/venv_modelopt/bin" ] && export PATH="${PROJECT_ROOT}/venv_modelopt/bin:${PATH}"
export CKPT_ROOT
export ROOT_SAVE_PATH="$CKPT_ROOT"
cd "$PROJECT_ROOT"

count_kv_fp16() { ls -d "$CKPT_ROOT"/*-kv_fp16 2>/dev/null | wc -l; }
# 只认实际跑 PTQ 的进程，避免把 nohup/tail -f 等误判为在跑
is_running() { pgrep -f "hf_ptq\.py" >/dev/null 2>&1 || pgrep -f "huggingface_example\.sh" >/dev/null 2>&1; }

log() { echo "[$(date -Iseconds)] $*"; }
restarts=0
CONTINUE_LOG="${PROJECT_ROOT}/outputs/continue_kv_fp16_remaining.log"
VENV_PIP="${PROJECT_ROOT}/venv_modelopt/bin/pip"
LAST_FIX_EPOCH_FILE="${PROJECT_ROOT}/outputs/.auto_kv_fp16_last_fix_epoch"

maybe_fix_env() {
    # 失败原因自动诊断与修复（只做“确定可修复”的依赖缺失）
    [ -x "$VENV_PIP" ] || return 0
    [ -f "$CONTINUE_LOG" ] || return 0

    local now last
    now="$(date +%s)"
    last="0"
    [ -f "$LAST_FIX_EPOCH_FILE" ] && last="$(cat "$LAST_FIX_EPOCH_FILE" 2>/dev/null || echo 0)"
    # 避免 30s 频率下重复 pip（2 分钟冷却）
    if [ $((now - last)) -lt 120 ]; then
        return 0
    fi

    local tail_log
    tail_log="$(tail -120 "$CONTINUE_LOG" 2>/dev/null || true)"

    if echo "$tail_log" | grep -q "No module named 'numpy'"; then
        log "Detected missing numpy; installing into venv_modelopt..."
        "$VENV_PIP" install -U numpy >/dev/null 2>&1 || true
        echo "$now" > "$LAST_FIX_EPOCH_FILE" || true
    fi

    if echo "$tail_log" | grep -q "have sentencepiece installed" || echo "$tail_log" | grep -q "No module named 'sentencepiece'"; then
        log "Detected missing sentencepiece; installing into venv_modelopt..."
        "$VENV_PIP" install -U sentencepiece >/dev/null 2>&1 || true
        echo "$now" > "$LAST_FIX_EPOCH_FILE" || true
    fi

    if echo "$tail_log" | grep -q "requires the protobuf library but it was not found" || echo "$tail_log" | grep -q "No module named 'google.protobuf'"; then
        log "Detected missing protobuf; installing into venv_modelopt..."
        "$VENV_PIP" install -U protobuf >/dev/null 2>&1 || true
        echo "$now" > "$LAST_FIX_EPOCH_FILE" || true
    fi
}

while true; do
    n=$(count_kv_fp16)
    if [ "$n" -ge "$TARGET_KV_FP16" ]; then
        log "Found $n *-kv_fp16. Running final check..."
        if python3 "$CHECK_SCRIPT" "$CKPT_ROOT" -v; then
            log "Check passed. Writing report to $REPORT"
            {
                echo "kv_fp16 ckpt generation completed at $(date -Iseconds)"
                echo "CKPT_ROOT=$CKPT_ROOT"
                echo "*-kv_fp16 count: $n"
                python3 "$CHECK_SCRIPT" "$CKPT_ROOT" -v 2>&1
            } > "$REPORT"
            log "Done. Report: $REPORT"
            exit 0
        else
            log "Check failed; will retry on next cycle."
        fi
    fi

    if is_running; then
        log "Generator still running (kv_fp16=$n/$TARGET_KV_FP16). Next check in ${INTERVAL}s."
    else
        restarts=$((restarts + 1))
        if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
            log "Max restarts ($MAX_RESTARTS) reached. Exit." >&2
            exit 1
        fi
        maybe_fix_env
        log "Generator not running (kv_fp16=$n/$TARGET_KV_FP16). Starting continue script (restart #$restarts)."
        nohup env PATH="$PATH" CKPT_ROOT="$CKPT_ROOT" ROOT_SAVE_PATH="$ROOT_SAVE_PATH" \
            bash "$CONTINUE_SCRIPT" >> "${PROJECT_ROOT}/outputs/continue_kv_fp16_remaining.log" 2>&1 &
        log "Started PID $!"
    fi
    sleep "$INTERVAL"
done
