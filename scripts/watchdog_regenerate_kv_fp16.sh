#!/bin/bash
# Watchdog: 每 3 分钟检查 regenerate 是否在跑；若进程不在且 *-kv_fp16 不足 6 个则自动重启。成功 6 个后跑校验并退出。
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_ROOT}/outputs/ckpts}"
LOG="${PROJECT_ROOT}/outputs/regenerate_kv_fp16.log"
REGENERATE_SCRIPT="${SCRIPT_DIR}/regenerate_kv_fp16_ckpts.sh"
CHECK_SCRIPT="${SCRIPT_DIR}/check_ckpt_quant_config.py"
INTERVAL=180
MAX_RESTARTS=5
TARGET_KV_FP16=6

cd "$PROJECT_ROOT"

count_kv_fp16() {
    ls -d "$CKPT_ROOT"/*-kv_fp16 2>/dev/null | wc -l
}

is_regenerate_running() {
    pgrep -f "regenerate_kv_fp16_ckpts.sh|hf_ptq.py" >/dev/null
}

restart_count=0
while true; do
    if is_regenerate_running; then
        echo "[$(date -Iseconds)] Watchdog: regenerate/ptq still running, next check in ${INTERVAL}s"
    else
        n=$(count_kv_fp16)
        if [ "$n" -ge "$TARGET_KV_FP16" ]; then
            echo "[$(date -Iseconds)] Watchdog: $n *-kv_fp16 found, running check then exit."
            python3 "$CHECK_SCRIPT" "$CKPT_ROOT" -v && echo "[$(date -Iseconds)] Watchdog: check passed, done." || echo "[$(date -Iseconds)] Watchdog: check failed." >&2
            exit 0
        fi
        if [ "$restart_count" -ge "$MAX_RESTARTS" ]; then
            echo "[$(date -Iseconds)] Watchdog: max restarts ($MAX_RESTARTS) reached, exit." >&2
            exit 1
        fi
        restart_count=$((restart_count + 1))
        echo "[$(date -Iseconds)] Watchdog: process not running, kv_fp16=$n/$TARGET_KV_FP16, restart #$restart_count"
        nohup bash -c "cd '$PROJECT_ROOT' && '$REGENERATE_SCRIPT'" >> "$LOG" 2>&1 &
        echo "[$(date -Iseconds)] Watchdog: started regenerate (PID $!)"
    fi
    sleep "$INTERVAL"
done
