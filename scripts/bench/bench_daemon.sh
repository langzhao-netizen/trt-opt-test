#!/bin/bash
# Fully automated: runs 15k-input/1-output (bs=1) benchmarks for the 12 final ckpts in outputs/ckpts
# (6 *-kv_fp8 + 6 *-kv_fp16) using trtllm==1.1.0 PyTorch backend. Monitors the runner,
# restarts on failure, and auto-installs missing dependencies.
#
# Usage:
#   nohup ./scripts/bench/bench_daemon.sh >> outputs/bench_daemon.log 2>&1 &
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_ROOT}/outputs/ckpts}"
OUT_DIR="${PROJECT_ROOT}/outputs/bench_trtllm110_pytorch"
RUNNER="${SCRIPT_DIR}/bench_impl.py"
VENV_PY="${PROJECT_ROOT}/venv_trtllm1.1.0/bin/python3"
VENV_PIP="${PROJECT_ROOT}/venv_trtllm1.1.0/bin/pip"

DAEMON_LOG="${PROJECT_ROOT}/outputs/bench_daemon.log"
RUNNER_LOG="${OUT_DIR}/runner.log"
RESULT_JSONL="${OUT_DIR}/results_15k1output.jsonl"

INTERVAL=30
TARGET_TOTAL=12

mkdir -p "$OUT_DIR"

log() { echo "[$(date -Iseconds)] $*"; }

count_ok() {
    [ -f "$RESULT_JSONL" ] || { echo 0; return; }
    python3 - <<'PY'
import json, sys, os
path=os.environ["RESULT_JSONL"]
ok=set()
with open(path) as f:
    for line in f:
        line=line.strip()
        if not line: continue
        r=json.loads(line)
        if r.get("status")=="ok":
            ok.add(r.get("ckpt"))
print(len(ok))
PY
}

is_running() {
    pgrep -f "bench_impl\.py" >/dev/null 2>&1
}

maybe_fix_env() {
    [ -x "$VENV_PIP" ] || return 0
    [ -f "$RUNNER_LOG" ] || return 0
    tail_log="$(tail -80 "$RUNNER_LOG" 2>/dev/null || true)"
    mod="$(echo "$tail_log" | sed -n \"s/.*ModuleNotFoundError: No module named '\\(.*\\)'.*/\\1/p\" | tail -1)"
    if [ -n "$mod" ]; then
        log "Detected missing module '$mod' in trtllm1.1.0 venv; installing..."
        "$VENV_PIP" install -U "$mod" >/dev/null 2>&1 || true
    fi
}

if [ ! -x "$VENV_PY" ]; then
    log "ERROR: $VENV_PY not found. Run ./scripts/setup.sh first." >&2
    exit 1
fi

export RESULT_JSONL

while true; do
    ok=$(RESULT_JSONL="$RESULT_JSONL" count_ok)
    if [ "$ok" -ge "$TARGET_TOTAL" ]; then
        log "All done: ok=$ok/$TARGET_TOTAL. Exit."
        exit 0
    fi

    if is_running; then
        log "Runner still running (ok=$ok/$TARGET_TOTAL). Next check in ${INTERVAL}s."
    else
        maybe_fix_env
        log "Runner not running (ok=$ok/$TARGET_TOTAL). Starting runner..."
        nohup "$VENV_PY" "$RUNNER" --ckpt-root "$CKPT_ROOT" --out-dir "$OUT_DIR" --resume >> "$RUNNER_LOG" 2>&1 &
        log "Started PID $!"
    fi
    sleep "$INTERVAL"
done

