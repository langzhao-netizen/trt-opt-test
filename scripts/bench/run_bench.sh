#!/bin/bash
# General trtllm-bench runner: input/output length and result dirs driven by env.
# Usage:
#   ./scripts/bench/run_bench.sh
#   TARGET_INPUT_TOKENS=8192 TARGET_OUTPUT_TOKENS=1 ./scripts/bench/run_bench.sh
#   TARGET_INPUT_TOKENS=32000 CUDA_VISIBLE_DEVICES=0 ./scripts/bench/run_bench.sh
# Env: CKPT_ROOT, TARGET_INPUT_TOKENS (default 15360), TARGET_OUTPUT_TOKENS (default 1),
#      NUM_REQUESTS (5), WARMUP (2), CUDA_VISIBLE_DEVICES (optional; for long context set to 0).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CKPT_ROOT="${CKPT_ROOT:-${PROJECT_ROOT}/outputs/ckpts}"
TARGET_INPUT_TOKENS="${TARGET_INPUT_TOKENS:-15360}"
TARGET_OUTPUT_TOKENS="${TARGET_OUTPUT_TOKENS:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-$(( TARGET_INPUT_TOKENS + TARGET_OUTPUT_TOKENS + 1 ))}"
NUM_REQUESTS="${NUM_REQUESTS:-5}"
WARMUP="${WARMUP:-2}"
KV_CACHE_FREE_GPU_MEM_FRACTION="${KV_CACHE_FREE_GPU_MEM_FRACTION:-0.9}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/outputs/bench_${TARGET_INPUT_TOKENS}_${TARGET_OUTPUT_TOKENS}}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/outputs/results_trtllm1.1.0_${TARGET_INPUT_TOKENS}_${TARGET_OUTPUT_TOKENS}_bs1_pytorch}"
MAIN_LOG="${RESULT_ROOT}/run_all.log"
LOCK_FILE="${RESULT_ROOT}/.runner.lock"
DATASET_NAME="dataset_${TARGET_INPUT_TOKENS}_${TARGET_OUTPUT_TOKENS}.jsonl"

mkdir -p "$RESULT_ROOT" "$DATA_ROOT"
touch "$MAIN_LOG"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$MAIN_LOG"; }

exec 9>>"$LOCK_FILE"
flock -n 9 || { log "Another runner is active, exit."; exit 0; }

pick_model_and_dataset() {
  local ckpt_name="$1"
  local model="" dataset="" tok=""
  if [[ "$ckpt_name" == llama-3.2-* ]]; then
    model="meta-llama/Llama-3.2-3B-Instruct"
    tok="$model"
    dataset="$DATA_ROOT/llama-3.2-3b/$DATASET_NAME"
  elif [[ "$ckpt_name" == llama-3.1-* ]]; then
    model="meta-llama/Llama-3.1-8B-Instruct"
    tok="$model"
    dataset="$DATA_ROOT/llama-3.1-8b/$DATASET_NAME"
  elif [[ "$ckpt_name" == mistral-7b* ]]; then
    model="mistralai/Mistral-7B-Instruct-v0.3"
    tok="$model"
    dataset="$DATA_ROOT/mistral-7b/$DATASET_NAME"
  elif [[ "$ckpt_name" == ministral-8b* ]]; then
    model="mistralai/Ministral-8B-Instruct-2410"
    tok="$model"
    dataset="$DATA_ROOT/ministral-8b/$DATASET_NAME"
  else
    echo "unknown_model"
    echo ""
    echo ""
    return 0
  fi
  echo "$model"
  echo "$dataset"
  echo "$tok"
}

source "${PROJECT_ROOT}/venv_trtllm1.1.0/bin/activate"
TRTLLM_BENCH="${PROJECT_ROOT}/venv_trtllm1.1.0/bin/trtllm-bench"
[ ! -x "$TRTLLM_BENCH" ] && { log "ERROR: trtllm-bench not found at $TRTLLM_BENCH"; exit 1; }

# Long context: use chunked prefill and single GPU if not set
CHUNKED_OPTS=""
if [[ "${TARGET_INPUT_TOKENS}" -gt 16000 ]] && [[ -f "${PROJECT_ROOT}/configs/chunked_prefill.yaml" ]]; then
  CHUNKED_OPTS="--extra_llm_api_options ${PROJECT_ROOT}/configs/chunked_prefill.yaml"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

export TARGET_INPUT_TOKENS TARGET_OUTPUT_TOKENS MAX_SEQ_LEN NUM_REQUESTS WARMUP KV_CACHE_FREE_GPU_MEM_FRACTION
log "Run trtllm-bench: input=${TARGET_INPUT_TOKENS}, output=${TARGET_OUTPUT_TOKENS}, max_seq_len=${MAX_SEQ_LEN}, requests=${NUM_REQUESTS}, warmup=${WARMUP}"
log "DATA_ROOT=$DATA_ROOT RESULT_ROOT=$RESULT_ROOT"

shopt -s nullglob
ckpts=( "$CKPT_ROOT"/*-trtllm-ckpt-wq_*-kv_* )
IFS=$'\n' ckpts=( $(printf "%s\n" "${ckpts[@]}" | sort) ); unset IFS

for ckpt_dir in "${ckpts[@]}"; do
  ckpt_name="$(basename "$ckpt_dir")"
  mapfile -t md < <(pick_model_and_dataset "$ckpt_name")
  model="${md[0]:-unknown_model}"
  dataset="${md[1]:-}"
  tok="${md[2]:-}"
  if [[ "$model" == "unknown_model" ]] || [[ -z "$dataset" ]] || [[ -z "$tok" ]]; then
    log "SKIP: no model/dataset for $ckpt_name"
    continue
  fi

  if [[ ! -f "$dataset" ]]; then
    log "Generate dataset: $dataset (tokenizer=$tok)"
    mkdir -p "$(dirname "$dataset")"
    BENCH_DATASET="$dataset" TOKENIZER_HF_ID="$tok" TARGET_INPUT_TOKENS="$TARGET_INPUT_TOKENS" TARGET_OUTPUT_TOKENS="$TARGET_OUTPUT_TOKENS" NUM_REQUESTS="$NUM_REQUESTS" \
      "${PROJECT_ROOT}/venv_trtllm1.1.0/bin/python3" "${PROJECT_ROOT}/scripts/bench/gen_dataset.py" >>"$MAIN_LOG" 2>&1
  fi
  [[ ! -f "$dataset" ]] && { log "ERROR: dataset not found $dataset"; exit 1; }

  out_dir="$RESULT_ROOT/$ckpt_name"
  report_json="$out_dir/latency.json"
  iter_log="$out_dir/iteration.log"
  ckpt_log="$out_dir/run.log"
  mkdir -p "$out_dir"

  if [[ -f "$report_json" ]]; then
    log "DONE (skip): $ckpt_name"
    continue
  fi

  log "=== $ckpt_name ==="
  set +e
  $TRTLLM_BENCH \
    --model "$model" \
    --model_path "$ckpt_dir" \
    --workspace "$out_dir/workspace" \
    latency \
      --backend pytorch \
      --dataset "$dataset" \
      --num_requests "$NUM_REQUESTS" \
      --warmup "$WARMUP" \
      --concurrency 1 \
      --kv_cache_free_gpu_mem_fraction "$KV_CACHE_FREE_GPU_MEM_FRACTION" \
      --max_seq_len "$MAX_SEQ_LEN" \
      $CHUNKED_OPTS \
      --report_json "$report_json" \
      --iteration_log "$iter_log" \
      2>&1 | tee -a "$ckpt_log"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ $rc -eq 0 ]] && [[ -f "$report_json" ]]; then
    log "  OK: $report_json"
  else
    log "  FAIL: rc=$rc (see $ckpt_log)"
    printf '%s\n' "{\"checkpoint\":\"$ckpt_name\",\"status\":\"fail\",\"exit_code\":$rc}" > "$out_dir/error.json"
  fi
done

log "Done. Results under $RESULT_ROOT"
deactivate || true
