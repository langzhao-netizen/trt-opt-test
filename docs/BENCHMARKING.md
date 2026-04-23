# Benchmark scripts

Use `venv_trtllm1.1.0` (TensorRT-LLM 1.1.0) and `trtllm-bench`. Checkpoints: `CKPT_ROOT` (default `outputs/ckpts`).

---

## General runner (recommended)

**`scripts/run_trtllm_bench_pytorch.sh`** — One script for any input/output length. Resumable; uses chunked prefill for long context (input > 16k) if `configs/llm_api_chunked_prefill.yaml` exists.

```bash
# Default: 15k input, 1 output (same as legacy 15k1)
./scripts/run_trtllm_bench_pytorch.sh

# Custom length
TARGET_INPUT_TOKENS=8192 TARGET_OUTPUT_TOKENS=1 ./scripts/run_trtllm_bench_pytorch.sh
TARGET_INPUT_TOKENS=32000 TARGET_OUTPUT_TOKENS=1 ./scripts/run_trtllm_bench_pytorch.sh

# Long context: pin to one GPU to avoid OOM
CUDA_VISIBLE_DEVICES=0 TARGET_INPUT_TOKENS=32000 ./scripts/run_trtllm_bench_pytorch.sh
```

Results: `outputs/results_trtllm1.1.0_<input>_<output>_bs1_pytorch/<ckpt>/latency.json`. Datasets under `outputs/bench_<input>_<output>/`.

---

## Script list (legacy / variants)

| Script | Purpose |
|--------|--------|
| `scripts/run_all_trtllm_bench_15k1_bs1_pytorch.sh` | **Main 15k1 runner**: 15k input / 1 output for every ckpt in `outputs/ckpts` (Llama-3.2-3B, Llama-3.1-8B, Mistral-7B, Ministral-8B). PyTorch backend, bs=1, 5 requests, 2 warmup. Resumable (skips ckpts with existing `latency.json`). |
| `scripts/run_20k1_bench_llama8b_ministral8b.sh` | 20k input / 1 output for Llama-3.1-8B and Ministral-8B ckpts only. Uses chunked prefill + `CUDA_VISIBLE_DEVICES=0`. |
| `scripts/run_25k1_bench_llama8b_ministral8b.sh` | 25k input / 1 output (Llama-3.1-8B + Ministral-8B). Chunked prefill + single GPU. |
| `scripts/run_32k1_bench_llama8b_ministral8b.sh` | 32k input / 1 output (Llama-3.1-8B + Ministral-8B). Chunked prefill + single GPU. |
| `scripts/run_50k1_bench_llama8b_ministral8b.sh` | 50k input / 1 output (Llama-3.1-8B + Ministral-8B). |
| `scripts/run_100k1_bench_llama8b_ministral8b.sh` | 100k input / 1 output (Llama-3.1-8B + Ministral-8B). |
| `scripts/bench_15k1output_trtllm110_pytorch.py` | **Python 15k1 bench**: Same 15k1 setup, implemented in Python (loads ckpt, runs inference, writes latency). Used as alternative to `trtllm-bench` for the same 12 ckpts. |
| `scripts/gen_bench_dataset.py` | **Dataset generator**: Produces JSONL with `input_ids` / `output_tokens` for trtllm-bench. Env: `BENCH_DATASET`, `TOKENIZER_HF_ID` or `TOKENIZER_PATH`, `TARGET_INPUT_TOKENS`, `TARGET_OUTPUT_TOKENS`, `NUM_REQUESTS`. |
| `scripts/auto_run_all_trtllm_bench_15k1_bs1_pytorch_daemon.sh` | **Watchdog**: Every 30s checks if 15k1 runner is running; if not, starts `run_all_trtllm_bench_15k1_bs1_pytorch.sh`. Stops when all ckpts have `latency.json`. |
| `scripts/auto_bench_15k1output_trtllm110_pytorch_daemon.sh` | **Watchdog for Python bench**: Same idea for `bench_15k1output_trtllm110_pytorch.py` (restarts until 12 results OK). |
| `configs/llm_api_chunked_prefill.yaml` | Enables chunked prefill for long-context runs (used by 20k/25k/32k scripts). |

---

## How to run

### 15k1 (all ckpts, recommended)

```bash
# One-shot
./scripts/run_all_trtllm_bench_15k1_bs1_pytorch.sh

# Or with daemon (restarts on failure until all done)
nohup ./scripts/auto_run_all_trtllm_bench_15k1_bs1_pytorch_daemon.sh >> outputs/auto_run_all_trtllm_bench_15k1_bs1_pytorch_daemon.log 2>&1 &
```

Results: `outputs/results_trtllm1.1.0_15k1_bs1_pytorch/<ckpt_name>/latency.json`, `iteration.log`, `run.log`.

### 20k1 / 32k1 (Llama-3.1-8B + Ministral-8B only)

Ensure no other bench is using the GPU (e.g. `pkill -f trtllm-bench`), then:

```bash
./scripts/run_20k1_bench_llama8b_ministral8b.sh
# or
./scripts/run_32k1_bench_llama8b_ministral8b.sh
```

Results: `outputs/results_trtllm1.1.0_20k1_bs1_pytorch/` or `_32k1_...`.

### Python 15k1 (alternative to trtllm-bench)

```bash
source venv_trtllm1.1.0/bin/activate
python3 scripts/bench_15k1output_trtllm110_pytorch.py --ckpt-root outputs/ckpts --out-dir outputs/bench_trtllm110_pytorch --resume
```

### Generate dataset only

```bash
export BENCH_DATASET=outputs/bench_15k1/llama-3.1-8b/dataset_15k_1.jsonl
export TOKENIZER_HF_ID=meta-llama/Llama-3.1-8B-Instruct
export TARGET_INPUT_TOKENS=15360
export TARGET_OUTPUT_TOKENS=1
export NUM_REQUESTS=5
python3 scripts/gen_bench_dataset.py
```

---

## Env vars (optional)

| Variable | Default | Meaning |
|----------|---------|--------|
| `CKPT_ROOT` | `outputs/ckpts` | Checkpoint root. |
| `DATA_ROOT` | e.g. `outputs/bench_15k1` | Where dataset JSONL files are (or generated). |
| `RESULT_ROOT` | e.g. `outputs/results_trtllm1.1.0_15k1_bs1_pytorch` | Per-ckpt results. |
| `TARGET_INPUT_TOKENS` | 15360 (15k1) / 20000 (20k1) / etc. | Input length. |
| `TARGET_OUTPUT_TOKENS` | 1 | Output length. |
| `NUM_REQUESTS` | 5 | Requests per ckpt. |
| `WARMUP` | 2 | Warmup requests. |
| `CUDA_VISIBLE_DEVICES` | (unset; 20k/25k/32k use 0) | GPU id for bench. |

---

## Bundle for another VM

To copy only benchmark-related files:

```bash
./scripts/pack_benchmark_scripts.sh
# -> outputs/benchmark_scripts_bundle.tar.gz
```

Then on the other VM: extract, ensure `venv_trtllm1.1.0` + `trtllm-bench` and `outputs/ckpts` (or set `CKPT_ROOT`), and run the same commands above.
