# Benchmarking

Uses `venv_trtllm1.1.0` (TensorRT-LLM 1.1.0) and `trtllm-bench`. Checkpoints from `CKPT_ROOT` (default `outputs/ckpts`).

## General runner

**`scripts/bench/run_bench.sh`** — Any input/output length, resumable, auto-generates dataset if missing. Uses chunked prefill for input > 16k tokens (`configs/chunked_prefill.yaml`).

```bash
./scripts/bench/run_bench.sh                                         # default: 15k input, 1 output
TARGET_INPUT_TOKENS=32000 TARGET_OUTPUT_TOKENS=1 ./scripts/bench/run_bench.sh
CUDA_VISIBLE_DEVICES=0 TARGET_INPUT_TOKENS=32000 ./scripts/bench/run_bench.sh  # long ctx, pin to GPU 0
```

Results: `outputs/results_trtllm1.1.0_<input>_<output>_bs1_pytorch/<ckpt>/latency.json`.

## Daemon (auto-resume on failure)

```bash
nohup ./scripts/bench/bench_daemon.sh >> outputs/bench_daemon.log 2>&1 &
```

Polls every 30s; restarts `bench_impl.py` until all ckpts have results.

## Python bench (alternative to trtllm-bench)

```bash
source venv_trtllm1.1.0/bin/activate
python3 scripts/bench/bench_impl.py --ckpt-root outputs/ckpts --out-dir outputs/bench_results --resume
```

## Generate dataset only

```bash
BENCH_DATASET=outputs/datasets/llama-3.1-8b/dataset_15k_1.jsonl \
TOKENIZER_HF_ID=meta-llama/Llama-3.1-8B-Instruct \
TARGET_INPUT_TOKENS=15360 TARGET_OUTPUT_TOKENS=1 NUM_REQUESTS=5 \
python3 scripts/bench/gen_dataset.py
```

## Env vars

| Variable | Default | Meaning |
|----------|---------|---------|
| `CKPT_ROOT` | `outputs/ckpts` | Checkpoint root |
| `TARGET_INPUT_TOKENS` | 15360 | Input token count |
| `TARGET_OUTPUT_TOKENS` | 1 | Output token count |
| `NUM_REQUESTS` | 5 | Requests per ckpt |
| `WARMUP` | 2 | Warmup requests |
| `CUDA_VISIBLE_DEVICES` | (unset) | Pin GPU for long context |
