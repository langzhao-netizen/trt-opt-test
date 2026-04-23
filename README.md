# trt-opt-test

Test matrix and scripts for TensorRT-LLM / Model Optimizer quantization and benchmarking.

## Layout

```
configs/          Runtime config (chunked_prefill.yaml)
docs/             Setup, test matrix, artifact layout, benchmarking guide, transfer guide
models/           Local HF model weights (gitignored)
notebooks/        toxicity_eval.ipynb — interactive accuracy eval
outputs/          Generated artifacts (gitignored): ckpts, engines, datasets, results
scripts/
  setup.sh        One-shot environment setup
  ptq/            PTQ & checkpoint management
  bench/          Benchmarking (trtllm-bench, dataset gen, daemon)
  eval/           Accuracy & prefix-KV evaluation
  upload_*.sh     HF Hub upload utilities
  pack_ckpts.sh   Pack ckpts to tar.gz for transfer
tools/            TensorRT-Model-Optimizer clone (gitignored, created by setup.sh)
```

Artifact layout details (PTQ ckpt / TRT-LLM ckpt / engine naming + env vars): see **`docs/ARTIFACTS.md`**.

## Quick setup

**Prerequisites:** Linux, Python 3, git; CUDA + TensorRT for build/serve.

```bash
git clone <repo-url> trt-opt-test && cd trt-opt-test
./scripts/setup.sh
```

Clones Model Optimizer + TRT-LLM 1.1.0 + 1.2.0, creates `venv_modelopt`, `venv_trtllm1.1.0`, `venv_trtllm1.2.0`, and output/model dirs.

Flags: `--no-trtllm-1.1.0`, `--no-trtllm-1.2.0` to skip optional clones.

Full checklist for new VM: see **`docs/SETUP.md`**.

## Usage

**PTQ (single run):**
```bash
./scripts/ptq/run_ptq_single.sh --model meta-llama/Llama-3.1-8B-Instruct --quant fp8 --kv_cache_quant none --tasks quant
```

**PTQ (batch — all models × quant combos):**
```bash
./scripts/ptq/run_ptq_all.sh
```

**Benchmark:**
```bash
./scripts/bench/run_bench.sh                             # default 15k/1 over all ckpts
TARGET_INPUT_TOKENS=32000 ./scripts/bench/run_bench.sh  # long context
./scripts/bench/bench_daemon.sh                          # auto-resume on failure
```

**Eval (prefix KV):**
```bash
./scripts/eval/run_prefix_kv.sh
```

**Key env vars:** `CKPT_ROOT` (default `outputs/ckpts`), `MODEL_ROOT` (default `models/`), `TARGET_INPUT_TOKENS`, `NUM_REQUESTS`, `HF_TOKEN` (for gated models). Full list: `docs/SETUP.md`.

**Transfer ckpts:** `docs/CKPT_TRANSFER.md` (tar.gz, HF Hub, rsync).

### Accuracy

`notebooks/toxicity_eval.ipynb` uses `AutoModelForCausalLM.from_pretrained` — works only on **FP16 unquantized** ckpts. For quantized (FP8/INT4) accuracy, use trtllm-serve + a client eval script.
