# Environment Setup on a New VM

## 1. Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Linux (Ubuntu 20.04/22.04) |
| Python | 3.10+, `python3 -m venv` available |
| CUDA | Must match TensorRT-LLM version (CUDA 12.x) |
| Disk | ~50 GB (deps + clones + venvs + optional local models) |
| Network | GitHub, HuggingFace, and pip must be reachable on first run |

---

## 2. One-shot deploy

```bash
git clone <repo-url> trt-opt-test && cd trt-opt-test
./scripts/setup.sh
```

By default clones TensorRT-Model-Optimizer + TRT-LLM 1.1.0 + 1.2.0, creates `venv_modelopt`, `venv_trtllm1.1.0`, `venv_trtllm1.2.0`, and `outputs/`, `models/` directories.

```bash
./scripts/setup.sh --no-trtllm-1.1.0   # skip 1.1.0 to save disk
./scripts/setup.sh --no-trtllm-1.2.0   # skip 1.2.0
```

---

## 3. Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `CKPT_ROOT` | `outputs/ckpts` | PTQ ckpt root directory |
| `TARGET_INPUT_TOKENS` | `15360` | Benchmark dataset input length |
| `NUM_REQUESTS` | `5` | Requests per ckpt in benchmark |
| `TOKENIZER_HF_ID` | `meta-llama/Llama-3.2-3B-Instruct` | Tokenizer used by gen_dataset.py |
| `TOKENIZER_PATH` | (unset) | Local tokenizer path; overrides HF id when set |
| `HF_TOKEN` | (unset) | Required for gated models |

---

## 4. Model sources

- **HF id (recommended)**: pass `meta-llama/Llama-3.1-8B-Instruct` etc. directly; scripts download automatically. Requires HF account agreement + `HF_TOKEN`.
- **Local**: download to `models/<model-name>/` (gitignored), then pass the local path to `--model`.

---

## 5. What you do NOT need to carry from an old machine

- `outputs/`: fully gitignored; starts empty on a new VM and is created by scripts as needed.
- `venv_*/`: gitignored; recreated by `setup.sh`.
- `TensorRT-LLM-*/`, `tools/`: gitignored; cloned by `setup.sh`.
- `models/`: gitignored; pull from HF or copy separately on the new VM.

---

## 6. Full reproduce flow

```bash
# 1. Clone and deploy
git clone <repo-url> trt-opt-test && cd trt-opt-test
./scripts/setup.sh

# 2. Set HF_TOKEN (gated models)
export HF_TOKEN=hf_xxx

# 3. PTQ
./scripts/ptq/run_ptq_all.sh

# 4. Benchmark
./scripts/bench/run_bench.sh

# 5. Accuracy eval (FP16 baseline only)
jupyter notebook notebooks/toxicity_eval.ipynb
```

> FP8/INT4 quantized ckpts cannot be loaded with `AutoModelForCausalLM.from_pretrained`. Accuracy evaluation for quantized models requires trtllm-serve + a client script.
