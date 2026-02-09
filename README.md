# trt-opt-test

Test matrix and scripts for TensorRT-LLM / Model Optimizer quantization and benchmarking.

## Layout

- **configs/** – Test matrix and config (e.g. `test_matrix_full.md`).
- **scripts/** – PTQ and TRT-LLM checkpoint scripts:
  - `generate_quant_ckpt.sh` – Single Model Optimizer PTQ run.
  - `run_all_ptq.sh` – Batch PTQ over test matrix.
  - `rename_ckpts_to_convention.sh` – Rename `saved_models_*` to convention names.
  - `gen_bench_dataset.py` – Benchmark dataset generation.
- **outputs/** – Generated checkpoints and engines (gitignored).
- **shared/** – Optional local HF model copies (gitignored, ~69GB). Scripts can use HF model IDs instead.
- **tools/** – Clone TensorRT-Model-Optimizer here (gitignored, see below).

## Quick setup on a new machine

**Prereqs:** Linux, Python 3, git, CUDA (and TensorRT if you will build/serve engines). Then:

Clone this repo, then run the one-shot setup (clones deps + creates venvs):

```bash
git clone <this-repo-url> trt-opt-test && cd trt-opt-test
./scripts/setup.sh
```

- **With TensorRT-LLM 1.1.0** (default): clones Model Optimizer + TensorRT-LLM v0.18.0 + v1.1.0, creates `venv_modelopt`, `venv_trtllm0.18.0`, `venv_trtllm1.1.0`.
- **Without 1.1.0** (save disk): `./scripts/setup.sh --no-trtllm-1.1.0`.

After setup, you can run PTQ without activating a venv (the script uses `venv_modelopt` if present):

```bash
./scripts/generate_quant_ckpt.sh --model meta-llama/Llama-3.2-3B-Instruct --quant int4_awq --kv_cache_quant fp8 --tasks quant
./scripts/run_all_ptq.sh
./scripts/rename_ckpts_to_convention.sh
```

For TensorRT-LLM build/serve, activate the right venv: `source venv_trtllm0.18.0/bin/activate` or `venv_trtllm1.1.0/bin/activate`.

## Usage

- **PTQ (Model Optimizer)**: `./scripts/generate_quant_ckpt.sh --model <HF_MODEL> --quant int4_awq --kv_cache_quant fp8 --tasks quant`  
  Batch: `./scripts/run_all_ptq.sh` then `./scripts/rename_ckpts_to_convention.sh`.
- See `configs/test_matrix_full.md` for the full test matrix.
