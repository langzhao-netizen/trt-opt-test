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
- **tools/** – Clone TensorRT-Model-Optimizer here (gitignored, see below).

## Clone dependency repos (not in this repo)

After cloning this repo, clone the following so paths used by the scripts exist:

```bash
# Model Optimizer (for PTQ / generate_quant_ckpt.sh)
git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer.git tools/TensorRT-Model-Optimizer

# TensorRT-LLM (for convert/build/serve; pick version as needed)
git clone -b v0.18.0 https://github.com/NVIDIA/TensorRT-LLM.git
# and/or
git clone -b v1.1.0 https://github.com/NVIDIA/TensorRT-LLM.git TensorRT-LLM-1.1.0
```

Create venvs and install deps per your setup (e.g. TensorRT-LLM docs / Model-Optimizer README).

## Usage

- **PTQ (Model Optimizer)**: `./scripts/generate_quant_ckpt.sh --model <HF_MODEL> --quant int4_awq --kv_cache_quant fp8 --tasks quant`  
  Batch: `./scripts/run_all_ptq.sh` then `./scripts/rename_ckpts_to_convention.sh`.
- See `configs/test_matrix_full.md` for the full test matrix.
