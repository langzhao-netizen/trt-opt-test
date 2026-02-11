# trt-opt-test

Test matrix and scripts for TensorRT-LLM / Model Optimizer quantization and benchmarking.

## Layout

- **configs/** – Test matrix and config (e.g. `test_matrix_full.md`).
- **scripts/** – PTQ and TRT-LLM checkpoint scripts:
  - `generate_quant_ckpt.sh` – Single Model Optimizer PTQ run.
  - `run_all_ptq.sh` – Batch PTQ over test matrix.
  - `rename_ckpts_to_convention.sh` – Rename `saved_models_*` to convention names.
  - `regenerate_kv_fp16_ckpts.sh` – Regenerate only *-kv_fp16 ckpts (PTQ with `--kv_cache_quant none`).
  - `gen_bench_dataset.py` – Benchmark dataset generation (uses `TOKENIZER_HF_ID` or `TOKENIZER_PATH`).
  - `run_llama8b_accuracy.py` – Accuracy eval on Llama 8B ckpts (see *Accuracy* below).
- **outputs/** – Generated checkpoints and engines (gitignored). On a new VM this dir is empty; `setup.sh` creates `outputs/ckpts` and `outputs/calib_data_tllm018`; other subdirs (e.g. `bench_*`, `engines_*`) are created by scripts as needed.
- **docs/** – `SETUP_VM.md` has a full checklist for replicating this environment on another VM.
- **shared/** – Tracked except `shared/models/` (gitignored, ~69GB). Use `shared/data/`, `shared/utils/` for small shared files; put local HF model copies in `shared/models/` or use HF model IDs in scripts.
- **tools/** – Clone TensorRT-Model-Optimizer here (gitignored; created by `setup.sh`).

### Version control (what is committed vs ignored)

| 上传 (tracked) | 不上传 (ignored / not committed) |
|----------------|----------------------------------|
| 代码与配置：`.gitignore`, `README.md`, `configs/`, `scripts/*.sh` 与 `scripts/*.py`, `shared/README.md`, `shared/data/.gitkeep`, `shared/utils/.gitkeep`, `docs/` | **outputs/**（ckpt、engine、bench 结果、日志）；**venv/**；**TensorRT-LLM/**、**tools/TensorRT-Model-Optimizer/**；**shared/models/**（~69GB）；`.env`、`.cursor/`、`.ipynb_checkpoints/`、`*.engine`、`*.log`、`model.cache` |
| 新 VM 只需 clone + `./scripts/setup.sh` 即可复现环境；不依赖本机生成物 | 大体积与生成物不进入仓库，避免仓库膨胀和机器相关差异 |

**为保障其他 VM 一键部署完整：** 请将以下文件一并提交（若尚未跟踪）：`docs/SETUP_VM.md`、`scripts/check_ckpt_quant_config.py`、`scripts/regenerate_kv_fp16_ckpts.sh`、`scripts/run_llama8b_accuracy.py`。`run_all_ptq.sh` 与 `regenerate_kv_fp16_ckpts.sh` 会调用 `check_ckpt_quant_config.py` 做校验；若该文件未提交，批量 PTQ 仍可跑完，但会跳过校验步骤。

## Quick setup on a new machine（一键部署）

**前置：** Linux、Python 3、git；若需 build/serve engine 则需 CUDA 与 TensorRT。

**一键部署（仅需两条命令）：**

```bash
git clone <本仓库-URL> trt-opt-test && cd trt-opt-test
./scripts/setup.sh
```

执行后：自动克隆 TensorRT-Model-Optimizer、TensorRT-LLM，创建 3 个 venv 与 `outputs/ckpts`、`outputs/calib_data_tllm018`。**无需**从本机拷贝 outputs 或模型，新 VM 上可直接跑 PTQ（见下）。

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

**Environment variables (optional):** `ROOT_SAVE_PATH` / `CKPT_ROOT` (ckpt output root), `BENCH_DATASET`, `TOKENIZER_HF_ID` or `TOKENIZER_PATH` (for `gen_bench_dataset.py`), `TARGET_INPUT_TOKENS`, `NUM_REQUESTS`. For gated HuggingFace models set `HF_TOKEN`. Full list: `docs/SETUP_VM.md`.

**Replicating on another VM:** See **`docs/SETUP_VM.md`** for a step-by-step checklist (prereqs, env vars, what to copy vs recreate, accuracy workflow).

### Accuracy and notebooks

- **`run_llama8b_accuracy.py`** and **`inference.ipynb`** load checkpoints with `AutoModelForCausalLM.from_pretrained`. They work for **FP16 (unquantized) baselines**. The **quantized ckpts** (FP8 / INT4 AWQ) produced by Model Optimizer use dtypes (e.g. Float8) that cause PyTorch loading to fail; for accuracy on those, use **trtllm-serve** and a small client script to run the same eval dataset and compute metrics.
- **`inference.ipynb`** is for interactive toxicity eval; it can use an optional `llm-finetuning` clone for `PromptTemplate`, or a built-in minimal template.
