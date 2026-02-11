# 在新 VM 上复现环境（测试不同模型 / 量化 / ckpt vs engine）

用于在另一台机器上搭建与当前项目**完全一致**的环境，以便测试：不同模型、不同量化、ckpt 与 engine 在不同机器上的效果。

---

## 1. 系统与依赖（Prereqs）

| 项目 | 说明 |
|------|------|
| 系统 | Linux（推荐 Ubuntu 20.04/22.04） |
| Python | 3.10+，`python3 -m venv` 可用 |
| CUDA | 与 TensorRT / TensorRT-LLM 版本匹配（如 CUDA 12.x） |
| TensorRT | 若需 build/serve engine，需安装对应 TensorRT |
| 磁盘 | 至少 ~50GB 可用（克隆依赖 + venv + 可选本地模型） |
| 网络 | 首次需能访问 GitHub、HuggingFace、pip |

---

## 2. 克隆与一键环境

```bash
git clone <本仓库 URL> trt-opt-test && cd trt-opt-test
./scripts/setup.sh
```

- 默认会克隆 **TensorRT-Model-Optimizer**、**TensorRT-LLM v0.18.0**、**TensorRT-LLM v1.1.0**，并创建 `venv_modelopt`、`venv_trtllm0.18.0`、`venv_trtllm1.1.0`。
- 若不需要 1.1.0（省磁盘）：`./scripts/setup.sh --no-trtllm-1.1.0`。

完成后无需手动激活 venv 即可跑 PTQ（脚本会自动使用 `venv_modelopt`）。

---

## 3. 环境变量（可选但建议统一）

| 变量 | 含义 | 默认 / 说明 |
|------|------|--------------|
| `ROOT_SAVE_PATH` / `CKPT_ROOT` | 量化 ckpt 输出根目录 | `$PROJECT_ROOT/outputs/ckpts` |
| `BENCH_DATASET` | benchmark 用 JSONL 路径 | `outputs/bench_<name>/dataset_15k_1.jsonl` |
| `TARGET_INPUT_TOKENS` | 生成 dataset 时目标 input 长度 | `15360` |
| `NUM_REQUESTS` | 生成 dataset 时请求条数 | `5` |
| `TOKENIZER_HF_ID` | `gen_bench_dataset.py` 使用的 tokenizer（HF id） | `meta-llama/Llama-3.2-3B-Instruct` |
| `TOKENIZER_PATH` | 若设，则用本地目录作为 tokenizer（覆盖 HF id） | 未设则用 `TOKENIZER_HF_ID` |
| `TRUST_REMOTE_CODE` | PTQ 脚本传参 | `--trust_remote_code` |
| `TASKS` | Model Optimizer 任务 | `quant` |
| `HF_TOKEN` | HuggingFace 读 gated 模型时需设置 | 无则 gated 模型会报错 |

建议在新 VM 上若使用 gated 模型，在 `~/.bashrc` 或当前 shell 中 `export HF_TOKEN=...`。

---

## 4. 模型来源：HF id vs 本地

- **推荐**：直接使用 HuggingFace 模型 id（如 `meta-llama/Llama-3.2-3B-Instruct`）。PTQ 脚本会按 id 下载；需在 HF 网站同意协议，并设置 `HF_TOKEN`。
- **可选**：大模型放本地以省流量、离线：将 HF 模型下载到 `shared/models/<模型目录名>`，PTQ 时 `--model` 传本地路径，例如 `shared/models/Llama-3.2-3B-Instruct`。注意 `shared/models/` 已 gitignore，不会随仓库同步，需在新 VM 自行拷贝或再下载。

---

## 5. 不需要从本机带过去的东西（可清理 / 不同步）

- **outputs/**：整个目录 gitignore，新 VM 上应为空；由 `setup.sh` 创建 `outputs/ckpts`、`outputs/calib_data_tllm018`，其余由各脚本按需生成（ckpt、engine、bench 结果等）。**无需**从旧机器拷贝 outputs。
- **venv_modelopt / venv_trtllm***：gitignore，在新 VM 上由 `setup.sh` 重新创建即可。
- **TensorRT-LLM / TensorRT-Model-Optimizer**：gitignore，由 `setup.sh` 克隆。
- **shared/models/**：gitignore（约 69GB），新 VM 要么用 HF id 在线拉取，要么单独拷贝/下载到 `shared/models/`。

---

## 6. 需要补充 / 在新 VM 上要做的

1. **HuggingFace**：若用 gated 模型，登录 HF 并设置 `HF_TOKEN`。
2. **Benchmark dataset**：若跑 bench，需先生成 dataset，例如：
   ```bash
   python3 scripts/gen_bench_dataset.py
   ```
   默认使用 HF id 取 tokenizer，无需本地模型。若系统 Python 未安装 `transformers`，请先激活 venv：`source venv_modelopt/bin/activate` 再执行。也可 `TOKENIZER_PATH=shared/models/Llama-3.2-3B-Instruct python3 scripts/gen_bench_dataset.py`。
3. **精度评测**：`scripts/run_llama8b_accuracy.py` 与 `inference.ipynb` 针对 **FP16 基线** 或可加载的 HF 格式 ckpt。运行 `run_llama8b_accuracy.py` 时若系统缺少依赖（datasets、transformers、torch、scipy、sklearn 等），请先 `source venv_modelopt/bin/activate`。当前 Model Optimizer 产出的 **FP8/INT4 量化 ckpt** 用 `AutoModelForCausalLM.from_pretrained` 会因 dtype 不匹配报错；量化模型的精度建议通过 **trtllm-serve + 请求脚本** 对同一数据集请求再算指标。详见 README 中「Accuracy」说明。

---

## 7. 建议的复现流程（新 VM）

1. 克隆仓库 → `./scripts/setup.sh`（可选 `--no-trtllm-1.1.0`）。
2. 设置 `HF_TOKEN`（若用 gated 模型）。
3. 跑 PTQ：`./scripts/run_all_ptq.sh` → `./scripts/rename_ckpts_to_convention.sh`。
4. （可选）生成 bench 用 dataset：`python3 scripts/gen_bench_dataset.py`。
5. 按需在对应 venv 下做 TensorRT-LLM convert / build / serve 与 bench，或跑 FP16 基线精度（`inference.ipynb` / `run_llama8b_accuracy.py` 仅适用于非量化或可加载的 ckpt）。

按此流程，新 VM 上可得到与当前项目一致、可复现的测试环境。
