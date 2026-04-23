# 在新 VM 上复现环境

## 1. 系统依赖

| 项目 | 说明 |
|------|------|
| 系统 | Linux（Ubuntu 20.04/22.04） |
| Python | 3.10+，`python3 -m venv` 可用 |
| CUDA | 与 TensorRT-LLM 版本匹配（CUDA 12.x） |
| 磁盘 | ~50GB（依赖克隆 + venv + 可选本地模型） |
| 网络 | 首次需访问 GitHub、HuggingFace、pip |

---

## 2. 一键部署

```bash
git clone <repo-url> trt-opt-test && cd trt-opt-test
./scripts/setup.sh
```

默认克隆 TensorRT-Model-Optimizer + TRT-LLM 1.1.0 + 1.2.0，创建 `venv_modelopt`、`venv_trtllm1.1.0`、`venv_trtllm1.2.0`，以及 `outputs/`、`models/` 目录。

```bash
./scripts/setup.sh --no-trtllm-1.1.0   # 跳过 1.1.0（省磁盘）
./scripts/setup.sh --no-trtllm-1.2.0   # 跳过 1.2.0
```

---

## 3. 环境变量

| 变量 | 默认 | 含义 |
|------|------|------|
| `CKPT_ROOT` | `outputs/ckpts` | PTQ ckpt 根目录 |
| `TARGET_INPUT_TOKENS` | `15360` | bench dataset input 长度 |
| `NUM_REQUESTS` | `5` | 每个 ckpt 请求数 |
| `TOKENIZER_HF_ID` | `meta-llama/Llama-3.2-3B-Instruct` | gen_dataset.py 使用的 tokenizer |
| `TOKENIZER_PATH` | (unset) | 本地 tokenizer 路径，覆盖 HF id |
| `HF_TOKEN` | (unset) | gated 模型必须设置 |

---

## 4. 模型来源

- **HF id**（推荐）：直接传 `meta-llama/Llama-3.1-8B-Instruct` 等，脚本自动下载。需 HF 账号同意协议 + `HF_TOKEN`。
- **本地**：下载到 `models/<model-name>/`（gitignored），传本地路径给 `--model`。

---

## 5. 不需要从旧机器带的东西

- `outputs/`：全部 gitignored，新 VM 上为空，由脚本按需创建。
- `venv_*/`：gitignored，`setup.sh` 重新创建。
- `TensorRT-LLM-*/`、`tools/`：gitignored，`setup.sh` 克隆。
- `models/`：gitignored，新 VM 用 HF id 在线拉或单独拷贝。

---

## 6. 复现流程

```bash
# 1. 克隆 + 部署
git clone <repo-url> trt-opt-test && cd trt-opt-test
./scripts/setup.sh

# 2. 设置 HF_TOKEN（gated 模型）
export HF_TOKEN=hf_xxx

# 3. PTQ
./scripts/ptq/run_ptq_all.sh

# 4. Benchmark
./scripts/bench/run_bench.sh

# 5. 精度评估（仅 FP16 baseline）
jupyter notebook notebooks/toxicity_eval.ipynb
```

> FP8/INT4 量化 ckpt 无法用 `AutoModelForCausalLM.from_pretrained` 加载，精度测试需通过 trtllm-serve + 客户端脚本。
