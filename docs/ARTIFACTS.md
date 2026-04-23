# 模型产物目录组织（TRT op ckpt / TRT-LLM ckpt / TRT engine）

项目中可能同时存在三类产物，建议按下面方式区分存放，便于脚本和环境变量统一指向。

---

## 三类产物说明

| 类型 | 来源 | 内容 | 典型用途 |
|------|------|------|----------|
| **PTQ ckpt（HF 风格）** | TensorRT Model Optimizer / TRT op 量化 | HF 格式：`config.json`、`*.safetensors`、`hf_quant_config.json` 等 | trtllm-bench **PyTorch 后端**（`--model_path`）、TRT-LLM `convert_checkpoint.py` 的**输入** |
| **TRT-LLM ckpt（原生）** | TRT-LLM `convert_checkpoint.py` 从 PTQ ckpt 转换 | TRT-LLM 原生目录：`config.json`、权重分片等（非 HF layout） | TRT-LLM **build** 的输入、部分推理工具直接读 ckpt |
| **TRT engine** | TRT-LLM `trtllm-build` 从 TRT-LLM ckpt 构建 | `.engine`、`config.json` 等 | `trtllm-run` / `trtllm-serve` 等** TensorRT 后端**推理 |

---

## 推荐目录结构

在 **outputs/** 下按类型分子目录，命名一致、便于用环境变量切换：

```
outputs/
├── ckpts/                    # PTQ ckpt（HF 风格），默认 CKPT_ROOT
│   ├── llama-3.1-8b-instruct-trtllm-ckpt-wq_fp8-kv_fp16/
│   ├── llama-3.1-8b-instruct-trtllm-ckpt-wq_fp8-kv_fp8/
│   ├── ministral-8b-instruct-2410-trtllm-ckpt-wq_fp8-kv_fp16/
│   └── ...
├── ckpts_trtllm/             # TRT-LLM 原生 ckpt（convert 后），CKPT_TRTLLM_ROOT
│   ├── llama-3.1-8b-instruct-trtllm-ckpt-wq_fp8-kv_fp16/
│   └── ...
├── engines/                  # 构建好的 TRT engine，ENGINE_ROOT
│   ├── llama-3.1-8b-instruct-wq_fp8-kv_fp16-tp1/    # 建议带 tp/pp 等配置
│   └── ...
├── bench_15k1/
├── results_trtllm1.1.0_15k1_bs1_pytorch/
└── ...
```

- **ckpts/**：当前脚本已在用（`generate_quant_ckpt.sh` → `ROOT_SAVE_PATH`，bench 用 `CKPT_ROOT`），保持为 **PTQ/HF 风格 ckpt** 根目录。
- **ckpts_trtllm/**：仅当你要跑 **convert_checkpoint → build → run** 时使用；convert 输出可统一放到这里，便于与 PTQ 区分。
- **engines/**：所有 `trtllm-build` 产出的 engine 目录放这里，子目录名建议包含模型+量化+tp（如 `llama-3.1-8b-wq_fp8-kv_fp16-tp1`）。

**另：models/ckpts_trtllm/**（供 TensorRT-LLM 0.18.0）

- 用于存放 **从 HF 模型直接 convert** 得到的 TRT-LLM 原生 ckpt（如 `convert_checkpoint.py --use_fp8 --fp8_kv_cache`），与 PTQ 流程无关。
- 这些 ckpt 作为 `trtllm-build` 的输入，构建出的 engine 供 **TensorRT-LLM 0.18.0** 使用。
- **命名与上表一致**：`<model>-trtllm-ckpt-wq_<quant>-kv_<kv>`（同 outputs/ckpts、rename_ckpts_to_convention.sh）。
- 脚本：`scripts/convert_hf_to_trtllm_ckpt.sh`，默认输出到 `models/ckpts_trtllm/`；详见 `models/ckpts_trtllm/README.md`。

---

## 环境变量约定

| 变量 | 默认路径 | 含义 |
|------|----------|------|
| `CKPT_ROOT` | `outputs/ckpts` | PTQ/HF 风格 ckpt 根目录（trtllm-bench PyTorch、convert 输入） |
| `CKPT_TRTLLM_ROOT` | `outputs/ckpts_trtllm` | TRT-LLM 原生 ckpt 根目录（build 输入） |
| `ENGINE_ROOT` | `outputs/engines` | TRT engine 根目录（trtllm-run / trtllm-serve） |

脚本约定：

- 所有 **PTQ 产出** 和 **基于 PTQ 的 bench**（如 `run_all_trtllm_bench_15k1_bs1_pytorch.sh`）只读 **CKPT_ROOT**，不读 ckpts_trtllm/engines。
- 若你写 **convert / build / serve** 脚本，从 `CKPT_ROOT` 读 PTQ ckpt，convert 结果写到 `CKPT_TRTLLM_ROOT`，build 结果写到 `ENGINE_ROOT`。

---

## 命名建议（与现有一致）

- **PTQ ckpt 目录名**（已在用）：`<model>-trtllm-ckpt-wq_<quant>-kv_<kv>`  
  例：`llama-3.1-8b-instruct-trtllm-ckpt-wq_fp8-kv_fp16`、`ministral-8b-instruct-2410-trtllm-ckpt-wq_int4_awq-kv_fp8`。
- **TRT-LLM 原生 ckpt**：可与 PTQ 同名，便于一一对应；或加后缀如 `-native`（按你习惯）。
- **Engine 目录**：建议包含 tp/pp，如 `llama-3.1-8b-wq_fp8-kv_fp16-tp1`，避免同一模型多配置覆盖。

---

## 流程对应关系

```
PTQ (Model Optimizer)     →  outputs/ckpts/<name>/          [CKPT_ROOT]
       ↓
convert_checkpoint.py     →  outputs/ckpts_trtllm/<name>/   [CKPT_TRTLLM_ROOT]
       ↓
trtllm-build              →  outputs/engines/<name>-tp1/    [ENGINE_ROOT]
       ↓
trtllm-run / trtllm-serve  读 ENGINE_ROOT
```

trtllm-bench **PyTorch 后端** 直接用 **CKPT_ROOT** 下的 PTQ ckpt，不经过 convert/build。

---

## 在其他机器使用

- **只跑 PyTorch 后端 bench**：只拷或下载 **outputs/ckpts/**（PTQ ckpt），设 `CKPT_ROOT` 即可。
- **要跑 TensorRT 后端**：需在同一机器上从 PTQ ckpt 做 convert → build，或拷 **ckpts_trtllm + engines**，并设 `CKPT_TRTLLM_ROOT`、`ENGINE_ROOT`。  
详见 `docs/CKPT_TRANSFER.md` 的打包与上传方式。
