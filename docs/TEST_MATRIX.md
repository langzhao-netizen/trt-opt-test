# 完整测试矩阵：所有要测的 case

三种模型 × 五类 GPU × 权重量化 × KV Cache，只列「可能最优」的 case。每个 case 一行。

---

## 必测 case（30 个）

| # | 模型 | GPU | 权重量化 (Weight) | KV Cache | 说明 |
|---|------|-----|-------------------|----------|------|
| 1 | Llama-3.2-3B | A100 | W4A16 AWQ | FP16 | Ampere 最佳权重量化，KV 不量化 |
| 2 | Llama-3.2-3B | A100 | W4A16 AWQ | FP8 | Ampere 最佳权重量化，KV 量化 |
| 3 | Llama-3.2-3B | A10 | W4A16 AWQ | FP16 | |
| 4 | Llama-3.2-3B | A10 | W4A16 AWQ | FP8 | |
| 5 | Llama-3.2-3B | H100 | FP8 | FP16 | Hopper/Ada 最佳权重量化，KV 不量化 |
| 6 | Llama-3.2-3B | H100 | FP8 | FP8 | Hopper/Ada 最佳权重量化，KV 量化 |
| 7 | Llama-3.2-3B | L4 | FP8 | FP16 | |
| 8 | Llama-3.2-3B | L4 | FP8 | FP8 | |
| 9 | Llama-3.2-3B | L40 | FP8 | FP16 | |
| 10 | Llama-3.2-3B | L40 | FP8 | FP8 | |
| 11 | Llama-3.1-8B | A100 | W4A16 AWQ | FP16 | |
| 12 | Llama-3.1-8B | A100 | W4A16 AWQ | FP8 | |
| 13 | Llama-3.1-8B | A10 | W4A16 AWQ | FP16 | |
| 14 | Llama-3.1-8B | A10 | W4A16 AWQ | FP8 | |
| 15 | Llama-3.1-8B | H100 | FP8 | FP16 | |
| 16 | Llama-3.1-8B | H100 | FP8 | FP8 | |
| 17 | Llama-3.1-8B | L4 | FP8 | FP16 | |
| 18 | Llama-3.1-8B | L4 | FP8 | FP8 | |
| 19 | Llama-3.1-8B | L40 | FP8 | FP16 | |
| 20 | Llama-3.1-8B | L40 | FP8 | FP8 | |
| 21 | Mistral-7B | A100 | W4A16 AWQ | FP16 | |
| 22 | Mistral-7B | A100 | W4A16 AWQ | FP8 | |
| 23 | Mistral-7B | A10 | W4A16 AWQ | FP16 | |
| 24 | Mistral-7B | A10 | W4A16 AWQ | FP8 | |
| 25 | Mistral-7B | H100 | FP8 | FP16 | |
| 26 | Mistral-7B | H100 | FP8 | FP8 | |
| 27 | Mistral-7B | L4 | FP8 | FP16 | |
| 28 | Mistral-7B | L4 | FP8 | FP8 | |
| 29 | Mistral-7B | L40 | FP8 | FP16 | |
| 30 | Mistral-7B | L40 | FP8 | FP8 | |

---

## 可选 baseline（3 个，做精度/性能参考）

| # | 模型 | GPU | 权重量化 (Weight) | KV Cache | 说明 |
|---|------|-----|-------------------|----------|------|
| 31 | Llama-3.2-3B | H100 | FP16 | FP16 | 无量化基线 |
| 32 | Llama-3.1-8B | H100 | FP16 | FP16 | 无量化基线 |
| 33 | Mistral-7B | H100 | FP16 | FP16 | 无量化基线 |

---

## 可选：A 系列 INT8（SmoothQuant W8A8）

A100/A10 无 FP8 权重量化，若需对比「8bit 权+激活」可加测 INT8 SmoothQuant。**注意**：Model Optimizer 的 `int8_sq` 对 LLaMA 3.x / Mistral 标为 ❌，产 INT8 ckpt 需走 **TRT-LLM SmoothQuant** 或其它工具链；产好后可按下表加测。

| # | 模型 | GPU | 权重量化 (Weight) | KV Cache |
|---|------|-----|-------------------|----------|
| 34 | Llama-3.2-3B | A100 | INT8 SmoothQuant (W8A8) | FP16 |
| 35 | Llama-3.2-3B | A100 | INT8 SmoothQuant (W8A8) | FP8 |
| 36 | Llama-3.2-3B | A10 | INT8 SmoothQuant (W8A8) | FP16 |
| 37 | Llama-3.2-3B | A10 | INT8 SmoothQuant (W8A8) | FP8 |
| 38 | Llama-3.1-8B | A100 | INT8 SmoothQuant (W8A8) | FP16 |
| 39 | Llama-3.1-8B | A100 | INT8 SmoothQuant (W8A8) | FP8 |
| 40 | Llama-3.1-8B | A10 | INT8 SmoothQuant (W8A8) | FP16 |
| 41 | Llama-3.1-8B | A10 | INT8 SmoothQuant (W8A8) | FP8 |
| 42 | Mistral-7B | A100 | INT8 SmoothQuant (W8A8) | FP16 |
| 43 | Mistral-7B | A100 | INT8 SmoothQuant (W8A8) | FP8 |
| 44 | Mistral-7B | A10 | INT8 SmoothQuant (W8A8) | FP16 |
| 45 | Mistral-7B | A10 | INT8 SmoothQuant (W8A8) | FP8 |

**合计**：12 个 case（3 模型 × 2 GPU × 2 KV）。需单独产出 INT8 权重量化 ckpt（TRT-LLM SmoothQuant 流程），再在 A100/A10 上按上表测。

---

## 合计

- **必测**：30 个 case
- **可选 baseline**：3 个 case
- **可选 A 系列 INT8**：12 个 case
- **总计（必测 + baseline）**：33 个 case
- **总计（含 A 系列 INT8）**：45 个 case

---

## 需要产出的 checkpoint（权重量化，先产再测）

| 模型 | 格式 | 用途 | 目录示例（统一命名） |
|------|------|------|----------------------|
| Llama-3.2-3B | fp8 | H100/L4/L40 | `llama-3.2-3b-instruct-trtllm-ckpt-wq_fp8-kv_fp16` |
| Llama-3.2-3B | int4_awq (W4A16 AWQ) | A100/A10/H100/L4/L40 | `llama-3.2-3b-instruct-trtllm-ckpt-wq_int4_awq-kv_fp16` |
| Llama-3.1-8B | fp8 | H100/L4/L40 | `llama-3.1-8b-instruct-trtllm-ckpt-wq_fp8-kv_fp16` |
| Llama-3.1-8B | int4_awq | 五类卡 | `llama-3.1-8b-instruct-trtllm-ckpt-wq_int4_awq-kv_fp16` |
| Mistral-7B | fp8 | H100/L4/L40 | `mistral-7b-instruct-v0.3-trtllm-ckpt-wq_fp8-kv_fp16` |
| Mistral-7B | int4_awq | 五类卡 | `mistral-7b-instruct-v0.3-trtllm-ckpt-wq_int4_awq-kv_fp16` |

**KV Cache：产 ckpt 时定 vs 运行时定（TRT-LLM 1.1.0）**  
- **TensorRT 后端**（`convert_checkpoint.py` → build → run）：`TrtLlmArgs` 里规定 `kv_cache_config.dtype` 必须为 `"auto"`，**不支持**运行时改 KV 精度。KV 行为由 **convert 时** 写进 ckpt 的 `quant_mode`（如 `--int8_kv_cache`）决定。要测 FP16 KV 与 INT8/FP8 KV，需 **产两份 ckpt**（一次不带 `--int8_kv_cache`，一次带）。  
- **PyTorch 后端**（LLM API，不 build engine）：`kv_cache_config.dtype` 可在 **运行时** 传入（如 `KvCacheConfig(dtype='fp8')` 或 `'auto'`）。同一份权重量化 ckpt，跑两次（dtype 不同）即可，**不需产两份 ckpt**。  
因此：用 **convert_checkpoint + TensorRT 跑** 时，两种方式**不一样**（KV 只能在 convert 时定）；用 **PyTorch 后端** 时，只需运行时定 KV，一份 ckpt 即可。

**若加测 A 系列 INT8**：需用 TRT-LLM SmoothQuant 流程为每个模型产出 INT8（W8A8）ckpt，目录示例：`*_int8_sq` 或 `*_w8a8`，再在 A100/A10 上测上表「可选：A 系列 INT8」的 12 个 case。

---

## 按 (模型, GPU) 索引：每个组合要测的 case

| 模型 | GPU | Case 1 | Case 2 |
|------|-----|--------|--------|
| Llama-3.2-3B | A100 | W4A16 AWQ + FP16 KV | W4A16 AWQ + FP8 KV |
| Llama-3.2-3B | A10 | W4A16 AWQ + FP16 KV | W4A16 AWQ + FP8 KV |
| Llama-3.2-3B | H100 | FP8 + FP16 KV | FP8 + FP8 KV |
| Llama-3.2-3B | L4 | FP8 + FP16 KV | FP8 + FP8 KV |
| Llama-3.2-3B | L40 | FP8 + FP16 KV | FP8 + FP8 KV |
| Llama-3.1-8B | A100 | W4A16 AWQ + FP16 KV | W4A16 AWQ + FP8 KV |
| Llama-3.1-8B | A10 | W4A16 AWQ + FP16 KV | W4A16 AWQ + FP8 KV |
| Llama-3.1-8B | H100 | FP8 + FP16 KV | FP8 + FP8 KV |
| Llama-3.1-8B | L4 | FP8 + FP16 KV | FP8 + FP8 KV |
| Llama-3.1-8B | L40 | FP8 + FP16 KV | FP8 + FP8 KV |
| Mistral-7B | A100 | W4A16 AWQ + FP16 KV | W4A16 AWQ + FP8 KV |
| Mistral-7B | A10 | W4A16 AWQ + FP16 KV | W4A16 AWQ + FP8 KV |
| Mistral-7B | H100 | FP8 + FP16 KV | FP8 + FP8 KV |
| Mistral-7B | L4 | FP8 + FP16 KV | FP8 + FP8 KV |
| Mistral-7B | L40 | FP8 + FP16 KV | FP8 + FP8 KV |

**可选 A 系列 INT8**（需先产 INT8 ckpt）：A100 / A10 每个组合再加 INT8 SmoothQuant + FP16 KV、INT8 SmoothQuant + FP8 KV（见上文「可选：A 系列 INT8」表）。
