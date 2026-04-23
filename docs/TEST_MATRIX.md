# Full Test Matrix

Three models × five GPU types × weight quantization × KV Cache. Only the likely-optimal cases per combination. One case per row.

---

## Required Cases (30)

| # | Model | GPU | Weight Quantization | KV Cache | Notes |
|---|-------|-----|---------------------|----------|-------|
| 1 | Llama-3.2-3B | A100 | W4A16 AWQ | FP16 | Best weight quant on Ampere, no KV quant |
| 2 | Llama-3.2-3B | A100 | W4A16 AWQ | FP8 | Best weight quant on Ampere, KV quantized |
| 3 | Llama-3.2-3B | A10 | W4A16 AWQ | FP16 | |
| 4 | Llama-3.2-3B | A10 | W4A16 AWQ | FP8 | |
| 5 | Llama-3.2-3B | H100 | FP8 | FP16 | Best weight quant on Hopper/Ada, no KV quant |
| 6 | Llama-3.2-3B | H100 | FP8 | FP8 | Best weight quant on Hopper/Ada, KV quantized |
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

## Optional Baselines (3, for accuracy/perf reference)

| # | Model | GPU | Weight Quantization | KV Cache | Notes |
|---|-------|-----|---------------------|----------|-------|
| 31 | Llama-3.2-3B | H100 | FP16 | FP16 | Unquantized baseline |
| 32 | Llama-3.1-8B | H100 | FP16 | FP16 | Unquantized baseline |
| 33 | Mistral-7B | H100 | FP16 | FP16 | Unquantized baseline |

---

## Optional: A-series INT8 (SmoothQuant W8A8)

A100/A10 do not support FP8 weight quantization. If comparing "8-bit weights + activations" is needed, add INT8 SmoothQuant cases. **Note**: Model Optimizer's `int8_sq` is marked ❌ for LLaMA 3.x / Mistral — producing INT8 ckpts requires the **TRT-LLM SmoothQuant** pipeline or another toolchain. Once produced, add the following cases:

| # | Model | GPU | Weight Quantization | KV Cache |
|---|-------|-----|---------------------|----------|
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

**Total**: 12 cases (3 models × 2 GPUs × 2 KV). Requires producing INT8 weight ckpts via TRT-LLM SmoothQuant, then benchmarking on A100/A10.

---

## Summary

- **Required**: 30 cases
- **Optional baselines**: 3 cases
- **Optional A-series INT8**: 12 cases
- **Total (required + baselines)**: 33 cases
- **Total (including A-series INT8)**: 45 cases

---

## Required Checkpoints (produce before benchmarking)

| Model | Format | GPU targets | Directory example (naming convention) |
|-------|--------|-------------|---------------------------------------|
| Llama-3.2-3B | fp8 | H100/L4/L40 | `llama-3.2-3b-instruct-trtllm-ckpt-wq_fp8-kv_fp16` |
| Llama-3.2-3B | int4_awq (W4A16 AWQ) | A100/A10/H100/L4/L40 | `llama-3.2-3b-instruct-trtllm-ckpt-wq_int4_awq-kv_fp16` |
| Llama-3.1-8B | fp8 | H100/L4/L40 | `llama-3.1-8b-instruct-trtllm-ckpt-wq_fp8-kv_fp16` |
| Llama-3.1-8B | int4_awq | All 5 GPU types | `llama-3.1-8b-instruct-trtllm-ckpt-wq_int4_awq-kv_fp16` |
| Mistral-7B | fp8 | H100/L4/L40 | `mistral-7b-instruct-v0.3-trtllm-ckpt-wq_fp8-kv_fp16` |
| Mistral-7B | int4_awq | All 5 GPU types | `mistral-7b-instruct-v0.3-trtllm-ckpt-wq_int4_awq-kv_fp16` |

**KV Cache: set at ckpt-produce time vs. runtime (TRT-LLM 1.1.0)**
- **TensorRT backend** (`convert_checkpoint.py` → build → run): `TrtLlmArgs` requires `kv_cache_config.dtype` to be `"auto"`. KV precision is fixed at **convert time** via `quant_mode` (e.g. `--int8_kv_cache`). Testing FP16 KV vs INT8/FP8 KV requires **two separate ckpts** (one pass without `--int8_kv_cache`, one with).
- **PyTorch backend** (LLM API, no engine build): `kv_cache_config.dtype` is passed at **runtime** (e.g. `KvCacheConfig(dtype='fp8')` or `'auto'`). One weight-quantized ckpt can cover both KV modes in two runs — **no need to produce two ckpts**.

**If adding A-series INT8**: use the TRT-LLM SmoothQuant pipeline to produce INT8 (W8A8) ckpts per model (e.g. `*_int8_sq` or `*_w8a8`), then run the 12 optional cases on A100/A10.

---

## By (Model, GPU): Cases per combination

| Model | GPU | Case 1 | Case 2 |
|-------|-----|--------|--------|
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

**Optional A-series INT8** (requires INT8 ckpts first): for each A100/A10 combination, add INT8 SmoothQuant + FP16 KV and INT8 SmoothQuant + FP8 KV (see the Optional A-series INT8 table above).
