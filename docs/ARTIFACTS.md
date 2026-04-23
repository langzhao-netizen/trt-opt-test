# Model Artifact Directory Layout (PTQ ckpt / TRT-LLM ckpt / TRT engine)

The project may contain three distinct artifact types. Keep them in separate directories so scripts and environment variables point to the right one.

---

## Artifact Types

| Type | Source | Contents | Typical Use |
|------|--------|----------|-------------|
| **PTQ ckpt (HF-format)** | TensorRT Model Optimizer / TRT-op quantization | HF layout: `config.json`, `*.safetensors`, `hf_quant_config.json`, etc. | trtllm-bench **PyTorch backend** (`--model_path`); input to TRT-LLM `convert_checkpoint.py` |
| **TRT-LLM native ckpt** | TRT-LLM `convert_checkpoint.py` converting from PTQ ckpt | TRT-LLM native layout: `config.json`, weight shards (not HF layout) | Input to TRT-LLM **build**; some inference tools read native ckpts directly |
| **TRT engine** | TRT-LLM `trtllm-build` from TRT-LLM native ckpt | `.engine`, `config.json`, etc. | `trtllm-run` / `trtllm-serve` **TensorRT backend** inference |

---

## Recommended Directory Layout

Under **outputs/**, separate by type with consistent names and environment variables:

```
outputs/
├── ckpts/                    # PTQ ckpts (HF-format), default CKPT_ROOT
│   ├── llama-3.1-8b-instruct-trtllm-ckpt-wq_fp8-kv_fp16/
│   ├── llama-3.1-8b-instruct-trtllm-ckpt-wq_fp8-kv_fp8/
│   ├── ministral-8b-instruct-2410-trtllm-ckpt-wq_fp8-kv_fp16/
│   └── ...
├── ckpts_trtllm/             # TRT-LLM native ckpts (after convert), CKPT_TRTLLM_ROOT
│   ├── llama-3.1-8b-instruct-trtllm-ckpt-wq_fp8-kv_fp16/
│   └── ...
├── engines/                  # Built TRT engines, ENGINE_ROOT
│   ├── llama-3.1-8b-instruct-wq_fp8-kv_fp16-tp1/    # include tp/pp in name
│   └── ...
├── bench_15k1/
├── results_trtllm1.1.0_15k1_bs1_pytorch/
└── ...
```

- **ckpts/**: used by current scripts (`scripts/ptq/run_ptq_single.sh` → `ROOT_SAVE_PATH`; bench via `CKPT_ROOT`). Keep as the **PTQ/HF-format ckpt** root.
- **ckpts_trtllm/**: only needed when running **convert_checkpoint → build → run**. Write convert output here to keep it separate from PTQ ckpts.
- **engines/**: all `trtllm-build` output goes here. Include model + quant + tp in the subdirectory name (e.g. `llama-3.1-8b-wq_fp8-kv_fp16-tp1`) to avoid collisions.

- **Naming matches the table above**: `<model>-trtllm-ckpt-wq_<quant>-kv_<kv>` (same as `outputs/ckpts/` and `scripts/ptq/rename_ckpts.sh`).

---

## Environment Variable Conventions

| Variable | Default path | Meaning |
|----------|-------------|---------|
| `CKPT_ROOT` | `outputs/ckpts` | PTQ/HF-format ckpt root (trtllm-bench PyTorch, convert input) |
| `CKPT_TRTLLM_ROOT` | `outputs/ckpts_trtllm` | TRT-LLM native ckpt root (trtllm-build input) |
| `ENGINE_ROOT` | `outputs/engines` | TRT engine root (trtllm-run / trtllm-serve) |

Script conventions:

- All **PTQ output** and **PTQ-based bench** (e.g. `scripts/bench/run_bench.sh`) only read **CKPT_ROOT**; they do not touch ckpts_trtllm/ or engines/.
- If writing **convert / build / serve** scripts: read PTQ ckpts from `CKPT_ROOT`, write convert output to `CKPT_TRTLLM_ROOT`, write build output to `ENGINE_ROOT`.

---

## Naming Conventions (consistent with existing scripts)

- **PTQ ckpt directory name** (in use): `<model>-trtllm-ckpt-wq_<quant>-kv_<kv>`  
  Examples: `llama-3.1-8b-instruct-trtllm-ckpt-wq_fp8-kv_fp16`, `ministral-8b-instruct-2410-trtllm-ckpt-wq_int4_awq-kv_fp8`.
- **TRT-LLM native ckpt**: same name as PTQ ckpt for 1:1 correspondence; optionally add `-native` suffix.
- **Engine directory**: include tp/pp suffix, e.g. `llama-3.1-8b-wq_fp8-kv_fp16-tp1`, to avoid overwriting when testing multiple configs.

---

## Pipeline Flow

```
PTQ (Model Optimizer)     →  outputs/ckpts/<name>/          [CKPT_ROOT]
       ↓
convert_checkpoint.py     →  outputs/ckpts_trtllm/<name>/   [CKPT_TRTLLM_ROOT]
       ↓
trtllm-build              →  outputs/engines/<name>-tp1/    [ENGINE_ROOT]
       ↓
trtllm-run / trtllm-serve  reads ENGINE_ROOT
```

trtllm-bench **PyTorch backend** consumes **CKPT_ROOT** PTQ ckpts directly — no convert or build step needed.

---

## Using on Another Machine

- **PyTorch backend bench only**: copy or download **outputs/ckpts/** (PTQ ckpts) and set `CKPT_ROOT`.
- **TensorRT backend**: must run convert → build on the same machine, or transfer **ckpts_trtllm + engines** and set `CKPT_TRTLLM_ROOT`, `ENGINE_ROOT`.  
  See `docs/CKPT_TRANSFER.md` for packaging and upload options.
