# Checkpoint Upload and Download

`outputs/ckpts/` holds 16 quantized ckpts (Llama-3.2-3B, Llama-3.1-8B, Mistral-7B, Ministral-8B, each with 4 wq/kv combinations). Three ways to transfer them to another machine:

## Option 1: tar.gz archives (universal)

**Pack on source machine:**

```bash
./scripts/transfer/pack_ckpts.sh
# Output in outputs/ckpts_archives/ — one <name>.tar.gz + .sha256 per ckpt
```

Copy the `.tar.gz` files to the target machine (USB, scp, object storage, etc.), then on the target:

```bash
mkdir -p outputs/ckpts
for f in *.tar.gz; do
  tar xzf "$f" -C outputs/ckpts
done
# Optional integrity check: sha256sum -c *.sha256
```

## Option 2: Hugging Face Hub (recommended for teams / multi-machine)

**Upload from source machine:**

1. Create a repo at https://huggingface.co (e.g. `your-username/trtllm-ckpts`); can be Private.
2. Authenticate:
   ```bash
   export HF_TOKEN="hf_xxx"
   # or: huggingface-cli login
   ```
3. Upload:
   ```bash
   export REPO_ID="your-username/trtllm-ckpts"
   ./scripts/transfer/upload_ckpts.sh
   ```

**Download on another machine:**

```bash
# Download all ckpts into ./ckpts/
huggingface-cli download your-username/trtllm-ckpts --local-dir ./ckpts

# Or download a single ckpt (e.g. llama fp8/fp16 only)
huggingface-cli download your-username/trtllm-ckpts --include "llama-3.1-8b-instruct-trtllm-ckpt-wq_fp8-kv_fp16/*" --local-dir ./ckpts
```

Point the project at the downloaded ckpts:

```bash
export CKPT_ROOT=/path/to/ckpts
# bench scripts read ckpts from CKPT_ROOT
```

## Option 3: rsync / scp direct copy

When you have SSH access to the target machine:

```bash
rsync -avz --progress outputs/ckpts/ user@other-host:/path/to/trt-opt-test/outputs/ckpts/
```

No extraction needed on the target — the directory is ready to use as `CKPT_ROOT`.

## Usage on any machine

- **trtllm-bench**: pass `--model_path` pointing to a ckpt directory, e.g.  
  `trtllm-bench --model meta-llama/Llama-3.1-8B-Instruct --model_path outputs/ckpts/llama-3.1-8b-instruct-trtllm-ckpt-wq_fp8-kv_fp16 ...`
- **Project scripts**: set `CKPT_ROOT` to the directory containing the 16 subdirectories, e.g.  
  `export CKPT_ROOT=/path/to/ckpts`, then run `scripts/bench/run_bench.sh` etc.

## Size reference

- Single ckpt: ~4–10 GB depending on model and quantization.
- All 16 ckpts: tens of GB total. Run `du -sh outputs/ckpts outputs/ckpts_archives` to check actual sizes before transferring.
