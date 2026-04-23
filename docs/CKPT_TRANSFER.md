# Checkpoint 上传与在其他机器下载使用

`outputs/ckpts/` 下有 16 个量化 ckpt（Llama-3.2-3B、Llama-3.1-8B、Mistral-7B、Ministral-8B 各 4 种 wq/kv 组合）。要在其他机器上下载使用，可用下面几种方式。

## 方式一：打包成 tar.gz 再传（通用）

**本机打包：**

```bash
./scripts/transfer/pack_ckpts.sh
# 输出在 outputs/ckpts_archives/，每个 ckpt 一个 <name>.tar.gz + .sha256
```

把 `outputs/ckpts_archives/` 里的 `.tar.gz` 拷到目标机（U 盘、scp、对象存储等），在目标机：

```bash
mkdir -p outputs/ckpts
for f in *.tar.gz; do
  tar xzf "$f" -C outputs/ckpts
done
# 可选校验：sha256sum -c *.sha256
```

## 方式二：Hugging Face Hub（适合团队/多机）

**本机上传：**

1. 在 https://huggingface.co 建一个 repo（如 `your-username/trtllm-ckpts`），可选 Private。
2. 登录并设置 token：
   ```bash
   export HF_TOKEN="hf_xxx"
   # 或: huggingface-cli login
   ```
3. 上传：
   ```bash
   export REPO_ID="your-username/trtllm-ckpts"
   ./scripts/transfer/upload_ckpts.sh
   ```

**其他机器下载：**

```bash
# 全部 ckpt 下载到当前目录下的 ckpts/
huggingface-cli download your-username/trtllm-ckpts --local-dir ./ckpts

# 或只下某一个 ckpt（如只下 llama fp8/fp16）
huggingface-cli download your-username/trtllm-ckpts --include "llama-3.1-8b-instruct-trtllm-ckpt-wq_fp8-kv_fp16/*" --local-dir ./ckpts
```

然后在本项目里把 ckpt 根目录指到 `./ckpts`：

```bash
export CKPT_ROOT=/path/to/ckpts
# 跑 bench 等脚本时会从 CKPT_ROOT 读 ckpt
```

## 方式三：rsync / scp 直拷

本机有 SSH 到目标机时，可直接同步目录：

```bash
rsync -avz --progress outputs/ckpts/ user@other-host:/path/to/trt-opt-test/outputs/ckpts/
```

目标机无需再解压，直接使用 `outputs/ckpts`。

## 使用方式（在任意机器上）

- **trtllm-bench**：`--model_path` 指向某个 ckpt 目录，例如  
  `trtllm-bench --model meta-llama/Llama-3.1-8B-Instruct --model_path outputs/ckpts/llama-3.1-8b-instruct-trtllm-ckpt-wq_fp8-kv_fp16 ...`
- **本项目脚本**：设置 `CKPT_ROOT` 为包含这 16 个子目录的根目录即可，例如  
  `export CKPT_ROOT=/path/to/ckpts` 再跑 `scripts/bench/run_bench.sh` 等。

## 体积参考

- 单 ckpt 约 4–10 GB（视模型与量化）。
- 16 个全打包约几十 GB，传之前可用 `du -sh outputs/ckpts outputs/ckpts_archives` 看实际大小。
