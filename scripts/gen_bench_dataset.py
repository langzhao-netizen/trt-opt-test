#!/usr/bin/env python3
"""生成 15k input / 1 output 的 benchmark 用 dataset（JSONL，每行一个请求）。"""
import json
import os
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    out_path = Path(os.environ.get("BENCH_DATASET", str(PROJECT_ROOT / "outputs/bench_l4/dataset_15k_1.jsonl")))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Engine build uses MAX_SEQ_LEN=15360 but executor default may be 8192; cap to 8192 for compatibility
    target_input_tokens = int(os.environ.get("TARGET_INPUT_TOKENS", "8192"))
    target_output_tokens = int(os.environ.get("TARGET_OUTPUT_TOKENS", "1"))
    num_lines = int(os.environ.get("NUM_REQUESTS", "5"))

    # 使用任一本地模型的 tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        PROJECT_ROOT / "shared/models/Llama-3.2-3B-Instruct",
        trust_remote_code=True,
    )
    # 构造一段重复文本直到约 target_input_tokens
    block = "The quick brown fox jumps over the lazy dog. "
    block_ids = tokenizer.encode(block, add_special_tokens=False)
    repeat = max(1, (target_input_tokens + len(block_ids) - 1) // len(block_ids))
    prompt = (block * repeat).strip()
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > target_input_tokens:
        prompt = tokenizer.decode(ids[:target_input_tokens])
    with open(out_path, "w") as f:
        for i in range(num_lines):
            line = json.dumps({"task_id": i, "prompt": prompt, "output_tokens": target_output_tokens})
            f.write(line + "\n")
    print(f"Wrote {num_lines} requests to {out_path} (input ~{len(tokenizer.encode(prompt))} tokens, output {target_output_tokens})")

if __name__ == "__main__":
    main()
