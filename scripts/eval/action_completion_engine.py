#!/usr/bin/env python3
"""Minimal: 1 sample from action_completion + 3B engine, report result in <5 min."""
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SYSTEM = """You will receive the complete chat history from a chatbot application between a user and an assistant.
Evaluate the full chat history. Check whether the assistant's final answer satisfies: coherent, factually correct, comprehensive, no contradiction with tool outputs, accurate summary.
Respond with a single word: "true" (no quotes) if every task satisfies all five conditions, and "false" (no quotes) otherwise. If there are no user asks, output "true" (no quotes)"""

def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from tensorrt_llm._tensorrt_engine import LLM
    from tensorrt_llm import SamplingParams

    engine_dir = PROJECT_ROOT / "outputs/engines/llama-3.2-3b-trtllm-ckpt-wq_int4_awq-kv_int8"
    ckpt_dir = PROJECT_ROOT / "outputs/ckpts/llama-3.2-3b-instruct-trtllm-ckpt-wq_fp8-kv_fp16"
    max_input = 128  # engine max_input_len

    print("Load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), trust_remote_code=True)
    print("Load dataset (1 sample)...")
    ds = load_dataset("rungalileo/action_completion", split="train", trust_remote_code=True)
    chat = ds[0]["chat_history"][:500]  # keep short
    user_text = f"Chat history:\n```\n{chat}\n```"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > max_input:
        prompt = tokenizer.decode(ids[:max_input])
    print(f"Input tokens: {len(tokenizer.encode(prompt, add_special_tokens=False))}")

    print("Load engine...")
    t0 = time.perf_counter()
    llm = LLM(model=str(engine_dir), tokenizer=str(ckpt_dir), trust_remote_code=True)
    load_s = time.perf_counter() - t0
    print(f"Engine loaded in {load_s:.1f}s")

    sampling = SamplingParams(max_tokens=8, temperature=0.0)
    print("Run inference...")
    t0 = time.perf_counter()
    out = llm.generate([prompt], sampling_params=sampling)
    infer_s = time.perf_counter() - t0
    text = (out[0].outputs[0].text or "").strip().lower()
    pred = "true" if "true" in text[:10] else ("false" if "false" in text[:10] else text[:20])

    result = {
        "dataset": "rungalileo/action_completion",
        "engine": str(engine_dir.name),
        "load_time_s": round(load_s, 2),
        "inference_time_s": round(infer_s, 2),
        "output": pred,
        "raw": text[:50],
    }
    out_path = PROJECT_ROOT / "outputs/bench_prefix_kv_action_completion/quick_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n--- Result ---\n{json.dumps(result, indent=2)}\nSaved: {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
