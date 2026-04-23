#!/usr/bin/env python3
"""Multi-sample: 同一 system prompt + 多条 action_completion，看 cold vs 后续请求延迟（prefix 需 trtllm engine 才有复用）。"""
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SYSTEM = """You will receive the complete chat history from a chatbot application between a user and an assistant.
Evaluate the full chat history. Check whether the assistant's final answer satisfies: coherent, factually correct, comprehensive, no contradiction with tool outputs, accurate summary.
Respond with a single word: "true" (no quotes) if every task satisfies all five conditions, and "false" (no quotes) otherwise. If there are no user asks, output "true" (no quotes)"""

NUM_SAMPLES = 5
MAX_INPUT_TOKENS = 8192  # 用完整 chat_history，不截断

def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print("Load tokenizer & model (Llama-3.2-3B)...")
    t0 = time.perf_counter()
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    load_s = time.perf_counter() - t0
    print(f"Loaded in {load_s:.1f}s")

    print(f"Load dataset ({NUM_SAMPLES} samples)...")
    ds = load_dataset("rungalileo/action_completion", split="train", trust_remote_code=True)
    prompts = []
    for i in range(NUM_SAMPLES):
        chat = ds[i]["chat_history"]  # 完整 chat_history，不截断
        user_text = f"Chat history:\n```\n{chat}\n```"
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > MAX_INPUT_TOKENS:
            prompt = tokenizer.decode(ids[:MAX_INPUT_TOKENS])
        prompts.append(prompt)

    latencies = []
    outputs = []
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS).to(model.device)
        t0 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        latencies.append(time.perf_counter() - t0)
        text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
        pred = "true" if "true" in text[:10] else ("false" if "false" in text[:10] else text[:20])
        outputs.append(pred)
        print(f"  req {i+1}/{NUM_SAMPLES}: {inputs.input_ids.shape[1]} tok, {latencies[-1]:.2f}s -> {pred}")

    cold_s = latencies[0]
    subsequent = latencies[1:] if len(latencies) > 1 else []
    mean_subsequent_s = sum(subsequent) / len(subsequent) if subsequent else None

    result = {
        "dataset": "rungalileo/action_completion",
        "model": model_id,
        "backend": "HuggingFace",
        "num_samples": NUM_SAMPLES,
        "load_time_s": round(load_s, 2),
        "latencies_s": [round(x, 2) for x in latencies],
        "cold_latency_s": round(cold_s, 2),
        "subsequent_mean_latency_s": round(mean_subsequent_s, 2) if mean_subsequent_s else None,
        "outputs": outputs,
        "note": "HF 无 prefix KV 复用；测 prefix 效果需 trtllm engine + enable_block_reuse，对比 cold vs subsequent",
    }
    out_path = PROJECT_ROOT / "outputs/bench_prefix_kv_action_completion/quick_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n--- Result ---\ncold={cold_s:.2f}s  subsequent_mean={mean_subsequent_s:.2f}s" if mean_subsequent_s else f"\n--- Result ---\n{result}")
    print(f"Saved: {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
