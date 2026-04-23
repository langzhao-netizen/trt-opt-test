#!/usr/bin/env python3
"""
Quick test of prefix KV effect on action_completion dataset.

Dataset: https://huggingface.co/datasets/rungalileo/action_completion
- System prompt: ACTION_COMPLETION_SYSTEM_PROMPT (same for all requests)
- User prompt: Chat history with {chat_history} (differs per request)

With enable_block_reuse=True, the system prompt prefix is cached; subsequent
requests should see lower first-token latency.

Uses trtllm 1.1.0, outputs/ckpts smallest model (llama-3.2-3b-instruct).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

ACTION_COMPLETION_SYSTEM_PROMPT = """\
You will receive the complete chat history from a chatbot application between a user and an assistant.

In the chat history, the user will ask questions, which are answered with words, or make requests that require calling tools and resolving actions. \
Sometimes these are given as orders; treat them as if they were questions or requests. \
Each assistant turn may involve several steps that combine internal reflections, planning steps, selecting tools, and calling tools, \
and should always end with the assistant replying back to the user.
Evaluate the full chat history between user and assistant. Follow these steps:

- Extract each user ask (question, command, request, clarification, follow-up) as a separate item, even within the same message.
- For each task, identify:
- The initial user ask and any refinements made to it.
- The final user ask combining the initial ask and all refinements.
- Identify any direct answers from the assistant:
- A direct answer addresses or resolves the final user ask explicitly.
- Identify any indirect answers:
- These include mentions of inability, tool failures, alternative offers, or clarification requests.
- If there's a direct answer:
- List any tools used, capturing both input and output.
- Check whether the assistant's final answer satisfies all of these five conditions:
1. It is coherent and internally consistent.
2. It is factually correct.
3. It comprehensively answers the full final user ask.
4. It does not contradict tool outputs.
5. It accurately summarizes tool outputs.

Respond with a single word. Respond with "true" (no quotes) if every task satisfies all five conditions, \
and "false" (no quotes) otherwise. If there are no user asks, output "true" (no quotes) \
"""

USER_PROMPT_TEMPLATE = """Chat history:
```
{chat_history}
```
"""


def build_messages(chat_history: str):
    return [
        {"role": "system", "content": ACTION_COMPLETION_SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(chat_history=chat_history)},
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Bench prefix KV effect on action_completion")
    ap.add_argument(
        "--ckpt",
        default="",
        help="Checkpoint path for tokenizer / PyTorch backend (default: see --engine or ckpts)",
    )
    ap.add_argument(
        "--engine",
        default=str(PROJECT_ROOT / "outputs/engines/llama-3.2-3b-trtllm-ckpt-wq_int4_awq-kv_int8"),
        help="TensorRT engine dir (if set and exists, use TRT backend; else use --ckpt with PyTorch)",
    )
    ap.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of dataset samples to run (default: 5)",
    )
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=8192,
        help="Max sequence length (default: 8192)",
    )
    ap.add_argument(
        "--output-tokens",
        type=int,
        default=8,
        help="Max output tokens for true/false (default: 8)",
    )
    ap.add_argument(
        "--no-prefix-kv",
        action="store_true",
        help="Disable prefix KV (enable_block_reuse=False) for comparison",
    )
    ap.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "outputs/bench_prefix_kv_action_completion"),
        help="Output directory for results",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup requests before measured runs (default: 1)",
    )
    args = ap.parse_args()

    engine_path = Path(args.engine) if args.engine else None
    ckpt_path = Path(args.ckpt) if args.ckpt else None
    use_engine = engine_path and engine_path.is_dir() and (engine_path / "rank0.engine").exists()
    if use_engine:
        model_path = engine_path
        tokenizer_path = ckpt_path or (PROJECT_ROOT / "outputs/ckpts/llama-3.2-3b-instruct-trtllm-ckpt-wq_fp8-kv_fp16")
        if not tokenizer_path.is_dir():
            tokenizer_path = "meta-llama/Llama-3.2-3B-Instruct"
    else:
        ckpt_path = ckpt_path or (PROJECT_ROOT / "outputs/ckpts/llama-3.2-3b-instruct-trtllm-ckpt-wq_fp8-kv_fp16")
        if not ckpt_path.is_dir():
            print(f"ERROR: checkpoint/engine not found: {ckpt_path}", file=sys.stderr)
            return 2
        model_path = ckpt_path
        tokenizer_path = ckpt_path

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    from datasets import load_dataset

    ds = load_dataset("rungalileo/action_completion", split="train", trust_remote_code=True)
    samples = list(ds.select(range(min(args.num_samples, len(ds)))))

    # Import trtllm inside venv
    from transformers import AutoTokenizer
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import BuildConfig, KvCacheConfig
    from tensorrt_llm.sampling_params import SamplingParams
    import torch
    import tensorrt_llm as tllm

    print(f"[bench] TensorRT-LLM version: {getattr(tllm, '__version__', 'unknown')}")
    print(f"[bench] backend={'TensorRT engine' if use_engine else 'PyTorch'}, model={model_path.name}")
    print(f"[bench] prefix_kv={not args.no_prefix_kv}")
    print(f"[bench] num_samples={args.num_samples} max_seq_len={args.max_seq_len} output_tokens={args.output_tokens}")

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)

    if use_engine:
        import json as _json
        with open(model_path / "config.json") as _f:
            _ec = _json.load(_f)
        _build = _ec.get("build_config", {})
        engine_max_seq = _build.get("max_seq_len") or _build.get("max_num_tokens") or 256
        max_seq = engine_max_seq
        max_input_tokens = max_seq - args.output_tokens
        print(f"[bench] engine max_seq_len={engine_max_seq}, using max_seq={max_seq}")
    else:
        max_input_tokens = args.max_seq_len - args.output_tokens
    prompts = []
    for ex in samples:
        messages = build_messages(ex["chat_history"])
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > max_input_tokens:
            text = tokenizer.decode(ids[:max_input_tokens])
        prompts.append(text)

    input_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    print(f"[bench] input token lengths: {input_lens}")

    if not use_engine:
        max_input = max(input_lens)
        max_seq = min(args.max_seq_len, max_input + args.output_tokens)
        if max_input > args.max_seq_len:
            print(f"[bench] WARNING: some inputs exceed max_seq_len; truncating to {args.max_seq_len}")

    sampling = SamplingParams(
        max_tokens=args.output_tokens,
        temperature=0.0,
        top_p=1.0,
        return_perf_metrics=True,
    )

    results = []
    if use_engine:
        from tensorrt_llm._tensorrt_engine import LLM as TrtLLM
        llm = TrtLLM(
            model=str(model_path),
            tokenizer=str(tokenizer_path),
            trust_remote_code=True,
        )
    else:
        build_config = BuildConfig(
            max_batch_size=1,
            max_num_tokens=max_seq,
            max_seq_len=max_seq,
            max_beam_width=1,
        )
        kv_cache_config = KvCacheConfig(
            free_gpu_memory_fraction=0.8,
            enable_block_reuse=not args.no_prefix_kv,
            dtype="auto",
        )
        try:
            llm = LLM(
                model=str(model_path),
                backend="pytorch",
                build_config=build_config,
                kv_cache_config=kv_cache_config,
                trust_remote_code=True,
            )
        except TypeError:
            llm = LLM(
                model=str(model_path),
                build_config=build_config,
                kv_cache_config=kv_cache_config,
                trust_remote_code=True,
            )

    with llm:
        # Warmup
        for _ in range(args.warmup):
            _ = llm.generate([prompts[0]], sampling_params=sampling)

        for i, prompt in enumerate(prompts):
            t0 = time.perf_counter()
            out = llm.generate([prompt], sampling_params=sampling)
            wall = time.perf_counter() - t0

            first_tok_s = None
            generated = ""
            try:
                gen = out[0].outputs[0].text
                generated = gen[0] if gen else ""
                rr = out[0].outputs[0].request_perf_metrics
                if rr is not None:
                    tm = rr.timing_metrics
                    first_tok_s = float(tm.first_token_time - tm.arrival_time)
            except Exception:
                pass

            rec = {
                "idx": i,
                "input_tokens": input_lens[i],
                "wall_s": wall,
                "first_token_s": first_tok_s,
                "output": generated.strip()[:20],
            }
            results.append(rec)
            print(
                f"[bench] req {i+1}/{len(prompts)}: input={input_lens[i]} tokens, "
                f"wall={wall:.3f}s, first_tok={first_tok_s:.3f}s if avail, out={generated.strip()[:20]!r}"
            )

    # Summary
    walls = [r["wall_s"] for r in results]
    first_toks = [r["first_token_s"] for r in results if r["first_token_s"] is not None]
    summary = {
        "prefix_kv": not args.no_prefix_kv,
        "model": str(model_path.name),
        "num_samples": len(results),
        "input_tokens": input_lens,
        "wall_s": walls,
        "first_token_s": first_toks,
        "mean_wall_s": sum(walls) / len(walls) if walls else 0,
        "mean_first_token_s": sum(first_toks) / len(first_toks) if first_toks else None,
        "cold_wall_s": walls[0] if walls else None,
        "cold_first_token_s": first_toks[0] if first_toks else None,
        "subsequent_mean_wall_s": sum(walls[1:]) / max(1, len(walls) - 1) if len(walls) > 1 else None,
        "subsequent_mean_first_token_s": (
            sum(first_toks[1:]) / max(1, len(first_toks) - 1) if len(first_toks) > 1 else None
        ),
    }
    if first_toks and len(first_toks) > 1:
        speedup_wall = walls[0] / (sum(walls[1:]) / (len(walls) - 1)) if walls[1:] else 1.0
        speedup_ft = first_toks[0] / (sum(first_toks[1:]) / (len(first_toks) - 1)) if first_toks[1:] else 1.0
        summary["speedup_wall_subsequent"] = speedup_wall
        summary["speedup_first_token_subsequent"] = speedup_ft
        print(
            f"\n[bench] Prefix KV {'enabled' if not args.no_prefix_kv else 'disabled'}: "
            f"cold wall={walls[0]:.3f}s, subsequent mean wall={summary['subsequent_mean_wall_s']:.3f}s "
            f"(speedup {speedup_wall:.2f}x), first_token speedup {speedup_ft:.2f}x"
        )

    out_json = out_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "per_request": results}, f, indent=2)
    print(f"\n[bench] Results written to {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
