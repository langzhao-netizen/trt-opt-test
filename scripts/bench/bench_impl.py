#!/usr/bin/env python3
"""
Run 15k input / 1 output benchmark on each ckpt under outputs/ckpts.

Requirements from user:
- trtllm==1.1.0
- PyTorch backend
- batch size 1
- input ~15k tokens, output 1 token
- record results per checkpoint
- resumable: writes results incrementally
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RequestResult:
    request_idx: int
    input_tokens: int
    output_tokens: int
    wall_time_s: float
    first_token_latency_s: Optional[float] = None
    last_token_latency_s: Optional[float] = None
    perf: Optional[Dict[str, Any]] = None


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs_sorted) - 1)
    if f == c:
        return xs_sorted[f]
    return xs_sorted[f] * (c - k) + xs_sorted[c] * (k - f)


def _make_prompt(tokenizer, target_input_tokens: int) -> Tuple[str, int]:
    block = "The quick brown fox jumps over the lazy dog. "
    block_ids = tokenizer.encode(block, add_special_tokens=False)
    repeat = max(1, (target_input_tokens + len(block_ids) - 1) // len(block_ids))
    prompt = (block * repeat).strip()
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > target_input_tokens:
        prompt = tokenizer.decode(ids[:target_input_tokens])
        ids = ids[:target_input_tokens]
    return prompt, len(ids)


def _load_completed(result_jsonl: Path) -> Dict[str, Dict[str, Any]]:
    completed: Dict[str, Dict[str, Any]] = {}
    if not result_jsonl.exists():
        return completed
    with result_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ckpt = rec.get("ckpt")
            status = rec.get("status")
            if ckpt and status == "ok":
                completed[ckpt] = rec
    return completed


def _iter_ckpts(ckpt_root: Path) -> List[Path]:
    dirs = [p for p in ckpt_root.iterdir() if p.is_dir()]
    # Only final ckpts: *-kv_fp8 or *-kv_fp16
    finals = [
        p for p in dirs if p.name.endswith("-kv_fp8") or p.name.endswith("-kv_fp16")
    ]
    return sorted(finals, key=lambda p: p.name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt-root",
        default=str(PROJECT_ROOT / "outputs/ckpts"),
        help="Checkpoint root (default: outputs/ckpts)",
    )
    ap.add_argument(
        "--target-input-tokens",
        type=int,
        default=int(os.environ.get("TARGET_INPUT_TOKENS", "15360")),
        help="Target input tokens (~15k). Default 15360.",
    )
    ap.add_argument(
        "--output-tokens",
        type=int,
        default=1,
        help="Output tokens. Default 1.",
    )
    ap.add_argument(
        "--num-requests",
        type=int,
        default=int(os.environ.get("NUM_REQUESTS", "5")),
        help="Requests per checkpoint (bs=1). Default 5.",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=int(os.environ.get("WARMUP", "2")),
        help="Warmup requests per checkpoint (not counted). Default 2.",
    )
    ap.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "outputs/bench_trtllm110_pytorch"),
        help="Output dir for logs/results.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume and skip ckpts already marked ok in results jsonl.",
    )
    args = ap.parse_args()

    ckpt_root = Path(args.ckpt_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_jsonl = out_dir / "results_15k1output.jsonl"
    result_csv = out_dir / "results_15k1output.csv"

    completed = _load_completed(result_jsonl) if args.resume else {}

    # Imports must happen inside the trtllm 1.1.0 venv.
    from transformers import AutoTokenizer  # type: ignore
    from tensorrt_llm import LLM  # type: ignore
    from tensorrt_llm.llmapi import BuildConfig, KvCacheConfig  # type: ignore
    from tensorrt_llm.sampling_params import SamplingParams  # type: ignore
    import torch  # type: ignore
    import tensorrt_llm as tllm  # type: ignore

    print(f"[bench] TensorRT-LLM version: {getattr(tllm, '__version__', 'unknown')}")
    print(f"[bench] ckpt_root={ckpt_root}")
    print(
        f"[bench] target_input_tokens={args.target_input_tokens} output_tokens={args.output_tokens} num_requests={args.num_requests} bs=1"
    )

    ckpts = _iter_ckpts(ckpt_root)
    if not ckpts:
        print(f"[bench] ERROR: no ckpts found under {ckpt_root}", file=sys.stderr)
        return 2

    # Ensure CSV header exists
    if not result_csv.exists():
        with result_csv.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "ckpt",
                    "kv_mode",
                    "status",
                    "input_tokens",
                    "output_tokens",
                    "num_requests",
                    "p50_wall_s",
                    "p90_wall_s",
                    "mean_wall_s",
                    "ts",
                    "error",
                ],
            )
            w.writeheader()

    for ckpt in ckpts:
        ckpt_name = ckpt.name
        ckpt_out = out_dir / ckpt_name
        ckpt_out.mkdir(parents=True, exist_ok=True)
        latency_json = ckpt_out / "latency.json"
        error_json = ckpt_out / "error.json"
        iteration_log = ckpt_out / "iteration.log"

        if args.resume and (ckpt_name in completed or latency_json.exists()):
            print(f"[bench] Skip (already ok): {ckpt_name}")
            continue

        kv_mode = "kv_fp8" if ckpt_name.endswith("-kv_fp8") else "kv_fp16"
        kv_dtype = "fp8" if kv_mode == "kv_fp8" else "auto"

        print(f"[bench] === {ckpt_name} ({kv_mode}, kv_dtype={kv_dtype}) ===")

        # Tokenizer from ckpt itself so 15k means tokens in that model's vocab.
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
        prompt, input_tokens = _make_prompt(tokenizer, args.target_input_tokens)

        build_config = BuildConfig(
            max_batch_size=1,
            max_num_tokens=args.target_input_tokens + args.output_tokens,
            max_seq_len=args.target_input_tokens + args.output_tokens,
            max_beam_width=1,
        )
        kv_cache_config = KvCacheConfig(
            free_gpu_memory_fraction=0.8,
            enable_block_reuse=True,
            dtype=kv_dtype,
        )

        sampling = SamplingParams(
            max_tokens=args.output_tokens,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            return_perf_metrics=True,
        )

        rec: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ckpt": ckpt_name,
            "ckpt_path": str(ckpt),
            "kv_mode": kv_mode,
            "kv_dtype": kv_dtype,
            "target_input_tokens": args.target_input_tokens,
            "input_tokens": input_tokens,
            "output_tokens": args.output_tokens,
            "num_requests": args.num_requests,
            "status": "running",
        }

        requests: List[RequestResult] = []

        try:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

            t0_init = time.perf_counter()
            try:
                llm = LLM(
                    model=str(ckpt),
                    backend="pytorch",
                    build_config=build_config,
                    kv_cache_config=kv_cache_config,
                    trust_remote_code=True,
                )
            except TypeError:
                # Some builds may not accept backend kwarg at top-level; fall back to default PyTorch backend.
                llm = LLM(
                    model=str(ckpt),
                    build_config=build_config,
                    kv_cache_config=kv_cache_config,
                    trust_remote_code=True,
                )
            rec["init_time_s"] = time.perf_counter() - t0_init

            # Run inside context manager for robust cleanup
            with llm:
                # Warmup (not counted)
                for _ in range(max(0, int(args.warmup))):
                    _ = llm.generate([prompt], sampling_params=sampling)

                for i in range(args.num_requests):
                    t0 = time.perf_counter()
                    out = llm.generate([prompt], sampling_params=sampling)
                    wall = time.perf_counter() - t0

                    perf_dict: Optional[Dict[str, Any]] = None
                    first_tok: Optional[float] = None
                    last_tok: Optional[float] = None
                    try:
                        rr = out[0].outputs[0].request_perf_metrics
                        if rr is not None:
                            tm = rr.timing_metrics
                            # These are absolute timestamps; convert to durations where possible.
                            first_tok = float(tm.first_token_time - tm.arrival_time)
                            last_tok = float(tm.last_token_time - tm.arrival_time)
                            perf_dict = {
                                "timing_metrics": {
                                    k: float(getattr(tm, k))
                                    for k in dir(tm)
                                    if k.endswith("_time")
                                },
                                "kv_cache_metrics": asdict(rr.kv_cache_metrics)
                                if rr.kv_cache_metrics is not None
                                else None,
                            }
                    except Exception:
                        pass

                    rr_obj = RequestResult(
                        request_idx=i,
                        input_tokens=input_tokens,
                        output_tokens=args.output_tokens,
                        wall_time_s=wall,
                        first_token_latency_s=first_tok,
                        last_token_latency_s=last_tok,
                        perf=perf_dict,
                    )
                    requests.append(rr_obj)
                    with iteration_log.open("a") as f:
                        f.write(json.dumps(asdict(rr_obj)) + "\n")
                    print(
                        f"[bench] {ckpt_name} req {i+1}/{args.num_requests}: wall={wall:.3f}s"
                    )
        except Exception as e:
            # Record failure for this ckpt, continue to next; daemon will keep retrying until all ok.
            rec["status"] = "fail"
            rec["error"] = repr(e)

            with result_jsonl.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            with error_json.open("w") as f:
                f.write(json.dumps(rec, indent=2) + "\n")
            with result_csv.open("a", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "ckpt",
                        "kv_mode",
                        "status",
                        "input_tokens",
                        "output_tokens",
                        "num_requests",
                        "p50_wall_s",
                        "p90_wall_s",
                        "mean_wall_s",
                        "ts",
                        "error",
                    ],
                )
                w.writerow(
                    {
                        "ckpt": ckpt_name,
                        "kv_mode": kv_mode,
                        "status": "fail",
                        "input_tokens": input_tokens,
                        "output_tokens": args.output_tokens,
                        "num_requests": args.num_requests,
                        "p50_wall_s": "",
                        "p90_wall_s": "",
                        "mean_wall_s": "",
                        "ts": rec["ts"],
                        "error": rec["error"],
                    }
                )
            print(f"[bench] FAIL {ckpt_name}: {e}", file=sys.stderr)
            continue

        wall_times = [r.wall_time_s for r in requests]
        rec["status"] = "ok"
        rec["requests"] = [asdict(r) for r in requests]
        rec["summary"] = {
            "p50_wall_s": _percentile(wall_times, 0.50),
            "p90_wall_s": _percentile(wall_times, 0.90),
            "mean_wall_s": sum(wall_times) / max(1, len(wall_times)),
        }

        try:
            rec["gpu_peak_mem_bytes"] = int(torch.cuda.max_memory_allocated())
        except Exception:
            pass

        # Append JSONL
        with result_jsonl.open("a") as f:
            f.write(json.dumps(rec) + "\n")

        # Write per-ckpt report (trtllm-bench-like)
        with latency_json.open("w") as f:
            f.write(json.dumps(rec, indent=2) + "\n")

        # Append CSV summary
        with result_csv.open("a", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "ckpt",
                    "kv_mode",
                    "status",
                    "input_tokens",
                    "output_tokens",
                    "num_requests",
                    "p50_wall_s",
                    "p90_wall_s",
                    "mean_wall_s",
                    "ts",
                    "error",
                ],
            )
            w.writerow(
                {
                    "ckpt": ckpt_name,
                    "kv_mode": kv_mode,
                    "status": "ok",
                    "input_tokens": input_tokens,
                    "output_tokens": args.output_tokens,
                    "num_requests": args.num_requests,
                    "p50_wall_s": rec["summary"]["p50_wall_s"],
                    "p90_wall_s": rec["summary"]["p90_wall_s"],
                    "mean_wall_s": rec["summary"]["mean_wall_s"],
                    "ts": rec["ts"],
                    "error": "",
                }
            )

    print(f"[bench] Done. Results: {result_jsonl} and {result_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

