#!/usr/bin/env python3
"""
Benchmark Time to First Token (TTFT) across input sizes.

Supports three backends:
  - TensorRT-LLM Executor  (pre-compiled engines via Executor API)
  - TensorRT-LLM PyTorch   (modelopt PTQ checkpoint via LLM API, no engine build needed)
  - vLLM                   (modelopt PTQ checkpoint or HF model via offline LLM API)

For a fair same-quantized-model comparison, point both --trt-pytorch and --vllm at
the same modelopt PTQ checkpoint directory. vLLM auto-detects quantization="modelopt"
when hf_quant_config.json is present.

Each model runs in its own subprocess for GPU memory isolation.

Usage — TRT-LLM PyTorch vs vLLM (same PTQ checkpoint, recommended for fair comparison):
    python scripts/bench/bench_ttft.py \\
        --trt-pytorch outputs/ckpts/llama-3.1-8b-instruct-trtllm-ckpt-wq_fp8-kv_fp16 \\
        --vllm outputs/ckpts/llama-3.1-8b-instruct-trtllm-ckpt-wq_fp8-kv_fp16 \\
        --trt-pytorch-python venv_trtllm1.1.0/bin/python \\
        --vllm-python venv_vllm/bin/python

Usage — TRT-LLM Executor (pre-compiled engine) vs vLLM:
    python scripts/bench/bench_ttft.py \\
        --engines outputs/engines/llama-3.1-8b-fp8-15k:models/llama-3.1-8b-instruct \\
        --vllm outputs/ckpts/llama-3.1-8b-instruct-trtllm-ckpt-wq_fp8-kv_fp16 \\
        --vllm-python venv_vllm/bin/python

Usage — auto-discover TRT-LLM engines only:
    python scripts/bench/bench_ttft.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_SEED_TEXT = (
    "Machine learning is a branch of artificial intelligence that focuses on "
    "building applications that learn from data and improve their accuracy over "
    "time without being programmed to do so. In data science, an algorithm is a "
    "sequence of statistical processing steps. In machine learning, algorithms "
    "are trained to find patterns and features in massive amounts of data in "
    "order to make decisions and predictions based on new data. The better the "
    "algorithm, the more accurate the decisions and predictions will become as "
    "it processes more data. A large language model is a type of machine learning "
    "model that can perform a variety of natural language processing tasks such "
    "as generating and classifying text, answering questions in a conversational "
    "manner, and translating text from one language to another. Large language "
    "models are trained on vast amounts of text data to learn patterns and "
    "relationships between words and phrases. They use a transformer architecture "
    "with self-attention mechanisms to understand context and generate coherent "
    "and contextually relevant responses. "
)

_LABEL_MAP = {
    500: "Small (500 tok)",
    2000: "Medium (2K tok)",
    15000: "Large (15K tok)",
    100000: "XL (100K tok)",
}

_MODEL_PATTERNS = [
    ("llama-3.2-3b", "llama-3.2-3b-instruct", "Llama-3.2 3B"),
    ("llama-3.1-8b", "llama-3.1-8b-instruct", "Llama-3.1 8B"),
    ("ministral-8b", "ministral-8b-instruct", "Ministral 8B"),
    ("mistral-7b", "mistral-7b-instruct-v0.3", "Mistral 7B"),
]

# LD_LIBRARY_PATH additions required for TRT-LLM venv at runtime
_TRTLLM_LD_EXTRA = [
    "/usr/local/cuda/targets/x86_64-linux/lib",
    "/usr/local/cuda/lib64",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def detect_gpu() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
        return out.split("\n")[0].strip()
    except Exception:
        return "Unknown GPU"


def get_engine_config(engine_dir: Path) -> dict:
    cfg_path = engine_dir / "config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        cfg = json.load(f)
    bc = cfg.get("build_config", {})
    return {
        "max_input_len": bc.get("max_input_len", 0),
        "max_seq_len": bc.get("max_seq_len", 0),
        "max_batch_size": bc.get("max_batch_size", 1),
        "paged_context_fmha": bc.get("plugin_config", {}).get(
            "use_paged_context_fmha", False
        ),
    }


def detect_ptq_quant(model_dir: Path) -> str | None:
    """Read quantization format from modelopt hf_quant_config.json."""
    for fname in ("hf_quant_config.json", "quantization_config.json"):
        p = model_dir / fname
        if p.exists():
            try:
                cfg = json.loads(p.read_text())
                q = cfg.get("quantization", {}).get("quant_algo") or cfg.get("quant_type")
                if q:
                    return str(q).lower()
            except Exception:
                pass
    cfg_p = model_dir / "config.json"
    if cfg_p.exists():
        try:
            cfg = json.loads(cfg_p.read_text())
            qc = cfg.get("quantization_config", {})
            if qc:
                return qc.get("quant_type") or qc.get("format") or "modelopt"
        except Exception:
            pass
    return None


def detect_ptq_label(model_dir: Path) -> str:
    """Return full quant label including KV cache, e.g. 'FP8-KV_FP8' or 'NVFP4-KV_BF16'."""
    for fname in ("hf_quant_config.json", "quantization_config.json"):
        p = model_dir / fname
        if p.exists():
            try:
                cfg = json.loads(p.read_text())
                qcfg = cfg.get("quantization", {})
                w = qcfg.get("quant_algo") or cfg.get("quant_type")
                kv = qcfg.get("kv_cache_quant_algo")
                if w:
                    kv_label = kv.upper() if kv else "BF16"
                    return f"{w.upper()}-KV_{kv_label}"
            except Exception:
                pass
    q = detect_ptq_quant(model_dir)
    return q.upper() if q else "unknown"


def clean_model_name(name: str) -> str:
    name_lower = name.lower()
    for pattern, _, display in _MODEL_PATTERNS:
        if pattern in name_lower:
            return display
    for sep in ("-fp8", "-fp16", "-trtllm", "-nvfp4", "-wq_fp8", "-wq_nvfp4", "-wq_int4"):
        if sep in name:
            return name[: name.index(sep)]
    return name


def resolve_tokenizer(engine_name: str, models_dir: Path) -> Path | None:
    name_lower = engine_name.lower()
    for pattern, tokenizer_dirname, _ in _MODEL_PATTERNS:
        if pattern in name_lower:
            p = models_dir / tokenizer_dirname
            if p.is_dir():
                return p
    return None


def discover_engines(
    engine_base: Path, models_dir: Path
) -> list[tuple[Path, Path, str]]:
    if not engine_base.is_dir():
        return []
    candidates: dict[str, tuple[Path, Path, str, int]] = {}
    quant_keywords = ("fp8", "nvfp4", "int4_awq", "fp16")
    for d in sorted(engine_base.iterdir()):
        if not d.is_dir():
            continue
        name_lower = d.name.lower()
        if not any(k in name_lower for k in quant_keywords):
            continue
        if not (d / "rank0.engine").exists():
            continue
        tokenizer = resolve_tokenizer(d.name, models_dir)
        if tokenizer is None:
            continue
        display = clean_model_name(d.name)
        mil = get_engine_config(d).get("max_input_len", 0)
        if display not in candidates or mil > candidates[display][3]:
            candidates[display] = (d, tokenizer, display, mil)
    return [(d, tok, name) for d, tok, name, _ in candidates.values()]


def make_token_ids(tokenizer, length: int) -> list[int]:
    ids = tokenizer.encode(_SEED_TEXT, add_special_tokens=False)
    if len(ids) < length:
        factor = (length // len(ids)) + 2
        ids = ids * factor
    return ids[:length]


def format_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1e6:.0f}us"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 10.0:
        return f"{seconds:.2f}s"
    return f"{seconds:.1f}s"


def _collect_results(times: list[float], length: int) -> dict:
    med = statistics.median(times)
    print(
        f"  [{length:>6} tok] median={format_time(med)} "
        f"(runs: {', '.join(format_time(t) for t in times)})"
    )
    return {
        "skipped": False,
        "median_s": round(med, 6),
        "mean_s": round(statistics.mean(times), 6),
        "min_s": round(min(times), 6),
        "max_s": round(max(times), 6),
        "stdev_s": (
            round(statistics.stdev(times), 6) if len(times) > 1 else 0.0
        ),
        "num_runs": len(times),
        "runs_s": [round(t, 6) for t in times],
    }


# ---------------------------------------------------------------------------
# TensorRT-LLM Executor backend (pre-compiled engines)
# ---------------------------------------------------------------------------


def run_benchmark(
    engine_dir: Path,
    tokenizer_path: Path,
    prefix_lengths: list[int],
    num_runs: int,
    warmup: int,
    kv_free_gpu_memory_fraction: float = 0.92,
) -> dict[int, dict]:
    from transformers import AutoTokenizer
    import tensorrt_llm.bindings.executor as trtexec

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    engine_cfg = get_engine_config(engine_dir)
    max_input = engine_cfg.get("max_input_len", 0)

    kv_config = trtexec.KvCacheConfig()
    kv_config.enable_block_reuse = False
    kv_config.free_gpu_memory_fraction = kv_free_gpu_memory_fraction

    executor_config = trtexec.ExecutorConfig()
    executor_config.kv_cache_config = kv_config
    executor_config.max_beam_width = 1
    executor_config.batching_type = trtexec.BatchingType.INFLIGHT

    print(f"  Loading engine: {engine_dir.name} ...")
    t0 = time.perf_counter()
    executor = trtexec.Executor(
        str(engine_dir),
        trtexec.ModelType.DECODER_ONLY,
        executor_config,
    )
    load_s = time.perf_counter() - t0
    print(f"  Engine loaded in {load_s:.1f}s (max_input_len={max_input})")

    sampling = trtexec.SamplingConfig()
    sampling.beam_width = 1
    sampling.temperature = 0.0
    sampling.top_p = 1.0

    output_config = trtexec.OutputConfig()
    output_config.exclude_input_from_output = True

    def run_once(input_ids: list[int]) -> float:
        request = trtexec.Request(
            input_token_ids=input_ids,
            max_tokens=1,
            sampling_config=sampling,
            output_config=output_config,
            streaming=False,
        )
        t_start = time.perf_counter()
        req_id = executor.enqueue_request(request)
        responses = executor.await_responses(req_id, timeout=300.0)
        wall_s = time.perf_counter() - t_start
        for resp in responses:
            if resp.has_error():
                raise RuntimeError(f"Executor error: {resp.error_msg}")
        return wall_s

    results = _run_length_loop(
        prefix_lengths, tokenizer, run_once, num_runs, warmup, max_input
    )
    executor.shutdown()
    return results


# ---------------------------------------------------------------------------
# TensorRT-LLM PyTorch backend (modelopt PTQ checkpoint, no engine needed)
# ---------------------------------------------------------------------------


def run_benchmark_trt_pytorch(
    model_path: Path,
    prefix_lengths: list[int],
    num_runs: int,
    warmup: int,
    dtype: str = "bfloat16",
    max_num_tokens: int | None = None,
) -> dict[int, dict]:
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.inputs.data import TokensPrompt
    from tensorrt_llm.llmapi.llm_utils import KvCacheConfig
    from transformers import AutoTokenizer

    max_len = max(prefix_lengths) + 256
    if max_num_tokens is None:
        max_num_tokens = max_len

    print(f"  Loading TRT-LLM PyTorch: {model_path.name} (dtype={dtype}) ...")
    t0 = time.perf_counter()
    llm = LLM(
        model=str(model_path),
        dtype=dtype,
        max_num_tokens=max_num_tokens,
        skip_tokenizer_init=False,
        kv_cache_config=KvCacheConfig(enable_block_reuse=False),
    )
    load_s = time.perf_counter() - t0
    print(f"  Model loaded in {load_s:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    sampling = SamplingParams(max_tokens=1, temperature=0.0)

    def run_once(input_ids: list[int]) -> float:
        prompt = TokensPrompt(prompt_token_ids=input_ids)
        t_start = time.perf_counter()
        llm.generate([prompt], sampling_params=sampling, use_tqdm=False)
        return time.perf_counter() - t_start

    results = _run_length_loop(prefix_lengths, tokenizer, run_once, num_runs, warmup)
    del llm
    return results


# ---------------------------------------------------------------------------
# vLLM backend
# ---------------------------------------------------------------------------


def run_benchmark_vllm(
    model_path: Path,
    prefix_lengths: list[int],
    num_runs: int,
    warmup: int,
    dtype: str = "bfloat16",
    quantization: str | None = None,
    kv_cache_dtype: str = "auto",
    gpu_memory_utilization: float = 0.92,
    enable_chunked_prefill: bool = True,
    enforce_eager: bool = False,
    max_num_seqs: int = 1,
    max_num_batched_tokens: int | None = None,
    attention_backend: str | None = None,
    fuse_passes: bool = False,
    disabled_kernels: str | None = None,
    compile_sizes: list[int] | None = None,
) -> dict[int, dict]:
    import os
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    if attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend
    if disabled_kernels:
        os.environ["VLLM_DISABLED_KERNELS"] = disabled_kernels

    # Auto-detect modelopt quantization if checkpoint has hf_quant_config.json
    if quantization is None:
        detected = detect_ptq_quant(model_path)
        if detected:
            quantization = "modelopt"
            print(f"  Auto-detected modelopt quantization: {detected}")

    max_len = max(prefix_lengths) + 256
    if max_num_batched_tokens is None:
        max_num_batched_tokens = max_len
    quant_label = quantization or "none"
    print(
        f"  Loading vLLM: {model_path.name} (dtype={dtype}, quant={quant_label}, "
        f"kv_dtype={kv_cache_dtype}, attn={attention_backend or 'default'}) ..."
    )
    t0 = time.perf_counter()

    llm_kwargs: dict = dict(
        model=str(model_path),
        dtype=dtype,
        max_model_len=max_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        enable_prefix_caching=False,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    if quantization and quantization != "none":
        llm_kwargs["quantization"] = quantization
    if kv_cache_dtype != "auto":
        llm_kwargs["kv_cache_dtype"] = kv_cache_dtype

    compilation_config: dict = {}
    if fuse_passes:
        compilation_config["pass_config"] = {
            "fuse_norm_quant": True,
            "fuse_act_quant": True,
            "fuse_attn_quant": True,
        }
    if compile_sizes:
        compilation_config["compile_sizes"] = compile_sizes
    if compilation_config:
        llm_kwargs["compilation_config"] = compilation_config

    llm = LLM(**llm_kwargs)
    load_s = time.perf_counter() - t0
    print(f"  Model loaded in {load_s:.1f}s (max_model_len={max_len})")

    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(max_tokens=1, temperature=0)

    def run_once(input_ids: list[int]) -> float:
        prompt = TokensPrompt(prompt_token_ids=input_ids)
        t_start = time.perf_counter()
        llm.generate([prompt], sampling_params=sampling, use_tqdm=False)
        return time.perf_counter() - t_start

    return _run_length_loop(prefix_lengths, tokenizer, run_once, num_runs, warmup)


# ---------------------------------------------------------------------------
# Shared measurement loop
# ---------------------------------------------------------------------------


def _run_length_loop(
    prefix_lengths: list[int],
    tokenizer,
    run_once_fn,
    num_runs: int,
    warmup: int,
    max_input: int | None = None,
) -> dict[int, dict]:
    results: dict[int, dict] = {}
    for length in prefix_lengths:
        if max_input is not None and length > max_input:
            print(f"  [{length:>6} tok] SKIP — exceeds max_input_len={max_input}")
            results[length] = {
                "skipped": True,
                "reason": f"exceeds max_input_len={max_input}",
            }
            continue

        input_ids = make_token_ids(tokenizer, length)

        for w in range(warmup):
            try:
                ws = run_once_fn(input_ids)
                if w == 0:
                    print(f"  [{length:>6} tok] warmup: {format_time(ws)}")
            except Exception as e:
                err = str(e).split("\n")[0][:120]
                print(f"  [{length:>6} tok] warmup FAILED: {err}")
                results[length] = {"skipped": True, "reason": err}
                break
        else:
            times = []
            for r in range(num_runs):
                try:
                    t = run_once_fn(input_ids)
                    times.append(t)
                except Exception as e:
                    err = str(e).split("\n")[0][:120]
                    print(f"  [{length:>6} tok] run {r + 1} FAILED: {err}")
                    break

            if times:
                results[length] = _collect_results(times, length)
            else:
                results[length] = {"skipped": True, "reason": "all runs failed"}

    return results


# ---------------------------------------------------------------------------
# Subprocess isolation
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)


def _wait_gpu_free(max_wait: int = 30, threshold_pct: float = 0.90):
    for _ in range(max_wait):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free,memory.total",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=5,
            ).strip().split("\n")[0]
            free, total = [int(x.strip()) for x in out.split(",")]
            if free / total >= threshold_pct:
                return
        except Exception:
            pass
        time.sleep(1)
    print("  WARNING: GPU memory not fully freed, proceeding anyway")


def _trtllm_ld_path(python_exe: str) -> str:
    """Build LD_LIBRARY_PATH including TRT-LLM venv site-packages cuda libs."""
    import site
    venv_dir = str(Path(python_exe).resolve().parent.parent)
    site_dirs = [
        f"{venv_dir}/lib/python3.12/site-packages",
        f"{venv_dir}/lib/python3.11/site-packages",
    ]
    lib_subdirs = ["tensorrt_libs", "nvidia/nccl/lib", "nvidia/nvjitlink/lib",
                   "nvidia/nvshmem/lib", "nvidia/cuda_runtime/lib"]
    paths = list(_TRTLLM_LD_EXTRA)
    for sd in site_dirs:
        if Path(sd).is_dir():
            for sub in lib_subdirs:
                p = Path(sd) / sub
                if p.is_dir():
                    paths.append(str(p))
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    return ":".join(paths) + (f":{existing}" if existing else "")


def _run_trt_subprocess(
    engine_dir: Path,
    tokenizer_dir: Path,
    display_name: str,
    prefix_lengths: list[int],
    num_runs: int,
    warmup: int,
    result_json: Path,
    kv_free_gpu_memory_fraction: float = 0.92,
) -> bool:
    child_code = f"""\
import sys, json; sys.path.insert(0, {_SCRIPTS_DIR!r})
from bench_ttft import run_benchmark, get_engine_config
from pathlib import Path
edir = Path({str(engine_dir)!r})
results = run_benchmark(edir, Path({str(tokenizer_dir)!r}), {prefix_lengths!r}, {num_runs!r}, {warmup!r}, {kv_free_gpu_memory_fraction!r})
ename = edir.name.lower()
kv_dtype = "fp8_e4m3" if "kv_fp8" in ename else ("int8" if "kv_int8" in ename else "fp16")
data = {{"model": {display_name!r}, "backend": "tensorrt-llm-executor", "engine": edir.name, "tokenizer": {str(tokenizer_dir)!r}, "engine_config": get_engine_config(edir), "kv_cache_dtype": kv_dtype, "kv_free_gpu_memory_fraction": {kv_free_gpu_memory_fraction!r}, "kv_enable_block_reuse": False, "max_output_tokens": 1, "results": {{str(k): v for k, v in results.items()}}}}
Path({str(result_json)!r}).parent.mkdir(parents=True, exist_ok=True)
with open({str(result_json)!r}, "w") as f: json.dump(data, f, indent=2)
"""
    r = subprocess.run([sys.executable, "-c", child_code], timeout=1800)
    return r.returncode == 0


def _run_trt_pytorch_subprocess(
    python_exe: str,
    model_path: Path,
    display_name: str,
    prefix_lengths: list[int],
    num_runs: int,
    warmup: int,
    dtype: str,
    result_json: Path,
) -> bool:
    child_code = f"""\
import sys, json; sys.path.insert(0, {_SCRIPTS_DIR!r})
from bench_ttft import run_benchmark_trt_pytorch, detect_ptq_quant
from pathlib import Path
mp = Path({str(model_path)!r})
results = run_benchmark_trt_pytorch(mp, {prefix_lengths!r}, {num_runs!r}, {warmup!r}, {dtype!r})
quant = detect_ptq_quant(mp) or "unknown"
data = {{"model": {display_name!r}, "backend": "tensorrt-llm-pytorch", "model_path": str(mp), "dtype": {dtype!r}, "quantization": quant, "max_output_tokens": 1, "results": {{str(k): v for k, v in results.items()}}}}
Path({str(result_json)!r}).parent.mkdir(parents=True, exist_ok=True)
with open({str(result_json)!r}, "w") as f: json.dump(data, f, indent=2)
"""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = _trtllm_ld_path(python_exe)
    venv_bin = str(Path(python_exe).resolve().parent)
    env["PATH"] = venv_bin + ":" + env.get("PATH", "")
    r = subprocess.run([python_exe, "-c", child_code], timeout=1800, env=env)
    return r.returncode == 0


def _run_vllm_subprocess(
    python_exe: str,
    model_path: Path,
    display_name: str,
    prefix_lengths: list[int],
    num_runs: int,
    warmup: int,
    dtype: str,
    quantization: str | None,
    kv_cache_dtype: str,
    result_json: Path,
    gpu_memory_utilization: float = 0.92,
    enable_chunked_prefill: bool = True,
    enforce_eager: bool = False,
    max_num_seqs: int = 1,
    attention_backend: str | None = None,
    fuse_passes: bool = False,
    disabled_kernels: str | None = None,
    compile_sizes: list[int] | None = None,
) -> bool:
    quant_repr = repr(quantization)
    attn_repr = repr(attention_backend)
    dk_repr = repr(disabled_kernels)
    cs_repr = repr(compile_sizes)
    kv_repr = repr(kv_cache_dtype)
    child_code = f"""\
import sys, json; sys.path.insert(0, {_SCRIPTS_DIR!r})
from bench_ttft import run_benchmark_vllm, detect_ptq_quant, detect_ptq_label
from pathlib import Path
mp = Path({str(model_path)!r})
results = run_benchmark_vllm(mp, {prefix_lengths!r}, {num_runs!r}, {warmup!r}, {dtype!r}, {quant_repr}, {kv_repr}, {gpu_memory_utilization!r}, {enable_chunked_prefill!r}, {enforce_eager!r}, {max_num_seqs!r}, None, {attn_repr}, {fuse_passes!r}, {dk_repr}, {cs_repr})
quant = {quant_repr} or detect_ptq_quant(mp) or "none"
# kv_cache_dtype_arg is what the user passed; vLLM may auto-detect a different actual value
# from hf_quant_config.json (e.g. "auto" -> "fp8_e4m3"). Record both for clarity.
kv_actual = detect_ptq_label(mp).split("-KV_", 1)[1] if "-KV_" in detect_ptq_label(mp) else {kv_repr}
data = {{"model": {display_name!r}, "backend": "vllm", "dtype": {dtype!r}, "quantization": quant, "kv_cache_dtype_arg": {kv_repr}, "kv_cache_dtype_actual": kv_actual, "gpu_memory_utilization": {gpu_memory_utilization!r}, "max_num_seqs": {max_num_seqs!r}, "enable_chunked_prefill": {enable_chunked_prefill!r}, "enforce_eager": {enforce_eager!r}, "enable_prefix_caching": False, "max_output_tokens": 1, "model_path": str(mp), "attention_backend": {attn_repr}, "fuse_passes": {fuse_passes!r}, "results": {{str(k): v for k, v in results.items()}}}}
Path({str(result_json)!r}).parent.mkdir(parents=True, exist_ok=True)
with open({str(result_json)!r}, "w") as f: json.dump(data, f, indent=2)
"""
    env = os.environ.copy()
    venv_bin = str(Path(python_exe).resolve().parent)
    env["PATH"] = venv_bin + ":" + env.get("PATH", "")
    r = subprocess.run([python_exe, "-c", child_code], timeout=1800, env=env)
    return r.returncode == 0


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_table(
    gpu_name: str,
    all_results: dict[str, dict[int, dict]],
    prefix_lengths: list[int],
    title: str = "TTFT Benchmark",
):
    col_w = 18
    name_w = max(24, max((len(n) for n in all_results), default=24))
    print(f"\n{'=' * 80}")
    print(f"  {gpu_name} — {title}")
    print(f"{'=' * 80}\n")

    header = f"  {'Model':<{name_w}}"
    for length in prefix_lengths:
        label = _LABEL_MAP.get(length, f"{length} tok")
        header += f"  {label:>{col_w}}"
    print(header)
    print("  " + "-" * (name_w + (col_w + 2) * len(prefix_lengths)))

    for display_name, results in all_results.items():
        row = f"  {display_name:<{name_w}}"
        for length in prefix_lengths:
            r = results.get(length, {})
            if r.get("skipped"):
                cell = "—"
            else:
                cell = format_time(r["median_s"])
            row += f"  {cell:>{col_w}}"
        print(row)

    print()


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark TTFT (TRT-LLM Executor / TRT-LLM PyTorch / vLLM)",
    )
    root = Path(__file__).resolve().parent.parent.parent

    # TRT-LLM Executor options (pre-compiled engines)
    trt_group = parser.add_argument_group("TensorRT-LLM Executor (pre-compiled engines)")
    trt_group.add_argument(
        "--engines",
        nargs="+",
        metavar="ENGINE:TOKENIZER",
        help="TRT engine-to-tokenizer pairs. If omitted, auto-discovers under --engine-base.",
    )
    trt_group.add_argument(
        "--engine-base", type=Path, default=root / "outputs" / "engines",
    )
    trt_group.add_argument(
        "--models-dir", type=Path, default=root / "models",
    )
    trt_group.add_argument(
        "--no-trt", action="store_true", help="Skip TRT-LLM Executor benchmarks",
    )
    trt_group.add_argument(
        "--trt-kv-mem-fraction", type=float, default=0.92, metavar="FRACTION",
    )

    # TRT-LLM PyTorch backend options (modelopt PTQ checkpoint, no engine needed)
    trtp_group = parser.add_argument_group(
        "TensorRT-LLM PyTorch backend (modelopt PTQ checkpoint)"
    )
    trtp_group.add_argument(
        "--trt-pytorch",
        nargs="+",
        metavar="MODEL_PATH[:DTYPE]",
        help="PTQ checkpoint dirs for TRT-LLM PyTorch backend. "
             "DTYPE default=bfloat16. "
             "Quantization is auto-detected from hf_quant_config.json.",
    )
    trtp_group.add_argument(
        "--trt-pytorch-python",
        default=None,
        help="Python executable for TRT-LLM venv (default: auto-detect venv_trtllm1.1.0).",
    )

    # vLLM options
    vllm_group = parser.add_argument_group("vLLM")
    vllm_group.add_argument(
        "--vllm",
        nargs="+",
        metavar="MODEL_PATH[:DTYPE[:QUANT]]",
        help="vLLM model specs. QUANT auto-detected from hf_quant_config.json if omitted. "
             "Use 'modelopt' for modelopt PTQ checkpoints, 'none' to force no quantization.",
    )
    vllm_group.add_argument(
        "--vllm-python", default=None,
        help="Python executable for vLLM venv (default: auto-detect venv_vllm).",
    )
    vllm_group.add_argument(
        "--vllm-gpu-mem-fraction", type=float, default=0.92, metavar="FRACTION",
    )
    vllm_group.add_argument(
        "--vllm-kv-cache-dtype", default="auto", metavar="DTYPE",
        help="KV cache dtype for vLLM: auto, fp8_e4m3, fp8_e5m2, int8 (default: auto).",
    )
    vllm_group.add_argument(
        "--vllm-no-chunked-prefill", action="store_true",
        help="Disable chunked prefill in vLLM.",
    )
    vllm_group.add_argument(
        "--vllm-enforce-eager", action="store_true",
        help="Disable torch.compile in vLLM.",
    )
    vllm_group.add_argument(
        "--vllm-attention-backend", default=None, metavar="BACKEND",
    )
    vllm_group.add_argument("--vllm-fuse-passes", action="store_true")
    vllm_group.add_argument("--vllm-disabled-kernels", default=None, metavar="KERNELS")
    vllm_group.add_argument(
        "--vllm-compile-sizes", nargs="+", type=int, default=None, metavar="N",
    )

    # Shared options
    parser.add_argument(
        "--prefix-lengths", nargs="+", type=int, default=[500, 2000, 15000, 100000],
    )
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    gpu_name = detect_gpu()
    print(f"GPU: {gpu_name}")
    print(f"Prefix lengths: {args.prefix_lengths}")
    print(f"Runs: {args.num_runs} (+{args.warmup} warmup)\n")

    # --- Resolve vLLM Python ---
    vllm_python = args.vllm_python
    if args.vllm and not vllm_python:
        for candidate in [root / "venv_vllm" / "bin" / "python"]:
            if candidate.exists():
                vllm_python = str(candidate)
                break
        if not vllm_python:
            print("ERROR: --vllm-python required (cannot auto-detect vLLM venv)", file=sys.stderr)
            return 1

    # --- Resolve TRT-LLM PyTorch Python ---
    trtp_python = args.trt_pytorch_python
    if args.trt_pytorch and not trtp_python:
        for candidate in [root / "venv_trtllm1.1.0" / "bin" / "python"]:
            if candidate.exists():
                trtp_python = str(candidate)
                break
        if not trtp_python:
            print("ERROR: --trt-pytorch-python required (cannot auto-detect TRT-LLM venv)",
                  file=sys.stderr)
            return 1

    # --- Collect TRT-LLM Executor configs ---
    trt_configs: list[tuple[Path, Path, str]] = []
    if not args.no_trt:
        if args.engines:
            for spec in args.engines:
                if ":" not in spec:
                    print(f"ERROR: invalid spec '{spec}'", file=sys.stderr)
                    return 1
                e_str, t_str = spec.split(":", 1)
                e_dir, t_dir = Path(e_str), Path(t_str)
                if not (e_dir / "rank0.engine").exists():
                    print(f"ERROR: engine not found: {e_dir}", file=sys.stderr)
                    return 1
                if not t_dir.is_dir():
                    print(f"ERROR: tokenizer not found: {t_dir}", file=sys.stderr)
                    return 1
                trt_configs.append((e_dir, t_dir, clean_model_name(e_dir.name)))
        elif not args.trt_pytorch:
            # Only auto-discover if no explicit trt-pytorch provided
            trt_configs = discover_engines(args.engine_base, args.models_dir)

    # --- Collect TRT-LLM PyTorch configs ---
    trtp_models: list[tuple[Path, str, str]] = []
    if args.trt_pytorch:
        for spec in args.trt_pytorch:
            parts = spec.split(":")
            mp = Path(parts[0])
            dtype = parts[1] if len(parts) > 1 else "bfloat16"
            if not mp.is_dir():
                print(f"ERROR: model not found: {mp}", file=sys.stderr)
                return 1
            trtp_models.append((mp, clean_model_name(mp.name), dtype))

    # --- Collect vLLM configs ---
    vllm_models: list[tuple[Path, str, str, str | None]] = []
    if args.vllm:
        for spec in args.vllm:
            parts = spec.split(":")
            mp = Path(parts[0])
            dtype = parts[1] if len(parts) > 1 else "bfloat16"
            quant = parts[2] if len(parts) > 2 else None
            if quant == "none":
                quant = None
            if not mp.is_dir():
                print(f"ERROR: model not found: {mp}", file=sys.stderr)
                return 1
            vllm_models.append((mp, clean_model_name(mp.name), dtype, quant))

    if not trt_configs and not trtp_models and not vllm_models:
        print("ERROR: no engines, --trt-pytorch, or --vllm models specified.", file=sys.stderr)
        return 1

    # Print plan
    if trt_configs:
        print(f"TRT-LLM Executor engines ({len(trt_configs)}):")
        for ed, td, dn in trt_configs:
            mil = get_engine_config(ed).get("max_input_len", 0)
            print(f"  {dn:<20} {ed.name}  (max_input_len={mil})")
    if trtp_models:
        print(f"TRT-LLM PyTorch models ({len(trtp_models)}):")
        for mp, dn, dt in trtp_models:
            q = detect_ptq_quant(mp) or "unknown"
            print(f"  {dn:<20} {mp.name}  (dtype={dt}, quant={q})")
    if vllm_models:
        print(f"vLLM models ({len(vllm_models)}):")
        for mp, dn, dt, qt in vllm_models:
            ql = qt or f"auto-detect({detect_ptq_quant(mp) or 'none'})"
            print(f"  {dn:<20} {mp.name}  (dtype={dt}, quant={ql})")
    print()

    # --- Run benchmarks ---
    all_results: dict[str, dict[int, dict]] = {}
    full_data = []

    with tempfile.TemporaryDirectory(prefix="bench_ttft_") as tmpdir:
        idx = 0

        # TRT-LLM Executor
        for engine_dir, tokenizer_dir, base_name in trt_configs:
            ename_lower = engine_dir.name.lower()
            if "fp8" in ename_lower:
                trt_quant = "FP8"
            elif "nvfp4" in ename_lower:
                trt_quant = "NVFP4"
            elif "int4_awq" in ename_lower or "awq" in ename_lower:
                trt_quant = "AWQ"
            elif "fp16" in ename_lower:
                trt_quant = "FP16"
            else:
                trt_quant = "?"
            display = f"{base_name} (TRT-Exec {trt_quant})"
            print(f"{'=' * 60}\n  {display}\n{'=' * 60}")

            rj = Path(tmpdir) / f"result_{idx}.json"
            idx += 1
            ok = _run_trt_subprocess(
                engine_dir, tokenizer_dir, display,
                args.prefix_lengths, args.num_runs, args.warmup, rj,
                kv_free_gpu_memory_fraction=args.trt_kv_mem_fraction,
            )
            if ok and rj.exists():
                data = json.loads(rj.read_text())
                all_results[display] = {int(k): v for k, v in data["results"].items()}
                full_data.append(data)
            else:
                print("  FAILED — subprocess error")
                all_results[display] = {
                    ln: {"skipped": True, "reason": "subprocess failed"}
                    for ln in args.prefix_lengths
                }
            _wait_gpu_free()
            print()

        # TRT-LLM PyTorch backend
        for model_path, base_name, dtype in trtp_models:
            q = detect_ptq_label(model_path)
            display = f"{base_name} (TRT-PyTorch {q})"
            print(f"{'=' * 60}\n  {display}\n{'=' * 60}")

            rj = Path(tmpdir) / f"result_{idx}.json"
            idx += 1
            ok = _run_trt_pytorch_subprocess(
                trtp_python, model_path, display,
                args.prefix_lengths, args.num_runs, args.warmup,
                dtype, rj,
            )
            if ok and rj.exists():
                data = json.loads(rj.read_text())
                all_results[display] = {int(k): v for k, v in data["results"].items()}
                full_data.append(data)
            else:
                print("  FAILED — subprocess error")
                all_results[display] = {
                    ln: {"skipped": True, "reason": "subprocess failed"}
                    for ln in args.prefix_lengths
                }
            _wait_gpu_free()
            print()

        # vLLM
        for model_path, base_name, dtype, quant in vllm_models:
            qlabel = detect_ptq_label(model_path)
            display = f"{base_name} (vLLM {qlabel})"
            print(f"{'=' * 60}\n  {display}\n{'=' * 60}")

            rj = Path(tmpdir) / f"result_{idx}.json"
            idx += 1
            ok = _run_vllm_subprocess(
                vllm_python, model_path, display,
                args.prefix_lengths, args.num_runs, args.warmup,
                dtype, quant, args.vllm_kv_cache_dtype, rj,
                gpu_memory_utilization=args.vllm_gpu_mem_fraction,
                enable_chunked_prefill=not args.vllm_no_chunked_prefill,
                enforce_eager=args.vllm_enforce_eager,
                attention_backend=args.vllm_attention_backend,
                fuse_passes=args.vllm_fuse_passes,
                disabled_kernels=args.vllm_disabled_kernels,
                compile_sizes=args.vllm_compile_sizes,
            )
            if ok and rj.exists():
                data = json.loads(rj.read_text())
                all_results[display] = {int(k): v for k, v in data["results"].items()}
                full_data.append(data)
            else:
                print("  FAILED — subprocess error")
                all_results[display] = {
                    ln: {"skipped": True, "reason": "subprocess failed"}
                    for ln in args.prefix_lengths
                }
            _wait_gpu_free()
            print()

    # Summary table
    backends = set()
    if trt_configs:
        backends.add("TRT-Executor")
    if trtp_models:
        backends.add("TRT-PyTorch")
    if vllm_models:
        backends.add("vLLM")
    title = "TTFT Benchmark — " + " vs ".join(sorted(backends)) if backends else "TTFT Benchmark"
    print_table(gpu_name, all_results, args.prefix_lengths, title)

    # Save JSON
    out_path = args.out
    if out_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = root / "outputs" / "results" / f"ttft_{slugify(gpu_name)}_{ts}.json"
    output_data = {
        "gpu": gpu_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prefix_lengths": args.prefix_lengths,
        "num_runs": args.num_runs,
        "warmup": args.warmup,
        "benchmarks": full_data,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
