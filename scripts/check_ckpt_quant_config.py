#!/usr/bin/env python3
"""
Check that each HF checkpoint under CKPT_ROOT has hf_quant_config.json
consistent with its directory name (wq_<quant>-kv_<kv>).

Expectations:
- *-kv_fp16: kv_cache_quant_algo must be null or absent (FP16 = no KV quant).
- *-kv_fp8:  kv_cache_quant_algo must be "FP8".
- wq_fp8:    quant_algo must be "FP8".
- wq_int4_awq: quant_algo must be "W4A16_AWQ".

Also checks that required files exist (config.json, hf_quant_config.json, weight files).
Exits with 0 if all pass, 1 if any check fails.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


# Dir name: ...-trtllm-ckpt-wq_<quant>-kv_<kv>  or  saved_models_*_<quant>_kv_<kv|none>
CONVENTION_PATTERN = re.compile(
    r".*-trtllm-ckpt-wq_(?P<wq>fp8|int4_awq)-kv_(?P<kv>fp16|fp8)$"
)
SAVED_MODELS_PATTERN = re.compile(
    r"^saved_models_.*_(?P<wq>fp8|int4_awq)_kv_(?P<kv>fp8|fp16|none)$"
)

EXPECTED_QUANT_ALGO = {"fp8": "FP8", "int4_awq": "W4A16_AWQ"}
EXPECTED_KV_FP16 = (None, "null")  # null in JSON, or key absent
EXPECTED_KV_FP8 = "FP8"


def parse_dir_name(name: str) -> tuple[str | None, str | None]:
    """Return (wq, kv) from directory name, or (None, None) if not a known pattern."""
    m = CONVENTION_PATTERN.match(name)
    if m:
        return m.group("wq"), m.group("kv")
    m = SAVED_MODELS_PATTERN.match(name)
    if m:
        kv = m.group("kv")
        if kv == "none":
            kv = "fp16"
        return m.group("wq"), kv
    return None, None


def check_ckpt(ckpt_dir: Path, strict_files: bool = True) -> list[str]:
    """Check one checkpoint. Return list of error messages (empty if OK)."""
    errors = []
    name = ckpt_dir.name
    wq, kv = parse_dir_name(name)
    if wq is None:
        return []  # skip non-matching dirs

    # Required files
    if strict_files:
        required = ["config.json", "hf_quant_config.json"]
        for f in required:
            if not (ckpt_dir / f).is_file():
                errors.append(f"{name}: missing {f}")
        if not any(ckpt_dir.glob("*.safetensors")):
            errors.append(f"{name}: no .safetensors weight files")
        if errors:
            return errors

    config_path = ckpt_dir / "hf_quant_config.json"
    if not config_path.is_file():
        errors.append(f"{name}: no hf_quant_config.json")
        return errors

    with open(config_path) as f:
        data = json.load(f)
    quant = data.get("quantization") or {}
    quant_algo = quant.get("quant_algo")
    kv_algo = quant.get("kv_cache_quant_algo")

    expected_wq = EXPECTED_QUANT_ALGO.get(wq)
    if expected_wq and quant_algo != expected_wq:
        errors.append(
            f"{name}: quant_algo={quant_algo!r} (expected {expected_wq!r} for wq_{wq})"
        )

    if kv == "fp16":
        if kv_algo is not None and kv_algo != "null" and str(kv_algo).upper() == "FP8":
            errors.append(
                f"{name}: kv_cache_quant_algo={kv_algo!r} (expected null/absent for kv_fp16)"
            )
    elif kv == "fp8":
        if kv_algo != EXPECTED_KV_FP8:
            errors.append(
                f"{name}: kv_cache_quant_algo={kv_algo!r} (expected {EXPECTED_KV_FP8!r} for kv_fp8)"
            )

    return errors


def main():
    p = argparse.ArgumentParser(description="Verify ckpt dirs: name vs hf_quant_config.json")
    p.add_argument(
        "ckpt_root",
        nargs="?",
        default=os.environ.get("CKPT_ROOT", "outputs/ckpts"),
        help="Checkpoint root directory",
    )
    p.add_argument(
        "--no-strict-files",
        action="store_true",
        help="Only check hf_quant_config.json; do not require config.json / safetensors",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print OK for each checked dir",
    )
    args = p.parse_args()
    root = Path(args.ckpt_root)
    if not root.is_dir():
        print(f"Error: not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    all_errors = []
    checked = 0
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        errs = check_ckpt(d, strict_files=not args.no_strict_files)
        if errs:
            all_errors.extend(errs)
        if errs or args.verbose:
            checked += 1
        if not errs and args.verbose:
            print(f"OK: {d.name}")

    for e in all_errors:
        print(f"CHECK FAIL: {e}", file=sys.stderr)
    if all_errors:
        print(f"\nTotal: {len(all_errors)} failure(s) in {checked} dir(s) checked.", file=sys.stderr)
        sys.exit(1)
    if checked:
        print(f"All {checked} checkpoint(s) passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
