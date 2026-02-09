#!/bin/bash
# Rename Model Optimizer output dirs (saved_models_*) to convention:
#   llama-3.2-3b-instruct-trtllm-ckpt-wq_fp8-kv_fp16
#   llama-3.1-8b-instruct-trtllm-ckpt-wq_int4_awq-kv_fp8
#   mistral-7b-instruct-v0.3-trtllm-ckpt-wq_fp8-kv_fp16
# Run from project root or set CKPT_ROOT.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_ROOT}/outputs/ckpts}"

cd "$CKPT_ROOT"

# Map: saved_models_* pattern -> convention name (lowercase, -trtllm-ckpt-wq_*-kv_*)
# Model Optimizer names: saved_models_<path_sanitized>_<quant>_kv_<kv>
rename_one() {
    local src="$1"
    local dest="$2"
    if [ -d "$src" ] && [ ! -e "$dest" ]; then
        echo "Renaming: $src -> $dest"
        mv "$src" "$dest"
    elif [ -d "$src" ] && [ -e "$dest" ]; then
        echo "Skip (dest exists): $src -> $dest"
    fi
}

# Llama-3.2-3B: MODEL_NAME = basename + sed 's/[^0-9a-zA-Z\-]/_/g' => Llama-3_2-3B-Instruct
for q in fp8 int4_awq; do
    for kv in fp16 fp8; do
        base="llama-3.2-3b-instruct-trtllm-ckpt-wq_${q}-kv_${kv}"
        for prefix in "Llama-3_2-3B-Instruct" "Llama-3.2-3B-Instruct" "meta-llama_Llama-3.2-3B-Instruct"; do
            src="saved_models_${prefix}_${q}_kv_${kv}"
            [ -d "$src" ] && rename_one "$src" "$base" && break
        done
    done
done

# Llama-3.1-8B
for q in fp8 int4_awq; do
    for kv in fp16 fp8; do
        base="llama-3.1-8b-instruct-trtllm-ckpt-wq_${q}-kv_${kv}"
        for prefix in "Llama-3_1-8B-Instruct" "Llama-3.1-8B-Instruct" "meta-llama_Llama-3.1-8B-Instruct"; do
            src="saved_models_${prefix}_${q}_kv_${kv}"
            [ -d "$src" ] && rename_one "$src" "$base" && break
        done
    done
done

# Mistral-7B: basename Mistral-7B-Instruct-v0.3 => Mistral-7B-Instruct-v0_3
for q in fp8 int4_awq; do
    for kv in fp16 fp8; do
        base="mistral-7b-instruct-v0.3-trtllm-ckpt-wq_${q}-kv_${kv}"
        for prefix in "Mistral-7B-Instruct-v0_3" "Mistral-7B-Instruct-v0.3" "mistralai_Mistral-7B-Instruct-v0.3"; do
            src="saved_models_${prefix}_${q}_kv_${kv}"
            [ -d "$src" ] && rename_one "$src" "$base" && break
        done
    done
done

# Fallback: any remaining saved_models_* -> try generic lowercase convention
for dir in saved_models_*; do
    [ ! -d "$dir" ] && continue
    # Strip saved_models_ and normalize to lowercase; replace _ with -
    rest="${dir#saved_models_}"
    # Heuristic: last part is _<quant>_kv_<kv>
    if [[ "$rest" =~ _(int4_awq|fp8)_kv_(fp8|fp16)$ ]]; then
        quant="${BASH_REMATCH[1]}"
        kv="${BASH_REMATCH[2]}"
        base="${rest%_${quant}_kv_${kv}}"
        base_lower=$(echo "$base" | tr '[:upper:]' '[:lower:]' | sed 's/[^0-9a-z]/-/g' | sed 's/--*/-/g' | sed 's/^-//;s/-$//')
        dest="${base_lower}-trtllm-ckpt-wq_${quant}-kv_${kv}"
        [ ! -e "$dest" ] && rename_one "$dir" "$dest"
    fi
done

echo "Done. Check $CKPT_ROOT"
