#!/usr/bin/env python3
"""
Run accuracy eval (toxicity binary) on each Llama 8B checkpoint under outputs/ckpts.
Writes outputs/llama8b_accuracy_results.json and prints a table.

Note: Model Optimizer quantized ckpts (FP8 / INT4 AWQ) use dtypes (e.g. Float8_e4m3fn) that
AutoModelForCausalLM.from_pretrained does not support; this script will fail on those. Use it
for FP16 (unquantized) baselines only. For quantized ckpts, run accuracy via trtllm-serve
and a client script that sends the same dataset and computes AUROC/accuracy.
"""
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Config
CKPT_ROOT = PROJECT_ROOT / "outputs/ckpts"
OUT_JSON = PROJECT_ROOT / "outputs/llama8b_accuracy_results.json"
LABEL_KEY = "label"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 4
MODE = "binary"

# Llama 8B checkpoint dir names (convention)
LLAMA8B_CKPT_PATTERNS = [
    "llama-3.1-8b-instruct-trtllm-ckpt-wq_fp8-kv_fp16",
    "llama-3.1-8b-instruct-trtllm-ckpt-wq_fp8-kv_fp8",
    "llama-3.1-8b-instruct-trtllm-ckpt-wq_int4_awq-kv_fp16",
    "llama-3.1-8b-instruct-trtllm-ckpt-wq_int4_awq-kv_fp8",
]


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import torch
    import numpy as np
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    # Minimal template (no llm-finetuning dep)
    class PromptTemplate:
        def __init__(self, name: str, template: str):
            self.name, self.template = name, template
        def format(self, example: dict) -> str:
            return self.template.format(**example)

    template_str = """\
The user will provide you with a text and you need to determine if it is toxic or not.

Message:

```
{text}
```

Respond with a single word. Respond with "true" (no quotes) if the text is toxic, and "false" (no quotes) if it is not.
"""
    template = PromptTemplate("toxicity", template_str)

    # Postprocessor: logits -> prob_true (binary)
    from scipy.special import softmax
    from typing import Set, Dict, List

    class BooleanProbabilityPostprocessor:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.vocab_ixs_counted_as_true = set()
            self.vocab_ixs_counted_as_false = set()
            vocab_ixs = list(range(tokenizer.vocab_size))
            hf_tokens = tokenizer.convert_ids_to_tokens(vocab_ixs)
            vocab_strings = [tokenizer.convert_tokens_to_string([t]) for t in hf_tokens]
            for ix, token in zip(vocab_ixs, vocab_strings):
                t = token.lower().strip()
                if t == "true":
                    self.vocab_ixs_counted_as_true.add(ix)
                elif t == "false":
                    self.vocab_ixs_counted_as_false.add(ix)

        def postprocess_logits(self, logits: List[float]) -> float:
            probs = softmax(logits)
            p_true = sum(probs[i] for i in self.vocab_ixs_counted_as_true)
            p_false = sum(probs[i] for i in self.vocab_ixs_counted_as_false)
            try:
                return p_true / (p_true + p_false)
            except ZeroDivisionError:
                return 0.0

    # Dataset that has "text" and label_key
    ds = load_dataset("rungalileo/automated-ft-luna-toxicity", split="test")
    # Map to "text" if needed (some datasets use "comment" or "content")
    if "text" not in ds.column_names and "comment" in ds.column_names:
        ds = ds.rename_column("comment", "text")
    if "text" not in ds.column_names:
        raise RuntimeError(f"Dataset must have 'text' or 'comment'. Columns: {ds.column_names}")

    def apply_template(ex, tokenizer, template):
        messages = [{"role": "user", "content": template.format(ex)}]
        ex["model_input_text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return ex

    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for ckpt_name in LLAMA8B_CKPT_PATTERNS:
        ckpt_path = CKPT_ROOT / ckpt_name
        if not ckpt_path.is_dir():
            print(f"Skip (not found): {ckpt_path}")
            results.append({"checkpoint": ckpt_name, "error": "not_found"})
            continue

        print(f"Loading {ckpt_name} ...")
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_path), device_map="auto", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path), trust_remote_code=True)
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ds_mapped = ds.map(
            apply_template,
            fn_kwargs={"tokenizer": tokenizer, "template": template},
            desc="Apply template",
        )

        ppp = BooleanProbabilityPostprocessor(tokenizer)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        class EvalDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.samples = []
                for ex in ds_mapped:
                    enc = tokenizer(
                        ex["model_input_text"],
                        truncation=True,
                        max_length=MAX_SEQ_LENGTH,
                        add_special_tokens=False,
                    )
                    label = 1 if ex.get(LABEL_KEY, 0) in (1, True, "1", "true") else 0
                    self.samples.append({
                        "input_ids": enc["input_ids"],
                        "attention_mask": enc["attention_mask"],
                        "label": label,
                    })

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, i):
                return self.samples[i]

        def collate_with_labels(batch):
            model_inputs = [{"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]} for x in batch]
            padded = data_collator(model_inputs)
            padded["label"] = torch.tensor([x["label"] for x in batch], dtype=torch.long)
            return padded

        eval_ds = EvalDataset()
        loader = DataLoader(
            eval_ds,
            batch_size=BATCH_SIZE,
            collate_fn=collate_with_labels,
            num_workers=0,
        )

        all_labels = []
        all_probs = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc=ckpt_name):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("label")
                all_labels.extend(labels.cpu().tolist())
                out = model(**batch)
                logits = out.logits
                positions = (batch["attention_mask"].sum(dim=1) - 1).long().to(device)
                batch_idx = torch.arange(logits.size(0), device=device)
                logits_last = logits[batch_idx, positions, :].cpu().float().numpy()
                for row in logits_last:
                    all_probs.append(ppp.postprocess_logits(row.tolist()))

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        preds = (all_probs >= 0.5).astype(int)

        auroc = roc_auc_score(all_labels, all_probs)
        acc = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds, zero_division=0)

        row = {
            "checkpoint": ckpt_name,
            "AUROC": round(auroc, 4),
            "accuracy": round(acc, 4),
            "F1": round(f1, 4),
        }
        results.append(row)
        print(f"  AUROC={auroc:.4f} accuracy={acc:.4f} F1={f1:.4f}")

        del model
        torch.cuda.empty_cache()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_JSON}")

    # Table
    print("\nLlama 8B quantization – accuracy (toxicity test set)")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"  {r['checkpoint']}: {r['error']}")
        else:
            print(f"  {r['checkpoint']}")
            print(f"    AUROC={r['AUROC']}  accuracy={r['accuracy']}  F1={r['F1']}")
    print("-" * 70)


if __name__ == "__main__":
    main()
