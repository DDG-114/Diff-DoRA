"""
src/eval/eval_ablation.py
--------------------------
Ablation: compare legacy engineering variants such as vanilla / RAG / RAG+CoT.
Each variant is identified by a run name and an adapter path.

For the paper-defined ablations (`w/o MoE`, `w/o CoT`, `w/o DoRA`, `Base Model`),
use `src.eval.eval_paper_ablation` instead.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.data.load_st_evcdp   import load_st_evcdp
from src.data.load_urbanev    import load_urbanev
from src.data.build_splits    import build_splits
from src.data.build_samples   import build_samples
from src.eval.metrics         import per_horizon_metrics
from src.models.qwen_peft     import load_model_and_tokenizer, load_peft_model, generate
from src.prompts.prompt_vanilla import build_vanilla_prompt
from src.prompts.parser       import parse_output


def eval_adapter(base_model, tokenizer, adapter_path, test_s, horizon, node_idx, splits, max_eval=300):
    model = load_peft_model(base_model, adapter_path)
    model.eval()
    preds, trues = [], []
    for s in test_s[:max_eval]:
        sys_msg, usr_msg = build_vanilla_prompt(s, node_idx, horizon)
        out = generate(model, tokenizer, sys_msg, usr_msg, max_new_tokens=horizon*12)
        arr = parse_output(out, expected_len=horizon)
        if arr is not None:
            preds.append(arr)
            trues.append(s["y"][:horizon, node_idx])
    if preds:
        return per_horizon_metrics(preds, trues, horizon, splits["norm_min"], splits["norm_max"])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="st_evcdp")
    parser.add_argument("--horizon",    type=int, default=6)
    parser.add_argument("--node_idx",   type=int, default=0)
    parser.add_argument("--output",     default="outputs/ablation_results.json")
    # Adapter paths for each variant
    parser.add_argument("--vanilla",    default=None)
    parser.add_argument("--rag",        default=None)
    parser.add_argument("--rag_cot",    default=None)
    parser.add_argument("--moe",        default=None)
    parser.add_argument("--diff_dora",  default=None)
    args = parser.parse_args()

    raw    = load_st_evcdp() if args.dataset == "st_evcdp" else load_urbanev()
    splits = build_splits(raw, args.dataset)
    test_map = build_samples(splits["test"], splits["timestamps_test"],
                             adj=splits.get("adj"), horizons=[args.horizon])
    test_s = test_map[args.horizon]

    base_model, tokenizer = load_model_and_tokenizer()

    variants = {
        "vanilla":   args.vanilla,
        "rag":       args.rag,
        "rag_cot":   args.rag_cot,
        "moe":       args.moe,
        "diff_dora": args.diff_dora,
    }
    results = {}
    for name, path in variants.items():
        if path is None:
            continue
        print(f"Evaluating {name} …")
        r = eval_adapter(base_model, tokenizer, path, test_s,
                         args.horizon, args.node_idx, splits)
        results[name] = r
        if r:
            print(f"  {name}: RMSE={r['overall']['rmse']:.4f}  MAE={r['overall']['mae']:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"dataset": args.dataset, "horizon": args.horizon,
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "ablation": results}, f, indent=2)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
