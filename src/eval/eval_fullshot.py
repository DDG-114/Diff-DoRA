"""
src/eval/eval_fullshot.py
--------------------------
Full-shot evaluation across all horizons (3/6/9/12).

Usage:
  python -m src.eval.eval_fullshot \
      --dataset st_evcdp \
      --adapter_dir outputs/single_lora_h6/adapter \
      --output outputs/fullshot_results.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.data.load_st_evcdp   import load_st_evcdp
from src.data.load_urbanev    import load_urbanev
from src.data.build_splits    import build_splits
from src.data.build_samples   import build_samples
from src.eval.metrics         import per_horizon_metrics
from src.models.qwen_peft     import load_model_and_tokenizer, load_peft_model, generate
from src.prompts.prompt_vanilla import build_vanilla_prompt
from src.prompts.parser       import parse_output

HORIZONS = [3, 6, 9, 12]


def eval_horizon(model, tokenizer, test_samples, horizon, node_idx, norm_min, norm_max, max_eval=300):
    preds, trues = [], []
    for s in test_samples[:max_eval]:
        sys_msg, usr_msg = build_vanilla_prompt(s, node_idx, horizon)
        out = generate(model, tokenizer, sys_msg, usr_msg, max_new_tokens=horizon * 12)
        arr = parse_output(out, expected_len=horizon)
        if arr is not None and len(arr) == horizon:
            preds.append(arr)
            trues.append(s["y"][:horizon, node_idx])
    if not preds:
        return None
    return per_horizon_metrics(preds, trues, horizon, norm_min, norm_max)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="st_evcdp")
    parser.add_argument("--adapter_dir",required=True)
    parser.add_argument("--node_idx",   type=int, default=0)
    parser.add_argument("--output",     default="outputs/fullshot_results.json")
    parser.add_argument("--max_eval",   type=int, default=300)
    args = parser.parse_args()

    raw    = load_st_evcdp() if args.dataset == "st_evcdp" else load_urbanev()
    splits = build_splits(raw, args.dataset)

    base_model, tokenizer = load_model_and_tokenizer()
    model = load_peft_model(base_model, args.adapter_dir)
    model.eval()

    all_results = {}
    for h in HORIZONS:
        print(f"Evaluating horizon={h} …")
        test_map   = build_samples(splits["test"], splits["timestamps_test"],
                                   adj=splits.get("adj"), horizons=[h])
        test_samples = test_map[h]
        r = eval_horizon(model, tokenizer, test_samples, h, args.node_idx,
                         splits["norm_min"], splits["norm_max"], args.max_eval)
        all_results[f"h{h}"] = r
        if r:
            print(f"  h={h}: RMSE={r['overall']['rmse']:.4f}  MAE={r['overall']['mae']:.4f}")

    output = {
        "dataset": args.dataset, "adapter": args.adapter_dir,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": all_results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
