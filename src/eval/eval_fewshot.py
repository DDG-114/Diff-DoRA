"""
src/eval/eval_fewshot.py
-------------------------
Few-shot evaluation: train on k% of training data, test on full test set.
Ratios: 5%, 10%, 20%, 40%, 100%.
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
from src.utils.node_context   import extract_node_static_context, normalise_domain_label
from src.eval.metrics         import per_horizon_metrics
from src.models.qwen_peft     import load_model_and_tokenizer, get_lora_model, generate
from src.prompts.prompt_vanilla import build_vanilla_prompt
from src.prompts.parser       import parse_output
from src.train.train_single   import EVDataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

RATIOS = [0.05, 0.10, 0.20, 0.40, 1.00]


def train_and_eval(base_model, tokenizer, train_s, test_s, horizon, node_idx, splits, out_dir, args, ratio):
    model = get_lora_model(base_model)
    ds    = EVDataset(
        train_s,
        tokenizer,
        horizon,
        node_idx=node_idx,
        node_meta=splits.get("node_meta"),
        node_ids=splits.get("node_ids"),
        poi=splits.get("poi"),
    )
    coll  = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8)
    ta    = TrainingArguments(
        output_dir=str(out_dir / f"ratio_{ratio:.2f}" / "ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=ta, train_dataset=ds, data_collator=coll)
    trainer.train()
    model.eval()
    preds, trues = [], []
    for s in test_s[:200]:
        static_context = extract_node_static_context(
            node_idx,
            node_ids=splits.get("node_ids"),
            node_meta=splits.get("node_meta"),
        )
        sys_msg, usr_msg = build_vanilla_prompt(
            s,
            node_idx,
            horizon,
            domain_label=normalise_domain_label(static_context.get("zone_type")),
            static_context=static_context,
        )
        out = generate(model, tokenizer, sys_msg, usr_msg, max_new_tokens=horizon * 12)
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
    parser.add_argument("--output_dir", default="outputs/fewshot")
    parser.add_argument("--epochs",     type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--history_len", type=int, default=12)
    parser.add_argument("--neighbor_k", type=int, default=7)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    raw     = load_st_evcdp() if args.dataset == "st_evcdp" else load_urbanev()
    splits  = build_splits(raw, args.dataset)
    train_map = build_samples(splits["train"], splits["timestamps_train"],
                              adj=splits.get("adj"), horizons=[args.horizon],
                              history_len=args.history_len, neighbor_k=args.neighbor_k)
    test_map  = build_samples(splits["test"],  splits["timestamps_test"],
                              adj=splits.get("adj"), horizons=[args.horizon],
                              history_len=args.history_len, neighbor_k=args.neighbor_k)
    all_train = train_map[args.horizon]
    test_s    = test_map[args.horizon]

    base_model, tokenizer = load_model_and_tokenizer()

    results = {}
    for ratio in RATIOS:
        n = max(1, int(len(all_train) * ratio))
        sub = all_train[:n]
        print(f"\n--- ratio={ratio:.2f}  n_train={n} ---")
        r = train_and_eval(base_model, tokenizer, sub, test_s, args.horizon,
                           args.node_idx, splits, out_dir, args, ratio)
        results[str(ratio)] = r
        if r:
            print(f"  RMSE={r['overall']['rmse']:.4f}  MAE={r['overall']['mae']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "fewshot_results.json", "w") as f:
        json.dump({"dataset": args.dataset, "horizon": args.horizon,
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "results": results}, f, indent=2)
    print(f"Saved to {out_dir}/fewshot_results.json")


if __name__ == "__main__":
    main()
