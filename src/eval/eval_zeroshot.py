"""
src/eval/eval_zeroshot.py
--------------------------
Zero-shot generalisation: train on source nodes, test on unseen target nodes.
Source ratios: 20%, 40%, 60%, 80% of nodes used as source.
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
from src.models.qwen_peft     import load_model_and_tokenizer, get_lora_model, generate
from src.prompts.prompt_vanilla import build_vanilla_prompt
from src.prompts.parser       import parse_output
from src.train.train_single   import EVDataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

SOURCE_RATIOS = [0.20, 0.40, 0.60, 0.80]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="st_evcdp")
    parser.add_argument("--horizon",    type=int, default=6)
    parser.add_argument("--output_dir", default="outputs/zeroshot")
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr",         type=float, default=1e-4)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    raw     = load_st_evcdp() if args.dataset == "st_evcdp" else load_urbanev()
    splits  = build_splits(raw, args.dataset)
    N       = splits["train"].shape[1]
    all_nodes = list(range(N))

    results = {}
    base_model, tokenizer = load_model_and_tokenizer()

    for ratio in SOURCE_RATIOS:
        n_src = max(1, int(N * ratio))
        src_nodes = all_nodes[:n_src]
        tgt_nodes = all_nodes[n_src:]
        if not tgt_nodes:
            continue
        print(f"\n--- source_ratio={ratio:.2f}  src={src_nodes[:5]}...  tgt={tgt_nodes[:5]}... ---")

        # Build training samples only from source nodes
        raw_train = build_samples(splits["train"], splits["timestamps_train"],
                                  adj=splits.get("adj"), horizons=[args.horizon])[args.horizon]
        train_s = []
        for s in raw_train:
            for n in src_nodes:
                train_s.append(dict(s, node_idx=n))

        # Train
        model = get_lora_model(base_model)
        ds    = EVDataset(train_s[:1000], tokenizer, args.horizon, node_idx=0)
        coll  = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8)
        ta    = TrainingArguments(
            output_dir=str(out_dir / f"ratio_{ratio:.2f}" / "ckpt"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            learning_rate=args.lr,
            fp16=True,
            logging_steps=50,
            save_strategy="no",
            report_to="none",
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )
        Trainer(model=model, args=ta, train_dataset=ds, data_collator=coll).train()
        model.eval()

        # Evaluate on target (unseen) nodes
        raw_test = build_samples(splits["test"], splits["timestamps_test"],
                                 adj=splits.get("adj"), horizons=[args.horizon])[args.horizon]
        preds, trues = [], []
        for s in raw_test[:200]:
            for n in tgt_nodes[:5]:
                sys_msg, usr_msg = build_vanilla_prompt(s, n, args.horizon)
                out  = generate(model, tokenizer, sys_msg, usr_msg, max_new_tokens=args.horizon*12)
                arr  = parse_output(out, args.horizon)
                if arr is not None:
                    preds.append(arr)
                    trues.append(s["y"][:args.horizon, n])
        if preds:
            r = per_horizon_metrics(preds, trues, args.horizon,
                                    splits["norm_min"], splits["norm_max"])
            results[str(ratio)] = r
            print(f"  RMSE={r['overall']['rmse']:.4f}  MAE={r['overall']['mae']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "zeroshot_results.json", "w") as f:
        json.dump({"dataset": args.dataset, "horizon": args.horizon,
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "results": results}, f, indent=2)
    print(f"Saved to {out_dir}/zeroshot_results.json")


if __name__ == "__main__":
    main()
