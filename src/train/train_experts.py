"""
src/train/train_experts.py
---------------------------
Train two independent expert adapters (CBD + Residential) using hard routing.

Usage:
  python -m src.train.train_experts \
      --dataset st_evcdp \
      --horizon 6 \
      --output_dir outputs/moe_experts_h6
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from src.data.load_st_evcdp   import load_st_evcdp
from src.data.load_urbanev    import load_urbanev
from src.data.build_splits    import build_splits
from src.data.build_samples   import build_samples
from src.eval.metrics         import per_horizon_metrics
from src.models.qwen_peft     import load_model_and_tokenizer, get_lora_model, generate
from src.prompts.prompt_vanilla import build_vanilla_prompt
from src.prompts.parser       import parse_output
from src.routing.build_labels import build_routing_labels
from src.routing.hard_router  import HardRouter
from src.train.train_single   import EVDataset   # re-use dataset class


def train_one_expert(
    expert_id: int,
    samples_with_node: list[dict],
    tokenizer,
    base_model,
    out_dir: Path,
    args,
):
    print(f"\n=== Training Expert {expert_id} ({len(samples_with_node)} samples) ===")
    peft_model = get_lora_model(base_model, use_dora=args.use_dora)

    # EVDataset expects per-node samples with "node_idx" baked in; we adapt here:
    # Each dict already has the correct "node_idx" attached
    train_ds = EVDataset(
        samples_with_node, tokenizer, args.horizon,
        node_idx=0  # override: x_hist is already sliced per node
    )
    # We need a version where build_vanilla_prompt uses the stored node_idx
    # so we override _build to use sample["node_idx"]
    # Easiest: just pass node_idx=0 and pre-slice x_hist/y
    # (x_hist here is already the full (12,N) – we access col node_idx in EVDataset)
    # To keep it simple, EVDataset accepts node_idx per-sample via "node_idx" key.

    collator = DataCollatorForSeq2Seq(tokenizer, model=peft_model, padding=True, pad_to_multiple_of=8)
    training_args = TrainingArguments(
        output_dir=str(out_dir / f"expert_{expert_id}" / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=20,
        save_strategy="epoch",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    trainer.train()
    save_path = str(out_dir / f"expert_{expert_id}" / "adapter")
    peft_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Expert {expert_id} saved → {save_path}")
    return peft_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="st_evcdp")
    parser.add_argument("--horizon",    type=int,   default=6)
    parser.add_argument("--output_dir", default="outputs/moe_experts_h6")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--use_dora",   action="store_true")
    parser.add_argument("--max_samples_per_expert", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data
    raw    = load_st_evcdp() if args.dataset == "st_evcdp" else load_urbanev()
    splits = build_splits(raw, args.dataset)

    # 2. Routing labels
    labels = build_routing_labels(splits["train"], raw.get("node_meta"))
    router = HardRouter(labels)
    N = splits["train"].shape[1]

    # 3. Build per-node samples tagged with node_idx
    print("Building per-node samples …")
    train_samples_map = build_samples(
        splits["train"], splits["timestamps_train"],
        adj=splits.get("adj"), horizons=[args.horizon]
    )
    raw_samples = train_samples_map[args.horizon]
    # Tag each sample with all nodes; we'll expand per-node
    tagged: dict[int, list] = {0: [], 1: []}
    for s in raw_samples:
        for n in range(N):
            eid = router.route(n)
            if len(tagged[eid]) < args.max_samples_per_expert:
                s_node = dict(s, node_idx=n)
                tagged[eid].append(s_node)

    # 4. Load base model once
    print("Loading base model …")
    base_model, tokenizer = load_model_and_tokenizer()

    # 5. Train each expert
    trained = {}
    for eid in (0, 1):
        trained[eid] = train_one_expert(
            eid, tagged[eid], tokenizer, base_model, out_dir, args
        )

    # 6. Evaluate both experts on test split
    test_map = build_samples(
        splits["test"], splits["timestamps_test"],
        adj=splits.get("adj"), horizons=[args.horizon]
    )
    test_samples = test_map[args.horizon]
    results = {}
    for eid in (0, 1):
        model = trained[eid]
        model.eval()
        nodes = router.nodes_for_expert(eid)
        preds, trues = [], []
        for s in test_samples[:200]:
            for n in nodes[:5]:   # limit for speed
                sys_msg, usr_msg = build_vanilla_prompt(s, n, args.horizon)
                out = generate(model, tokenizer, sys_msg, usr_msg, max_new_tokens=128)
                arr = parse_output(out, args.horizon)
                if arr is not None:
                    preds.append(arr)
                    trues.append(s["y"][:args.horizon, n])
        if preds:
            m = per_horizon_metrics(preds, trues, args.horizon,
                                    splits["norm_min"], splits["norm_max"])
            results[f"expert_{eid}"] = m
            print(f"Expert {eid}: {m['overall']}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"dataset": args.dataset, "horizon": args.horizon,
                   "results": results,
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print(f"Saved to {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
