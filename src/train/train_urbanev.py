"""
src/train/train_urbanev.py
--------------------------
UrbanEV dedicated training script with fixed defaults:
- split: 8:1:1 (via build_splits for dataset='urbanev')
- epochs: 2
- learning_rate: 2e-4
- max_length: 2560
- LoRA rank: 32
- history window: 12
- RAG top-k: 2
- neighbour top-k: 7

Usage:
  /home/kaga/diffdora/.venv/bin/python -m src.train.train_urbanev \
      --horizon 6 \
      --output_dir outputs/urbanev_r32_h6
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from src.data.load_urbanev import load_urbanev
from src.data.build_splits import build_splits
from src.data.build_samples import build_samples
from src.eval.metrics import per_horizon_metrics
from src.models.qwen_peft import load_model_and_tokenizer, get_lora_model
from src.retrieval.knn_retriever import KNNRetriever
from src.train.train_single import EVDataset, run_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--node_idx", type=int, default=0)
    parser.add_argument("--output_dir", default="outputs/urbanev_r32_h6")

    # Required defaults from user request
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=2560)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--history_len", type=int, default=12)
    parser.add_argument("--retrieval_top_k", type=int, default=2)
    parser.add_argument("--neighbor_k", type=int, default=7)

    # Runtime knobs
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_checkpointing", dest="gradient_checkpointing",
                        action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                        action="store_false")
    parser.add_argument("--retrieval_cache", default="",
                        help="Path to retrieval cache pkl; default data/retrieval_cache/urbanev_h{horizon}.pkl")
    parser.add_argument("--max_train_samples", type=int, default=0,
                        help="If >0, cap train samples for quick runs")
    parser.add_argument("--eval_max_samples", type=int, default=20,
                        help="Maximum test samples for evaluation")
    parser.add_argument("--eval_max_new_tokens", type=int, default=160)
    parser.add_argument("--eval_sampling", choices=["head", "random"], default="random")
    parser.add_argument("--eval_seed", type=int, default=42)

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Data: UrbanEV (split = 8:1:1 in build_splits)
    print("Loading UrbanEV …")
    raw = load_urbanev()
    splits = build_splits(raw, "urbanev")

    # 2) Build samples with history=12 and neighbour top-k=7
    print("Building samples …")
    train_map = build_samples(
        splits["train"],
        splits["timestamps_train"],
        adj=splits.get("adj"),
        horizons=[args.horizon],
        history_len=args.history_len,
        neighbor_k=args.neighbor_k,
    )
    test_map = build_samples(
        splits["test"],
        splits["timestamps_test"],
        adj=splits.get("adj"),
        horizons=[args.horizon],
        history_len=args.history_len,
        neighbor_k=args.neighbor_k,
    )

    train_samples = train_map[args.horizon]
    if args.max_train_samples > 0:
        train_samples = train_samples[:args.max_train_samples]
    test_samples = test_map[args.horizon]

    # 3) RAG retriever top-k=2
    if args.retrieval_cache:
        cache_path = Path(args.retrieval_cache)
    else:
        cache_path = Path(f"data/retrieval_cache/urbanev_h{args.horizon}.pkl")

    if cache_path.exists():
        retriever = KNNRetriever.load(cache_path)
        retriever.top_k = args.retrieval_top_k
        print(f"Loaded retrieval cache: {cache_path} (top_k={retriever.top_k})")
    else:
        retriever = KNNRetriever(train_samples, top_k=args.retrieval_top_k)
        print(f"Built in-memory retriever (top_k={args.retrieval_top_k})")

    # 4) Model with LoRA rank=32
    print("Loading model …")
    model, tokenizer = load_model_and_tokenizer()
    model = get_lora_model(model, r=args.rank, lora_alpha=args.lora_alpha)

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # 5) Dataset/tokenization with max_length=2560
    train_ds = EVDataset(
        train_samples,
        tokenizer,
        args.horizon,
        max_length=args.max_length,
        node_idx=args.node_idx,
        use_rag=True,
        retriever=retriever,
        weather=splits.get("weather"),
        price=splits.get("price"),
        node_meta=splits.get("node_meta"),
        node_ids=splits.get("node_ids"),
        poi=splits.get("poi"),
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=20,
        save_strategy="epoch",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    print("Training …")
    trainer.train()

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Adapter saved: {adapter_dir}")

    # 6) Quick evaluation
    print("Evaluating …")
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.use_cache = True

    preds, trues = run_inference(
        model,
        tokenizer,
        test_samples,
        args.horizon,
        args.node_idx,
        max_samples=args.eval_max_samples,
        max_new_tokens=args.eval_max_new_tokens,
        use_rag=True,
        retriever=retriever,
        node_meta=splits.get("node_meta"),
        node_ids=splits.get("node_ids"),
        sampling=args.eval_sampling,
        seed=args.eval_seed,
    )

    result = {
        "dataset": "urbanev",
        "split": "8:1:1",
        "horizon": args.horizon,
        "node_idx": args.node_idx,
        "epochs": args.epochs,
        "lr": args.lr,
        "max_length": args.max_length,
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "history_len": args.history_len,
        "retrieval_top_k": args.retrieval_top_k,
        "neighbor_k": args.neighbor_k,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": None,
    }

    if preds:
        metrics = per_horizon_metrics(preds, trues, args.horizon, splits["norm_min"], splits["norm_max"])
        result["metrics"] = metrics
        print(json.dumps(metrics, indent=2))
    else:
        print("No parseable predictions.")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
