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
import gc
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
from src.models.qwen_peft     import load_model_and_tokenizer, get_lora_model, load_peft_model, generate
from src.prompts.prompt_vanilla import build_vanilla_prompt
from src.prompts.prompt_cot   import build_cot_prompt
from src.prompts.parser       import parse_output
from src.retrieval.knn_retriever import KNNRetriever
from src.retrieval.diff_features import compute_diff_features
from src.routing.build_labels import build_routing_labels
from src.routing.hard_router  import HardRouter
from src.train.train_single   import EVDataset   # re-use dataset class


def train_one_expert(
    expert_id: int,
    samples_with_node: list[dict],
    out_dir: Path,
    args,
    retriever: KNNRetriever | None = None,
    weather=None,
    price=None,
):
    print(f"\n=== Training Expert {expert_id} ({len(samples_with_node)} samples) ===")
    base_model, tokenizer = load_model_and_tokenizer()
    peft_model = get_lora_model(base_model, use_dora=args.use_dora)
    if args.gradient_checkpointing and hasattr(peft_model, "gradient_checkpointing_enable"):
        peft_model.gradient_checkpointing_enable()

    train_ds = EVDataset(
        samples_with_node,
        tokenizer,
        args.horizon,
        max_length=args.max_length,
        node_idx=0,
        use_rag=args.use_rag,
        retriever=retriever,
        weather=weather,
        price=price,
    )

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
    del trainer
    del collator
    del train_ds
    del peft_model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return save_path


def evaluate_one_expert(
    expert_id: int,
    adapter_path: str,
    tokenizer,
    test_samples: list[dict],
    router: HardRouter,
    splits: dict,
    args,
    retriever: KNNRetriever | None = None,
):
    print(f"\n=== Evaluating Expert {expert_id} ===")
    base_model, _ = load_model_and_tokenizer()
    model = load_peft_model(base_model, adapter_path)
    model.eval()

    nodes = router.nodes_for_expert(expert_id)
    preds, trues = [], []
    for s in test_samples[:args.eval_max_samples]:
        for n in nodes[:args.eval_nodes_per_expert]:
            if args.use_rag and retriever is not None:
                retrieved = retriever.query(s, exclude_t_start=s.get("t_start"))
                diff = compute_diff_features(query_sample=s, retrieved_samples=retrieved)
                sys_msg, usr_msg = build_cot_prompt(s, retrieved, diff, n, args.horizon)
            else:
                sys_msg, usr_msg = build_vanilla_prompt(s, n, args.horizon)
            out = generate(model, tokenizer, sys_msg, usr_msg, max_new_tokens=128)
            arr = parse_output(out, args.horizon)
            if arr is not None and len(arr) == args.horizon:
                preds.append(arr)
                trues.append(s["y"][:args.horizon, n])

    del model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if not preds:
        return None
    return per_horizon_metrics(
        preds,
        trues,
        args.horizon,
        splits["norm_min"],
        splits["norm_max"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="st_evcdp", choices=["st_evcdp", "urbanev"])
    parser.add_argument("--horizon",    type=int,   default=6)
    parser.add_argument("--output_dir", default="outputs/moe_experts_h6")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--use_dora",   action="store_true")
    parser.add_argument("--use_rag",    action="store_true",
                        help="Enable retrieval-augmented CoT prompts for expert training/eval")
    parser.add_argument("--max_length", type=int, default=384,
                        help="Tokenization max length (reduce for lower VRAM)")
    parser.add_argument("--gradient_checkpointing", dest="gradient_checkpointing",
                        action="store_true", default=True,
                        help="Enable gradient checkpointing to reduce VRAM (default: enabled)")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                        action="store_false",
                        help="Disable gradient checkpointing")
    parser.add_argument("--retrieval_cache", default="",
                        help="Path to retrieval cache pkl; default data/retrieval_cache/{dataset}_h{horizon}.pkl")
    parser.add_argument("--max_samples_per_expert", type=int, default=1000)
    parser.add_argument("--eval_max_samples", type=int, default=200,
                        help="Cap evaluation samples for speed")
    parser.add_argument("--eval_nodes_per_expert", type=int, default=5,
                        help="Cap number of nodes per expert during evaluation for speed")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data
    raw    = load_st_evcdp() if args.dataset == "st_evcdp" else load_urbanev()
    splits = build_splits(raw, args.dataset)

    retriever = None
    if args.use_rag:
        if args.retrieval_cache:
            cache_path = Path(args.retrieval_cache)
        else:
            cache_path = Path(f"data/retrieval_cache/{args.dataset}_h{args.horizon}.pkl")
        if cache_path.exists():
            retriever = KNNRetriever.load(cache_path)
            print(f"Loaded retrieval cache: {cache_path}")
        else:
            print(f"Retrieval cache not found, building in-memory retriever from train pool.")

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

    if args.use_rag and retriever is None:
        retriever = KNNRetriever(raw_samples, top_k=2)

    print(f"Expert sample counts: expert_0={len(tagged[0])}, expert_1={len(tagged[1])}")

    _, tokenizer = load_model_and_tokenizer()

    # 5. Train each expert
    trained = {}
    for eid in (0, 1):
        trained[eid] = train_one_expert(
            eid,
            tagged[eid],
            out_dir,
            args,
            retriever=retriever,
            weather=splits.get("weather"),
            price=splits.get("price"),
        )

    # 6. Evaluate both experts on test split
    test_map = build_samples(
        splits["test"], splits["timestamps_test"],
        adj=splits.get("adj"), horizons=[args.horizon]
    )
    test_samples = test_map[args.horizon]
    results = {}
    for eid in (0, 1):
        m = evaluate_one_expert(
            eid,
            trained[eid],
            tokenizer,
            test_samples,
            router,
            splits,
            args,
            retriever=retriever,
        )
        if m is not None:
            results[f"expert_{eid}"] = m
            print(f"Expert {eid}: {m['overall']}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"dataset": args.dataset, "horizon": args.horizon,
                   "results": results,
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print(f"Saved to {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
