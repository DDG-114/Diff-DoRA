"""
src/train/train_experts.py
---------------------------
Train two independent expert adapters (CBD + Residential) using hard routing.

Usage:
  python -m src.train.train_experts \
      --dataset st_evcdp \
      --horizon 6 \
      --output_dir outputs/moe_experts_h6 \
      --use_dora --use_diff_dora --use_rag
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
from src.utils.node_context   import extract_node_static_context
from src.eval.metrics         import per_horizon_metrics
from src.models.qwen_peft     import load_model_and_tokenizer, get_lora_model, load_peft_model, generate
from src.models.diff_dora     import DiffDoRAModel, set_diff_context
from src.prompts.prompt_vanilla import build_vanilla_prompt
from src.prompts.prompt_cot   import build_cot_prompt
from src.prompts.parser       import parse_output
from src.retrieval.knn_retriever import KNNRetriever
from src.retrieval.diff_features import compute_diff_features
from src.routing.build_labels import build_routing_labels
from src.routing.hard_router  import HardRouter
from src.train.train_single   import EVDataset   # re-use dataset class


class DiffDoRATrainer(Trainer):
    """Trainer that injects per-batch diff context for Diff-DoRA hooks."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        diff_vec = inputs.pop("diff_vec", None)
        if isinstance(diff_vec, torch.Tensor):
            if diff_vec.ndim == 2:
                diff_ctx = diff_vec.float().mean(dim=0)
            else:
                diff_ctx = diff_vec.float().view(-1)
            set_diff_context(diff_ctx.detach())
        else:
            set_diff_context(None)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)


def train_one_expert(
    expert_id: int,
    samples_with_node: list[dict],
    out_dir: Path,
    args,
    retriever: KNNRetriever | None = None,
    weather=None,
    price=None,
    node_meta=None,
    node_ids: list[str] | None = None,
    poi=None,
):
    print(f"\n=== Training Expert {expert_id} ({len(samples_with_node)} samples) ===")
    base_model, tokenizer = load_model_and_tokenizer()
    peft_model = get_lora_model(base_model, use_dora=args.use_dora,
                                r=args.lora_rank, lora_alpha=args.lora_alpha)
    model_for_training = peft_model
    if args.use_diff_dora:
        model_for_training = DiffDoRAModel(
            peft_model,
            diff_input_dim=3,
            hidden_dim=args.diff_hidden_dim,
            scale=args.diff_scale,
        )
    use_gc = args.gradient_checkpointing
    if args.use_diff_dora and args.gradient_checkpointing:
        print("[DiffDoRA] Gradient checkpointing enabled (low-VRAM mode).")
    if use_gc and hasattr(peft_model, "gradient_checkpointing_enable"):
        peft_model.gradient_checkpointing_enable()
    elif hasattr(peft_model, "gradient_checkpointing_disable"):
        peft_model.gradient_checkpointing_disable()

    # Split train/val (85/15)
    val_split_idx = int(len(samples_with_node) * 0.85)
    train_samples = samples_with_node[:val_split_idx]
    val_samples = samples_with_node[val_split_idx:]
    
    print(f"  Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    train_ds = EVDataset(
        train_samples,
        tokenizer,
        args.horizon,
        max_length=args.max_length,
        node_idx=0,
        use_rag=args.use_rag,
        retriever=retriever,
        weather=weather,
        price=price,
        node_meta=node_meta,
        node_ids=node_ids,
        poi=poi,
        include_diff_vec=args.use_diff_dora,
    )
    
    val_ds = EVDataset(
        val_samples,
        tokenizer,
        args.horizon,
        max_length=args.max_length,
        node_idx=0,
        use_rag=args.use_rag,
        retriever=retriever,
        weather=weather,
        price=price,
        node_meta=node_meta,
        node_ids=node_ids,
        poi=poi,
        include_diff_vec=args.use_diff_dora,
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=peft_model, padding=True, pad_to_multiple_of=8)
    save_strategy = "no" if args.use_diff_dora else "steps"
    load_best = False if args.use_diff_dora else True
    training_args = TrainingArguments(
        output_dir=str(out_dir / f"expert_{expert_id}" / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy=save_strategy,
        save_steps=25,
        load_best_model_at_end=load_best,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer_cls = DiffDoRATrainer if args.use_diff_dora else Trainer
    trainer = trainer_cls(
        model=model_for_training,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.label_names = ["labels"]
    trainer.train()
    save_path = str(out_dir / f"expert_{expert_id}" / "adapter")
    peft_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    if args.use_diff_dora and hasattr(model_for_training, "controller"):
        torch.save(
            model_for_training.controller.state_dict(),
            out_dir / f"expert_{expert_id}" / "diff_controller.pt",
        )
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
    eval_model = model
    if args.use_diff_dora:
        eval_model = DiffDoRAModel(
            model,
            diff_input_dim=3,
            hidden_dim=args.diff_hidden_dim,
            scale=args.diff_scale,
        )
        eval_model.to(next(model.parameters()).device)
        ctrl_path = Path(adapter_path).parent / "diff_controller.pt"
        if ctrl_path.exists():
            eval_model.controller.load_state_dict(torch.load(ctrl_path, map_location="cpu"))
            eval_model.controller.to(next(model.parameters()).device)
        else:
            print(f"Warning: DiffDoRA controller not found at {ctrl_path}, using randomly initialized controller.")
        eval_model.eval()
    else:
        eval_model.eval()
    model.eval()

    from tqdm import tqdm
    nodes = router.nodes_for_expert(expert_id)
    preds, trues = [], []
    eval_cap = min(args.eval_max_samples, 50)
    n_nodes = min(args.eval_nodes_per_expert, len(nodes))
    total_calls = eval_cap * n_nodes
    print(f"  Inference: {eval_cap} samples × {n_nodes} nodes = {total_calls} calls")
    domain_name = "CBD" if expert_id == 0 else "Residential"
    bar = tqdm(total=total_calls, desc=f"  Expert {expert_id} ({domain_name})",
               unit="call", ncols=90, leave=True)
    for si, s in enumerate(test_samples[:eval_cap]):
        for n in nodes[:n_nodes]:
            if args.use_rag and retriever is not None:
                retrieved = retriever.query(s, exclude_t_start=None)
                diff = compute_diff_features(query_sample=s, retrieved_samples=retrieved)
                static_context = extract_node_static_context(
                    n,
                    node_ids=splits.get("node_ids"),
                    node_meta=splits.get("node_meta"),
                )
                domain_label = domain_name
                if args.use_diff_dora:
                    set_diff_context(torch.tensor([
                        float(diff.get("diff_occ", 0.0) or 0.0),
                        float(diff.get("diff_temp", 0.0) or 0.0),
                        float(diff.get("diff_price", 0.0) or 0.0),
                    ], dtype=torch.float32))
                sys_msg, usr_msg = build_cot_prompt(
                    s,
                    retrieved,
                    diff,
                    n,
                    args.horizon,
                    domain_label=domain_label,
                    static_context=static_context,
                )
            else:
                static_context = extract_node_static_context(
                    n,
                    node_ids=splits.get("node_ids"),
                    node_meta=splits.get("node_meta"),
                )
                if args.use_diff_dora:
                    set_diff_context(torch.zeros(3, dtype=torch.float32))
                sys_msg, usr_msg = build_vanilla_prompt(
                    s,
                    n,
                    args.horizon,
                    domain_label=domain_name,
                    static_context=static_context,
                )
            out = generate(eval_model, tokenizer, sys_msg, usr_msg, max_new_tokens=128)
            arr = parse_output(out, args.horizon)
            ok = arr is not None and len(arr) == args.horizon
            if ok:
                preds.append(arr)
                trues.append(s["y"][:args.horizon, n])
            bar.update(1)
            bar.set_postfix(parsed=len(preds), node=n,
                            ok="✓" if ok else "✗", refresh=False)
    bar.close()

    del eval_model
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
    parser.add_argument("--epochs",     type=int,   default=2)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--lora_rank",  type=int, default=32,
                        help="LoRA rank; the paper uses 32.")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (scaling factor); the paper uses 32.")
    parser.add_argument("--history_len", type=int, default=12,
                        help="Historical observation window; the paper uses 12.")
    parser.add_argument("--neighbor_k", type=int, default=7,
                        help="Neighbour top-k used for spatial context; the paper uses 7.")
    parser.add_argument("--use_dora",   action="store_true")
    parser.add_argument("--use_diff_dora", action="store_true",
                        help="Enable Diff-DoRA magnitude modulation (requires DoRA)")
    parser.add_argument("--diff_hidden_dim", type=int, default=32)
    parser.add_argument("--diff_scale", type=float, default=0.5)
    parser.add_argument("--use_rag",    action="store_true",
                        help="Enable retrieval-augmented CoT prompts for expert training/eval")
    parser.add_argument("--max_length", type=int, default=2560,
                        help="Tokenization max length; the paper uses 2560.")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps.")
    parser.add_argument("--gradient_checkpointing", dest="gradient_checkpointing",
                        action="store_true", default=True,
                        help="Enable gradient checkpointing to reduce VRAM (default: enabled)")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                        action="store_false",
                        help="Disable gradient checkpointing")
    parser.add_argument("--retrieval_cache", default="",
                        help="Path to retrieval cache pkl; default data/retrieval_cache/{dataset}_h{horizon}.pkl")
    parser.add_argument("--max_samples_per_expert", type=int, default=1000)
    parser.add_argument("--eval_max_samples", type=int, default=3,
                        help="Number of test samples per expert after training (default 3)")
    parser.add_argument("--eval_nodes_per_expert", type=int, default=5,
                        help="Number of nodes per expert after training (default 5)")
    args = parser.parse_args()

    if args.use_diff_dora and not args.use_dora:
        raise ValueError("--use_diff_dora requires --use_dora")

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
        adj=splits.get("adj"),
        horizons=[args.horizon],
        history_len=args.history_len,
        neighbor_k=args.neighbor_k,
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

    print(f"Expert sample counts: expert_0={len(tagged[0])}, expert_1={len(tagged[1])}")

    _, tokenizer = load_model_and_tokenizer()

    # Build per-expert retrieval banks for isolation (Issue 2 fix)
    retrievers = {0: None, 1: None}
    if args.use_rag:
        try:
            if args.retrieval_cache:
                cache_path = Path(args.retrieval_cache)
            else:
                cache_path = Path(f"data/retrieval_cache/{args.dataset}_h{args.horizon}.pkl")
            
            if cache_path.exists():
                global_retriever = KNNRetriever.load(cache_path)
                print(f"Loaded retrieval cache: {cache_path}")
                # Create per-expert retrievers by filtering the global pool
                for eid in (0, 1):
                    expert_samples = tagged[eid][:800]  # Use subset for retrieval bank
                    retrievers[eid] = KNNRetriever(expert_samples, top_k=2)
                    print(f"Built per-expert retriever for expert_{eid} with {len(expert_samples)} reference samples")
            else:
                # Build from per-expert tagged samples
                print(f"Retrieval cache not found, building per-expert in-memory retrievers.")
                for eid in (0, 1):
                    retrievers[eid] = KNNRetriever(tagged[eid], top_k=2)
                    print(f"Built per-expert retriever for expert_{eid} with {len(tagged[eid])} reference samples")
        except Exception as e:
            print(f"Warning: Failed to build per-expert retrievers, falling back to shared retriever: {e}")
            shared_retriever = KNNRetriever(raw_samples, top_k=2)
            retrievers = {0: shared_retriever, 1: shared_retriever}

    # 5. Train each expert (with per-expert retriever isolation)
    trained = {}
    for eid in (0, 1):
        trained[eid] = train_one_expert(
            eid,
            tagged[eid],
            out_dir,
            args,
            retriever=retrievers[eid],  # Pass per-expert retriever
            weather=splits.get("weather"),
            price=splits.get("price"),
            node_meta=splits.get("node_meta"),
            node_ids=splits.get("node_ids"),
            poi=splits.get("poi"),
        )

    # 6. Evaluate both experts on test split
    test_map = build_samples(
        splits["test"], splits["timestamps_test"],
        adj=splits.get("adj"),
        horizons=[args.horizon],
        history_len=args.history_len,
        neighbor_k=args.neighbor_k,
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
            retriever=retrievers[eid],  # Use per-expert retriever for evaluation
        )
        if m is not None:
            results[f"expert_{eid}"] = m
            print(f"Expert {eid}: {m['overall']}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "run_id": out_dir.name,
            "dataset": args.dataset,
            "horizon": args.horizon,
            "results": results,
            "config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "max_length": args.max_length,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "history_len": args.history_len,
                "neighbor_k": args.neighbor_k,
                "use_dora": args.use_dora,
                "use_diff_dora": args.use_diff_dora,
                "use_rag": args.use_rag,
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    print(f"Saved to {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
