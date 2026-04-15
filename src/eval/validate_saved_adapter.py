"""
src/eval/validate_saved_adapter.py
---------------------------------
Validate a saved adapter on val/test split and save full outputs.

Saved content includes:
- run config
- aggregate metrics
- parse statistics
- per-sample records (prompt, raw generation, parsed prediction, target)

Usage:
  /home/kaga/diffdora/.venv/bin/python -m src.eval.validate_saved_adapter \
      --dataset urbanev \
      --horizon 6 \
      --split val \
      --adapter_dir outputs/urbanev_r32_h6/adapter \
      --use_rag \
      --max_eval 100 \
      --max_new_tokens 160 \
      --sampling random \
      --seed 42 \
      --output outputs/urbanev_r32_h6/val_outputs.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from src.data.load_st_evcdp import load_st_evcdp
from src.data.load_urbanev import load_urbanev
from src.data.build_splits import build_splits
from src.data.build_samples import build_samples
from src.utils.node_context import extract_node_static_context, normalise_domain_label
from src.eval.metrics import per_horizon_metrics
from src.models.qwen_peft import load_model_and_tokenizer, load_peft_model, generate
from src.prompts.prompt_vanilla import build_vanilla_prompt
from src.prompts.prompt_cot import build_cot_prompt
from src.prompts.parser import parse_output
from src.retrieval.diff_features import compute_diff_features
from src.retrieval.knn_retriever import KNNRetriever


def _load_dataset(dataset: str) -> dict:
    return load_st_evcdp() if dataset == "st_evcdp" else load_urbanev()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="urbanev", choices=["st_evcdp", "urbanev"])
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--node_idx", type=int, default=0)
    parser.add_argument("--history_len", type=int, default=12)
    parser.add_argument("--neighbor_k", type=int, default=7)

    parser.add_argument("--use_rag", action="store_true")
    parser.add_argument("--retrieval_cache", default="",
                        help="Path to retrieval cache pkl; default data/retrieval_cache/{dataset}_h{horizon}.pkl")

    parser.add_argument("--max_eval", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--sampling", choices=["head", "random"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_prompts", action="store_true", default=True)

    parser.add_argument("--output", default="outputs/validate_saved_adapter_outputs.json")
    args = parser.parse_args()

    raw = _load_dataset(args.dataset)
    splits = build_splits(raw, args.dataset)

    split_key = args.split
    ts_key = f"timestamps_{split_key}"
    sample_map = build_samples(
        splits[split_key],
        splits[ts_key],
        adj=splits.get("adj"),
        horizons=[args.horizon],
        history_len=args.history_len,
        neighbor_k=args.neighbor_k,
    )
    all_samples = sample_map[args.horizon]

    if args.sampling == "random":
        rng = random.Random(args.seed)
        subset = rng.sample(all_samples, min(args.max_eval, len(all_samples)))
    else:
        subset = all_samples[:args.max_eval]

    retriever = None
    if args.use_rag:
        if args.retrieval_cache:
            cache_path = Path(args.retrieval_cache)
        else:
            cache_path = Path(f"data/retrieval_cache/{args.dataset}_h{args.horizon}.pkl")
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Retrieval cache not found: {cache_path}. Build cache first or pass --retrieval_cache."
            )
        retriever = KNNRetriever.load(cache_path)

    print("Loading model …")
    base_model, tokenizer = load_model_and_tokenizer()
    model = load_peft_model(base_model, args.adapter_dir)
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.use_cache = True

    preds, trues = [], []
    records = []

    print(f"Validating on {args.split} split …")
    for i, sample in enumerate(subset):
        sample_node_idx = int(sample.get("node_idx", args.node_idx))
        static_context = extract_node_static_context(
            sample_node_idx,
            node_ids=splits.get("node_ids"),
            node_meta=splits.get("node_meta"),
        )
        domain_label = normalise_domain_label(static_context.get("zone_type"))

        if args.use_rag and retriever is not None:
            retrieved = retriever.query(sample, exclude_t_start=None)
            diff = compute_diff_features(query_sample=sample, retrieved_samples=retrieved)
            sys_msg, usr_msg = build_cot_prompt(
                sample,
                retrieved,
                diff,
                sample_node_idx,
                args.horizon,
                domain_label=domain_label,
                static_context=static_context,
            )
            retrieved_t_starts = [int(s.get("t_start", -1)) for s in retrieved]
        else:
            retrieved_t_starts = []
            sys_msg, usr_msg = build_vanilla_prompt(
                sample,
                sample_node_idx,
                args.horizon,
                domain_label=domain_label,
                static_context=static_context,
            )

        raw_output = generate(
            model,
            tokenizer,
            sys_msg,
            usr_msg,
            max_new_tokens=args.max_new_tokens,
        )

        parsed = parse_output(raw_output, expected_len=args.horizon)
        parse_ok = parsed is not None and len(parsed) == args.horizon

        target = sample["y"][:args.horizon, sample_node_idx]
        if parse_ok:
            preds.append(parsed)
            trues.append(target)

        row = {
            "sample_index": i,
            "t_start": int(sample.get("t_start", -1)),
            "node_idx": sample_node_idx,
            "parse_ok": parse_ok,
            "retrieved_t_starts": retrieved_t_starts,
            "raw_generation": raw_output,
            "parsed_prediction": parsed.tolist() if parse_ok else None,
            "target": target.tolist(),
        }
        if args.save_prompts:
            row["system_prompt"] = sys_msg
            row["user_prompt"] = usr_msg
        records.append(row)

    requested = len(subset)
    parsed_n = len(preds)

    result = {
        "dataset": args.dataset,
        "split": args.split,
        "horizon": args.horizon,
        "node_idx": args.node_idx,
        "adapter_dir": args.adapter_dir,
        "use_rag": args.use_rag,
        "history_len": args.history_len,
        "neighbor_k": args.neighbor_k,
        "max_eval": args.max_eval,
        "sampling": args.sampling,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "requested_samples": requested,
        "evaluated_samples": parsed_n,
        "parse_failures": requested - parsed_n,
        "parse_success_rate": parsed_n / max(requested, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": None,
        "records": records,
    }

    if preds:
        metrics = per_horizon_metrics(preds, trues, args.horizon, splits["norm_min"], splits["norm_max"])
        result["metrics"] = metrics
        print(json.dumps(metrics, indent=2))
    else:
        print("No parseable predictions.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
