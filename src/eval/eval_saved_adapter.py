"""
src/eval/eval_saved_adapter.py
-----------------------------
Evaluate a saved PEFT adapter directly on a chosen dataset/horizon.

Usage:
  python -m src.eval.eval_saved_adapter \
      --dataset st_evcdp \
      --horizon 6 \
      --adapter_dir outputs/single_dora_rag_h6/adapter \
      --use_rag \
      --output outputs/single_dora_rag_h6/eval_metrics.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.data.load_st_evcdp import load_st_evcdp
from src.data.load_urbanev import load_urbanev
from src.data.build_splits import build_splits
from src.data.build_samples import build_samples
from src.eval.metrics import per_horizon_metrics
from src.models.qwen_peft import load_model_and_tokenizer, load_peft_model, generate
from src.prompts.prompt_vanilla import build_vanilla_prompt
from src.prompts.prompt_cot import build_cot_prompt
from src.prompts.parser import parse_output
from src.retrieval.diff_features import compute_diff_features
from src.retrieval.knn_retriever import KNNRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="st_evcdp", choices=["st_evcdp", "urbanev"])
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--adapter_dir", required=True,
                        help="Path to saved adapter directory")
    parser.add_argument("--node_idx", type=int, default=0)
    parser.add_argument("--use_rag", action="store_true",
                        help="Enable retrieval-augmented evaluation")
    parser.add_argument("--retrieval_cache", default="",
                        help="Path to retrieval cache pkl; default data/retrieval_cache/{dataset}_h{horizon}.pkl")
    parser.add_argument("--max_eval", type=int, default=200,
                        help="Maximum number of test samples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Generation max_new_tokens during evaluation")
    parser.add_argument("--save_generations", action="store_true", default=True,
                        help="Save raw model generations and parse status into output JSON")
    parser.add_argument("--output", default="outputs/eval_saved_adapter_metrics.json")
    args = parser.parse_args()

    raw = load_st_evcdp() if args.dataset == "st_evcdp" else load_urbanev()
    splits = build_splits(raw, args.dataset)

    test_map = build_samples(
        splits["test"],
        splits["timestamps_test"],
        adj=splits.get("adj"),
        horizons=[args.horizon],
    )
    test_samples = test_map[args.horizon]

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
            raise FileNotFoundError(
                f"Retrieval cache not found: {cache_path}. Build cache first or pass --retrieval_cache."
            )

    print("Loading base model …")
    base_model, tokenizer = load_model_and_tokenizer()
    print(f"Loading adapter … {args.adapter_dir}")
    model = load_peft_model(base_model, args.adapter_dir)
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.use_cache = True

    print("Evaluating saved adapter …")
    preds, trues = [], []
    generation_records = []
    subset = test_samples[:args.max_eval]
    for idx, sample in enumerate(subset):
        sample_node_idx = int(sample.get("node_idx", args.node_idx))
        if args.use_rag and retriever is not None:
            retrieved = retriever.query(sample, exclude_t_start=sample.get("t_start"))
            diff = compute_diff_features(query_sample=sample, retrieved_samples=retrieved)
            sys_msg, usr_msg = build_cot_prompt(sample, retrieved, diff, sample_node_idx, args.horizon)
        else:
            sys_msg, usr_msg = build_vanilla_prompt(sample, sample_node_idx, args.horizon)

        raw_output = generate(
            model,
            tokenizer,
            sys_msg,
            usr_msg,
            max_new_tokens=args.max_new_tokens,
        )
        parsed = parse_output(raw_output, expected_len=args.horizon)
        parse_ok = parsed is not None and len(parsed) == args.horizon
        if parse_ok:
            preds.append(parsed)
            trues.append(sample["y"][:args.horizon, sample_node_idx])

        if args.save_generations:
            generation_records.append({
                "sample_index": idx,
                "t_start": int(sample.get("t_start", -1)),
                "node_idx": sample_node_idx,
                "parse_ok": parse_ok,
                "raw_generation": raw_output,
                "parsed_prediction": parsed.tolist() if parse_ok else None,
                "target": sample["y"][:args.horizon, sample_node_idx].tolist(),
            })

    result = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "node_idx": args.node_idx,
        "adapter_dir": args.adapter_dir,
        "use_rag": args.use_rag,
        "evaluated_samples": len(preds),
        "requested_samples": min(args.max_eval, len(test_samples)),
        "parse_failures": min(args.max_eval, len(test_samples)) - len(preds),
        "parse_success_rate": len(preds) / max(min(args.max_eval, len(test_samples)), 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": None,
        "generations": generation_records if args.save_generations else None,
    }

    if preds:
        metrics = per_horizon_metrics(
            preds,
            trues,
            args.horizon,
            splits["norm_min"],
            splits["norm_max"],
        )
        result["metrics"] = metrics
        print(json.dumps(metrics, indent=2))
    else:
        print("No parseable predictions.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
