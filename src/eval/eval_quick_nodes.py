"""
src/eval/eval_quick_nodes.py
-----------------------------
快速节点测试：固定 5 个节点（含 CBD 和 Residential）、2 个样本，
验证两个专家的路由、推理和解析是否正常。

Usage:
  /home/kaga/diffdora/.venv/bin/python -m src.eval.eval_quick_nodes \
      --dataset st_evcdp \
      --horizon 3 \
      --expert_0_dir outputs/<run_name>/expert_0/adapter \
      --expert_1_dir outputs/<run_name>/expert_1/adapter \
      --use_rag --use_diff_dora
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.data.load_st_evcdp import load_st_evcdp
from src.data.load_urbanev import load_urbanev
from src.data.build_splits import build_splits
from src.data.build_samples import build_samples
from src.utils.node_context import extract_node_static_context
from src.models.qwen_peft import load_model_and_tokenizer, load_peft_model, generate
from src.models.diff_dora import DiffDoRAModel, set_diff_context
from src.prompts.prompt_cot import build_cot_prompt
from src.prompts.prompt_vanilla import build_vanilla_prompt
from src.prompts.parser import parse_output
from src.retrieval.diff_features import compute_diff_features
from src.retrieval.knn_retriever import KNNRetriever
from src.routing.build_labels import build_routing_labels
from src.routing.hard_router import HardRouter


N_NODES  = 5   # 固定测试节点数（含 CBD + Residential）
N_SAMPLES = 2  # 每个节点的样本数


def pick_test_nodes(router: HardRouter, n_cbd: int = 2, n_res: int = 3) -> list[int]:
    """从 CBD 和 Residential 各取若干节点，共 n_cbd+n_res 个。"""
    cbd_nodes = router.nodes_for_expert(0)
    res_nodes = router.nodes_for_expert(1)
    chosen = cbd_nodes[:n_cbd] + res_nodes[:n_res]
    return chosen


def load_expert(base_model, adapter_dir: str, use_diff_dora: bool,
                diff_hidden_dim: int, diff_scale: float, device):
    model = load_peft_model(base_model, adapter_dir)
    if use_diff_dora:
        wrapper = DiffDoRAModel(model, diff_input_dim=3,
                                hidden_dim=diff_hidden_dim, scale=diff_scale)
        wrapper.to(device)
        ctrl_path = Path(adapter_dir).parent / "diff_controller.pt"
        if ctrl_path.exists():
            wrapper.controller.load_state_dict(
                torch.load(ctrl_path, map_location="cpu"))
            wrapper.controller.to(device)
        wrapper.eval()
        return wrapper
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      default="st_evcdp",
                        choices=["st_evcdp", "urbanev"])
    parser.add_argument("--horizon",      type=int, required=True)
    parser.add_argument("--split",        default="test",
                        choices=["val", "test"])
    parser.add_argument("--expert_0_dir", required=True,
                        help="CBD expert adapter path")
    parser.add_argument("--expert_1_dir", required=True,
                        help="Residential expert adapter path")
    parser.add_argument("--use_rag",      action="store_true")
    parser.add_argument("--use_diff_dora",action="store_true")
    parser.add_argument("--diff_hidden_dim", type=int, default=32)
    parser.add_argument("--diff_scale",   type=float, default=0.5)
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max tokens for generation; CoT reasoning needs ~200+")
    parser.add_argument("--retrieval_cache", default="",
                        help="path to .pkl cache; auto-detected if empty")
    parser.add_argument("--save_prompts", action="store_true", default=True,
                        help="Save system and user prompts in output JSON (default: True)")
    parser.add_argument("--output",       default="outputs/quick_nodes_eval.json")
    args = parser.parse_args()

    # ── 1. 数据 ─────────────────────────────────────────────────────────────
    raw    = load_st_evcdp() if args.dataset == "st_evcdp" else load_urbanev()
    splits = build_splits(raw, args.dataset)
    labels = build_routing_labels(splits["train"], raw.get("node_meta"))
    router = HardRouter(labels)

    split_key = args.split
    sample_map = build_samples(
        splits[split_key],
        splits[f"timestamps_{split_key}"],
        adj=splits.get("adj"),
        horizons=[args.horizon],
    )
    all_samples = sample_map[args.horizon]

    # ── 2. 节点选择（2 CBD + 3 Residential）────────────────────────────────
    test_nodes = pick_test_nodes(router, n_cbd=2, n_res=3)
    cbd_nodes = router.nodes_for_expert(0)
    res_nodes = router.nodes_for_expert(1)
    print(f"\nTest nodes: {test_nodes}")
    for n in test_nodes:
        domain = "CBD" if n in cbd_nodes else "Residential"
        print(f"  node {n:3d} → {domain}")

    # ── 3. RAG ──────────────────────────────────────────────────────────────
    retriever = None
    if args.use_rag:
        cache = args.retrieval_cache or \
            f"data/retrieval_cache/{args.dataset}_h{args.horizon}.pkl"
        cache_path = Path(cache)
        if cache_path.exists():
            retriever = KNNRetriever.load(cache_path)
            print(f"Loaded retrieval cache: {cache_path}")
        else:
            print(f"[WARN] No retrieval cache at {cache_path}; running without RAG.")

    # ── 4. 加载两个专家 ──────────────────────────────────────────────────────
    print("\nLoading base model …")
    base0, tokenizer = load_model_and_tokenizer()
    device = next(base0.parameters()).device
    expert_0 = load_expert(base0, args.expert_0_dir, args.use_diff_dora,
                           args.diff_hidden_dim, args.diff_scale, device)

    import gc, torch as _torch
    del base0
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    gc.collect()

    base1, _ = load_model_and_tokenizer()
    expert_1 = load_expert(base1, args.expert_1_dir, args.use_diff_dora,
                           args.diff_hidden_dim, args.diff_scale, device)

    experts = {0: expert_0, 1: expert_1}
    domain_names = {0: "CBD", 1: "Residential"}

    # ── 5. 推理 ──────────────────────────────────────────────────────────────
    samples_subset = all_samples[:N_SAMPLES]
    records = []
    total = len(test_nodes) * len(samples_subset)
    done = 0

    print(f"\nRunning {total} inferences "
          f"({len(test_nodes)} nodes × {len(samples_subset)} samples) …\n")

    for node_idx in test_nodes:
        expert_id  = router.route(node_idx)
        domain     = domain_names[expert_id]
        expert     = experts[expert_id]

        for si, sample in enumerate(samples_subset):
            done += 1
            static_context = extract_node_static_context(
                node_idx,
                node_ids=splits.get("node_ids"),
                node_meta=splits.get("node_meta"),
            )
            if args.use_rag and retriever is not None:
                retrieved = retriever.query(sample, exclude_t_start=None)
                diff = compute_diff_features(
                    query_sample=sample, retrieved_samples=retrieved)
                if args.use_diff_dora:
                    set_diff_context(torch.tensor([
                        float(diff.get("diff_occ", 0.0) or 0.0),
                        float(diff.get("diff_temp", 0.0) or 0.0),
                        float(diff.get("diff_price", 0.0) or 0.0),
                    ], dtype=torch.float32))
                sys_msg, usr_msg = build_cot_prompt(
                    sample, retrieved, diff, node_idx, args.horizon,
                    domain_label=domain,
                    static_context=static_context)
            else:
                if args.use_diff_dora:
                    set_diff_context(torch.zeros(3, dtype=torch.float32))
                sys_msg, usr_msg = build_vanilla_prompt(
                    sample, node_idx, args.horizon,
                    domain_label=domain,
                    static_context=static_context)

            raw_out = generate(expert, tokenizer, sys_msg, usr_msg,
                               max_new_tokens=args.max_new_tokens)
            parsed  = parse_output(raw_out, expected_len=args.horizon)
            parse_ok = parsed is not None and len(parsed) == args.horizon
            target   = sample["y"][:args.horizon, node_idx].tolist()

            rec = {
                "node_idx":   node_idx,
                "domain":     domain,
                "expert_id":  expert_id,
                "sample_idx": si,
                "t_start":    int(sample.get("t_start", -1)),
                "parse_ok":   parse_ok,
                "predicted":  parsed.tolist() if parse_ok else None,
                "target":     target,
                "raw_output": raw_out,
            }
            if args.save_prompts:
                rec["system_prompt"] = sys_msg
                rec["user_prompt"]   = usr_msg
            records.append(rec)

            status = "✓" if parse_ok else "✗"
            print(f"[{done:2d}/{total}] node={node_idx:3d} ({domain:12s}) "
                  f"sample={si}  {status}  "
                  f"pred={rec['predicted']}  target={[round(v,3) for v in target]}")

    # ── 6. 汇总 ──────────────────────────────────────────────────────────────
    n_ok   = sum(1 for r in records if r["parse_ok"])
    by_dom = {d: {"ok": 0, "total": 0} for d in ("CBD", "Residential")}
    for r in records:
        by_dom[r["domain"]]["total"] += 1
        if r["parse_ok"]:
            by_dom[r["domain"]]["ok"] += 1

    print(f"\n── Summary ────────────────────────────────────────")
    print(f"  Total inferences : {total}")
    print(f"  Parse success    : {n_ok}/{total} "
          f"({100*n_ok/max(total,1):.0f}%)")
    for d, v in by_dom.items():
        print(f"  {d:13s}: {v['ok']}/{v['total']}")

    result = {
        "dataset":      args.dataset,
        "horizon":      args.horizon,
        "split":        args.split,
        "test_nodes":   test_nodes,
        "use_rag":      args.use_rag,
        "use_diff_dora":args.use_diff_dora,
        "n_samples":    N_SAMPLES,
        "parse_success": n_ok / max(total, 1),
        "by_domain":    by_dom,
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "records":      records,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
