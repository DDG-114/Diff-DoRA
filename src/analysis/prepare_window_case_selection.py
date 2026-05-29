from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.data.build_samples import build_samples
from src.data.build_splits import build_splits
from src.data.load_st_evcdp import load_st_evcdp
from src.data.load_urbanev import load_urbanev


def _load_dataset(dataset: str) -> dict:
    return load_st_evcdp() if dataset == "st_evcdp" else load_urbanev()


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Prepare scheme-A/B window selections from ablation records.")
    parser.add_argument("--dataset", default="st_evcdp", choices=["st_evcdp", "urbanev"])
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--ablation-json", required=True)
    parser.add_argument("--history-len", type=int, default=12)
    parser.add_argument("--neighbor-k", type=int, default=7)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw = _load_dataset(args.dataset)
    splits = build_splits(raw, args.dataset)
    test_samples = build_samples(
        splits["test"],
        splits["timestamps_test"],
        adj=splits.get("adj"),
        horizons=[args.horizon],
        history_len=args.history_len,
        neighbor_k=args.neighbor_k,
    )[args.horizon]
    max_index = len(test_samples) - 1

    data = _load_json(Path(args.ablation_json))
    full_records = [r for r in data["results"]["full"]["records"] if r.get("parse_ok")]
    wo_lookup = {
        (r["sample_dataset_index"], r["node_idx"]): r
        for r in data["results"]["wo_diffdora"]["records"]
        if r.get("parse_ok")
    }

    node_adv = defaultdict(list)
    window_adv = defaultdict(list)
    node_best_window = {}

    for full in full_records:
        key = (full["sample_dataset_index"], full["node_idx"])
        wo = wo_lookup.get(key)
        if wo is None:
            continue
        target = full["target"]
        full_pred = full["parsed_prediction"]
        wo_pred = wo["parsed_prediction"]
        full_mae = sum(abs(a - b) for a, b in zip(full_pred, target)) / len(target)
        wo_mae = sum(abs(a - b) for a, b in zip(wo_pred, target)) / len(target)
        advantage = wo_mae - full_mae
        domain = full["domain"]
        node_idx = int(full["node_idx"])
        sample_idx = int(full["sample_dataset_index"])
        node_adv[(domain, node_idx)].append(advantage)
        window_adv[(domain, sample_idx)].append(advantage)
        best_key = (domain, node_idx)
        prev = node_best_window.get(best_key)
        if prev is None or advantage > prev["advantage"]:
            node_best_window[best_key] = {"sample_dataset_index": sample_idx, "advantage": advantage}

    selected_nodes = {}
    single_windows = {}
    collage_windows = {}

    for domain in ("CBD", "Residential"):
        domain_nodes = []
        for (d, node_idx), vals in node_adv.items():
            if d == domain:
                domain_nodes.append((sum(vals) / len(vals), node_idx))
        domain_nodes.sort(reverse=True)
        top_nodes = [int(node_idx) for _, node_idx in domain_nodes[:3]]
        selected_nodes[domain] = top_nodes

        single_windows[domain] = [
            {
                "node_idx": node_idx,
                "sample_dataset_index": node_best_window[(domain, node_idx)]["sample_dataset_index"],
                "advantage": node_best_window[(domain, node_idx)]["advantage"],
            }
            for node_idx in top_nodes
        ]

        candidate_windows = []
        for (d, sample_idx), vals in window_adv.items():
            if d != domain:
                continue
            if sample_idx + 12 > max_index:
                continue
            candidate_windows.append((sum(vals) / len(vals), sample_idx))
        candidate_windows.sort(reverse=True)
        anchor = candidate_windows[0][1] if candidate_windows else 0
        collage_windows[domain] = {
            "node_indices": top_nodes,
            "window_indices": [anchor, anchor + 6, anchor + 12],
        }

    payload = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "history_len": args.history_len,
        "neighbor_k": args.neighbor_k,
        "selected_nodes": selected_nodes,
        "single_windows": single_windows,
        "collage_windows": collage_windows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
