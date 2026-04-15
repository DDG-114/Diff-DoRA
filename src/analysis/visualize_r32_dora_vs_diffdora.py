from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file
import torch


def _load_eval(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _calc_metrics(records: list[dict]) -> dict:
    maes, rmses = [], []
    for r in records:
        if not r.get("parse_ok"):
            continue
        p = np.array(r["predicted"], dtype=np.float32)
        t = np.array(r["target"], dtype=np.float32)
        err = p - t
        maes.append(np.mean(np.abs(err)))
        rmses.append(np.sqrt(np.mean(err ** 2)))
    if not maes:
        return {"mae": np.nan, "rmse": np.nan}
    return {
        "mae": float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
    }


def _domain_metrics(records: list[dict], domain: str) -> dict:
    return _calc_metrics([r for r in records if r.get("domain") == domain])


def _count_adapter_params(adapter_dir: Path) -> int:
    sf = adapter_dir / "adapter_model.safetensors"
    state = load_file(str(sf))
    return int(sum(v.numel() for v in state.values()))


def _count_controller_params(controller_path: Path) -> int:
    state = torch.load(controller_path, map_location="cpu")
    return int(sum(v.numel() for v in state.values()))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a DoRA vs DiffDoRA comparison from evaluation JSON files."
    )
    parser.add_argument("--dora-eval", required=True,
                        help="Path to the DoRA evaluation JSON file.")
    parser.add_argument("--diff-eval", required=True,
                        help="Path to the DiffDoRA evaluation JSON file.")
    parser.add_argument("--adapter-dir", default="",
                        help="Optional adapter directory for parameter counting.")
    parser.add_argument("--controller-path", default="",
                        help="Optional Diff-DoRA controller checkpoint for parameter counting.")
    parser.add_argument("--output-dir", default="outputs/figures",
                        help="Directory where the figure and summary JSON will be written.")
    args = parser.parse_args()

    dora_eval_path = Path(args.dora_eval)
    diff_eval_path = Path(args.diff_eval)
    if not dora_eval_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {dora_eval_path}")
    if not diff_eval_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {diff_eval_path}")

    dora_eval = _load_eval(dora_eval_path)
    diff_eval = _load_eval(diff_eval_path)

    dora_records = dora_eval.get("records", [])
    diff_records = diff_eval.get("records", [])

    dora_m = _calc_metrics(dora_records)
    diff_m = _calc_metrics(diff_records)

    dora_cbd = _domain_metrics(dora_records, "CBD")
    dora_res = _domain_metrics(dora_records, "Residential")
    diff_cbd = _domain_metrics(diff_records, "CBD")
    diff_res = _domain_metrics(diff_records, "Residential")

    adapter_params = None
    controller_params = None
    if args.adapter_dir:
        adapter_dir = Path(args.adapter_dir)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
        adapter_params = _count_adapter_params(adapter_dir)
    if args.controller_path:
        controller_path = Path(args.controller_path)
        if not controller_path.exists():
            raise FileNotFoundError(f"Controller checkpoint not found: {controller_path}")
        controller_params = _count_controller_params(controller_path)

    fig = plt.figure(figsize=(12, 5.2), dpi=170)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.26)

    # (A) Global metrics (DiffDoRA first for presentation order)
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ["RMSE", "MAE"]
    x = np.arange(len(labels))
    width = 0.36
    fvals = [diff_m["rmse"], diff_m["mae"]]
    dvals = [dora_m["rmse"], dora_m["mae"]]

    ax1.bar(x - width / 2, fvals, width, label="DoRA", color="#f39c75")
    ax1.bar(x + width / 2, dvals, width, label="DiffDoRA", color="#74a9cf")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Global Comparison (r32 quick eval)")
    ax1.legend(frameon=False)
    ax1.grid(axis="y", alpha=0.25)

    # (B) Domain MAE comparison (DiffDoRA first for presentation order)
    ax2 = fig.add_subplot(gs[0, 1])
    f_mae = [diff_cbd["mae"], diff_res["mae"]]
    d_mae = [dora_cbd["mae"], dora_res["mae"]]
    dom = np.arange(2)
    ax2.bar(dom - width / 2, f_mae, width, label="DoRA", color="#f39c75")
    ax2.bar(dom + width / 2, d_mae, width, label="DiffDoRA", color="#74a9cf")
    ax2.set_xticks(dom)
    ax2.set_xticklabels(["CBD", "Residential"])
    ax2.set_title("Domain MAE Comparison")
    ax2.legend(frameon=False)
    ax2.grid(axis="y", alpha=0.25)

    title = "DoRA vs DiffDoRA Comparison"
    if adapter_params is not None and controller_params is not None:
        title += f" | Adapter params: {adapter_params:,} | Controller params: {controller_params:,}"
    elif adapter_params is not None:
        title += f" | Adapter params: {adapter_params:,}"
    elif controller_params is not None:
        title += f" | Controller params: {controller_params:,}"
    fig.suptitle(title, fontsize=11, y=1.02)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "dora_vs_diffdora_structure.png"
    fig.savefig(fig_path, bbox_inches="tight")

    summary = {
        "inputs": {
            "dora_eval": str(dora_eval_path),
            "diff_eval": str(diff_eval_path),
            "adapter_dir": args.adapter_dir or None,
            "controller_path": args.controller_path or None,
        },
        "dora": {
            "metrics": dora_m,
            "CBD": dora_cbd,
            "Residential": dora_res,
            "parse_success": dora_eval.get("parse_success"),
        },
        "diffdora": {
            "metrics": diff_m,
            "CBD": diff_cbd,
            "Residential": diff_res,
            "parse_success": diff_eval.get("parse_success"),
        },
        "params": {
            "adapter_params_expert0": adapter_params,
            "diff_controller_params_expert0": controller_params,
        },
        "figure": str(fig_path),
    }
    summary_path = out_dir / "dora_vs_diffdora_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved figure: {fig_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
