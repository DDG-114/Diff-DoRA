"""
src/analysis/compare_inference_track.py
---------------------------------------
Compare fixed and shape-aware inference-track outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.build_splits import build_splits
from src.data.load_st_evcdp import load_st_evcdp
from src.data.load_urbanev import load_urbanev
from src.eval.diagnostics import compute_diagnostics_from_records

RUN_ORDER = [
    "quick_seed42",
    "official_seed42",
    "official_seed43",
    "official_seed44",
]
OFFICIAL_RUNS = [run for run in RUN_ORDER if run.startswith("official_")]
LOADERS = {
    "st_evcdp": load_st_evcdp,
    "urbanev": load_urbanev,
}
SUMMARY_METRICS = (
    "overall_rmse",
    "overall_mae",
    "cbd_mae",
    "residential_mae",
    "constant_prediction_rate",
    "mean_prediction_range",
)

_NORM_CACHE: dict[str, tuple[float, float]] = {}


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _metric_value(block: dict | None, *keys, default: float | None = None) -> float | None:
    cur = block
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            if key in cur:
                cur = cur[key]
            elif str(key) in cur:
                cur = cur[str(key)]
            else:
                return default
        else:
            return default
    if cur is None:
        return default
    return float(cur)


def _norm_bounds(dataset: str) -> tuple[float, float]:
    if dataset not in _NORM_CACHE:
        raw = LOADERS[dataset]()
        splits = build_splits(raw, dataset)
        _NORM_CACHE[dataset] = (float(splits["norm_min"]), float(splits["norm_max"]))
    return _NORM_CACHE[dataset]


def _ensure_diagnostics(payload: dict) -> dict:
    norm_min = payload.get("norm_min")
    norm_max = payload.get("norm_max")
    if norm_min is None or norm_max is None:
        norm_min, norm_max = _norm_bounds(payload["dataset"])

    for variant_payload in payload.get("results", {}).values():
        if variant_payload.get("diagnostics") is None or variant_payload.get("domain_diagnostics") is None:
            diagnostics, domain_diagnostics = compute_diagnostics_from_records(
                variant_payload.get("records", []),
                float(norm_min),
                float(norm_max),
            )
            if variant_payload.get("diagnostics") is None:
                variant_payload["diagnostics"] = diagnostics
            if variant_payload.get("domain_diagnostics") is None:
                variant_payload["domain_diagnostics"] = domain_diagnostics
    return payload


def _variant_summary(variant_payload: dict | None) -> dict | None:
    if variant_payload is None:
        return None
    diagnostics = variant_payload.get("diagnostics") or {}
    return {
        "overall_rmse": _metric_value(variant_payload.get("metrics"), "overall", "rmse"),
        "overall_mae": _metric_value(variant_payload.get("metrics"), "overall", "mae"),
        "cbd_mae": _metric_value(variant_payload.get("domain_metrics", {}).get("CBD"), "overall", "mae"),
        "residential_mae": _metric_value(variant_payload.get("domain_metrics", {}).get("Residential"), "overall", "mae"),
        "constant_prediction_rate": _metric_value(diagnostics, "constant_prediction_rate"),
        "mean_prediction_range": _metric_value(diagnostics, "mean_prediction_range"),
    }


def _delta_summary(new: dict | None, old: dict | None) -> dict | None:
    if new is None or old is None:
        return None
    delta = {}
    for metric in SUMMARY_METRICS:
        if new.get(metric) is None or old.get(metric) is None:
            delta[metric] = None
        else:
            delta[metric] = float(new[metric] - old[metric])
    return delta


def _mean_std(values: list[float | None]) -> dict | None:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    arr = np.asarray(clean, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
    }


def _aggregate_official(runs: dict) -> dict:
    aggregate = {}
    variant_names = sorted({
        variant
        for run_name in OFFICIAL_RUNS
        for variant in runs.get(run_name, {}).get("variants", {}).keys()
    })

    for variant in variant_names:
        baseline_metrics = {metric: [] for metric in SUMMARY_METRICS}
        shapeaware_metrics = {metric: [] for metric in SUMMARY_METRICS}
        delta_metrics = {metric: [] for metric in SUMMARY_METRICS}

        for run_name in OFFICIAL_RUNS:
            variant_payload = runs.get(run_name, {}).get("variants", {}).get(variant, {})
            baseline = variant_payload.get("baseline")
            shapeaware = variant_payload.get("shapeaware")
            delta = variant_payload.get("delta")
            for metric in SUMMARY_METRICS:
                baseline_metrics[metric].append(None if baseline is None else baseline.get(metric))
                shapeaware_metrics[metric].append(None if shapeaware is None else shapeaware.get(metric))
                delta_metrics[metric].append(None if delta is None else delta.get(metric))

        aggregate[variant] = {
            "baseline": {metric: _mean_std(values) for metric, values in baseline_metrics.items()},
            "shapeaware": {metric: _mean_std(values) for metric, values in shapeaware_metrics.items()},
            "delta": {metric: _mean_std(values) for metric, values in delta_metrics.items()},
        }
    return aggregate


def _load_run(path: Path) -> dict | None:
    if not path.exists():
        return None
    return _ensure_diagnostics(_load_json(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_root", required=True)
    parser.add_argument("--shapeaware_root", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    baseline_root = Path(args.baseline_root)
    shapeaware_root = Path(args.shapeaware_root)

    runs = {}
    for run_name in RUN_ORDER:
        baseline_payload = _load_run(baseline_root / run_name / "expert_ablation.json")
        shapeaware_payload = _load_run(shapeaware_root / run_name / "expert_ablation.json")
        if baseline_payload is None and shapeaware_payload is None:
            continue

        variant_names = sorted({
            *(baseline_payload or {}).get("results", {}).keys(),
            *(shapeaware_payload or {}).get("results", {}).keys(),
        })
        run_summary = {
            "baseline_path": str(baseline_root / run_name / "expert_ablation.json"),
            "shapeaware_path": str(shapeaware_root / run_name / "expert_ablation.json"),
            "variants": {},
        }
        for variant in variant_names:
            baseline_variant = None if baseline_payload is None else baseline_payload["results"].get(variant)
            shapeaware_variant = None if shapeaware_payload is None else shapeaware_payload["results"].get(variant)
            baseline_summary = _variant_summary(baseline_variant)
            shapeaware_summary = _variant_summary(shapeaware_variant)
            run_summary["variants"][variant] = {
                "baseline": baseline_summary,
                "shapeaware": shapeaware_summary,
                "delta": _delta_summary(shapeaware_summary, baseline_summary),
            }
        runs[run_name] = run_summary

    summary = {
        "baseline_root": str(baseline_root),
        "shapeaware_root": str(shapeaware_root),
        "runs": runs,
        "official_mean_std": _aggregate_official(runs),
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for run_name in RUN_ORDER:
        if run_name not in runs:
            continue
        print(f"\n=== {run_name} ===")
        for variant, variant_payload in runs[run_name]["variants"].items():
            baseline = variant_payload["baseline"]
            shapeaware = variant_payload["shapeaware"]
            delta = variant_payload["delta"]
            if baseline is None or shapeaware is None or delta is None:
                continue
            print(
                f"{variant}: "
                f"old_mae={baseline['overall_mae']:.4f}, new_mae={shapeaware['overall_mae']:.4f}, "
                f"delta_mae={delta['overall_mae']:+.4f}, "
                f"old_const={baseline['constant_prediction_rate']:.4f}, "
                f"new_const={shapeaware['constant_prediction_rate']:.4f}, "
                f"delta_range={delta['mean_prediction_range']:+.4f}"
            )
    print(f"\nSaved comparison -> {out_path}")


if __name__ == "__main__":
    main()
