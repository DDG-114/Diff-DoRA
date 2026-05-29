#!/usr/bin/env python3
"""Train a non-LLM day-ahead electricity price baseline on 2025 data.

This script uses:
  - quarter-hourly GS market features from ``data/GS(1).csv``
  - slow-moving supply / demand / policy features from ``supply_demand_data``

The target is the next-day 96-point price curve. The implementation avoids
same-day price leakage by only using historical prices plus target-day
exogenous variables that are assumed known in day-ahead forecasting.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


PRICE_COL = "Price"
EXOG_COLS = [
    "发电总出力预测",
    "竞价空间",
    "统一负荷预测",
    "抽蓄",
    "统一新能源预测",
    "联络线计划",
]
REQUIRED_COLS = ["Date", PRICE_COL, *EXOG_COLS]


@dataclass
class RegimeConfig:
    name: str
    slot_start: int
    slot_end: int
    wraps: bool = False

    def matches(self, slot: int) -> bool:
        if self.wraps:
            return slot >= self.slot_start or slot <= self.slot_end
        return self.slot_start <= slot <= self.slot_end


REGIMES = [
    RegimeConfig(name="floor", slot_start=78, slot_end=17, wraps=True),
    RegimeConfig(name="solar", slot_start=18, slot_end=39),
    RegimeConfig(name="mid", slot_start=40, slot_end=65),
    RegimeConfig(name="evening", slot_start=66, slot_end=77),
]


def _read_csv(path: Path) -> pd.DataFrame:
    last_exc: Exception | None = None
    for encoding in (None, "utf-8-sig", "gb18030", "gbk"):
        try:
            kwargs = {"low_memory": False}
            if encoding is not None:
                kwargs["encoding"] = encoding
            return pd.read_csv(path, **kwargs)
        except UnicodeDecodeError as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to read CSV: {path}")


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).lstrip("\ufeff").strip() for col in out.columns]
    return out


def _load_market_frame(source_csv: Path) -> pd.DataFrame:
    df = _normalise_columns(_read_csv(source_csv))
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {source_csv}: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="mixed")
    for col in [PRICE_COL, *EXOG_COLS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.loc[df["Date"].notna(), ["Date", PRICE_COL, *EXOG_COLS]].copy()
    df = df.sort_values("Date").reset_index(drop=True)
    df[[PRICE_COL, *EXOG_COLS]] = (
        df[[PRICE_COL, *EXOG_COLS]]
        .replace([np.inf, -np.inf], np.nan)
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
    )
    df["year"] = df["Date"].dt.year
    df["day"] = df["Date"].dt.normalize()
    df["slot"] = df["Date"].dt.hour * 4 + df["Date"].dt.minute // 15
    return df.loc[df["year"] == 2025].reset_index(drop=True)


def _load_daily_supply_demand_features(
    timeline_csv: Path,
    policy_csv: Path,
    priority_monthly_csv: Path,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    timeline = _normalise_columns(_read_csv(timeline_csv))
    policy = _normalise_columns(_read_csv(policy_csv))
    monthly = _normalise_columns(_read_csv(priority_monthly_csv))

    timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce")
    policy["date"] = pd.to_datetime(policy["date"], errors="coerce")
    daily = timeline.merge(policy, on="date", how="outer").sort_values("date")
    daily = daily.loc[daily["date"].dt.year == 2025].copy()
    daily = daily.set_index("date")
    daily = daily.apply(pd.to_numeric, errors="coerce").ffill().fillna(0.0)
    keep_cols = [col for col in daily.columns if float(np.abs(daily[col]).sum()) > 0.0]
    daily = daily[keep_cols]

    monthly["month"] = pd.to_numeric(monthly["month"], errors="coerce").astype("Int64")
    month_features: dict[int, np.ndarray] = {}
    for _, row in monthly.dropna(subset=["month"]).iterrows():
        month = int(row["month"])
        values = pd.to_numeric(row.drop(labels=["month"]), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        month_features[month] = values
    if len(month_features) != 12:
        raise ValueError("Expected 12 monthly priority-generation rows.")
    return daily, month_features


def _build_daily_matrices(market: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    price = market.pivot(index="day", columns="slot", values=PRICE_COL).sort_index()
    exog = {
        col: market.pivot(index="day", columns="slot", values=col).sort_index()
        for col in EXOG_COLS
    }
    if price.shape[1] != 96:
        raise ValueError(f"Expected 96 quarter-hour slots, got {price.shape[1]}")
    return price, exog


def _feature_names(daily_features: pd.DataFrame, month_features: dict[int, np.ndarray]) -> list[str]:
    names = [
        "slot_index",
        "day_of_week",
        "month",
        "is_weekend",
        "slot_sin",
        "slot_cos",
        "prev_day_same_slot_price",
        "prev_2day_same_slot_price",
        "prev_7day_same_slot_price",
        "prev_day_price_mean",
        "prev_day_price_std",
        "prev_day_price_max",
        "prev_day_price_min",
        "prev_2day_price_mean",
        "prev_7day_price_mean",
    ]
    for col in EXOG_COLS:
        prefix = {
            "发电总出力预测": "gen_forecast",
            "竞价空间": "bidding_space",
            "统一负荷预测": "load_forecast",
            "抽蓄": "pumped_storage",
            "统一新能源预测": "renewable_forecast",
            "联络线计划": "tie_line_plan",
        }[col]
        names.extend(
            [
                f"{prefix}_target_slot",
                f"{prefix}_prev_day_slot",
                f"{prefix}_slot_delta_vs_prev_day",
                f"{prefix}_target_day_mean",
                f"{prefix}_target_day_std",
                f"{prefix}_prev_day_mean",
            ]
        )
    names.extend([f"daily_{col}" for col in daily_features.columns])
    month_width = len(next(iter(month_features.values())))
    names.extend([f"priority_month_feature_{idx:02d}" for idx in range(month_width)])
    return names


def _build_point_dataset(
    price: pd.DataFrame,
    exog: dict[str, pd.DataFrame],
    daily_features: pd.DataFrame,
    month_features: dict[int, np.ndarray],
    *,
    history_days: int = 7,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[str]]:
    days = list(price.index)
    rows: list[np.ndarray] = []
    targets: list[float] = []
    meta: list[dict] = []
    feature_names = _feature_names(daily_features, month_features)

    for day_idx in range(history_days, len(days)):
        day = pd.Timestamp(days[day_idx])
        prev_day = price.iloc[day_idx - 1]
        prev_2day = price.iloc[day_idx - 2]
        prev_7day = price.iloc[day_idx - 7]
        prev_day_stats = np.array(
            [
                float(prev_day.mean()),
                float(prev_day.std()),
                float(prev_day.max()),
                float(prev_day.min()),
                float(prev_2day.mean()),
                float(prev_7day.mean()),
            ],
            dtype=np.float32,
        )
        slow_values = daily_features.reindex([day]).ffill().fillna(0.0).iloc[0].to_numpy(dtype=np.float32)
        month_values = month_features[int(day.month)]

        for slot in range(96):
            feat = [
                float(slot),
                float(day.dayofweek),
                float(day.month),
                float(day.dayofweek >= 5),
                float(math.sin(2.0 * math.pi * slot / 96.0)),
                float(math.cos(2.0 * math.pi * slot / 96.0)),
                float(prev_day.iloc[slot]),
                float(prev_2day.iloc[slot]),
                float(prev_7day.iloc[slot]),
                *prev_day_stats.tolist(),
            ]
            for col in EXOG_COLS:
                target_curve = exog[col].iloc[day_idx]
                prev_curve = exog[col].iloc[day_idx - 1]
                feat.extend(
                    [
                        float(target_curve.iloc[slot]),
                        float(prev_curve.iloc[slot]),
                        float(target_curve.iloc[slot] - prev_curve.iloc[slot]),
                        float(target_curve.mean()),
                        float(target_curve.std()),
                        float(prev_curve.mean()),
                    ]
                )
            feat.extend(slow_values.tolist())
            feat.extend(month_values.tolist())
            rows.append(np.asarray(feat, dtype=np.float32))
            targets.append(float(price.iloc[day_idx, slot]))
            meta.append(
                {
                    "day": day,
                    "slot": slot,
                    "target_price": float(price.iloc[day_idx, slot]),
                }
            )

    return np.stack(rows), np.asarray(targets, dtype=np.float32), meta, feature_names


def _split_days(meta: list[dict], train_ratio: float, val_ratio: float) -> tuple[set[pd.Timestamp], set[pd.Timestamp], set[pd.Timestamp]]:
    all_days = sorted({item["day"] for item in meta})
    n_days = len(all_days)
    train_end = int(n_days * train_ratio)
    val_end = int(n_days * (train_ratio + val_ratio))
    train_days = set(all_days[:train_end])
    val_days = set(all_days[train_end:val_end])
    test_days = set(all_days[val_end:])
    if not train_days or not test_days:
        raise ValueError("Split produced empty train or test days.")
    return train_days, val_days, test_days


def _train_regime_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[HistGradientBoostingRegressor, dict[str, float]]:
    best_model: HistGradientBoostingRegressor | None = None
    best_metrics: dict[str, float] | None = None
    best_mae = float("inf")

    for max_depth in (4, 6, 8):
        for min_leaf in (10, 20, 40):
            model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_depth=max_depth,
                max_iter=400,
                min_samples_leaf=min_leaf,
                l2_regularization=0.05,
                random_state=42,
            )
            model.fit(x_train, y_train)
            if len(x_val) == 0:
                pred_val = model.predict(x_train)
                ref = y_train
            else:
                pred_val = model.predict(x_val)
                ref = y_val
            mae = float(np.mean(np.abs(pred_val - ref)))
            rmse = float(np.sqrt(np.mean((pred_val - ref) ** 2)))
            if mae < best_mae:
                best_mae = mae
                best_model = model
                best_metrics = {
                    "val_mae": mae,
                    "val_rmse": rmse,
                    "max_depth": float(max_depth),
                    "min_samples_leaf": float(min_leaf),
                }

    assert best_model is not None
    assert best_metrics is not None
    return best_model, best_metrics


def _train_binary_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[HistGradientBoostingClassifier, dict[str, float]]:
    best_model: HistGradientBoostingClassifier | None = None
    best_metrics: dict[str, float] | None = None
    best_logloss_proxy = float("inf")

    for max_depth in (4, 6, 8):
        for min_leaf in (10, 20, 40):
            model = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=max_depth,
                max_iter=300,
                min_samples_leaf=min_leaf,
                random_state=42,
            )
            model.fit(x_train, y_train)
            if len(x_val) == 0:
                proba = model.predict_proba(x_train)[:, 1]
                ref = y_train
            else:
                proba = model.predict_proba(x_val)[:, 1]
                ref = y_val
            # use Brier score as a cheap stable proxy
            score = float(np.mean((proba - ref) ** 2))
            if score < best_logloss_proxy:
                best_logloss_proxy = score
                best_model = model
                best_metrics = {
                    "val_brier": score,
                    "max_depth": float(max_depth),
                    "min_samples_leaf": float(min_leaf),
                }

    assert best_model is not None
    assert best_metrics is not None
    return best_model, best_metrics


def _apply_two_stage_predictions(
    base_regression: np.ndarray,
    floor_proba: np.ndarray,
    peak_proba: np.ndarray,
    *,
    floor_threshold: float,
    peak_threshold: float,
    price_floor: float,
    price_cap: float,
) -> np.ndarray:
    pred = np.asarray(base_regression, dtype=np.float32).copy()
    pred = np.clip(pred, price_floor, price_cap)
    floor_mask = floor_proba >= floor_threshold
    peak_mask = (peak_proba >= peak_threshold) & (~floor_mask)
    pred[floor_mask] = float(price_floor)
    pred[peak_mask] = float(price_cap)
    return pred


def _mean_daily_relative_accuracy(
    meta_rows: list[dict],
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    price_floor: float,
) -> float:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for item, pred_value, target_value in zip(meta_rows, predictions, targets):
        day = str(item["day"].date())
        grouped.setdefault(day, []).append((float(pred_value), float(target_value)))

    scores: list[float] = []
    for rows in grouped.values():
        pred = np.asarray([row[0] for row in rows], dtype=np.float32)
        true = np.asarray([row[1] for row in rows], dtype=np.float32)
        relative_mape = float(np.mean(np.abs(pred - true) / np.maximum(np.abs(true), price_floor)))
        scores.append(max(0.0, 1.0 - relative_mape))
    return float(np.mean(scores)) if scores else 0.0


def _evaluate_daily(
    predictions: list[dict],
    *,
    price_floor: float,
    price_cap: float,
) -> dict:
    rows: list[dict] = []
    grouped: dict[str, list[dict]] = {}
    for item in predictions:
        grouped.setdefault(item["day"], []).append(item)

    for day, items in sorted(grouped.items()):
        pred = np.asarray([row["prediction"] for row in sorted(items, key=lambda row: row["slot"])], dtype=np.float32)
        true = np.asarray([row["target"] for row in sorted(items, key=lambda row: row["slot"])], dtype=np.float32)
        err = pred - true
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        relative_mape = float(np.mean(np.abs(err) / np.maximum(np.abs(true), price_floor)))
        relative_mape_accuracy = max(0.0, 1.0 - relative_mape)
        market_range_accuracy = max(0.0, 1.0 - mae / max(price_cap - price_floor, 1e-6))
        daily_mean_accuracy = max(0.0, 1.0 - abs(float(pred.mean()) - float(true.mean())) / max(abs(float(true.mean())), price_floor))
        peak_slot_hit = int(int(np.argmax(pred)) == int(np.argmax(true)))
        rows.append(
            {
                "day": day,
                "mae": mae,
                "rmse": rmse,
                "relative_mape": relative_mape,
                "relative_mape_accuracy": relative_mape_accuracy,
                "market_range_accuracy": market_range_accuracy,
                "daily_mean_accuracy": daily_mean_accuracy,
                "peak_slot_hit": peak_slot_hit,
                "pred_mean": float(pred.mean()),
                "true_mean": float(true.mean()),
                "pred_max": float(pred.max()),
                "true_max": float(true.max()),
            }
        )

    daily_df = pd.DataFrame(rows)
    summary = {
        "daily_rows": rows,
        "metrics": {
            "mean_day_mae": float(daily_df["mae"].mean()),
            "mean_day_rmse": float(daily_df["rmse"].mean()),
            "mean_relative_mape": float(daily_df["relative_mape"].mean()),
            "mean_relative_mape_accuracy": float(daily_df["relative_mape_accuracy"].mean()),
            "median_relative_mape_accuracy": float(daily_df["relative_mape_accuracy"].median()),
            "share_days_relative_mape_accuracy_ge_0_8": float((daily_df["relative_mape_accuracy"] >= 0.8).mean()),
            "mean_market_range_accuracy": float(daily_df["market_range_accuracy"].mean()),
            "median_market_range_accuracy": float(daily_df["market_range_accuracy"].median()),
            "share_days_market_range_accuracy_ge_0_8": float((daily_df["market_range_accuracy"] >= 0.8).mean()),
            "mean_daily_mean_accuracy": float(daily_df["daily_mean_accuracy"].mean()),
            "peak_slot_hit_rate": float(daily_df["peak_slot_hit"].mean()),
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_csv", default="data/GS(1).csv")
    parser.add_argument("--timeline_csv", default="supply_demand_data/features/shaanxi_2025_power_timeline_daily_asof.csv")
    parser.add_argument("--policy_csv", default="supply_demand_data/features/supply_demand_policy_calendar_daily.csv")
    parser.add_argument("--priority_monthly_csv", default="supply_demand_data/features/priority_generation_2025_monthly_features.csv")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--price_floor", type=float, default=40.0)
    parser.add_argument("--price_cap", type=float, default=1000.0)
    parser.add_argument("--floor_threshold", type=float, default=50.0)
    parser.add_argument("--peak_threshold", type=float, default=700.0)
    parser.add_argument("--output_dir", default="outputs/gs_price_2025_supply_demand_baseline")
    args = parser.parse_args()

    market = _load_market_frame(Path(args.source_csv))
    daily_features, month_features = _load_daily_supply_demand_features(
        Path(args.timeline_csv),
        Path(args.policy_csv),
        Path(args.priority_monthly_csv),
    )
    price, exog = _build_daily_matrices(market)
    x, y, meta, feature_names = _build_point_dataset(price, exog, daily_features, month_features)
    train_days, val_days, test_days = _split_days(meta, args.train_ratio, args.val_ratio)

    predictions: list[dict] = []
    val_predictions: list[dict] = []
    regime_reports: list[dict] = []
    feature_importance_frames: list[pd.DataFrame] = []

    for regime in REGIMES:
        train_mask = np.asarray([item["day"] in train_days and regime.matches(item["slot"]) for item in meta], dtype=bool)
        val_mask = np.asarray([item["day"] in val_days and regime.matches(item["slot"]) for item in meta], dtype=bool)
        test_mask = np.asarray([item["day"] in test_days and regime.matches(item["slot"]) for item in meta], dtype=bool)

        model, report = _train_regime_model(
            x_train=x[train_mask],
            y_train=y[train_mask],
            x_val=x[val_mask],
            y_val=y[val_mask],
        )
        floor_train = (y[train_mask] <= args.floor_threshold).astype(np.int32)
        floor_val = (y[val_mask] <= args.floor_threshold).astype(np.int32)
        floor_model, floor_report = _train_binary_classifier(
            x_train=x[train_mask],
            y_train=floor_train,
            x_val=x[val_mask],
            y_val=floor_val,
        )
        peak_train = (y[train_mask] >= args.peak_threshold).astype(np.int32)
        peak_val = (y[val_mask] >= args.peak_threshold).astype(np.int32)
        peak_model, peak_report = _train_binary_classifier(
            x_train=x[train_mask],
            y_train=peak_train,
            x_val=x[val_mask],
            y_val=peak_val,
        )

        chosen_floor_prob = 0.5
        chosen_peak_prob = 0.5
        if int(val_mask.sum()) > 0:
            val_pred_reg = model.predict(x[val_mask])
            val_floor_proba = floor_model.predict_proba(x[val_mask])[:, 1]
            val_peak_proba = peak_model.predict_proba(x[val_mask])[:, 1]
            val_meta = [meta[idx] for idx in np.flatnonzero(val_mask)]
            best_score = float("-inf")
            for floor_prob in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                for peak_prob in (0.2, 0.3, 0.4, 0.5, 0.6):
                    tuned_val = _apply_two_stage_predictions(
                        base_regression=val_pred_reg,
                        floor_proba=val_floor_proba,
                        peak_proba=val_peak_proba,
                        floor_threshold=floor_prob,
                        peak_threshold=peak_prob,
                        price_floor=args.price_floor,
                        price_cap=args.price_cap,
                    )
                    score = _mean_daily_relative_accuracy(
                        meta_rows=val_meta,
                        predictions=tuned_val,
                        targets=y[val_mask],
                        price_floor=args.price_floor,
                    )
                    if score > best_score:
                        best_score = score
                        chosen_floor_prob = floor_prob
                        chosen_peak_prob = peak_prob

        regime_reports.append(
            {
                "regime": regime.name,
                "train_rows": int(train_mask.sum()),
                "val_rows": int(val_mask.sum()),
                "test_rows": int(test_mask.sum()),
                **report,
                "floor_classifier_brier": float(floor_report["val_brier"]),
                "peak_classifier_brier": float(peak_report["val_brier"]),
                "floor_probability_threshold": float(chosen_floor_prob),
                "peak_probability_threshold": float(chosen_peak_prob),
            }
        )

        val_idx = np.flatnonzero(val_mask)
        val_pred = _apply_two_stage_predictions(
            base_regression=model.predict(x[val_mask]),
            floor_proba=floor_model.predict_proba(x[val_mask])[:, 1],
            peak_proba=peak_model.predict_proba(x[val_mask])[:, 1],
            floor_threshold=chosen_floor_prob,
            peak_threshold=chosen_peak_prob,
            price_floor=args.price_floor,
            price_cap=args.price_cap,
        )
        for idx, pred_value in zip(val_idx, val_pred):
            val_predictions.append(
                {
                    "day": str(meta[idx]["day"].date()),
                    "slot": int(meta[idx]["slot"]),
                    "prediction": float(pred_value),
                    "target": float(y[idx]),
                    "regime": regime.name,
                }
            )

        test_idx = np.flatnonzero(test_mask)
        pred = _apply_two_stage_predictions(
            base_regression=model.predict(x[test_mask]),
            floor_proba=floor_model.predict_proba(x[test_mask])[:, 1],
            peak_proba=peak_model.predict_proba(x[test_mask])[:, 1],
            floor_threshold=chosen_floor_prob,
            peak_threshold=chosen_peak_prob,
            price_floor=args.price_floor,
            price_cap=args.price_cap,
        )
        for idx, pred_value in zip(test_idx, pred):
            predictions.append(
                {
                    "day": str(meta[idx]["day"].date()),
                    "slot": int(meta[idx]["slot"]),
                    "prediction": float(pred_value),
                    "target": float(y[idx]),
                    "regime": regime.name,
                }
            )

        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            top = pd.DataFrame({"feature": feature_names, "importance": importances})
            top["regime"] = regime.name
            feature_importance_frames.append(top.sort_values("importance", ascending=False).head(40))

    evaluation = _evaluate_daily(
        predictions,
        price_floor=args.price_floor,
        price_cap=args.price_cap,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    val_predictions_df = pd.DataFrame(val_predictions).sort_values(["day", "slot"])
    predictions_df = pd.DataFrame(predictions).sort_values(["day", "slot"])
    val_predictions_df.to_csv(output_dir / "val_predictions.csv", index=False)
    predictions_df.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(evaluation["daily_rows"]).to_csv(output_dir / "daily_metrics.csv", index=False)
    pd.DataFrame(regime_reports).to_csv(output_dir / "regime_reports.csv", index=False)
    if feature_importance_frames:
        pd.concat(feature_importance_frames, ignore_index=True).to_csv(output_dir / "feature_importance_top40.csv", index=False)

    summary = {
        "dataset": "gs_price_2025_supply_demand_baseline",
        "source_csv": args.source_csv,
        "timeline_csv": args.timeline_csv,
        "policy_csv": args.policy_csv,
        "priority_monthly_csv": args.priority_monthly_csv,
        "price_floor": args.price_floor,
        "price_cap": args.price_cap,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "total_rows": int(len(meta)),
        "unique_days": int(len(sorted({item["day"] for item in meta}))),
        "train_days": {
            "count": int(len(train_days)),
            "start": str(min(train_days).date()),
            "end": str(max(train_days).date()),
        },
        "val_days": {
            "count": int(len(val_days)),
            "start": str(min(val_days).date()) if val_days else None,
            "end": str(max(val_days).date()) if val_days else None,
        },
        "test_days": {
            "count": int(len(test_days)),
            "start": str(min(test_days).date()),
            "end": str(max(test_days).date()),
        },
        "regime_reports": regime_reports,
        "metrics": evaluation["metrics"],
        "objective_audit": {
            "metric_name": "mean_daily_mean_accuracy",
            "metric_description": "Daily average price accuracy: 1 - abs(pred_day_mean - true_day_mean) / max(true_day_mean, price_floor)",
            "threshold": 0.8,
            "value": float(evaluation["metrics"]["mean_daily_mean_accuracy"]),
            "passed": bool(evaluation["metrics"]["mean_daily_mean_accuracy"] >= 0.8),
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
