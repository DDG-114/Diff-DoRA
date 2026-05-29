#!/usr/bin/env python3
"""Sequence baseline for 2025 day-ahead electricity price forecasting.

This model predicts a 96-point next-day price curve from:
  - previous-day / previous-week price curves
  - target-day exogenous curves from GS(1).csv
  - daily supply-demand and policy features

The implementation is fully within 2025 data and keeps same-day price leakage
out of the input features.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


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
    df = df.loc[df["Date"].dt.year == 2025].reset_index(drop=True)
    return df


def _load_daily_features(
    timeline_csv: Path,
    policy_csv: Path,
) -> pd.DataFrame:
    timeline = _normalise_columns(_read_csv(timeline_csv))
    policy = _normalise_columns(_read_csv(policy_csv))
    timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce")
    policy["date"] = pd.to_datetime(policy["date"], errors="coerce")
    daily = timeline.merge(policy, on="date", how="outer").sort_values("date")
    daily = daily.loc[daily["date"].dt.year == 2025].copy()
    daily = daily.set_index("date")
    daily = daily.apply(pd.to_numeric, errors="coerce").ffill().fillna(0.0)
    keep = [col for col in daily.columns if float(np.abs(daily[col]).sum()) > 0.0]
    return daily[keep]


class DayAheadSequenceDataset(Dataset):
    def __init__(
        self,
        *,
        frame: pd.DataFrame,
        daily_features: pd.DataFrame,
        price_floor: float,
        price_cap: float,
        exog_min: np.ndarray,
        exog_max: np.ndarray,
        starts: list[int],
    ) -> None:
        self.frame = frame
        self.daily_features = daily_features
        self.price_floor = float(price_floor)
        self.price_cap = float(price_cap)
        self.price_scale = self.price_cap - self.price_floor
        self.exog_min = exog_min.astype(np.float32)
        self.exog_max = exog_max.astype(np.float32)
        self.starts = np.asarray(starts, dtype=np.int64)

        self.price = frame[PRICE_COL].to_numpy(dtype=np.float32)
        self.price_norm = (self.price - self.price_floor) / self.price_scale
        self.exog = frame[EXOG_COLS].to_numpy(dtype=np.float32)
        self.exog_norm = (self.exog - self.exog_min) / (self.exog_max - self.exog_min + 1e-6)
        self.timestamps = frame["Date"]

        slow_min = daily_features.min(axis=0)
        slow_max = daily_features.max(axis=0)
        self.slow_norm = (daily_features - slow_min) / (slow_max - slow_min + 1e-6)

    def __len__(self) -> int:
        return len(self.starts)

    def _slow_vector(self, day: pd.Timestamp) -> np.ndarray:
        row = self.slow_norm.reindex([day.normalize()]).ffill().fillna(0.0).iloc[0]
        return row.to_numpy(dtype=np.float32)

    def __getitem__(self, idx: int):
        t = int(self.starts[idx])
        hist_price = self.price_norm[t - 96 : t]
        prev2_price = self.price_norm[t - 192 : t - 96]
        prev7_price = self.price_norm[t - 672 : t - 576]

        future_exog = self.exog_norm[t : t + 96]
        prev_day_exog = self.exog_norm[t - 96 : t]
        exog_delta = future_exog - prev_day_exog

        exog_day_mean = future_exog.mean(axis=0)
        exog_day_std = future_exog.std(axis=0)
        exog_prev_mean = prev_day_exog.mean(axis=0)

        price_stats = np.array(
            [
                float(hist_price.mean()),
                float(hist_price.std()),
                float(hist_price.max()),
                float(hist_price.min()),
                float(prev2_price.mean()),
                float(prev7_price.mean()),
            ],
            dtype=np.float32,
        )
        stats_rep = np.repeat(price_stats[None, :], 96, axis=0)

        future_ts = self.timestamps.iloc[t : t + 96]
        slot = (future_ts.dt.hour * 4 + future_ts.dt.minute // 15).to_numpy(dtype=np.float32)
        dow = future_ts.dt.dayofweek.to_numpy(dtype=np.float32)
        month = future_ts.dt.month.to_numpy(dtype=np.float32)
        time_feat = np.stack(
            [
                np.sin(2.0 * math.pi * slot / 96.0),
                np.cos(2.0 * math.pi * slot / 96.0),
                np.sin(2.0 * math.pi * dow / 7.0),
                np.cos(2.0 * math.pi * dow / 7.0),
                np.sin(2.0 * math.pi * month / 12.0),
                np.cos(2.0 * math.pi * month / 12.0),
            ],
            axis=-1,
        ).astype(np.float32)

        slow = self._slow_vector(pd.Timestamp(future_ts.iloc[0]))
        slow_rep = np.repeat(slow[None, :], 96, axis=0)

        floor_share = np.full((96, 1), float((self.price[t - 96 : t] <= 50.0).mean()), dtype=np.float32)
        peak_share = np.full((96, 1), float((self.price[t - 96 : t] >= 700.0).mean()), dtype=np.float32)
        exog_mean_rep = np.repeat(exog_day_mean[None, :], 96, axis=0)
        exog_std_rep = np.repeat(exog_day_std[None, :], 96, axis=0)
        exog_prev_mean_rep = np.repeat(exog_prev_mean[None, :], 96, axis=0)

        features = np.concatenate(
            [
                hist_price[:, None],
                prev2_price[:, None],
                prev7_price[:, None],
                future_exog,
                prev_day_exog,
                exog_delta,
                exog_mean_rep,
                exog_std_rep,
                exog_prev_mean_rep,
                stats_rep,
                floor_share,
                peak_share,
                time_feat,
                slow_rep,
            ],
            axis=1,
        ).astype(np.float32)

        base_price = self.price[t - 96 : t].astype(np.float32)
        target_price = self.price[t : t + 96].astype(np.float32)
        target_floor = (target_price <= 50.0).astype(np.float32)
        target_peak = (target_price >= 700.0).astype(np.float32)
        return features, base_price, target_price, target_floor, target_peak


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2 * dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2 * dilation, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.conv1(x))
        y = self.conv2(y)
        return F.gelu(self.norm(x + y))


class DayAheadSequenceNet(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 192) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(feature_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dilation=1),
            ResidualBlock(hidden_dim, dilation=2),
            ResidualBlock(hidden_dim, dilation=4),
            ResidualBlock(hidden_dim, dilation=8),
        )
        self.delta_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.floor_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.peak_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = F.gelu(self.input_proj(x.transpose(1, 2)))
        hidden = self.blocks(hidden)
        delta = self.delta_head(hidden).squeeze(1)
        floor = self.floor_head(hidden).squeeze(1)
        peak = self.peak_head(hidden).squeeze(1)
        return delta, floor, peak


def _daily_metrics(pred: np.ndarray, true: np.ndarray, *, price_floor: float, price_cap: float) -> dict:
    mae = np.mean(np.abs(pred - true), axis=1)
    rmse = np.sqrt(np.mean((pred - true) ** 2, axis=1))
    relative_mape = np.mean(np.abs(pred - true) / np.maximum(np.abs(true), price_floor), axis=1)
    relative_acc = np.maximum(0.0, 1.0 - relative_mape)
    daily_mean_acc = np.maximum(
        0.0,
        1.0 - np.abs(pred.mean(axis=1) - true.mean(axis=1)) / np.maximum(np.abs(true.mean(axis=1)), price_floor),
    )
    range_acc = np.maximum(0.0, 1.0 - mae / max(price_cap - price_floor, 1e-6))
    return {
        "mean_day_mae": float(mae.mean()),
        "mean_day_rmse": float(rmse.mean()),
        "mean_relative_mape_accuracy": float(relative_acc.mean()),
        "median_relative_mape_accuracy": float(np.median(relative_acc)),
        "share_days_relative_mape_accuracy_ge_0_8": float((relative_acc >= 0.8).mean()),
        "mean_daily_mean_accuracy": float(daily_mean_acc.mean()),
        "mean_market_range_accuracy": float(range_acc.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_csv", default="data/GS(1).csv")
    parser.add_argument("--timeline_csv", default="supply_demand_data/features/shaanxi_2025_power_timeline_daily_asof.csv")
    parser.add_argument("--policy_csv", default="supply_demand_data/features/supply_demand_policy_calendar_daily.csv")
    parser.add_argument("--price_floor", type=float, default=40.0)
    parser.add_argument("--price_cap", type=float, default=1000.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--output_dir", default="outputs/gs_price_2025_supply_demand_seq")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frame = _load_market_frame(Path(args.source_csv))
    daily_features = _load_daily_features(Path(args.timeline_csv), Path(args.policy_csv))
    train_cut = int(len(frame) * 0.8)
    exog = frame[EXOG_COLS].to_numpy(dtype=np.float32)
    exog_min = exog[:train_cut].min(axis=0)
    exog_max = exog[:train_cut].max(axis=0)

    timestamps = frame["Date"]
    train_starts: list[int] = []
    val_starts: list[int] = []
    test_starts: list[int] = []
    for t in range(672, len(frame) - 96 + 1, 8):
        day = pd.Timestamp(timestamps.iloc[t].normalize())
        if day <= pd.Timestamp("2025-10-20"):
            train_starts.append(t)
    for t in range(672, len(frame) - 96 + 1):
        dt = pd.Timestamp(timestamps.iloc[t])
        if dt.hour == 0 and dt.minute == 0 and pd.Timestamp("2025-10-20") < dt.normalize() <= pd.Timestamp("2025-11-25"):
            val_starts.append(t)
        if dt.hour == 0 and dt.minute == 0 and dt.normalize() > pd.Timestamp("2025-11-25"):
            test_starts.append(t)

    train_ds = DayAheadSequenceDataset(
        frame=frame,
        daily_features=daily_features,
        price_floor=args.price_floor,
        price_cap=args.price_cap,
        exog_min=exog_min,
        exog_max=exog_max,
        starts=train_starts,
    )
    val_ds = DayAheadSequenceDataset(
        frame=frame,
        daily_features=daily_features,
        price_floor=args.price_floor,
        price_cap=args.price_cap,
        exog_min=exog_min,
        exog_max=exog_max,
        starts=val_starts,
    )
    test_ds = DayAheadSequenceDataset(
        frame=frame,
        daily_features=daily_features,
        price_floor=args.price_floor,
        price_cap=args.price_cap,
        exog_min=exog_min,
        exog_max=exog_max,
        starts=test_starts,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    feature_dim = train_ds[0][0].shape[1]
    model = DayAheadSequenceNet(feature_dim=feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    floor_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0, device=device))
    peak_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8.0, device=device))

    def run_epoch(loader: DataLoader, *, training: bool) -> tuple[float, dict, np.ndarray | None, np.ndarray | None]:
        if training:
            model.train()
        else:
            model.eval()
        losses: list[float] = []
        preds: list[np.ndarray] = []
        trues: list[np.ndarray] = []

        with torch.set_grad_enabled(training):
            for feat, base, true, floor_target, peak_target in loader:
                feat = feat.to(device)
                base = base.to(device)
                true = true.to(device)
                floor_target = floor_target.to(device)
                peak_target = peak_target.to(device)

                delta, floor_logit, peak_logit = model(feat)
                pred = base + delta * (args.price_cap - args.price_floor)
                rel = torch.abs(pred - true) / torch.clamp(true, min=args.price_floor)
                reg_loss = rel.mean() + 0.002 * F.smooth_l1_loss(pred, true)
                loss = reg_loss + 0.12 * floor_loss(floor_logit, floor_target) + 0.08 * peak_loss(peak_logit, peak_target)

                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                losses.append(float(loss.item()))

                if not training:
                    floor_prob = torch.sigmoid(floor_logit)
                    peak_prob = torch.sigmoid(peak_logit)
                    pred_adj = pred.clone()
                    pred_adj[floor_prob >= 0.3] = float(args.price_floor)
                    pred_adj[(peak_prob >= 0.6) & (floor_prob < 0.3)] = float(args.price_cap)
                    preds.append(pred_adj.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())

        if training:
            return float(np.mean(losses)), {}, None, None

        pred_arr = np.concatenate(preds, axis=0)
        true_arr = np.concatenate(trues, axis=0)
        metrics = _daily_metrics(
            pred_arr,
            true_arr,
            price_floor=args.price_floor,
            price_cap=args.price_cap,
        )
        return float(np.mean(losses)), metrics, pred_arr, true_arr

    best_score = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    history: list[dict] = []
    patience = 0

    for epoch in range(args.epochs):
        train_loss, _, _, _ = run_epoch(train_loader, training=True)
        val_loss, val_metrics, _, _ = run_epoch(val_loader, training=False)
        scheduler.step(val_metrics["mean_relative_mape_accuracy"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **val_metrics,
            }
        )
        print(json.dumps(history[-1], ensure_ascii=False))
        score = float(val_metrics["mean_relative_mape_accuracy"])
        if score > best_score + 1e-4:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    _, val_metrics, pred_val, true_val = run_epoch(val_loader, training=False)
    _, test_metrics, pred_test, true_test = run_epoch(test_loader, training=False)
    assert pred_test is not None and true_test is not None
    assert pred_val is not None and true_val is not None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_rows: list[dict] = []
    for idx, start in enumerate(val_starts):
        day = str(pd.Timestamp(timestamps.iloc[start]).date())
        for slot in range(96):
            val_rows.append(
                {
                    "day": day,
                    "slot": slot,
                    "prediction": float(pred_val[idx, slot]),
                    "target": float(true_val[idx, slot]),
                }
            )
    test_rows: list[dict] = []
    for idx, start in enumerate(test_starts):
        day = str(pd.Timestamp(timestamps.iloc[start]).date())
        for slot in range(96):
            test_rows.append(
                {
                    "day": day,
                    "slot": slot,
                    "prediction": float(pred_test[idx, slot]),
                    "target": float(true_test[idx, slot]),
                }
            )
    pd.DataFrame(val_rows).to_csv(output_dir / "val_predictions.csv", index=False)
    pd.DataFrame(test_rows).to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    summary = {
        "dataset": "gs_price_2025_supply_demand_seq",
        "source_csv": args.source_csv,
        "timeline_csv": args.timeline_csv,
        "policy_csv": args.policy_csv,
        "device": device,
        "feature_dim": int(feature_dim),
        "train_samples": int(len(train_ds)),
        "val_samples": int(len(val_ds)),
        "test_days": int(len(test_ds)),
        "best_epoch": int(best_epoch),
        "val_metrics": val_metrics,
        "metrics": test_metrics,
        "objective_audit": {
            "metric_name": "mean_daily_mean_accuracy",
            "threshold": 0.8,
            "value": float(test_metrics["mean_daily_mean_accuracy"]),
            "passed": bool(test_metrics["mean_daily_mean_accuracy"] >= 0.8),
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
