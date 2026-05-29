"""
scripts/inspect_stev_data.py
----------------------------
快速查看处理好的 ST-EVCDP 数据集的内容与结构。
用法：
  python scripts/inspect_stev_data.py
"""
import pickle
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_st_evcdp import load_st_evcdp
from src.data.build_samples import build_samples

SEP = "=" * 60

# ── 1. 加载 processed pkl ──────────────────────────────────────
print(SEP)
print("1. 加载 data/processed/st_evcdp.pkl")
print(SEP)
data = load_st_evcdp()

occ       = data["occupancy"]        # (T, N) 归一化
occ_raw   = data["occupancy_raw"]    # (T, N) 原始
ts        = data["timestamps"]       # DatetimeIndex
node_meta = data["node_meta"]        # DataFrame
adj       = data["adj"]              # (N, N)
norm_min  = data["norm_min"]
norm_max  = data["norm_max"]

T, N = occ.shape
print(f"时间步数 T       : {T}")
print(f"节点数   N       : {N}")
print(f"时间范围          : {ts[0]}  →  {ts[-1]}")
print(f"采样间隔 (推断)   : {ts[1] - ts[0]}")
print(f"归一化范围        : [{norm_min:.4f},  {norm_max:.4f}]")

# ── 2. 占用率矩阵统计 ──────────────────────────────────────────
print()
print(SEP)
print("2. occupancy 矩阵统计（归一化后）")
print(SEP)
print(f"shape    : {occ.shape}   dtype={occ.dtype}")
print(f"min/max  : {occ.min():.4f}  /  {occ.max():.4f}")
print(f"mean±std : {occ.mean():.4f} ± {occ.std():.4f}")
print(f"NaN 数量  : {np.isnan(occ).sum()}")

print()
print("前 5 个时间步 × 前 6 个节点（归一化占用率）：")
df_preview = pd.DataFrame(
    occ[:5, :6],
    index=ts[:5],
    columns=[f"N{i}" for i in range(6)],
)
print(df_preview.to_string())

# ── 3. 原始占用率 ──────────────────────────────────────────────
print()
print(SEP)
print("3. occupancy_raw 统计（反归一化原始值）")
print(SEP)
print(f"shape    : {occ_raw.shape}   dtype={occ_raw.dtype}")
print(f"min/max  : {occ_raw.min():.4f}  /  {occ_raw.max():.4f}")
print(f"mean±std : {occ_raw.mean():.4f} ± {occ_raw.std():.4f}")

# ── 4. 节点元数据 ──────────────────────────────────────────────
print()
print(SEP)
print("4. node_meta（节点元数据）")
print(SEP)
if node_meta.empty:
    print("（nodes.csv 未找到，node_meta 为空 DataFrame）")
else:
    print(f"shape   : {node_meta.shape}")
    print(f"columns : {list(node_meta.columns)}")
    print()
    print("前 5 行：")
    print(node_meta.head().to_string())

    # 如果有 zone_type 字段，统计分布
    if "zone_type" in node_meta.columns:
        print()
        print("zone_type 分布：")
        print(node_meta["zone_type"].value_counts().to_string())

# ── 5. 邻接矩阵 ──────────────────────────────────────────────
print()
print(SEP)
print("5. adj（邻接矩阵）")
print(SEP)
print(f"shape       : {adj.shape}   dtype={adj.dtype}")
non_zero = (adj > 0).sum()
total_entries = adj.size
sparsity = 1.0 - non_zero / total_entries
print(f"非零元素数   : {non_zero}  /  {total_entries}  （稀疏度 {sparsity:.2%}）")
print(f"权重范围     : [{adj[adj > 0].min():.4f},  {adj.max():.4f}]")
print()
print(f"左上角 6×6 子矩阵：")
print(np.array2string(adj[:6, :6], precision=3, suppress_small=True))

# ── 6. 滑动窗口样本统计 ──────────────────────────────────────
print()
print(SEP)
print("6. build_samples 统计（horizons=[3,6]）")
print(SEP)
samples = build_samples(occ, ts, adj=adj, horizons=[3, 6])

for h in [3, 6]:
    s_list = samples[h]
    print(f"\n  horizon={h}  样本数={len(s_list)}")
    if s_list:
        s = s_list[0]
        print(f"    第0个样本字段：")
        for k, v in s.items():
            if isinstance(v, np.ndarray):
                print(f"      {k:12s}: shape={v.shape}  dtype={v.dtype}"
                      f"  range=[{v.min():.3f}, {v.max():.3f}]")
            else:
                print(f"      {k:12s}: {v}")

# ── 7. 按节点汇总：各节点时序均值 top/bottom 5 ────────────────
print()
print(SEP)
print("7. 各节点时序均值（归一化）— 最高 / 最低 5 个节点")
print(SEP)
node_means = occ.mean(axis=0)        # (N,)
top5_idx    = np.argsort(node_means)[-5:][::-1]
bot5_idx    = np.argsort(node_means)[:5]
print("  最高占用率节点：")
for i in top5_idx:
    name = node_meta.index[i] if not node_meta.empty else i
    print(f"    节点 {str(name):>10s}  mean={node_means[i]:.4f}")
print("  最低占用率节点：")
for i in bot5_idx:
    name = node_meta.index[i] if not node_meta.empty else i
    print(f"    节点 {str(name):>10s}  mean={node_means[i]:.4f}")

print()
print(SEP)
print("检查完毕 ✓")
print(SEP)
