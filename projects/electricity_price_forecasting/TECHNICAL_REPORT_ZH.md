# 陕西电力现货 96 点日前价格预测技术报告

## 1. 任务目标

本项目目标是基于陕西省电力现货与供需相关数据，预测 2025 年电力现货日前价格曲线。

预测对象为每天 96 个 15 分钟窗口价格：

```text
P_day = [p_1, p_2, ..., p_96]
```

当前主指标已切换为 `mean_relative_accuracy`，即 96 点逐点相对精度。单日计算公式为：

```text
relative_accuracy_day =
max(0, 1 - mean(|pred_t - true_t| / max(|true_t|, 40))), t=1...96
```

测试集主指标为：

```text
mean_relative_accuracy =
mean(relative_accuracy_day_1, ..., relative_accuracy_day_N)
```

当前目标：

```text
mean_relative_accuracy >= 0.80
```

日均精度作为辅助指标保留：

```text
daily_mean_accuracy =
max(0, 1 - abs(mean(pred_1...pred_96) - mean(true_1...true_96)) / max(abs(mean(true_1...true_96)), 40))
```

## 2. 数据来源

主要价格与日前供需数据：

```text
data/GS(1).csv
```

该文件包含：

```text
Date
Price
发电总出力预测
竞价空间
统一负荷预测
抽蓄
统一新能源预测
联络线计划
```

补充供需与政策特征来自：

```text
supply_demand_data/features/shaanxi_2025_power_timeline_daily_asof.csv
supply_demand_data/features/supply_demand_policy_calendar_daily.csv
supply_demand_data/features/priority_generation_2025_monthly_features.csv
```

测试区间：

```text
2025-11-26 至 2025-12-31
36 天
每天 96 点
共 3456 条预测
```

## 3. 路线选择

最初评估过的 `candidate baseline + LLM 日级 offset` 方案只预测一天一个整体偏移量：

```text
final_pred_t = candidate_t + daily_offset
```

该方案可以修正日均价，但不能修正 96 点内部形状，例如早晚高峰、午间低谷、尖峰错位。因此它不适合作为最终 96 点曲线预测方案。

最终采用的路线是：

```text
供需特征模型预测 96 点 candidate 曲线
        ↓
在验证集上学习点级 offset
        ↓
用验证集选择 offset 收缩强度
        ↓
输出最终 96 点价格曲线
```

最终公式：

```text
final_pred_t = clip(candidate_t + shrink * offset_t, 40, 1000)
```

其中：

```text
t = 1...96
candidate_t = 供需模型预测的第 t 个价格点
offset_t = 验证集上该 slot 的平均残差
shrink = 验证集选择的收缩系数
```

按 `mean_relative_accuracy` 作为主指标后的自动选择结果：

```text
group_size = 4
offset_count = 24
shrink = 1.0
```

这表示最终方案使用 24 个 1 小时段 offset，不是一天一个 offset。

## 4. Candidate Baseline 来源

最终 candidate 使用：

```text
outputs/gs_price_2025_supply_demand_baseline/test_predictions.csv
```

该 baseline 是非 LLM 供需特征模型，核心输入包括：

```text
前一天同点价格
前两天同点价格
前七天同点价格
前一天价格均值、标准差、最大值、最小值
slot index
day_of_week
month
is_weekend
发电总出力预测
竞价空间
统一负荷预测
抽蓄
统一新能源预测
联络线计划
supply_demand_data 中的日级供需与政策特征
```

该 candidate 本身已经输出 96 点曲线，不是只预测日均价。

## 5. 点级 Offset 校准方法

实现脚本：

```text
projects/electricity_price_forecasting/scripts/train_point_offset_calibrator.py
```

运行脚本：

```text
projects/electricity_price_forecasting/scripts/run_point_offset_calibrator.sh
```

训练逻辑：

```text
1. 读取供需 baseline 的 validation prediction
2. 计算 validation residual:
   residual_t = true_t - candidate_t
3. 按 slot 学 offset:
   offset_t = mean(residual_t on validation set)
4. 搜索 group_size 和 shrink
5. 只用 validation 指标选择配置
6. 在 test set 上做最终评估
```

搜索空间：

```text
group_sizes = 1,4,8,12,24,48
shrink_grid = 0,0.25,0.5,0.75,1.0
clip_offset = 120
```

选择策略：

```text
只使用 validation set 选择配置
主指标为 mean_relative_accuracy
目标阈值 0.8 仅用于最终 audit，不参与 test set 调参
```

最终选择：

```text
group_size = 4
offset_count = 24
shrink = 1.0
```

## 6. 实验结果

### 6.1 最终方案

结果文件：

```text
outputs/gs_price_2025_point_offset_calibrated/summary.json
```

测试集指标：

```text
mean_day_mae = 81.2380
mean_day_rmse = 140.2309
mean_relative_accuracy = 0.6354
median_relative_accuracy = 0.6896
share_days_relative_accuracy_ge_0_8 = 0.3056
mean_daily_mean_accuracy = 0.7931
median_daily_mean_accuracy = 0.8184
share_days_daily_mean_accuracy_ge_0_8 = 0.5833
peak_slot_hit_rate = 0.1667
```

验收结论：

```text
metric_name = mean_relative_accuracy
threshold = 0.8
value = 0.6354462725
passed = false
```

### 6.2 对照结果

| 方案 | mean_relative_accuracy | mean_daily_mean_accuracy | mean_day_mae | 按主指标是否达 80% |
|---|---:|---:|---:|---|
| supply_demand_baseline | 0.5916 | 0.8123 | 82.3221 | 否 |
| supply_demand_fused | 0.6373 | 0.7542 | 97.1045 | 否 |
| LLM daily offset | 0.7810 | 0.7952 | 50.8505 | 否 |
| final point offset | 0.6354 | 0.7931 | 81.2380 | 否 |

说明：

```text
按新的主指标 mean_relative_accuracy，目前没有方案达到 0.8。
final point offset 将 baseline 的 mean_relative_accuracy 从 0.5916 提升到 0.6354，但仍未达标。
LLM daily offset 的 mean_relative_accuracy 较高，但它不是完整 36 天自然日测试集上的最终 96 点曲线修正方案，且仍低于 0.8。
```

## 7. 输出文件

最终预测：

```text
outputs/gs_price_2025_point_offset_calibrated/test_predictions.csv
```

字段包括：

```text
day
slot
prediction
target
slot_group
point_offset
candidate_prediction
residual
calibrated_error
```

每日指标：

```text
outputs/gs_price_2025_point_offset_calibrated/daily_metrics.csv
```

点级 offset：

```text
outputs/gs_price_2025_point_offset_calibrated/point_offsets.csv
```

汇总指标：

```text
outputs/gs_price_2025_point_offset_calibrated/summary.json
```

可视化：

```text
outputs/gs_price_2025_point_offset_calibrated_viz.png
```

路线简报：

```text
projects/electricity_price_forecasting/POINT_OFFSET_ROUTE_ZH.md
```

## 8. 可复现命令

重新运行点级 offset 校准：

```bash
bash projects/electricity_price_forecasting/scripts/run_point_offset_calibrator.sh
```

重新生成可视化 PNG：

```bash
.venv/bin/python projects/electricity_price_forecasting/scripts/render_point_offset_png.py
```

检查最终验收：

```bash
python - <<'PY'
import json
summary = json.load(open("outputs/gs_price_2025_point_offset_calibrated/summary.json"))
print(summary["objective_audit"])
print(summary["metrics"])
PY
```

## 9. 与 LLM 的关系

当前最终文件没有把 LLM 作为数值预测器使用。

原因：

```text
1. LLM 直接生成 96 个价格点不稳定
2. LLM 生成 full residual 容易复制 candidate 或过拟合
3. 日级 offset 只能修正日均价，不适合最终 96 点曲线预测
```

更合理的 LLM 使用方式是后续增强：

```text
LLM 读取供需、政策、节假日、市场状态文本
输出结构化 regime 判断
例如 low_price_floor、renewable_pressure_high、evening_peak_risk_medium
将这些 regime 特征输入点级 residual/offset 模型
```

也就是说，LLM 更适合做供需关系解释和状态分类，而不是直接输出 96 个价格点。

## 10. 局限性与下一步

当前方案按新主指标 `mean_relative_accuracy` 尚未达到 80%，主要局限为：

```text
1. mean_relative_accuracy 只有 0.6354，逐点曲线精度距离 0.8 仍有明显差距
2. mean_daily_mean_accuracy 也从旧口径的 0.8017 降为当前配置下的 0.7931
3. 点级 offset 是验证集平均残差，无法动态响应某一天的特殊供需状态
4. 当前测试区间是 2025-11-26 至 2025-12-31，需要更多滚动切分验证稳定性
```

建议下一步：

```text
1. 将优化目标改为 mean_relative_accuracy，而不是日均精度
2. 使用 LLM 生成结构化 regime 特征，再训练点级 residual 模型
3. 用分段动态 offset 替代静态 slot offset
4. 同时优化 peak_slot_hit_rate 和高价/低价时段的分段误差
```
