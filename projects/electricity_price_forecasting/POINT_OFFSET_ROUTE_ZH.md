# 96 点曲线预测路线：供需 candidate + 点级 offset 校准

## 结论

当前主指标已切换为 `mean_relative_accuracy`。在这个更严格的 96 点逐点相对精度口径下，当前点级 offset 路线尚未达到 0.8，但它仍然是比日级 offset 更合理的曲线修正路线。

```text
供需特征模型预测 96 点 candidate 曲线
        ↓
在验证集上学习每个 slot 的点级 offset
        ↓
用验证集选择 offset 收缩强度
        ↓
最终输出 96 点预测曲线
```

当前测试集结果：

```text
primary_metric = mean_relative_accuracy
mean_relative_accuracy = 0.6354462725
target_metric = 0.8
passed = false
mean_daily_mean_accuracy = 0.7931450952
mean_day_mae = 81.2380
```

验收文件：

```text
outputs/gs_price_2025_point_offset_calibrated/summary.json
outputs/gs_price_2025_point_offset_calibrated/test_predictions.csv
outputs/gs_price_2025_point_offset_calibrated/daily_metrics.csv
outputs/gs_price_2025_point_offset_calibrated/point_offsets.csv
outputs/gs_price_2025_point_offset_calibrated_viz.png
```

## Candidate 怎么来

这版 candidate 使用 `outputs/gs_price_2025_supply_demand_baseline/test_predictions.csv`。

它是非 LLM 的供需特征模型，输入包括：

```text
前一天/前两天/前七天同点价格
slot、星期、月份、是否周末
发电总出力预测
竞价空间
统一负荷预测
抽蓄
统一新能源预测
联络线计划
supply_demand_data 里的日级和政策特征
```

它本身已经是 96 点预测曲线，不是日均值预测。

## 点级 offset 怎么学

训练脚本：

```text
projects/electricity_price_forecasting/scripts/train_point_offset_calibrator.py
```

运行脚本：

```text
projects/electricity_price_forecasting/scripts/run_point_offset_calibrator.sh
```

核心公式：

```text
residual_slot = true_slot - candidate_slot
offset_slot = mean(validation residual_slot)
final_pred_slot = clip(candidate_slot + shrink * offset_slot, 40, 1000)
```

按 `mean_relative_accuracy` 自动选择结果：

```text
group_size = 4
offset_count = 24
shrink = 1.0
```

这表示它学习的是 24 个 1 小时段 offset，不是一天一个 offset。

## 为什么不是日级 offset

日级 offset 的公式是：

```text
final_pred_t = candidate_t + same_daily_offset
```

它只能整体上移或下移曲线，不能修复早晚峰、午间谷、尖峰错位。

当前点级 offset 的公式是：

```text
final_pred_t = candidate_t + offset_t
```

其中每个 `t` 可以不同，所以它确实是在 96 点层面修正曲线。

## LLM 的位置

当前最终文件没有把 LLM 作为数值预测器使用。原因是前面实验显示，LLM 直接生成 96 点或 residual 容易不稳定；当前供需模型 + 点级/分段 offset 能改善逐点相对精度，但按 `mean_relative_accuracy >= 0.8` 尚未达标。

更合理的 LLM 使用方式是作为下一步增强：

```text
LLM 读取供需、政策、节假日和市场状态
输出结构化 regime 判断
把 regime 作为额外特征喂给点级 offset / residual 模型
```

当前已验证版本说明：非 LLM 的 96 点 candidate 和点级/分段 offset 可以提升逐点相对精度，但还不能作为 `mean_relative_accuracy >= 0.8` 的达标证据。
