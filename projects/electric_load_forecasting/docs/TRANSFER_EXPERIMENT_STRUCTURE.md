# 迁移实验结构设计

## 1. 目标

本文中的迁移实验旨在回答以下问题：

```text
源域上训练得到的时序知识，能否迁移到目标域上完成有效预测？
```

针对当前项目，迁移实验的主线为：

```text
source dataset  ->  target dataset
renewable_solar ->  wotai_evcdp.pv_total_power
```

同时可以扩展为对照实验：

```text
renewable_wind  -> wotai_evcdp.pv_total_power
st_evcdp        -> wotai_evcdp.pv_total_power
```

## 2. 实验层次

迁移实验建议分成四层结构组织。

### 2.1 数据层

记录源域与目标域的数据来源、处理方式和目标变量。

建议固定记录：

```text
source_dataset
target_dataset
source_task
target_task
source_node_scope
target_node_id
target_signal_name
time_frequency
horizon
history_len
window_stride
```

当前主线可写为：

```text
source_dataset   = renewable_solar
target_dataset   = wotai_evcdp
source_task      = solar power forecasting
target_task      = pv power forecasting
source_node_scope = all solar nodes
target_node_id   = pv_total_power
time_frequency   = 15min
horizon          = 6
history_len      = 12
window_stride    = 6
```

### 2.2 训练层

记录源域训练阶段的模型形式与配置。

建议固定记录：

```text
training_mode
base_model
adapter_type
epochs
batch_size
learning_rate
gradient_checkpointing
use_rag
use_diff_dora
prompt_style
retrieval_device
item_sampling
max_train_items
train_runtime
train_loss
```

当前主线通常为：

```text
training_mode = shared_adapter
base_model    = Qwen2.5-1.5B-Instruct
adapter_type  = DoRA
prompt_style  = cot 或 direct_physical
```

### 2.3 迁移评估层

记录目标域测试时的冻结迁移设置。

建议固定记录：

```text
adapter_dir
evaluation_mode
max_eval
sampling
use_rag
use_diff_dora
prompt_style
retrieval_cache
max_new_tokens
parse_success_rate
```

建议明确说明：

```text
是否冻结模型参数
是否在目标域仅重建 retrieval cache
是否只评估单一目标节点
```

当前主线可写为：

```text
evaluation_mode = frozen transfer
adapter_dir     = outputs/renewable_solar_shared_h6_v3/adapter
target_node_id  = pv_total_power
use_rag         = true
use_diff_dora   = true
prompt_style    = cot 或 direct_physical
retrieval_cache = data/retrieval_cache/wotai_evcdp_h6_step6.pkl
```

### 2.4 可视化层

记录最终用于展示的图、表、案例分析对象。

建议固定分成三类：

```text
总体指标图
逐时间步误差图
样本级案例图
```

## 3. 结果文件结构

迁移实验建议统一输出三种结果文件。

### 3.1 summary 文件

用于保存整体实验配置和关键指标。

建议字段：

```json
{
  "source_dataset": "...",
  "target_dataset": "...",
  "target_node_id": "...",
  "training_mode": "...",
  "adapter_dir": "...",
  "prompt_style": "...",
  "use_rag": true,
  "use_diff_dora": true,
  "max_eval": 200,
  "parse_success_rate": 1.0,
  "metrics": {
    "overall": {"mae": 0.0, "rmse": 0.0},
    "1": {"mae": 0.0, "rmse": 0.0},
    "2": {"mae": 0.0, "rmse": 0.0}
  }
}
```

### 3.2 sample-level 文件

用于保存逐样本预测结果，便于后续画图和做误差分析。

建议字段：

```json
{
  "sample_index": 0,
  "t_start": 12,
  "node_id": "pv_total_power",
  "history_raw": [...],
  "target_raw": [...],
  "prediction_raw": [...],
  "parse_ok": true,
  "retrieved_t_starts": [...],
  "raw_generation": "...",
  "system_prompt": "...",
  "user_prompt": "..."
}
```

### 3.3 visualization-ready 文件

如果后续图较多，建议单独导出一份轻量级绘图数据表。

建议字段：

```text
sample_index
t_start
node_id
horizon_step
true_value
pred_value
abs_error
experiment_tag
source_dataset
target_dataset
prompt_style
```

这样后续可以直接画：

```text
逐步误差图
连续轨迹图
不同实验方案对比图
```

## 4. 指标设计

迁移实验建议至少报告三类指标。

### 4.1 基础精度指标

```text
MAE
RMSE
nMAE
nRMSE
```

### 4.2 序列行为指标

```text
direction_accuracy
prediction_range_mean
constant_prediction_ratio
peak_abs_error
peak_time_error_steps
```

### 4.3 结果可用性指标

```text
parse_success_rate
parse_failures
```

如果使用 CoT，`parse_success_rate` 必须单独报告，否则高误差可能只是因为输出格式不稳定。

## 5. 可视化建议

### 5.1 总体对比图

建议画柱状图，对比不同源域迁移效果：

```text
renewable_solar -> wotai pv
renewable_wind  -> wotai pv
st_evcdp        -> wotai pv
```

横轴：

```text
实验方案
```

纵轴：

```text
overall MAE / overall RMSE
```

### 5.2 逐时间步误差图

建议画 horizon 误差折线图：

```text
h=1 到 h=6
```

用于展示误差是否随预测步长增加而增长。

### 5.3 行为分析图

建议至少补两张：

```text
constant prediction ratio
prediction range distribution
```

这样可以区分：

```text
模型是真的学到了趋势
还是只是输出保守的平坦序列
```

### 5.4 案例曲线图

建议从样本级结果中挑三类：

```text
最佳样本
典型样本
失败样本
```

每张图至少画：

```text
history
true future
predicted future
```

如果是 CoT 版本，可以在图旁边附简化的 reasoning 片段。

## 6. 推荐实验矩阵

建议按如下矩阵组织迁移实验。

### 6.1 主实验

```text
source: renewable_solar
target: wotai_evcdp.pv_total_power
prompt: cot
RAG: on
Diff-DoRA: on
```

### 6.2 Prompt 对照

```text
renewable_solar -> wotai pv
direct_physical
cot
```

### 6.3 Source 对照

```text
renewable_solar -> wotai pv
renewable_wind  -> wotai pv
st_evcdp        -> wotai pv
```

### 6.4 RAG 对照

```text
no RAG
history-only RAG
history+future exemplar RAG
```

### 6.5 Diff 对照

```text
Diff T / Diff P only
renewable-enhanced differentials
```

## 7. 当前主线建议

如果只保留一条最核心的迁移主线，建议为：

```text
renewable_solar(shared adapter, DoRA, RAG, Diff-DoRA, CoT)
-> wotai_evcdp.pv_total_power
```

然后至少配两条对照：

```text
renewable_solar + direct_physical -> wotai pv
renewable_wind  + same setting    -> wotai pv
```

这样你最终能够回答：

1. 光伏源域知识能否迁移到沃太光伏功率预测？
2. 显式 CoT 是否优于 direct physical prompt？
3. 语义更接近的源域是否比 wind 或 EV 需求源域更适合迁移？

## 8. 一句话总结

迁移实验的结构应当围绕：

```text
源域训练配置
目标域冻结评估配置
样本级预测轨迹
逐时间步误差
行为分析指标
```

这五部分展开。只有同时保留“总体指标”和“样本级轨迹”，迁移实验的可解释性和说服力才足够强。
