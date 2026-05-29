# Renewable Solar To Wotai Transfer Workflow

## 1. 任务目标

本文实验主线不是直接在沃太数据上训练，而是采用跨数据集迁移方案：

```text
源域训练：renewable_solar
目标域测试：wotai_evcdp 中的 pv_total_power
```

目标是验证：在可再生能源光伏发电数据上训练得到的时序知识，能否迁移到沃太项目中的真实光伏功率预测任务。

## 2. 数据集来源与任务划分

### 2.1 源域：renewable_solar

原始来源：

```text
data/raw/renewable_generation/source/
```

其中包含多个太阳能站点 Excel 文件。预处理后得到：

```text
data/raw/renewable_solar/
data/processed/renewable_solar.pkl
```

当前保留的站点：

```text
solar_site_01
solar_site_02
solar_site_04
```

当前规模：

```text
occupancy shape: (70175, 3)
weather shape: (70175, 6)
adj shape: (3, 3)
```

目标变量：

```text
各光伏站点的 Power (MW)
```

外生变量：

```text
total_solar_irradiance
direct_normal_irradiance
global_horizontal_irradiance
temperature
pressure
humidity
```

### 2.2 目标域：wotai_evcdp

原始来源：

```text
data/raw/wotai_source/
```

适配后格式：

```text
data/raw/wotai_evcdp/
data/processed/wotai_evcdp.pkl
```

当前功能节点：

```text
actual_load
pv_total_power
storage_ac_power
grid_active_power
```

当前规模：

```text
occupancy shape: (38208, 4)
weather shape: (38208, 15)
adj shape: (4, 4)
```

在本实验中，只评估：

```text
node_id = pv_total_power
```

原因是它与 `renewable_solar` 的任务语义最一致，都是光伏功率预测。

## 3. 数据预处理思路

### 3.1 renewable_solar 的处理方式

原始 renewable 数据不是 `time x node` 矩阵，而是“每个站点一个 Excel 时序表”。因此需要先将单站点数据转成项目统一格式。

预处理脚本：

```text
scripts/prepare_renewable_generation.py
```

主要处理步骤：

1. 读取每个 solar station 的 Excel 表。
2. 统一字段名和时间戳格式。
3. 取 `Power (MW)` 作为目标序列。
4. 对多站点时间轴做公共重叠区间对齐。
5. 剔除时间覆盖不足的站点。
6. 构造：
   - `occupancy.csv`：时间 × 站点功率矩阵
   - `weather.csv`：外生特征矩阵
   - `nodes.csv`：站点容量、类型、源文件名等元数据
   - `adjacency.csv`：按训练段相关性构造的站点关系矩阵

### 3.2 wotai_evcdp 的处理方式

沃太原始数据是单项目综合能源数据，不是传统充电站点图数据。因此采用“功能节点”方式接入：

预处理脚本：

```text
scripts/prepare_wotai_evcdp.py
```

主要处理步骤：

1. 从负荷预测、光伏预测、储能、电表、天气等原始文件中抽取可对齐信号。
2. 按 15 分钟粒度重采样。
3. 构造 4 个功能节点：
   - `actual_load`
   - `pv_total_power`
   - `storage_ac_power`
   - `grid_active_power`
4. 构造天气特征和相关性邻接矩阵。

## 4. 数据划分与样本构造

项目统一使用时间切分，而不是随机切分。

### 4.1 renewable_solar 划分

划分比例：

```text
train / val / test = 0.6 / 0.2 / 0.2
```

当前得到：

```text
train: (12211, 8)
val:   (4070, 8)
test:  (4071, 8)
```

样本构造参数：

```text
horizon = 16
history_len = 16
context_history_len = 672
window_stride = 16
neighbor_k = 7
```

#### 4h 预测的数据处理说明

- 当前 `renewable_solar` 的一个时间步对应 `15 min`，因此 `horizon=16` 对应 `16 x 15 min = 4 h` 多步预测；`history_len=16` 表示输入最近 4 小时历史，`context_history_len=672` 表示额外保留最近 7 天长上下文。
- 原始 solar 站点 Excel 会先由 `scripts/prepare_renewable_generation.py` 统一字段名，并同时兼容文本时间戳与 Excel serial 日期；随后对每个站点执行去重、排序和数值化，避免时间列格式不一致导致有效样本被误删。
- 8 个站点随后被对齐到共同的 15 分钟时间网格；功率序列按时间插值并做 `ffill/bfill`，气象变量沿同一时间轴对齐后聚合，邻接矩阵则由前 60% 时段的站间绝对相关系数构造，并显式移除自环。
- `src/data/load_renewable_generation.py` 会把对齐后的功率矩阵做 min-max 归一化，并写入 `data/processed/renewable_solar.pkl`；当前主链路默认使用 full-data 统计量，然后再由 `src/data/build_splits.py` 按时间顺序切分为 `0.6 / 0.2 / 0.2` 的 train / val / test。
- `src/data/build_samples.py` 对每个窗口都会生成 `x_hist`、`time_feat`、`nbr_feat` 和 `y`，当 `context_history_len > history_len` 时还会附带 `x_context`、`time_feat_context`、`nbr_feat_context`。其中 `time_feat` 使用 `hour` 与 `day-of-week` 的正余弦编码，`nbr_feat` 使用 top-`k=7` 邻居的按行归一化平均功率。
- 对 4h 主实验而言，`window_stride=16` 表示窗口每次前进 16 个时间步，也就是使用非重叠的 4h 预测块；如果命令行传入 `window_stride=0`，`src/data/windowing.py` 会自动把它解析成 `stride=horizon=16`，与这里显式写 `16` 的语义完全一致。

在训练集上：

```text
train windows = 721
```

如果同时构建 `include_test=True` 的 sample cache，则测试集还会得到：

```text
test windows = 212
```

由于共享训练时会对每个窗口展开全部 8 个站点，因此训练条目数为：

```text
721 x 8 = 5768
```

### 4.2 wotai_evcdp 划分

同样采用：

```text
train / val / test = 0.6 / 0.2 / 0.2
```

当前得到：

```text
train: (22924, 4)
val:   (7642, 4)
test:  (7642, 4)
```

在目标域测试中，使用：

```text
horizon = 6
history_len = 12
window_stride = 6
```

测试窗口总数：

```text
1271
```

如果不显式修改 `max_eval`，`eval_saved_adapter.py` 默认只评估前 200 条。

## 5. 源域训练策略

### 5.1 为什么使用 shared adapter

`renewable_solar` 现在已经保留全部 8 个太阳能站点，且语义仍然高度一致。如果强行用双 expert MoE 训练，路由语义依旧偏弱。因此主实验继续采用共享适配器训练：

```text
一个 base model
+ 一个 DoRA adapter
+ 所有 solar 站点共享参数
```

训练入口：

```text
src/train/train_shared_adapter.py
```

### 5.2 训练配置

当前采用的新主配置为：

```bash
bash scripts/run_renewable_shared_dual_history.sh
```

等价的显式命令为：

```bash
.venv/bin/python -m src.train.train_shared_adapter \
  --dataset renewable_solar \
  --horizon 16 \
  --output_dir outputs/renewable_solar_shared_h16_hist16_ctx672 \
  --epochs 2 \
  --batch_size 16 \
  --lr 2e-4 \
  --history_len 16 \
  --context_history_len 672 \
  --neighbor_k 7 \
  --window_stride 16 \
  --use_dora \
  --use_rag \
  --use_diff_dora \
  --prompt_style cot \
  --item_sampling shuffled_pairs \
  --max_train_items 0 \
  --max_eval 0 \
  --retrieval_device auto
```

关键含义：

- `use_dora`：使用 DoRA 适配器训练
- `use_rag`：源域训练时启用检索增强
- `use_diff_dora`：启用差分驱动上下文
- `prompt_style cot`：使用显式三阶段推理模板
- `item_sampling shuffled_pairs`：打乱 `(窗口, 节点)` 对，提高训练覆盖

### 5.3 RAG 与 CoT 设计

这条主线的 prompt 已经做过增强，不再只是 EV 负荷模板，而是根据任务语义做了自适配。

对于 `renewable_solar -> wotai pv_total_power`：

- system prompt 采用 `PV power generation forecaster`
- 历史序列表述为 `power output`
- 检索样本采用：

```text
Ref history + Ref future
```

而不是只展示检索历史。

同时，差分分析也从传统的：

```text
Diff T / Diff P
```

扩展为适合 solar 任务的：

```text
Diff T
Diff SI
Diff DNI
Diff GHI
Diff W
```

其中：

- `Diff SI`：太阳总辐照差分
- `Diff DNI`：法向直射辐照差分
- `Diff GHI`：全球水平辐照差分
- `Diff W`：风速差分

### 5.4 训练结果

本次 `renewable_solar_shared_h6_v3` 训练输出为：

```text
adapter_dir: outputs/renewable_solar_shared_h6_v3/adapter
train_items: 21045
epochs: 2
train_runtime: 9278.9441 s
train_loss: 0.0485
```

## 6. 目标域迁移评估策略

### 6.1 评估目标

迁移阶段不再更新模型参数，只将源域训练好的 adapter 直接应用于：

```text
dataset = wotai_evcdp
node_id  = pv_total_power
```

也就是说，这是一个：

```text
frozen transfer evaluation
```

### 6.2 评估命令

最终可用的 CoT 评估命令如下：

```bash
.venv/bin/python -m src.eval.eval_saved_adapter \
  --dataset wotai_evcdp \
  --horizon 6 \
  --adapter_dir outputs/renewable_solar_shared_h6_v3/adapter \
  --node_id pv_total_power \
  --history_len 12 \
  --neighbor_k 7 \
  --window_stride 6 \
  --use_rag \
  --use_diff_dora \
  --prompt_style cot \
  --retrieval_cache data/retrieval_cache/wotai_evcdp_h6_step6.pkl \
  --max_eval 200 \
  --sampling head \
  --max_new_tokens 512 \
  --output outputs/renewable_solar_shared_h6_v3_to_wotai_pv_h6_cot512.json
```

### 6.3 为什么需要 `max_new_tokens = 512`

由于 CoT prompt 中同时包含：

- 当前 12 步历史
- 两个检索参考的 `history + future`
- 趋势摘要
- 多项 solar 差分特征
- 三阶段推理文本

所以如果 `max_new_tokens` 太小，模型会在 Stage 3 之前被截断，导致大量：

```text
parsed_prediction = null
```

将 `max_new_tokens` 提升到 `512` 后，解析成功率恢复正常。

## 7. 最新目标域评估结果

结果文件：

```text
outputs/renewable_solar_shared_h6_v3_to_wotai_pv_h6_cot512.json
```

当前摘要：

```text
requested_samples = 200
evaluated_samples = 200
parse_failures    = 0
parse_success_rate = 1.0
```

主要指标：

```text
overall RMSE = 193.65
overall MAE  = 99.90
```

分步误差呈现正常的多步递增趋势：

```text
h1 MAE ≈ 40.92
h6 MAE ≈ 156.15
```

说明：

当前迁移结果已经可以作为正式实验保留，因为：

1. prompt 语义已与 solar 功率任务对齐；
2. CoT + RAG(history+future) + enhanced differential analysis 已经生效；
3. parse success rate 已达到 100%；
4. 指标可稳定计算。

## 8. 实验链路总结

整个流程可以概括为：

```text
renewable_generation/source
    -> 预处理为 renewable_solar
    -> 构建 train/val/test
    -> 构建 sliding windows
    -> 在 renewable_solar 上训练 shared adapter
    -> 冻结 adapter
    -> 在 wotai_evcdp 的 pv_total_power 上做 transfer evaluation
    -> 用目标域 retrieval cache + solar-aware CoT prompt 完成推理
```

这条链路的核心价值在于：

```text
将源域光伏发电时序知识迁移到目标域真实项目光伏功率预测，
从而验证 Diff-DoRA 框架在跨数据集、跨场景条件下的泛化能力。
```
