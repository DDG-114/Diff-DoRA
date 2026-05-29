# GS 2025 Electricity Price Single-Adapter Fine-Tuning

## 任务定义

当前实验不再使用多专家划分，只预测一个目标变量：`Price`。其余电力系统变量不作为预测节点，而作为已知辅助特征输入模型，包括：

- `total_generation_forecast`：发电总出力预测
- `bidding_space`：竞价空间
- `unified_load_forecast`：统一负荷预测
- `pumped_storage`：抽蓄
- `unified_renewable_forecast`：统一新能源预测
- `tie_line_schedule`：联络线计划

预测粒度为 15 分钟，`horizon=96` 表示连续预测未来一天 96 个电价点。

## 数据划分

使用 `gs_price_2025` 数据集，也就是只在 2025 年内部做训练、验证和测试：

- 训练集：2025 年前 80%
- 验证集：2025 年接下来的 10%
- 测试集：2025 年最后 10%

电价按固定边界 `0-1000` 归一化，因此模型内部输出的 `0.040` 对应原始电价约 `40 CNY/MWh`。这个固定边界不会从测试集重新估计，可以减少归一化口径变化。

## 历史窗口设计

两周历史在 15 分钟粒度下是：

```text
14 days * 24 hours/day * 4 points/hour = 1344 points
```

本实验采用双层历史输入：

```text
history_len = 96
context_history_len = 1344
```

含义是：

- `history_len=96`：最近一天的电价逐点写入 prompt，模型可以直接看到完整日内曲线。
- `context_history_len=1344`：过去两周作为长历史上下文，不逐点全部写入，而是采样成少量 anchor 并给出均值、标准差、最大值和最小值等摘要。

这样做比直接设置 `history_len=1344` 更稳。直接把 1344 个历史点全部写入 prompt，会显著增加上下文长度，并且还要叠加检索样例、辅助变量和 96 步输出，容易造成截断或让模型只学到局部复制。

## 检索增强

RAG 检索从训练窗口中找到相似历史样例，并在 prompt 中展示：

- reference history：相似样例的历史电价窗口
- reference future：该相似样例之后真实发生的未来 96 步电价

当前检索向量已经改为 context-aware。除了最近一天的价格形态和未来辅助变量摘要，还会纳入两周长历史上下文的统计信息，包括均值、波动、范围和前后半段变化。这样检索更接近“找相似的近期曲线 + 相似的两周背景 + 相似的系统条件”。

## Prompt 与监督目标

推荐使用：

```text
prompt_style = direct_physical
target_style = numeric_only
```

`direct_physical` 会把以下信息交给模型：

- 最近一天电价历史
- 两周长历史摘要
- 检索到的相似 history-to-future 样例
- 当前窗口和检索样例之间的差分信息
- 已知未来系统辅助变量摘要

`numeric_only` 表示训练时只监督最终 96 个数值，而不是让模型生成长 CoT 解释。这样更适合 96 步连续输出，因为长 CoT 很容易造成解析失败或输出长度不稳定。

## 主动样本选择

训练时使用：

```text
active_selection = price_dynamic
active_budget_ratio = 0.8
```

它会优先选择未来一天电价变化更明显的窗口，例如：

- 未来 96 步方差更大
- 日内最大最小价差更大
- 明显偏离低价地板
- 存在峰值或坡度变化

这个设计是为了减少模型继续坍缩到常数低价，例如一直输出 `0.040`。

## 推荐运行命令

```bash
projects/electricity_price_forecasting/scripts/run_single_h96_2week.sh
```

等价训练核心配置如下：

```bash
.venv/bin/python -m src.train.train_shared_adapter \
  --dataset gs_price_2025 \
  --horizon 96 \
  --output_dir outputs/gs_price_2025_single_h96_2week_direct_v1 \
  --epochs 3 \
  --batch_size 2 \
  --grad_accum 8 \
  --lr 1e-4 \
  --history_len 96 \
  --context_history_len 1344 \
  --neighbor_k 0 \
  --window_stride 24 \
  --use_dora \
  --use_rag \
  --use_diff_dora \
  --prompt_style direct_physical \
  --target_style numeric_only \
  --active_selection price_dynamic \
  --active_budget_ratio 0.8 \
  --max_length 4096 \
  --max_new_tokens 1024 \
  --retrieval_device auto
```

评估使用非重叠日窗口：

```bash
.venv/bin/python -m src.eval.eval_saved_adapter \
  --dataset gs_price_2025 \
  --horizon 96 \
  --adapter_dir outputs/gs_price_2025_single_h96_2week_direct_v1/adapter \
  --node_id Price \
  --history_len 96 \
  --context_history_len 1344 \
  --neighbor_k 0 \
  --window_stride 96 \
  --use_rag \
  --use_diff_dora \
  --prompt_style direct_physical \
  --max_eval 0 \
  --sampling head \
  --max_new_tokens 1024 \
  --output outputs/gs_price_2025_single_h96_2week_direct_v1_eval_full.json
```

## 微调学到的内容

这个微调不是让模型预测所有电力系统变量，也不是按专家拆节点。它学习的是一个单目标映射：

```text
最近一天电价曲线
+ 过去两周电价背景
+ 已知未来系统变量
+ 相似历史日的后续走势
-> 未来一天 96 点电价
```

如果结果仍然出现大量常数低价，说明主要瓶颈不是多专家划分，而是 96 步长序列生成本身过难。下一步更合理的备选方案是把一天拆成 `horizon=24` 或 `horizon=16` 的滚动预测，再拼接成 96 点。
