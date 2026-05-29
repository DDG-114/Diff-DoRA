# 导师汇报简稿：基于 LLM 的陕西电力现货价格预测

## 1. 研究目标

目标是使用 `2025` 年陕西电力现货数据，预测日前 `96` 个时间窗口（一日 `96` 个 `15min` 点）的电价曲线，并探索 LLM 在该任务中的可行技术路线。

## 2. 文献调研结论

结合近期时序 / LLM / 电价预测相关论文，当前最重要的判断是：

1. 直接让 LLM 一次性生成 `96` 个未来价格点，稳定性较差。
2. 更合理的路线是：
   - 短窗口预测
   - 检索增强
   - 候选曲线（candidate skeleton）
   - 残差修正（residual refinement）
3. 对于电价这种带尖峰、地板价、强时段结构的任务，LLM 更适合作为“局部修正器”，而不是主预测器。

相关文献与工程路线已整理在：

- [LLM_EP_PLAN.md](/root/Diff-DoRA/projects/electricity_price_forecasting/LLM_EP_PLAN.md:1)

## 3. 当前实现路线

基于现有项目代码，已经实现以下 LLM 路线：

1. 使用非 LLM 强基线先生成全天候选曲线（candidate skeleton）
2. 将 candidate 注入 LLM prompt
3. 使用 `h16` 短窗口预测并滚动拼接为 `96` 点
4. 让 LLM 学习 candidate 的残差修正，而不是直接生成全部价格
5. 引入检索增强（RAG）与系统侧辅助特征

当前相关代码已实现：

- [projects/electricity_price_forecasting/scripts/run_llm_h16_candidate_rolling_day.sh](/root/Diff-DoRA/projects/electricity_price_forecasting/scripts/run_llm_h16_candidate_rolling_day.sh:1)
- [src/utils/price_candidate.py](/root/Diff-DoRA/src/utils/price_candidate.py:1)
- [src/train/train_single.py](/root/Diff-DoRA/src/train/train_single.py:1)
- [src/prompts/prompt_vanilla.py](/root/Diff-DoRA/src/prompts/prompt_vanilla.py:1)
- [projects/electricity_price_forecasting/scripts/eval_gs_price_rolling_day.py](/root/Diff-DoRA/projects/electricity_price_forecasting/scripts/eval_gs_price_rolling_day.py:1)

## 4. 已解决的技术问题

当前 LLM 路线已经解决了以下问题：

1. 直接 `96` 点生成不现实，已改为 `h16 rolling`
2. 原始 LLM 会塌缩为近似常数曲线
3. residual 训练中 candidate 与目标量纲不一致，已修正
4. residual 输出曾出现数值爆炸，已通过归一化和裁剪控制
5. prompt 语义与监督目标不一致，已改为显式 residual 修正提示

## 5. 当前实验结果

### 5.1 当前最强 candidate baseline（同一批 10 天）

- `mean_day_mae ≈ 94.80`
- `mean_daily_mean_accuracy ≈ 0.6897`
- `share_days_daily_mean_accuracy_ge_0.8 = 0.2`

补充说明：

- 如果改用“自然日 `00:00-23:45`”口径评估同一测试尾段（`2025-11-26` 至 `2025-12-31`），candidate baseline 可达到：
  - `mean_daily_mean_accuracy ≈ 0.7542`
  - `share_days_daily_mean_accuracy_ge_0.8 = 0.5`
- 这说明评估窗口定义会显著影响结论，后续汇报时必须明确口径。

进一步分析表明：

- 如果只针对“日均价精度”优化，在 candidate 曲线基础上加入一个“日级整体偏移量”修正，理论上可将该口径显著拉高。
- 对 `2025-11-26` 至 `2025-12-31` 的自然日测试段，oracle 式日级偏移修正结果为：
  - `mean_daily_mean_accuracy ≈ 0.9876`
  - `share_days_daily_mean_accuracy_ge_0.8 ≈ 0.9722`
- 这说明：如果导师验收核心更偏向“日均价精度”，那么“candidate 曲线 + LLM 日级偏移量预测”比“LLM 全量改写 96 点”更贴近目标。

### 5.2 当前 LLM residual 路线（同一批 10 天）

结果文件：

- [gs_price_2025_llm_h16_candidate_residual_normfix_10day_rolling_day96.json](/root/Diff-DoRA/outputs/gs_price_2025_llm_h16_candidate_residual_normfix_10day_rolling_day96.json:1)

对比汇总：

- [summary.json](/root/Diff-DoRA/outputs/gs_price_2025_candidate_vs_llm/summary.json:1)

当前结果：

- `mean_day_mae ≈ 94.80`
- `mean_day_rmse ≈ 171.14`
- `mean_daily_mean_accuracy ≈ 0.6897`
- `share_days_daily_mean_accuracy_ge_0.8 = 0.2`

### 5.3 当前最接近目标的 LLM 路线：日级 offset 校准

结果文件：

- [gs_price_2025_llm_daily_offset_smoke_eval_fixed_v2.json](/root/Diff-DoRA/outputs/gs_price_2025_llm_daily_offset_smoke_eval_fixed_v2.json:1)

当前结果：

- `mean_day_mae ≈ 50.85`
- `mean_day_rmse ≈ 98.70`
- `mean_daily_mean_accuracy ≈ 0.7952`
- `share_days_daily_mean_accuracy_ge_0.8 = 0.6`

解释：

- 与逐点/逐块 LLM refinement 相比，LLM 只预测“日级整体偏移量”的路线更贴近导师所关心的“日均精度”口径。
- 该路线目前已经显著逼近 `0.8`，是当前最值得继续加训练预算验证的一条 LLM 方案。

进一步分析表明：

- 将日级 offset 离散为少数几个档位后，从上限实验看，仍然可以保持很高的日均价精度：
  - 5 档位 offset 的 oracle 结果：
    - `mean_daily_mean_accuracy ≈ 0.9193`
    - `share_days_daily_mean_accuracy_ge_0.8 ≈ 0.9167`
- 这说明：若后续继续坚持 LLM 路线，最值得继续的不是 offset 回归，而是“LLM 选择 offset 档位”的分类式方案。

## 6. 当前阶段结论

当前可以明确得出以下结论：

1. LLM 路线已经从“不可用”推进到了“稳定可运行”
2. 但在当前配置和预算下，LLM residual refinement 还没有体现出超越 candidate baseline 的增益
3. 当前更合理的解释是：LLM 已经学会稳定复制 candidate 骨架，但尚未学会提供实质性的附加修正收益
4. 因此，当前阶段不宜把 LLM 当作“整条 96 点曲线主预测器”
5. 更合理的角色是：
   - candidate baseline 的局部修正器
   - retrieval-aware 的辅助修正模块
6. 如果导师的验收更接近“日均价精度”，那么目前最有希望过线的 LLM 路线不是全曲线生成，而是“candidate baseline + LLM 日级 offset 校准”

## 7. 下一步建议

建议后续工作重点转向：

1. selective refinement
   - 只让 LLM 修正 candidate 明显不可靠的时段
   - 例如峰值段、陡坡段、candidate 与检索样本差异大的时段

2. 保留 candidate baseline 的主体作用
   - 不再让 LLM 尝试重写整条未来曲线

3. 若继续做大规模实验，应先验证：
   - selective refinement 是否优于 full residual refinement
   - 日级 offset 预测是否能在稳定训练后把“日均价精度”继续推高

## 8. 当前最客观的判断

到目前为止：

- “使用 LLM 做电力现货价格预测”的技术路线已经建立并实现
- 但“LLM 路线在当前实验设置下达到日均精度 80%”尚未被验证
- 当前最稳健的系统仍然是 candidate baseline
- LLM 当前更多体现为“稳定复制 candidate”的能力，而不是“稳定改进 candidate”的能力
- 因此，LLM 更适合作为其上层局部修正模块继续研究

补充判断：

- 从目标口径上看，最值得继续的 LLM 方向不是逐点重写，而是“日级 offset 预测”
- 理由是：若只对 candidate 曲线做整体均值修正，自然日口径下理论上可以显著逼近或超过导师要求的日均价精度目标
