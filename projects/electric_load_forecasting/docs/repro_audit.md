# Reproduction Audit

## 1. Positioning

当前仓库保留两条路径：

- `legacy baseline`
  - 当前 prompt-only Diff-DoRA 的工程基线
  - 用于快速冒烟、快速消融、结果回退
- `strict repro`
  - 后续正式论文复现候选路径
  - 用于全量训练与完整测试集评测

## 2. 已对齐

- Diff-DoRA 已切换为 prompt-only 实现，不再使用 controller / gate / 数值 side-channel
- `full` 与 `wo_diffdora` 的核心差异已经收敛为：是否在 retrieval prompt 中注入环境差分
- prompt 中只显式注入 `Diff T` / `Diff P`
- `diff_occ` 保留为内部 gap reasoning 信号，不再作为环境差分字段直接显示

## 3. 工程近似

- 当前实现依赖标准 DoRA + prompt supervision 隐式体现“幅值分量更重要”
- 仓库没有显式验证或诊断 magnitude / direction 的参数更新分布
- 这意味着当前实现可以称为“高保真工程复现”，但还不能单凭代码结构声称完成了内部学习动力学层面的严格论文等价

## 4. 协议偏差

### legacy baseline

- `train_experts` 默认 `max_samples_per_expert=1000`
- 有 retrieval cache 时，expert-local retrieval bank 默认截断为 `800`
- quick launcher `scripts/run_train_eval_diffdora.sh` 默认 `max_eval=6`
- quick ablation 常用 `balanced_random` 节点子集，而不是全节点

这些设置适合快速比较 `full` 与 `wo_diffdora`，但不应直接作为最终论文表格协议。

### strict repro

- `scripts/run_train_eval_diffdora_strict.sh`
  - `max_samples_per_expert=0`，表示全量 expert 训练样本
  - `retrieval_bank_max_samples_per_expert=0`，表示完整 expert-local retrieval bank
- `scripts/run_eval_diffdora_strict.sh`
  - `max_eval=0`，表示完整测试集
  - `node_sampling=all`，表示全节点 routed evaluation

## 5. 当前推荐口径

- 当前 quick 路径：`legacy baseline`
  - 可写成“prompt-only Diff-DoRA 的高保真工程复现”
  - 适合用于快速验证 `full > wo_diffdora` 的趋势
- 当前 strict 路径：`strict repro`
  - 才是后续正式论文复现的主候选入口
  - 需要以 full test / all nodes 结果作为主要报告依据
