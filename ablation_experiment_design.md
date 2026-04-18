# Diff-DoRA 消融实验设计

## 1. 目标

这套消融实验的目标不是只做一个结果表，而是回答下面几个因果问题：

1. `MoE` 的收益到底来自“按域路由”还是来自别的训练细节。
2. `CoT` 三阶段提示到底有没有稳定提升，还是只是在单次抽样上看起来更好。
3. `DoRA / Diff-DoRA` 的收益是否真实存在，还是被评测子集波动掩盖了。
4. 模型收益主要出现在哪个维度：`整体误差`、`长预测步`、`CBD/Residential` 某一类节点，还是仅仅体现在 `parse success rate`。

当前仓库已经具备较完整的专家训练、固定子集评测和可视化能力，因此建议直接基于现有入口组织实验，而不是另起一套脚本。

## 2. 先说结论：应该怎么做

建议把消融分成两层：

1. `当前仓库可立即执行的消融`
2. `论文级严格消融`

原因是当前代码里有两个会影响结论解释的现实约束：

1. `train_experts.py` 没有显式暴露训练随机种子，当前多次运行不能严格分离“训练随机性”。
2. `eval_paper_ablation.py` 的 `--seed` 控制的是评测样本和评测节点的随机抽样，所以现有 `seed42/43/44` 结果混合了“评测子集变化”。
3. `train_single.py` 默认是 `--node_idx 0` 的单节点训练入口，不能直接当成“去掉 MoE 后的全节点共享 adapter”基线。
4. 当前共享单 adapter 流程没有 `Diff-DoRA` 训练入口，所以 `wo_moe` 如果直接拿现有单模型脚本跑，会同时混入“去掉 MoE”和“去掉 Diff-DoRA”两个变化。

因此：

1. 你现在可以先跑一个 `专家级核心消融`，用来验证 `CoT` 和 `DoRA` 在专家路由框架内的作用。
2. 如果要写成更严格的论文或报告，必须补一个“共享 adapter 的全节点训练基线”，并把训练种子和评测种子拆开。

## 3. 当前仓库里各变体的精确定义

下面这个表先把“变量到底变了什么”说明白。

| 变体 | 路由 | Prompt | Adapter | Diff-DoRA | 解释 |
|---|---|---|---|---|---|
| `full` | 有 | `cot` | `DoRA` | 开 | 完整 Diff-DoRA + 专家路由 |
| `wo_cot` | 有 | `direct_physical` | `DoRA` | 开 | 去掉显式三阶段推理，但保留 RAG、diff、邻域和静态物理信息 |
| `wo_dora` | 有 | `cot` | `LoRA` | 关 | 去掉 DoRA 后，Diff-DoRA 也自然不可用 |
| `wo_moe` | 无 | `cot` | 理想上应为共享 `DoRA` | 理想上可开/可关 | 用于验证专家路由是否必要 |
| `base_model` | 无 | `cot` | 无 adapter | 关 | 大模型原始能力基线 |

这里有两个关键说明：

1. 当前代码中的 `wo_cot` 是一个比较干净的消融，因为它只去掉了显式分阶段 reasoning，保留了检索信息和物理上下文。
2. 当前代码中的 `wo_moe` 还不够干净，因为现有 `train_single.py` 不是“全节点共享 adapter”训练器。

## 4. 推荐的实验分层

### 4.1 第一层：当前仓库可立即执行的核心消融

这层先不追求完美，只追求结论清晰、能直接跑通。

建议先做：

1. `full`
2. `wo_cot`
3. `wo_dora`

为什么先做这三个：

1. 这三个都可以直接用 [`train_experts.py`](/root/Diff-DoRA/src/train/train_experts.py) 和 [`eval_paper_ablation.py`](/root/Diff-DoRA/src/eval/eval_paper_ablation.py) 完成。
2. 三者的比较能直接回答“显式推理”和“DoRA/Diff-DoRA”在专家框架里是否有效。
3. 现有 [`scripts/run_ablation_expert_only.sh`](/root/Diff-DoRA/scripts/run_ablation_expert_only.sh) 已经覆盖这三项。

### 4.2 第二层：论文级严格消融

这一层是最终应该汇报的版本，推荐包含：

1. `full`
2. `wo_cot`
3. `wo_dora`
4. `wo_moe`
5. `base_model`

其中 `wo_moe` 必须满足下面条件才算公平：

1. 训练数据覆盖全部节点，而不是单个 `node_idx`。
2. 训练预算与专家模型尽量一致。
3. Prompt 仍然使用 `RAG + CoT`。
4. 如果目标是纯粹测 `MoE`，最好让共享 adapter 也支持 `DoRA`，必要时再补一版“共享 Diff-DoRA”作附加对照。

## 5. 实验问题与假设

### 5.1 主问题

1. `full` 是否在 `overall RMSE / MAE` 上稳定优于 `wo_cot` 和 `wo_dora`。
2. `CoT` 的收益是否主要体现在长预测步，比如 `h4-h6`。
3. `DoRA / Diff-DoRA` 的收益是否主要体现在某一个域，比如 `CBD`。
4. `MoE` 是否缩小了 `CBD` 与 `Residential` 的误差差距。

### 5.2 我的预期

如果实现是健康的，理论上应该看到：

1. `full` 最优或至少与最优非常接近。
2. `wo_cot` 在短 horizon 上不一定明显变差，但在长 horizon 上更容易退化。
3. `wo_dora` 的退化通常比 `wo_cot` 更平滑，更多表现为整体误差上升。
4. `wo_moe` 的主要损失应出现在跨域异质性较强的数据上。

如果没有看到这个趋势，优先怀疑：

1. 评测子集采样噪声过大。
2. 训练随机性没有控制。
3. `parse failure` 干扰了指标。
4. `wo_moe` 基线不公平。

## 6. 固定控制变量

下面这些变量建议全程固定，不要在消融里随意改：

| 项目 | 建议值 |
|---|---|
| dataset | `st_evcdp` |
| 主 horizon | `6` |
| history_len | `12` |
| neighbor_k | `7` |
| epochs | `2` |
| batch_size | `8` |
| lr | `2e-4` |
| max_length | `2560` |
| lora_rank | `32` |
| lora_alpha | `32` |
| retrieval top-k | `2` |
| eval split | `test` |
| node sampling | `balanced_random` |
| max_nodes_per_domain | `12` |
| max_eval | `60` |
| max_new_tokens | `512` |
| infer_batch_size | `12` |

额外建议：

1. `训练随机种子` 和 `评测抽样种子` 必须分开。
2. 在正式报告里，所有变体都必须使用同一批 `selected_sample_indices` 和 `selected_nodes`。
3. 如果 `parse_success_rate < 0.98`，该轮结果要单独标记。

## 7. 指标设计

### 7.1 主指标

1. `overall RMSE`
2. `overall MAE`

### 7.2 辅助指标

1. `h1-h6` 的逐步 `RMSE / MAE`
2. `CBD` 与 `Residential` 的 `overall RMSE / MAE`
3. `parse_success_rate`
4. `requested_predictions / parsed_predictions`

### 7.3 诊断指标

1. `CBD-Residential` 的误差差值
2. 各变体跨重复实验的 `std`
3. 出错样本的 `raw_generation` 模式

## 8. 实验组织建议

### 8.1 先跑一个最小可交付版本

建议第一轮只做：

1. 数据集：`st_evcdp`
2. horizon：`6`
3. 变体：`full / wo_cot / wo_dora`
4. 重复次数：`3`

这能最快告诉你当前实现是否真有“完整模型优于消融”的趋势。

### 8.2 再做完整报告版本

建议完整版本按下面顺序扩展：

1. `st_evcdp, h=6, 5个变体, 3~5次重复`
2. `st_evcdp, h=3/9/12, 只跑关键变体 full / wo_cot / wo_dora / wo_moe`
3. `urbanev, h=6, 至少跑 full / wo_cot / wo_dora`

原因很简单：先验证主结论，再验证 horizon 泛化，最后验证跨数据集稳定性。

## 9. 当前结果对实验设计的启发

从现有输出目录里的三个专家级结果文件看：

1. `seed42/43/44` 的结果波动比较大。
2. 当前 `full` 并没有稳定优于 `wo_cot` 和 `wo_dora`。
3. 至少有一轮出现了明显的 `parse_success_rate` 下降。

这说明你后面写结论时不能只报单次最好结果，必须报：

1. `mean ± std`
2. `parse_success_rate`
3. 固定评测子集

否则结论很容易被人质疑成“抽样刚好有利”。

## 10. 立即可执行版命令

### 10.1 专家级核心消融

直接使用现有脚本：

```bash
cd /root/Diff-DoRA
source .venv/bin/activate

bash scripts/run_ablation_expert_only.sh st_evcdp 6
```

这会生成：

1. `full`
2. `wo_cot`
3. `wo_dora`
4. `expert_ablation.json`
5. 对应可视化图

输出目录默认是：

```text
outputs/ablation_expert_st_evcdp_h6_fixed
```

### 10.2 建议的重复实验组织方式

现有脚本默认只跑一轮。为了做重复实验，建议你手动组织为：

```text
outputs/ablation_suite_st_evcdp_h6/
  repeat_1/
  repeat_2/
  repeat_3/
```

每一轮内部都保存：

1. `full/`
2. `wo_cot/`
3. `wo_dora/`
4. `expert_ablation.json`
5. `figures/`

注意：当前代码没有显式训练种子，所以“repeat_1/2/3”更像独立重复运行，而不是严格受控 seed。

## 11. 论文级严格版应补的两个小改动

如果你要把消融写进正式论文或答辩，建议先补下面两个功能再跑完整套件。

### 11.1 给训练入口加独立随机种子

建议给下面两个入口都加：

1. [`train_experts.py`](/root/Diff-DoRA/src/train/train_experts.py)
2. [`train_single.py`](/root/Diff-DoRA/src/train/train_single.py)

至少要控制：

1. `random.seed`
2. `numpy.random.seed`
3. `torch.manual_seed`
4. `torch.cuda.manual_seed_all`
5. `TrainingArguments(seed=..., data_seed=...)`

### 11.2 增加“共享 adapter 的全节点训练”入口

严格的 `wo_moe` 不应使用当前默认的 `--node_idx 0` 单节点流程。推荐做法：

1. 在 `train_single.py` 增加 `--all_nodes`，把每个 sample 展开成所有节点的训练样本。
2. 或新建一个 `train_shared_adapter.py`，复用 `EVDataset` 的 `sample_node_idx` 机制。

只有这样，`wo_moe` 才真的是“去掉专家路由”，而不是“换成单节点基线”。

## 12. 建议增加的机制级扩展消融

这些不是第一优先级，但一旦主消融跑通，非常值得补。

### 12.1 Diff 特征分量消融

目标：验证 `diff_occ / diff_temp / diff_price` 哪个贡献最大。

建议变体：

1. `full`
2. `full_wo_diff_occ`
3. `full_wo_diff_temp`
4. `full_wo_diff_price`
5. `full_diff_occ_only`

这组实验需要给 diff 特征注入位置加开关。

### 12.2 Diff-DoRA 控制器容量消融

目标：验证当前 `DiffController` 是否过强或过弱。

建议配置：

1. `hidden_dim = 8 / 32 / 64`
2. `scale = 0.1 / 0.5 / 1.0`

这组实验仓库已经暴露了 `diff_hidden_dim` 和 `diff_scale`，实现成本较低。

### 12.3 路由标签消融

目标：验证收益来自路由本身，还是来自当前 `CBD/Residential` 标签恰好有效。

建议变体：

1. `真实物理标签路由`
2. `随机路由`
3. `全部送 expert_0`
4. `全部送 expert_1`

如果随机路由也能接近 `full`，说明当前 MoE 解释站不住。

## 13. 推荐的结果表

### 13.1 主表

| Variant | RMSE↓ | MAE↓ | Parse↑ | CBD MAE↓ | Residential MAE↓ | h1 RMSE↓ | h6 RMSE↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| full |  |  |  |  |  |  |  |
| wo_cot |  |  |  |  |  |  |  |
| wo_dora |  |  |  |  |  |  |  |
| wo_moe |  |  |  |  |  |  |  |
| base_model |  |  |  |  |  |  |  |

正式版建议填 `mean ± std`。

### 13.2 扩展表

| Horizon | Variant | RMSE↓ | MAE↓ | Parse↑ |
|---|---|---:|---:|---:|
| 3 | full |  |  |  |
| 3 | wo_cot |  |  |  |
| 6 | full |  |  |  |
| 6 | wo_cot |  |  |  |
| 9 | full |  |  |  |
| 9 | wo_cot |  |  |  |
| 12 | full |  |  |  |
| 12 | wo_cot |  |  |  |

## 14. 最终结论怎么写才稳

建议用下面这个判断标准： 

1. 如果 `full` 只在单次结果里最优，但 `mean ± std` 不占优，不能写“显著优于”。
2. 如果 `full` 的 `parse_success_rate` 明显更低，即使 RMSE 更低，也要把格式稳定性作为负面因素写出来。
3. 如果 `wo_cot` 与 `full` 差距主要集中在 `h5-h6`，结论应写成“CoT 主要提升长步预测稳定性”。
4. 如果 `wo_dora` 的退化集中在某一域，结论应写成“DoRA/Diff-DoRA 对域内异质性更敏感的节点更有帮助”。

## 15. 我建议你实际执行的顺序

最务实的顺序是：

1. 先跑 `st_evcdp, h=6, full / wo_cot / wo_dora, 3次重复`
2. 看 `full` 是否稳定占优
3. 如果没有稳定占优，先修训练种子和 `wo_moe` 基线
4. 再补 `wo_moe` 和 `base_model`
5. 最后扩到 `h=3/9/12` 与 `urbanev`

这样不会一开始就把算力花在一套解释不干净的结果上。
