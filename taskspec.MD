
---

下面是**详细任务清单**。我按“你自己做”或者“交给 agent 做”都方便执行的格式写。

# 详细任务清单

## P0：初始化与对齐
- [ ] 建仓库 `lr_moe_repro`
- [ ] 创建 `repro_spec.md`
- [ ] 创建 `README.md`
- [ ] 固定 Python / CUDA / Transformers / PEFT 版本
- [ ] 建立实验日志规范
- [ ] 建立 `configs/`、`src/`、`scripts/`、`outputs/` 目录
- [ ] 写 `paper_alignment.md`，逐项记录论文设定与自己实现差异

**产出物**
- `repro_spec.md`
- `paper_alignment.md`
- `environment.yml` 或 `requirements.txt`

---

## P1：数据集打通
论文实验明确使用 ST-EVCDP 和 UrbanEV 两个真实 EV 充电数据集，并给出了 ST-EVCDP 6:2:2、UrbanEV 8:1:1 的划分方式。:contentReference[oaicite:1]{index=1}

### ST-EVCDP
- [ ] 下载原始数据
- [ ] 识别 occupancy 主表
- [ ] 读取时间索引
- [ ] 读取节点/区域元数据
- [ ] 读取空间关系
- [ ] 构建 train/val/test
- [ ] 验证长度、节点数、缺失值比例
- [ ] 输出标准化后的 `processed/st_evcdp.pkl`

### UrbanEV
- [ ] 下载原始数据
- [ ] 读取 occupancy
- [ ] 读取天气、价格、POI、邻接或距离信息
- [ ] 构建 train/val/test
- [ ] 验证长度、节点数、缺失值比例
- [ ] 输出 `processed/urbanev.pkl`

**产出物**
- `src/data/load_st_evcdp.py`
- `src/data/load_urbanev.py`
- `src/data/build_splits.py`
- `data/processed/*.pkl`

---

## P2：样本构造与评估脚本
论文把问题重构成条件概率生成任务，输入包括历史负荷、环境状态、空间拓扑和检索差分信号。:contentReference[oaicite:2]{index=2}

- [ ] 写滑动窗口采样器
- [ ] 历史窗口固定为 12
- [ ] 支持预测步长 3/6/9/12
- [ ] 构造时间特征
- [ ] 构造邻居汇总特征
- [ ] 写 RMSE / MAE 评估函数
- [ ] 写 sanity check：随机样本可视化

**产出物**
- `src/data/build_samples.py`
- `src/eval/metrics.py`
- `notebooks/check_samples.ipynb`

---

## P3：单专家 baseline
先不要碰 MoE，先跑通单专家。

- [ ] 接入 Qwen2.5-1.5B-Instruct
- [ ] 用 LoRA 做第一版微调
- [ ] 写 vanilla prompt
- [ ] 固定输出为数值列表
- [ ] 写 parser，保证输出可解析
- [ ] 跑 ST-EVCDP 上的 baseline
- [ ] 保存 per-horizon RMSE / MAE

**产出物**
- `src/prompts/prompt_vanilla.py`
- `src/prompts/parser.py`
- `src/models/qwen_peft.py`
- `src/train/train_single.py`

**验收标准**
- 训练可跑完
- 推理可输出
- 解析成功率高
- RMSE/MAE 正常

---

## P4：RAG + 差分信息
论文明确写了：对当前上下文先构造特征向量，再用 KNN 检索 top-k=2 历史片段，并计算环境差分。:contentReference[oaicite:3]{index=3}

- [ ] 定义 retrieval vector
- [ ] 实现 KNN 检索
- [ ] 限制检索池不泄漏未来信息
- [ ] 计算 `delta_temp`、`delta_price` 等差分
- [ ] 把差分写入 prompt
- [ ] 记录 retrieval 样本与当前样本的对应关系

**产出物**
- `src/retrieval/knn_retriever.py`
- `src/retrieval/diff_features.py`
- `data/retrieval_cache/*.pkl`

**验收标准**
- 检索逻辑无时间泄漏
- 差分特征数值正确
- prompt 中差分字段稳定生成

---

## P5：三阶段 CoT
论文的 CoT 固定为“差距分析—空间逻辑—数值预测”三阶段。:contentReference[oaicite:4]{index=4}

- [ ] 设计 CoT prompt 模板
- [ ] 设计 target 输出模板
- [ ] 强制最终输出 `Numerical Prediction: [...]`
- [ ] 让 parser 只抽最后数值
- [ ] 对比 vanilla vs RAG vs RAG+CoT

**产出物**
- `src/prompts/prompt_cot.py`
- `scripts/run_ablation_cot.sh`

**验收标准**
- CoT 输出结构稳定
- RAG+CoT 至少不差于 vanilla
- 解析错误率可控

---

## P6：硬路由 MoE
论文核心是硬路由混合专家，用物理标签把样本分流到不同专家。:contentReference[oaicite:5]{index=5}

- [ ] 定义 routing label
- [ ] ST-EVCDP 优先尝试直接使用已有区域属性
- [ ] UrbanEV 若无直接标签，则做代理划分
- [ ] 实现两个独立专家 checkpoint
- [ ] 训练时按 label 分流
- [ ] 推理时按 label 选择专家
- [ ] 统计各专家数据量和效果

**产出物**
- `src/routing/build_labels.py`
- `src/routing/hard_router.py`
- `src/train/train_experts.py`
- `src/models/lr_moe_wrapper.py`

**验收标准**
- 双专家版本优于单专家
- 路由逻辑可解释
- 各专家不会数据极端失衡

---

## P7：DoRA 与 Diff-DoRA
论文先基于 DoRA，再加入差分信号驱动的幅度适配。:contentReference[oaicite:6]{index=6}

- [ ] 先把 LoRA 切到 DoRA
- [ ] 复现实验：single + DoRA
- [ ] 实现 `g(diff_features)` 控制器
- [ ] 用控制器调制 magnitude 分量
- [ ] 比较 LoRA / DoRA / Diff-DoRA

**产出物**
- `src/models/dora_adapter.py`
- `src/models/diff_dora.py`
- `scripts/run_ablation_dora.sh`

**验收标准**
- Diff-DoRA 跑通
- 不引入训练崩溃
- 对高波动场景有增益

---

## P8：正式实验
论文实验分为全样本、少样本、零样本和消融。:contentReference[oaicite:7]{index=7}

### 全样本
- [ ] 3/6/9/12 步长全部评测
- [ ] RMSE/MAE 表格化输出

### 少样本
- [ ] 5/10/20/40/100% 训练比例
- [ ] 保持测试集不变
- [ ] 画比例-指标曲线

### 零样本
- [ ] 20/40/60/80% source ratio
- [ ] 未见 target nodes 测试
- [ ] 明确 source/target 节点 ID

### 消融
- [ ] w/o MoE
- [ ] w/o CoT
- [ ] w/o Diff-DoRA 或退回 DoRA/LoRA
- [ ] base model / untuned

**产出物**
- `src/eval/eval_fullshot.py`
- `src/eval/eval_fewshot.py`
- `src/eval/eval_zeroshot.py`
- `src/eval/eval_ablation.py`
- `notebooks/report_tables.ipynb`

---

## P9：复现报告与问题登记
- [ ] 汇总结果表
- [ ] 记录与论文一致/不一致处
- [ ] 单独记录所有实现近似
- [ ] 记录失败实验
- [ ] 记录开放问题

**产出物**
- `repro_report.md`
- `open_issues.md`

---

# 我建议你现在立刻开工的顺序

先不要全铺开，按下面这个顺序最稳：

## 第一步
先做：
- P0
- P1 的 ST-EVCDP
- P2
- P3

也就是：
**先把单专家 baseline 跑通。**

## 第二步
再做：
- P4
- P5

也就是：
**先证明 RAG + CoT 是否有帮助。**

## 第三步
再做：
- P6
- P7

也就是：
**把论文最核心的 MoE 和 Diff-DoRA 接上。**

## 第四步
最后做：
- P8
- P9

也就是：
**正式出表和写复现总结。**

---

# 两个你必须提前记住的风险

第一，论文里表 1 的电价差分示例有明显小错误：前面写当前 1.2、历史 0.9、差分应为 +0.3，但提示词示例里却写成 `Diff P: +0.0`。这说明你不能无脑照搬表格格式，要以方法逻辑为准。:contentReference[oaicite:8]{index=8}

第二，论文虽然写了 CBD 与 Res 的硬路由思想，但真正落到公开数据时，标签构造方式未必能一模一样恢复，所以你需要在 `paper_alignment.md` 里把“完全一致”和“工程近似”分开写清楚。论文关于硬路由专家、两类模式分离和用元数据标签直接激活对应专家的总体逻辑是明确的。:contentReference[oaicite:9]{index=9}

如果你要，我下一条可以直接继续给你：
**一个可直接建仓库的目录树 + 每个 Python 文件的职责说明 + 首批要写的 5 个脚本骨架。**
::contentReference[oaicite:10]{index=10}