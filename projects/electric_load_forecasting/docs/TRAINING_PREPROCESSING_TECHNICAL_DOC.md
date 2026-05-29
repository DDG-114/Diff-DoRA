# 训练与数据预处理技术文档

## 1. 文档范围

本文档面向当前仓库中的 `Diff-DoRA / LR-MoE` 训练主链路，重点说明：

- 原始数据放在哪里
- 原始数据如何被加载、清洗、归一化和切分
- sliding-window 样本、sample cache、retrieval cache、retrieval-result cache、tokenized cache 分别是什么
- 单专家训练、双专家训练、strict repro 训练分别如何消费这些数据
- 当前工作区里已经存在的数据规模与缓存规模

本文档描述的实现均以当前仓库 `src/` 与 `scripts/` 中的代码为准。

## 2. 当前工作区数据资产

当前环境里，`st_evcdp` 和 `urbanev` 两套数据都已经存在，且可以被当前代码正常加载。

### 2.1 ST-EVCDP

- 原始数据目录：`data/raw/st_evcdp/`
- 处理后缓存：`data/processed/st_evcdp.pkl`
- 归一化后 occupancy 形状：`(8640, 247)`
- 时间步数：`8640`
- 节点数：`247`
- 节点元数据形状：`(247, 10)`
- 邻接矩阵形状：`(247, 247)`
- 天气表形状：`(8640, 5)`
- 电价表形状：`(8640, 247)`
- 数据切分比例：`6:2:2`
- 切分结果：
  - train：`(5184, 247)`
  - val：`(1728, 247)`
  - test：`(1728, 247)`

在当前专家训练默认配置 `history_len=12`、`horizon=6`、`neighbor_k=7` 下：

- train sliding-window 样本数：`5167`
- 当 `include_test=True` 时，test sliding-window 样本数：`1711`
- 路由标签统计：
  - CBD 节点：`62`
  - Residential 节点：`185`
- 展开后的专家样本数：
  - `expert_0`：`320354` = `5167 x 62`
  - `expert_1`：`955895` = `5167 x 185`
- 专家内部再按 `85/15` 划分 train/val 后：
  - `expert_0`：train `272300`，val `48054`
  - `expert_1`：train `812510`，val `143385`

### 2.2 UrbanEV

- 原始数据目录：`data/raw/urbanev/`
- 处理后缓存：`data/processed/urbanev.pkl`
- 归一化后 occupancy 形状：`(4344, 275)`
- 时间步数：`4344`
- 节点数：`275`
- 节点元数据形状：`(275, 6)`
- 邻接矩阵形状：`(275, 275)`
- 天气表形状：`(4344, 18)`
- 电价表形状：`(4344, 275)`
- POI 表形状：`(712135, 3)`
- 数据切分比例：`8:1:1`
- 切分结果：
  - train：`(3475, 275)`
  - val：`(434, 275)`
  - test：`(435, 275)`

在当前专家训练默认配置 `history_len=12`、`horizon=6`、`neighbor_k=7` 下：

- train sliding-window 样本数：`3458`
- 由于缺少可直接用的 zone label，代码回退到 occupancy-based proxy label
- 路由标签统计：
  - CBD 节点：`138`
  - Residential 节点：`137`
- 展开后的专家样本数：
  - `expert_0`：`477204`
  - `expert_1`：`473746`
- 专家内部再按 `85/15` 划分 train/val 后：
  - `expert_0`：train `405623`，val `71581`
  - `expert_1`：train `402684`，val `71062`

## 3. 原始数据布局与加载逻辑

### 3.1 ST-EVCDP 加载器

代码入口：`src/data/load_st_evcdp.py`

期望的原始文件通常包括：

- `occupancy.csv`
- `time.csv`
- `nodes.csv` 或 `information.csv`
- `adjacency.csv` 或 `adjacency.npy`
- 可选 `price.csv`
- 可选 `weather.csv` 或 `SZweather20220619-20220718.xls`

主要预处理步骤如下：

- 读取 occupancy 时间序列，并尽量对齐时间索引
- 对 occupancy 中的缺失值做前向填充与后向填充
- 对完整 occupancy 矩阵执行 min-max 归一化
- 加载节点元数据；如果只有 CBD 标记，也会推导出 `zone_type`
- 加载邻接矩阵
- 将天气、电价按 occupancy 时间轴重新对齐，并做 `ffill/bfill`
- 将处理结果落盘到 `data/processed/st_evcdp.pkl`

这里有一个很重要的实现细节：

- 归一化发生在 `train/val/test` 切分之前
- 也就是说当前实现里的 `norm_min` 和 `norm_max` 来自整套数据，而不是只来自 train split

### 3.2 UrbanEV 加载器

代码入口：`src/data/load_urbanev.py`

期望的原始文件通常包括：

- `occupancy.csv`
- `adjacency.csv` 或 `adjacency.npy`
- `price.csv`，或 `e_price.csv + s_price.csv`
- `weather.csv`，或 `weather_airport.csv + weather_central.csv`
- `poi.csv`
- `inf.csv`

主要预处理步骤如下：

- 读取 occupancy，并构造成 `(T, N)` 的浮点矩阵
- 对 occupancy 缺失值做前向填充与后向填充
- 对完整 occupancy 矩阵执行 min-max 归一化
- 从 `inf.csv` 聚合站点元数据
- 合并多路天气数据，并构造统一的逻辑字段
- 将天气和电价重新对齐到 occupancy 时间轴
- 加载 POI 表
- 加载邻接矩阵
- 将处理结果落盘到 `data/processed/urbanev.pkl`

## 4. 数据切分

代码入口：`src/data/build_splits.py`

当前切分比例是写死在代码里的：

- `st_evcdp`：`0.6 / 0.2 / 0.2`
- `urbanev`：`0.8 / 0.1 / 0.1`

切分对象中保留的信息包括：

- `train`、`val`、`test`
- `timestamps_train`、`timestamps_val`、`timestamps_test`
- `norm_min`、`norm_max`
- `node_ids`、`node_meta`
- `adj`、`weather`、`price`、`poi`

因此后续训练阶段不只拿到 occupancy，还能拿到路由标签、RAG 检索、环境差分和静态上下文所需的全部辅助信息。

## 5. Sliding-Window 样本构造

代码入口：`src/data/build_samples.py`

每个样本都是一个字典，主要字段如下：

- `x_hist`：历史 occupancy，形状 `(history_len, N)`
- `time_feat`：时间特征，形状 `(history_len, 4)`
- `nbr_feat`：邻居平均 occupancy，形状 `(history_len, N)`
- `y`：未来目标，形状 `(horizon, N)`
- `t_start`：预测起点在当前 split 内的索引

当前专家训练主链路的默认参数是：

- `history_len=12`
- `neighbor_k=7`
- 常见 horizon：`3`、`6`、`9`、`12`

时间特征包含：

- `hour_sin`
- `hour_cos`
- `dow_sin`
- `dow_cos`

邻居特征的构造方式：

- 先从邻接矩阵中移除自环
- 如果设置了 `neighbor_k`，则每个节点只保留 top-k 邻居
- 对邻接矩阵做按行归一化
- 用归一化后的邻接矩阵聚合得到邻居平均 occupancy

对于长度为 `T_split` 的一个 split，在固定 horizon `h` 下，样本数公式是：

- `T_split - history_len - h + 1`

例如 ST-EVCDP 的 train split 在 `h=6` 时：

- `5184 - 12 - 6 + 1 = 5167`

## 6. 缓存层级

当前仓库不是只有一个“数据缓存”，而是分成了多层，每层的职责不同。

### 6.1 处理后数据缓存

路径：

- `data/processed/st_evcdp.pkl`
- `data/processed/urbanev.pkl`

用途：

- 存放归一化后的 occupancy、时间戳、元数据、邻接矩阵与对齐后的外部协变量

### 6.2 全局 retrieval cache

代码入口：`src/retrieval/build_cache.py`

默认路径：

- `data/retrieval_cache/{dataset}_h{horizon}.pkl`

用途：

- 在 train split 的 sliding-window 样本上预建一个全局 `KNNRetriever`
- 供训练或评测阶段快速复用

当前检索器实现位于 `src/retrieval/knn_retriever.py`，其特点是：

- `top_k` 默认是 `2`
- 当前特征版本是 `shapeaware_v1`
- 检索向量包含：
  - 均值 occupancy
  - occupancy 标准差
  - 最后一个时刻的 occupancy
  - 短时变化量
  - 中期变化量
  - 平均时间特征

为了避免时间泄漏，查询时会屏蔽掉：

- `t_start >= exclude_t_start` 的候选样本

### 6.3 Expert sample cache

代码入口：`src/data/sample_cache.py`

默认根目录：

- `${DIFFDORA_CACHE_ROOT}/sample_cache/`

当前代码在未显式设置环境变量时，默认会落到：

- `/root/autodl-tmp/Diff-DoRA-cache/sample_cache/`

命名规则：

- `train_experts_{dataset}_h{horizon}_hist{history_len}_nbr{neighbor_k}.pkl`
- 如果包含 test 样本，则在文件名后追加 `_with_test`

用途：

- 把 sliding-window 样本缓存下来
- 避免每次训练都重新跑 `build_samples(...)`

当前工作区里已经存在或已构建过的典型文件：

- `/root/autodl-tmp/Diff-DoRA-cache/sample_cache/train_experts_st_evcdp_h6_hist12_nbr7.pkl`
- `/root/autodl-tmp/Diff-DoRA-cache/sample_cache/train_experts_st_evcdp_h6_hist12_nbr7_with_test.pkl`
- `/root/autodl-tmp/Diff-DoRA-cache/sample_cache/train_experts_urbanev_h6_hist12_nbr7.pkl`

### 6.4 Retrieval-result cache

代码入口：`src/retrieval/retrieval_result_cache.py`

默认根目录：

- `${DIFFDORA_CACHE_ROOT}/retrieval_result_cache/`

用途：

- 为每个“已经展开到具体节点”的专家样本保存检索结果
- 每条缓存记录通常包含：
  - 被检索到的 pool 索引
  - 预先计算好的 diff features

这层缓存位于：

- 专家样本展开之后
- prompt/tokenization 物化之前

它的价值在于：

- 避免反复做 KNN 检索
- 避免反复计算环境差分

### 6.5 Tokenized cache

代码入口：

- `src/train/tokenized_cache.py`
- `src/train/build_tokenized_expert_cache.py`

默认根目录：

- `${DIFFDORA_CACHE_ROOT}/tokenized_cache/`

用途：

- 直接缓存训练用的 `input_ids`、`attention_mask`、`labels`
- 后续训练如果命中这层缓存，可以跳过：
  - 专家样本展开
  - expert-local retriever 构建
  - prompt 生成
  - tokenizer 编码

这一层缓存按训练变体区分：

- `full`
- `wo_diffdora`

这也是为什么“同一份原始数据、同一份 sample cache”最后仍然需要两套 tokenized cache：

- `full` 会在 prompt 里注入环境差分
- `wo_diffdora` 不会注入
- 因而最终文本不同，token 序列也不同，不能共用

## 7. 路由与专家样本展开

代码入口：

- `src/routing/build_labels.py`
- `src/routing/hard_router.py`

当前双专家语义是：

- `expert_0`：CBD / high-demand
- `expert_1`：Residential / low-demand

标签优先级如下：

- 优先使用 `node_meta.zone_type` 或类似元数据字段
- 如果缺少稳定元数据，就退化为 occupancy-based proxy label
- proxy 的规则是：节点平均 occupancy 高于中位数的归入 CBD，否则归入 Residential

样本展开逻辑如下：

- sample cache 存的仍然是“多节点窗口样本”
- 真正进入 expert 训练前，会把每个样本按节点展开，并补上 `node_idx`
- 每个节点只会被路由到一个 expert

因此，专家样本规模会远大于原始 sliding-window 样本规模。

以 ST-EVCDP、`h=6` 为例：

- sample cache 里的 train 样本只有 `5167`
- 展开到节点之后变成：
  - `expert_0`: `320354`
  - `expert_1`: `955895`

然后，`train_experts.py` 会在每个 expert 内部再次做：

- `85%` train
- `15%` val

所以你日志里看到的 `272300`，并不是新的原始样本数，而是：

- `expert_0` 展开后总样本 `320354`
- 再乘以 `0.85`
- 向下取整得到 `272300`

## 8. Prompt 与 Diff-DoRA 语义

相关代码：

- `src/train/train_single.py`
- `src/prompts/prompt_cot.py`
- `src/prompts/prompt_vanilla.py`
- `src/retrieval/diff_features.py`

当前主线实现采用的是 prompt-only Diff-DoRA 语义：

- `DoRA` 仍然是模型侧 adapter 机制
- `Diff-DoRA` 在当前仓库里表示“在 RAG prompt 中注入环境差分”
- 当前主线不再使用 controller MLP 或额外数值 side-channel

当前显式注入到 prompt 中的环境差分只有：

- `Diff T`：当前温度减去检索样本历史均值温度
- `Diff P`：当前电价减去检索样本历史均值电价

仍然会参与内部推理、但不作为环境差分字段显式展示的是：

- `Diff Occ`

`EVDataset` 当前支持的 prompt 风格包括：

- `auto`
- `cot`
- `direct_physical`
- `vanilla`

在 strict/full 专家训练路径中，launcher 固定使用：

- `use_rag=True`
- `prompt_style=cot`

## 9. 训练路径

### 9.1 单专家基线

启动脚本：

- `scripts/run_train_single.sh`

训练入口：

- `src/train/train_single.py`

典型命令：

```bash
source .venv/bin/activate
bash scripts/run_train_single.sh st_evcdp 6

source .venv/bin/activate
bash scripts/run_train_single.sh st_evcdp 6 dora
```

这条路径不做 expert routing，而是直接在单套数据上训练一个 adapter。

### 9.2 双专家训练

训练入口：

- `src/train/train_experts.py`

主流程可以概括为：

1. 加载处理后数据
2. 构造 `train/val/test` split
3. 加载或构造 sample cache
4. 生成 routing labels
5. 将 sliding-window 样本展开为 per-node expert 样本
6. 构建 expert-local retriever
7. 加载或生成 retrieval-result cache
8. 加载或生成 tokenized cache
9. 分别训练 `expert_0` 与 `expert_1`
10. 如果开启 quick eval，则在训练后做小规模评测

当前 paper-aligned 训练主路径中的关键超参是：

- 基座模型：`models/Qwen2.5-1.5B-Instruct`
- `epochs=2`
- `lr=2e-4`
- `max_length=2560`
- `lora_rank=32`
- `lora_alpha=32`
- gradient checkpointing 默认开启
- 训练精度为 fp16

输出目录通常是：

- `outputs/<run_name>/expert_0/adapter`
- `outputs/<run_name>/expert_1/adapter`

### 9.3 Strict reproduction 路径  

启动脚本：

- `scripts/run_train_eval_diffdora_strict.sh`  

这条路径会顺序训练两套变体：   

- `full`：显式开启 `--use_diff_dora`   
- `wo_diffdora`：不加 `--use_diff_dora`

当前脚本里的默认配置是：

- `epochs=2`
- `batch_size=16`
- `max_length=2560`
- `history_len=12`
- `neighbor_k=7`
- `use_dora`
- `use_rag`
- `prompt_style=cot`
- `max_samples_per_expert=0`
- `retrieval_bank_max_samples_per_expert=0`
- `eval_max_samples=0`
- `eval_nodes_per_expert=0`

这里两个“0 上限”非常关键：

- `max_samples_per_expert=0` 表示使用完整 expert 样本池
- `retrieval_bank_max_samples_per_expert=0` 表示使用完整 expert-local retrieval bank

### 9.4 双 GPU 并行全量训练

启动脚本：

- `scripts/run_train_full_repro_parallel.sh`

行为如下：

- GPU `0` 跑 `full`
- GPU `1` 跑 `wo_diffdora`
- 不自动进入评测
- 如果对应 tokenized cache 已经存在，就直接复用

### 9.5 预构建 tokenized cache

启动脚本：

- `scripts/prebuild_train_full_repro_tokenized_cache.sh`

典型命令：

```bash
source .venv/bin/activate
export DIFFDORA_CACHE_ROOT=/root/autodl-tmp/Diff-DoRA-cache
bash scripts/prebuild_train_full_repro_tokenized_cache.sh st_evcdp 6 0 0 6 5000 25000
```

参数含义依次为：

- dataset
- horizon
- sample cap
- retrieval-bank cap
- worker 数量
- materialize chunk size
- tokenized-cache shard size

脚本会做的事情包括：

- 构建或复用 sample cache
- 为 `full` 预构建 tokenized cache
- 为 `wo_diffdora` 预构建 tokenized cache
- 同时为两套变体生成 retrieval-result cache

为什么这里要处理两套变体：

- 它们共享同一份原始数据
- 它们共享同一份 sliding-window sample cache
- 但它们不共享最终 prompt 文本
- 所以不能共享最终 tokenized dataset

## 10. 目录与产物汇总

### 10.1 原始与处理后数据

- `data/raw/st_evcdp/`
- `data/raw/urbanev/`
- `data/processed/st_evcdp.pkl`
- `data/processed/urbanev.pkl`

### 10.2 检索与样本缓存

- `data/retrieval_cache/{dataset}_h{horizon}.pkl`
- `${DIFFDORA_CACHE_ROOT}/sample_cache/...`
- `${DIFFDORA_CACHE_ROOT}/retrieval_result_cache/...`
- `${DIFFDORA_CACHE_ROOT}/tokenized_cache/...`

### 10.3 训练输出

- `outputs/<run_name>/expert_0/adapter`
- `outputs/<run_name>/expert_1/adapter`
- `outputs/<run_name>/metrics.json`
- `outputs/<run_name>/logs/`

## 11. 推荐执行顺序

如果要跑一套完整的全量专家训练，推荐顺序是：

1. 准备好原始数据和基座模型
2. 激活 `.venv`
3. 可选：先构建全局 retrieval cache
4. 可选：先预构建 sample cache 和 tokenized cache
5. 跑 strict full/wo_diffdora 训练
6. 再跑 strict 评测

推荐命令串如下：

```bash
source .venv/bin/activate
export DIFFDORA_CACHE_ROOT=/root/autodl-tmp/Diff-DoRA-cache

python -m src.retrieval.build_cache \
  --datasets st_evcdp \
  --horizons 6 \
  --output_path data/retrieval_cache/st_evcdp_h6.pkl

bash scripts/prebuild_train_full_repro_tokenized_cache.sh st_evcdp 6 0 0 6 5000 25000

bash scripts/run_train_full_repro_parallel.sh st_evcdp 6 16
```

如果你想走“一键 strict train + eval”路径，也可以直接执行：

```bash
source .venv/bin/activate
bash scripts/run_train_eval_diffdora_strict.sh st_evcdp 6
```

## 12. 当前实现的注意事项

- 当前 loader 会在切分前先对整套 occupancy 做归一化。如果未来要切到“只用 train split 统计量归一化”的论文协议，需要显式改 loader。
- expert 样本量会因为 per-node 展开迅速膨胀，这也是 strict 全量训练非常吃时间和缓存空间的根本原因。
- `sample_cache` 存的是“路由前”的 sliding-window 样本，而不是已经按 expert 拆好的样本。
- `retrieval_result_cache` 和 `tokenized_cache` 都是按变体区分的，即使底层原始样本相同也不会共用。
- strict 启动脚本通过把 `eval_max_samples=0` 和 `eval_nodes_per_expert=0` 设为零，关闭了训练结束后的 quick eval。
- 当前主线实现里，Diff-DoRA 的语义是“DoRA + prompt 中的环境差分注入”，不是旧版 controller-based 模块。
