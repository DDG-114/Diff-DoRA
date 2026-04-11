# LR-MoE 启动指南

> 当前环境：Python 3.10 · PyTorch 2.11.0+cu130 · Transformers 5.5.3 · PEFT 0.18.1  
> GPU：NVIDIA GeForce RTX 4070 Ti SUPER (16.7 GB VRAM)  
> 虚拟环境：`.venv/`  
> 模型：`models/Qwen2.5-1.5B-Instruct/`（已下载到本地）

---

## 目录

1. [激活虚拟环境](#1-激活虚拟环境)
2. [准备数据](#2-准备数据)
3. [快速冒烟测试（合成数据）](#3-快速冒烟测试合成数据)
4. [P3：单专家 LoRA 训练](#4-p3单专家-lora-训练)
5. [评估：全步长](#5-评估全步长)
6. [P4–P5：RAG + CoT 推理演示](#6-p4p5rag--cot-推理演示)
7. [P6：双专家 MoE 训练](#7-p6双专家-moe-训练)
8. [P7：DoRA / Diff-DoRA 切换](#8-p7dora--diff-dora-切换)
9. [P8：消融实验脚本](#9-p8消融实验脚本)
10. [查看结果](#10-查看结果)
11. [Notebook 交互入口](#11-notebook-交互入口)
12. [常见错误](#12-常见错误)

---

## 1 激活虚拟环境

```bash
cd /home/kaga/diffdora
source .venv/bin/activate
```

之后所有命令均在该目录执行，无需再加 `./.venv/bin/python`。

---

## 2 准备数据

### 选项 A：使用真实数据（推荐用于正式实验）

| 数据集 | 放置路径 | 必须文件 |
|--------|---------|---------|
| ST-EVCDP | `data/raw/st_evcdp/` | `occupancy.csv`（T×N，列为节点ID，行索引为时间戳）<br>`nodes.csv`（含 `zone_type` 列：`cbd`/`residential`）<br>`adjacency.csv`（列：`src,dst,weight`）|
| UrbanEV  | `data/raw/urbanev/`  | `occupancy.csv`<br>`weather.csv`（含 `temperature` 列）<br>`price.csv`（含 `price` 列）|

### 选项 B：生成合成数据（用于测试代码流程）

```bash
python - << 'EOF'
import pandas as pd, numpy as np
from pathlib import Path

def make(raw, T, N, seed=42):
    raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=T, freq="h")
    occ = np.clip(rng.normal(0.5, 0.15, (T, N)), 0, 1)
    pd.DataFrame(occ, index=idx, columns=[f"node_{i}" for i in range(N)]).to_csv(raw/"occupancy.csv")
    pd.DataFrame({"node_id":[f"node_{i}" for i in range(N)],
                  "zone_type":["cbd" if i<N//2 else "residential" for i in range(N)],
                  "lat": rng.uniform(30,31,N), "lon": rng.uniform(120,121,N)
                 }).set_index("node_id").to_csv(raw/"nodes.csv")
    pd.DataFrame([(i,(i+1)%N,1.0) for i in range(N)], columns=["src","dst","weight"]).to_csv(raw/"adjacency.csv",index=False)
    print(f"Synthetic {raw.name}: T={T} N={N}")

make(Path("data/raw/st_evcdp"), T=2000, N=20)
make(Path("data/raw/urbanev"),  T=3000, N=30)
EOF
```

---

## 3 快速冒烟测试（合成数据）

验证整条管线（数据→样本→模型→推理→指标）能跑通，只用 50 个样本、1 个 epoch：

```bash
python -m src.train.train_single \
    --dataset    st_evcdp \
    --horizon    6 \
    --epochs     1 \
    --batch_size 2 \
    --max_samples 50 \
    --output_dir outputs/smoke_test
```

预期输出末尾：

```
Results saved to outputs/smoke_test/metrics.json
```

---

## 4 P3：单专家 LoRA 训练

### 正式单专家训练（推荐配置，RTX 4070 Ti SUPER 可运行）

```bash
python -m src.train.train_single \
    --dataset     st_evcdp \
    --horizon     6 \
    --epochs      3 \
    --batch_size  4 \
    --lr          1e-4 \
    --max_samples 2000 \
    --output_dir  outputs/single_lora_h6
```

| 参数 | 说明 |
|------|------|
| `--horizon` | 预测步长，可选 3 / 6 / 9 / 12 |
| `--max_samples` | 训练样本上限（2000 约跑 20 分钟）|
| `--node_idx` | 默认 0；指定单节点训练/评估 |
| `--use_dora` | 切换到 DoRA（见 P7）|

或直接用 shell 脚本（自动循环多个 horizon）：

```bash
bash scripts/run_train_single.sh st_evcdp 6
bash scripts/run_train_single.sh st_evcdp 12
```

产出：

```
outputs/single_lora_h6/
  adapter/          ← PEFT adapter 权重
  metrics.json      ← per-step RMSE/MAE
  checkpoints/      ← Trainer checkpoints
```

---

## 5 评估：全步长

```bash
python -m src.eval.eval_fullshot \
    --dataset     st_evcdp \
    --adapter_dir outputs/single_lora_h6/adapter \
    --output      outputs/single_lora_h6/fullshot.json \
    --max_eval    300
```

少样本评估（5%/10%/20%/40%/100% 训练量）：

```bash
python -m src.eval.eval_fewshot \
    --dataset    st_evcdp \
    --horizon    6 \
    --output_dir outputs/fewshot_h6
```

零样本（跨节点泛化）：

```bash
python -m src.eval.eval_zeroshot \
    --dataset    st_evcdp \
    --horizon    6 \
    --output_dir outputs/zeroshot_h6
```

---

## 6 P4–P5：RAG + CoT 推理演示

以下代码片段可在 Python REPL 或 notebook 中直接运行，演示 RAG 检索 + CoT prompt 生成：

```python
from src.data.load_st_evcdp import load_st_evcdp
from src.data.build_splits   import build_splits
from src.data.build_samples  import build_samples
from src.retrieval.knn_retriever import KNNRetriever
from src.retrieval.diff_features import compute_diff_features, format_diff_block
from src.prompts.prompt_cot      import build_cot_prompt

data   = load_st_evcdp()
splits = build_splits(data, "st_evcdp")
train_s = build_samples(splits["train"], splits["timestamps_train"], horizons=[6])[6]
test_s  = build_samples(splits["test"],  splits["timestamps_test"],  horizons=[6])[6]

retriever = KNNRetriever(train_s, top_k=2)
query     = test_s[0]
retrieved = retriever.query(query, exclude_t_start=query["t_start"])

diff = compute_diff_features(query, retrieved)
sys_msg, usr_msg = build_cot_prompt(query, retrieved, diff, node_idx=0, horizon=6)
print(usr_msg)
```

---

## 7 P6：双专家 MoE 训练

```bash
python -m src.train.train_experts \
    --dataset    st_evcdp \
    --horizon    6 \
    --epochs     3 \
    --batch_size 4 \
    --max_samples_per_expert 1000 \
    --output_dir outputs/moe_experts_h6
```

产出：

```
outputs/moe_experts_h6/
  expert_0/adapter/   ← CBD 专家
  expert_1/adapter/   ← Residential 专家
  metrics.json
```

MoE 推理：

```python
from src.data.load_st_evcdp   import load_st_evcdp
from src.data.build_splits    import build_splits
from src.routing.build_labels import build_routing_labels
from src.models.qwen_peft     import load_model_and_tokenizer
from src.models.lr_moe_wrapper import LRMoEWrapper
import numpy as np

data   = load_st_evcdp()
splits = build_splits(data, "st_evcdp")
labels = build_routing_labels(splits["train"], data["node_meta"])

base_model, tokenizer = load_model_and_tokenizer()
wrapper = LRMoEWrapper(
    base_model, tokenizer, labels,
    expert_paths=["outputs/moe_experts_h6/expert_0/adapter",
                  "outputs/moe_experts_h6/expert_1/adapter"]
)
```

---

## 8 P7：DoRA / Diff-DoRA 切换

### DoRA（只需加 `--use_dora`）

```bash
python -m src.train.train_single \
    --dataset    st_evcdp \
    --horizon    6 \
    --epochs     3 \
    --use_dora \
    --output_dir outputs/single_dora_h6
```

### Diff-DoRA（代码级）

```python
from src.models.qwen_peft      import load_model_and_tokenizer, get_lora_model
from src.models.diff_dora      import DiffDoRAModel, set_diff_context
import torch

base, tok = load_model_and_tokenizer()
dora      = get_lora_model(base, use_dora=True)
model     = DiffDoRAModel(dora, diff_input_dim=3)

diff_vec = torch.tensor([[0.1, -0.05, 0.2]])  # [diff_occ, diff_temp, diff_price]
set_diff_context(diff_vec)
outputs = model(**inputs)
```

消融对比脚本：

```bash
bash scripts/run_ablation_dora.sh st_evcdp 6
```

---

## 9 P8：消融实验脚本

```bash
# CoT 消融（vanilla vs RAG vs RAG+CoT）
bash scripts/run_ablation_cot.sh st_evcdp 6

# DoRA 消融（LoRA vs DoRA vs Diff-DoRA）
bash scripts/run_ablation_dora.sh st_evcdp 6

# 全量对比评估（需先提供各 adapter 路径）
python -m src.eval.eval_ablation \
    --dataset   st_evcdp \
    --horizon   6 \
    --vanilla   outputs/single_lora_h6/adapter \
    --rag       outputs/single_rag_h6/adapter \
    --rag_cot   outputs/single_rag_cot_h6/adapter \
    --output    outputs/ablation_cot.json
```

---

## 10 查看结果

所有指标文件都是 JSON 格式，可直接查看：

```bash
# 单次训练结果
cat outputs/single_lora_h6/metrics.json

# 用 Python 汇总所有结果
python - << 'EOF'
import json
from pathlib import Path

for p in sorted(Path("outputs").rglob("metrics.json")):
    d = json.loads(p.read_text())
    m = d.get("metrics", {}).get("overall") or d.get("results", {})
    print(f"{p.parent.name:35s}  {m}")
EOF
```

或打开 notebook 查看汇总表：

```bash
# 在已激活 .venv 的终端
jupyter notebook notebooks/report_tables.ipynb
```

---

## 11 Notebook 交互入口

```bash
source .venv/bin/activate
jupyter notebook notebooks/check_samples.ipynb
```

Notebook 涵盖：
- 环境验证
- 合成/真实数据加载 & 可视化
- 样本 shape 检查
- Prompt & Parser 演示
- LoRA 模型加载 & 冒烟训练
- 结果导出

---

## 12 常见错误

| 错误信息 | 原因 | 解决方法 |
|----------|------|---------|
| `FileNotFoundError: data/raw/st_evcdp/occupancy.csv` | 未放数据 | 参考[第 2 节](#2-准备数据)放置真实数据或生成合成数据 |
| `ImportError: Using SOCKS proxy, but 'socksio' not installed` | 代理环境缺依赖 | `.venv/bin/pip install socksio` |
| `CUDA out of memory` | batch_size 过大 | 降低 `--batch_size` 到 1，并增大 `--gradient_accumulation_steps` 到 8 |
| `parse_output returns None` | 模型输出格式不对 | 检查 `src/prompts/parser.py`，适当增大 `--max_new_tokens` |
| `ModuleNotFoundError: src` | 不在项目根目录 | 确保在 `/home/kaga/diffdora` 下运行，并已激活 `.venv` |

---

## 推荐执行顺序速查

```
第一步（跑通 baseline）
  ① 生成合成数据（或放置真实数据）
  ② python -m src.train.train_single --dataset st_evcdp --horizon 6 ...

第二步（RAG + CoT）
  ③ 参考第 6 节代码片段在 REPL 验证 RAG 逻辑
  ④ 微调 RAG-CoT 版本 adapter（修改 train_single 传入 CoT prompt）

第三步（MoE + Diff-DoRA）
  ⑤ python -m src.train.train_experts ...
  ⑥ python -m src.train.train_single --use_dora ...

第四步（正式出表）
  ⑦ python -m src.eval.eval_fullshot ...
  ⑧ jupyter notebook notebooks/report_tables.ipynb
```
