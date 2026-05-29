# 启动指南

这个项目是 `LR-MoE: Logic-Reasoning Mixture-of-Experts for EV Charging Demand Prediction` 的复现仓库。

## 1. 环境准备

在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 服务器建议

如果你准备租服务器，推荐直接选：

- 系统镜像：`Ubuntu 22.04 LTS`
- CUDA：`12.1`
- Python：`3.10`
- 运行时类型：优先选带 `PyTorch + CUDA` 的官方/平台预装镜像；如果平台没有现成 PyTorch 镜像，就选纯 `Ubuntu 22.04` 后手动装依赖

更具体一点：

- 最稳妥的镜像组合：`Ubuntu 22.04 + CUDA 12.1 + Python 3.10`
- 如果云平台提供预装模板，优先选：`PyTorch 2.2/2.3 + CUDA 12.1 + Ubuntu 22.04`
- 不建议为这个仓库优先选 CPU-only 镜像

GPU 建议：

- 推荐：`1 x A100 80GB` 或 `1 x L40S 48GB`
- 可以尝试：`1 x RTX 4090 24GB`，但通常需要显式降低 `batch_size`、`max_length` 或训练样本数
- 如果你想尽量接近当前仓库里的论文默认值，优先租 `>=48GB` 显存

## 3. 从 git clone 开始的完整步骤

如果你拿到的是“干净的 Ubuntu 22.04”镜像，先执行：

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip build-essential
```

然后再执行：

```bash
git clone <你的仓库地址> diffdora
cd diffdora
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

安装完成后，建议先做 3 个检查：

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```

如果 `torch.cuda.is_available()` 返回 `False`，先不要开始训练，优先检查：

- 服务器是否真的挂载了 NVIDIA GPU
- 驱动是否正常
- 选的镜像是不是 CPU-only

## 4. 数据与模型准备

确保以下资源已经放到仓库默认位置：

- 基座模型：`models/Qwen2.5-1.5B-Instruct`
- ST-EVCDP 原始数据：`data/raw/st_evcdp/`
- UrbanEV 原始数据：`data/raw/urbanev/`

常见原始文件包括：

- `occupancy.csv`
- `nodes.csv` 或其他节点元数据
- `adjacency.csv` / `adjacency.npy`
- UrbanEV 额外文件：`weather.csv`、`price.csv`、`poi.csv`

## 5. 推荐启动顺序

### 5.1 单专家基线

LoRA：

```bash
source .venv/bin/activate
bash scripts/run_train_single.sh st_evcdp 6
```

DoRA：

```bash
source .venv/bin/activate
bash scripts/run_train_single.sh st_evcdp 6 dora
```

### 5.2 UrbanEV 论文对齐训练

```bash
source .venv/bin/activate
python -m src.train.train_urbanev \
  --horizon 6 \
  --output_dir outputs/urbanev_r32_h6
```

### 5.3 双专家 MoE / Diff-DoRA

当前仓库保留两条运行路径：

- `legacy baseline`：当前默认 quick 路径，保留为可回退工程基线
- `strict repro`：新增加的严格复现路径，用于全量专家训练和完整测试集评测

`legacy baseline` 的典型特征：

- `max_samples_per_expert=1000`
- 有 cache 时 expert-local retrieval bank 默认截断到 `800`
- quick ablation 常用 `max_eval=6`

这些设置适合快速比较，不应直接作为最终论文表格协议。

这里的 `--use_diff_dora` 现在表示：

- 保持 `DoRA` 作为唯一模型侧适配器
- 仅在 `RAG` 结构化 prompt 中注入环境差分（温度 / 电价）
- 不再使用数值 side-channel、controller MLP 或 `diff_controller.pt`

```bash
source .venv/bin/activate
python -m src.train.train_experts \
  --dataset st_evcdp \
  --horizon 6 \
  --output_dir outputs/st_evcdp_moe_diffdora_h6 \
  --use_dora \
  --use_diff_dora \
  --use_rag
```

如果你要保留当前 quick 协议并一键跑 `full/wo_diffdora`：

```bash
source .venv/bin/activate
bash scripts/run_train_eval_diffdora.sh st_evcdp 6
```

如果你要跑严格复现候选版本：

```bash
source .venv/bin/activate
bash scripts/run_train_eval_diffdora_strict.sh st_evcdp 6
```

当前这条 strict 入口默认就是：

- `batch_size=16`
- `max_samples_per_expert=0`，即完整训练集
- `retrieval_bank_max_samples_per_expert=0`，即完整 retrieval bank

如果你想手动覆盖，参数顺序是：

```bash
bash scripts/run_train_eval_diffdora_strict.sh <dataset> <horizon> <batch_size> <sample_cap> <retrieval_bank_cap>
```

如果你要只做 `full` / `wo_diffdora` 的全量训练，不自动进入评测，并且用两张 GPU 并行跑：

```bash
source .venv/bin/activate
bash scripts/run_train_full_repro_parallel.sh st_evcdp 6 16
```

这条入口固定为：

- `full` 放在 GPU 0
- `wo_diffdora` 放在 GPU 1
- `batch_size=16`
- `max_samples_per_expert=0`
- `retrieval_bank_max_samples_per_expert=0`
- `eval_steps=0`

## 6. 常用评测命令

### 6.1 保存后 adapter 验证

```bash
source .venv/bin/activate
python -m src.eval.validate_saved_adapter \
  --dataset st_evcdp \
  --horizon 6 \
  --split test \
  --adapter_dir outputs/single_lora_h6/adapter \
  --output outputs/validate_saved_adapter_outputs.json
```

### 6.2 路由 MoE 评测

```bash
source .venv/bin/activate
python -m src.eval.eval_moe_routed \
  --dataset st_evcdp \
  --horizon 6 \
  --expert_0_dir outputs/st_evcdp_moe_diffdora_h6/expert_0/adapter \
  --expert_1_dir outputs/st_evcdp_moe_diffdora_h6/expert_1/adapter \
  --use_rag \
  --use_diff_dora
```

### 6.3 Quick Nodes 冒烟测试

```bash
source .venv/bin/activate
python -m src.eval.eval_quick_nodes \
  --dataset st_evcdp \
  --horizon 6 \
  --expert_0_dir outputs/<run_name>/expert_0/adapter \
  --expert_1_dir outputs/<run_name>/expert_1/adapter \
  --use_rag \
  --use_diff_dora
```

### 6.4 论文版消融

```bash
source .venv/bin/activate
bash scripts/run_ablation_paper.sh st_evcdp 6
```

### 6.5 Strict 全测试集评测

在 `strict` 训练完成后，可单独跑完整测试集 + 全节点评测：

```bash
source .venv/bin/activate
bash scripts/run_eval_diffdora_strict.sh st_evcdp 6
```

### 6.6 Few-shot 少样本预测

Few-shot 现在按“训练时间前缀比例”运行：

- 测试集保持不变
- 训练只取 train split 最前面的一部分时间窗口
- 站点集合保持不变
- 默认训练样本上限为 `4000`

快捷入口：

```bash
source .venv/bin/activate
bash scripts/run_fewshot.sh st_evcdp 6
```

如果你想先单独把 strict few-shot / zero-shot 预处理产物准备好，再启动实验：

```bash
source .venv/bin/activate
bash scripts/run_preprocess_shots.sh st_evcdp 6 all
```

只预处理 few-shot：

```bash
source .venv/bin/activate
bash scripts/run_preprocess_shots.sh st_evcdp 6 fewshot
```

只预处理 zero-shot：

```bash
source .venv/bin/activate
bash scripts/run_preprocess_shots.sh st_evcdp 6 zeroshot
```

这条预处理入口会提前构建：

- `data/processed/st_evcdp_trainnorm_h6.pkl`
- `data/manifests/st_evcdp/fewshot_st_evcdp_h6_step6_seed42.json`
- `data/manifests/st_evcdp/zeroshot_st_evcdp_h6_step6_seed42.json`
- `data/retrieval_cache/shot/*.pkl`（few-shot）
- `data/sample_cache/shot/*.pkl`（zero-shot 的 masked train windows）

如果你要自定义比例、节点数或评测规模：

```bash
source .venv/bin/activate
python -m src.eval.eval_fewshot \
  --dataset st_evcdp \
  --horizon 6 \
  --fewshot_ratios 0.05,0.10,0.20 \
  --max_train_items 4000 \
  --use_dora \
  --use_diff_dora \
  --use_rag \
  --prompt_style cot
```

### 6.7 Zero-shot 零样本预测

Zero-shot 现在统一按“hard-routing MoE 训练 + target 节点零样本评测”运行：

- source 节点先按 `CBD / Residential` 路由到两个 expert
- 训练阶段只用 source 节点训练对应 expert
- train split 会对 target 节点负荷与邻接影响做屏蔽，避免训练泄漏
- 测试阶段仅在 target 节点上计算指标，并按同一路由规则激活 expert
- strict 版本默认使用 `train_only` 归一化、`window_stride=6`、`source_masked` retrieval query view
- 默认总训练样本上限为 `4000`

快捷入口：

```bash
source .venv/bin/activate
bash scripts/run_zeroshot.sh st_evcdp 6
```

如果你要自定义 source 比例、target 节点数或评测规模：

```bash
source .venv/bin/activate
python -m src.eval.eval_zeroshot \
  --strict_protocol \
  --dataset st_evcdp \
  --horizon 6 \
  --source_ratios 0.20,0.40,0.60,0.80 \
  --max_train_items 4000 \
  --use_dora \
  --use_diff_dora \
  --use_rag \
  --prompt_style cot
```

## 7. 论文默认超参

当前主训练入口默认已经对齐论文表 4：

- `epochs=2`
- `batch_size=8`
- `lr=2e-4`
- `max_length=2560`
- `lora_rank=32`
- `lora_alpha=32`
- `history_len=12`
- `neighbor_k=7`

如果只是做快速冒烟测试，可以在命令行显式覆盖这些参数。

## 8. 输出位置

训练和评测结果默认写到：

- `outputs/<run_name>/`
- `outputs/*.json`

其中 adapter 常见位置为：

- `outputs/<run_name>/adapter`
- `outputs/<run_name>/expert_0/adapter`
- `outputs/<run_name>/expert_1/adapter`

## 9. 一个最短可运行流程

如果你只是想先确认项目能跑起来，推荐按这个顺序：

```bash
source .venv/bin/activate
bash scripts/run_train_single.sh st_evcdp 6
python -m src.eval.validate_saved_adapter \
  --dataset st_evcdp \
  --horizon 6 \
  --split test \
  --adapter_dir outputs/single_lora_h6/adapter
```

跑通这两步后，再继续跑 MoE、Diff-DoRA 和论文版消融。

## 10. 参考兼容性说明

这个镜像建议基于两点：

- NVIDIA 官方 CUDA Linux 安装文档支持 `Ubuntu 22.04 LTS`
- PyTorch 官方历史安装页提供了 Linux `CUDA 12.1` 的 wheel 方案


## 11. 修复版 Expert Ablation 复现命令

下面这条命令用于跑当前论文对齐版的 expert ablation。

特点：

- `Diff-DoRA` 采用 prompt-only 环境差分注入
- 所有变体统一导出 best snapshot
- 旧的 controller 版输出仅保留作历史对照，不应与当前结果直接混用
- 评测协议固定为 `max_eval=60`、`12 CBD + 12 Residential`、`max_new_tokens=512`、`infer_batch_size=12`
- 这一节描述的仍是当前 quick/legacy 协议，不等于 strict full-test 复现协议

```bash
cd /root/Diff-DoRA
source .venv/bin/activate

python -m src.eval.eval_paper_ablation \
  --dataset st_evcdp \
  --horizon 6 \
  --use_rag \
  --retrieval_cache data/retrieval_cache/st_evcdp_h6.pkl \
  --max_eval 60 \
  --sampling random \
  --seed 42 \
  --node_sampling balanced_random \
  --max_nodes_per_domain 12 \
  --max_new_tokens 512 \
  --infer_batch_size 12 \
  --skip_base_model \
  --full_expert_0_dir outputs/ablation_expert_st_evcdp_h6_fixed/full/expert_0/adapter \
  --full_expert_1_dir outputs/ablation_expert_st_evcdp_h6_fixed/full/expert_1/adapter \
  --wo_cot_expert_0_dir outputs/ablation_expert_st_evcdp_h6_fixed/wo_cot/expert_0/adapter \
  --wo_cot_expert_1_dir outputs/ablation_expert_st_evcdp_h6_fixed/wo_cot/expert_1/adapter \
  --wo_dora_expert_0_dir outputs/ablation_expert_st_evcdp_h6_fixed/wo_dora/expert_0/adapter \
  --wo_dora_expert_1_dir outputs/ablation_expert_st_evcdp_h6_fixed/wo_dora/expert_1/adapter \
  --output outputs/ablation_expert_st_evcdp_h6_fixed/expert_ablation.json
```

一键重训 + 重评 + 重新作图则使用：

```bash
cd /root/Diff-DoRA
source .venv/bin/activate
bash scripts/run_ablation_expert_only.sh st_evcdp 6
```

## 12. 审计与双轨说明

- 当前 quick 路径的差异审计见 `repro_audit.md`
- `legacy baseline` 继续保留脚本和结果，便于回退和快速比对
- `strict repro` 入口单独放在：
  - `scripts/run_train_eval_diffdora_strict.sh`
  - `scripts/run_eval_diffdora_strict.sh`
