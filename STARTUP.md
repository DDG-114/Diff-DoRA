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
  --expert_0_dir outputs/moe_experts_h6/expert_0/adapter \
  --expert_1_dir outputs/moe_experts_h6/expert_1/adapter \
  --use_rag
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
