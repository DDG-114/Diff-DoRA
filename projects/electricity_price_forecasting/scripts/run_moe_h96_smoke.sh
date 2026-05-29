#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

.venv/bin/python -m src.train.train_experts \
  --dataset gs_market_2025 \
  --horizon 96 \
  --output_dir outputs/gs_market_2025_moe_h96_smoke \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum 2 \
  --lr 2e-4 \
  --history_len 96 \
  --context_history_len 96 \
  --neighbor_k 7 \
  --window_stride 96 \
  --use_dora \
  --use_rag \
  --use_diff_dora \
  --prompt_style cot \
  --max_length 4096 \
  --max_samples_per_expert 240 \
  --retrieval_bank_max_samples_per_expert 240 \
  --eval_max_samples 3 \
  --eval_nodes_per_expert 4 \
  --eval_steps 100000 \
  --retrieval_device auto

.venv/bin/python -m src.eval.eval_moe_routed \
  --dataset gs_market_2025 \
  --horizon 96 \
  --split test \
  --expert_0_dir outputs/gs_market_2025_moe_h96_smoke/expert_0/adapter \
  --expert_1_dir outputs/gs_market_2025_moe_h96_smoke/expert_1/adapter \
  --history_len 96 \
  --neighbor_k 7 \
  --window_stride 96 \
  --use_rag \
  --use_diff_dora \
  --max_eval 3 \
  --max_nodes 7 \
  --max_new_tokens 1536 \
  --infer_batch_size 1 \
  --sampling head \
  --output outputs/gs_market_2025_moe_h96_smoke/eval_head3.json
