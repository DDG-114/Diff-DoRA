# Reproduction Specification

## Goal
Reproduce key results of "LR-MoE: Logic-Reasoning Mixture-of-Experts for EV Charging Demand Prediction".

## Environment (pinned)

| Package | Version |
|---------|---------|
| Python | 3.10 |
| PyTorch | 2.11.0 |
| transformers | 5.5.3 |
| peft | ≥0.11.0 |
| accelerate | 1.13.0 |
| sentencepiece | 0.2.1 |

## Model
- Base: `Qwen/Qwen2.5-1.5B-Instruct` (local: `models/Qwen2.5-1.5B-Instruct`)

## Datasets
- ST-EVCDP: 6:2:2 (train/val/test)
- UrbanEV : 8:1:1 (train/val/test)

## Key hyper-parameters (from paper)
| Param | Value |
|-------|-------|
| History window | 12 |
| Forecast horizons | 3, 6, 9, 12 |
| KNN top-k | 2 |
| LoRA rank | 32 |
| LoRA alpha | 32 |
| Batch size | 8 |
| Learning rate | 2e-4 |
| Epochs | 2 |
| Max length | 2560 |
| Neighbour top-k | 7 |

## Experiment log convention
Every run writes a JSON file to `outputs/<run_id>/metrics.json` with keys:
`run_id, dataset, horizon, timestamp, config, metrics`
