# LR-MoE Repro

Reproduction of **"LR-MoE: Logic-Reasoning Mixture-of-Experts for EV Charging Demand Prediction"**.

## Quick-start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Recommended execution order

| Phase | Content |
|-------|---------|
| P0 | Initialization & alignment |
| P1 | Dataset pipeline (ST-EVCDP → UrbanEV) |
| P2 | Sample construction & metrics |
| P3 | Single-expert LoRA baseline |
| P4 | RAG + diff features |
| P5 | 3-stage CoT prompts |
| P6 | Hard-routing MoE |
| P7 | DoRA / Diff-DoRA |
| P8 | Full / few / zero-shot experiments |
| P9 | Reproduction report |

## Directory layout

```
configs/          YAML experiment configs
src/
  data/           Data loading & sample building
  eval/           Metrics & evaluation scripts
  prompts/        Prompt templates & parser
  models/         Model wrappers (PEFT, DoRA, Diff-DoRA)
  train/          Training entry-points
  retrieval/      KNN retriever & diff features
  routing/        Label construction & hard router
scripts/          Shell launchers & ablation runners
data/
  raw/            Original datasets
  processed/      Serialised .pkl files
  retrieval_cache/
outputs/          Checkpoints & result JSON files
notebooks/        EDA & reporting notebooks
```
