# Reproduction Report (Draft)

## 1. Objective
Reproduce LR-MoE for EV charging demand forecasting on ST-EVCDP and UrbanEV.

## 2. Environment
- Python: 3.10
- Torch: 2.11.0
- Transformers: 5.5.3
- PEFT: 0.18.1

## 3. Data Pipeline Status
- [x] ST-EVCDP loader
- [x] UrbanEV loader (schema adapter)
- [x] 6:2:2 / 8:1:1 splits
- [x] sample builder (history=12, horizons=3/6/9/12)

## 4. Baseline Status
- [x] Qwen2.5-1.5B-Instruct local load
- [x] LoRA training script
- [x] prompt + parser
- [ ] final baseline numbers (pending full run)

## 5. RAG / CoT / MoE / Diff-DoRA
- [x] code skeleton complete
- [ ] full experiments pending

## 6. Alignment with paper
See paper_alignment.md

## 7. Current best metrics
(To be filled after experiments)

## 8. Failure cases / notes
(To be filled)
