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
- [x] paper-aligned default hyper-parameters wired into training entry points
- [ ] final baseline numbers (pending full run)

## 5. RAG / CoT / MoE / Diff-DoRA
- [x] code skeleton complete
- [x] paper-style prompt metadata and static-context injection where dataset fields exist
- [x] parser compatibility for placeholder formats such as `[v1, ..., vH] = [...]`
- [ ] full experiments pending

## 6. Alignment with paper
See paper_alignment.md

## 7. Current best metrics
- Existing `outputs/` contains legacy sanity-check runs and partial expert metrics.
- Do not treat cached `outputs/` artifacts as the final paper reproduction tables unless they are regenerated with the current paper-aligned configuration.

## 8. Failure cases / notes
- UrbanEV station-level POI / area / road-length context is still limited by the available raw schema.
- Full-scope paper experiments still require running: full-shot, few-shot, zero-shot, and paper-defined ablations.
