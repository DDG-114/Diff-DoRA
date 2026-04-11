# Paper Alignment Notes

## Fully consistent with the paper

| Item | Paper | This implementation |
|------|-------|---------------------|
| History window | 12 steps | ✅ 12 |
| Forecast horizons | 3/6/9/12 | ✅ same |
| KNN top-k | 2 | ✅ 2 |
| CoT structure | gap→spatial→numeric | ✅ 3-stage template |
| Hard routing | physical label → expert selection | ✅ |

## Engineering approximations

| Item | Paper description | Our approximation | Reason |
|------|-------------------|-------------------|--------|
| Routing labels (ST-EVCDP) | CBD vs. Residential from metadata | Use zone/region column if present; else k-means on usage pattern | Metadata availability unclear |
| Routing labels (UrbanEV) | Not specified | Proxy: high-demand vs. low-demand percentile split | No explicit label |
| Diff-DoRA controller `g(·)` | Described as scalar gating on magnitude | 2-layer MLP with sigmoid | Implementation detail absent from paper |
| Table 1 price diff example | Claims diff = +0.0 but current=1.2, hist=0.9 → should be +0.3 | We compute `current − historical` correctly; ignore the table typo | Paper typo confirmed |

## Open questions
See `open_issues.md`.
