# Open Issues

## Implementation unknowns

1. **Routing label for ST-EVCDP** – The paper mentions CBD vs. Residential labels but does not describe the exact column name in the raw dataset. We will use `zone_type` if available, else fall back to unsupervised clustering.

2. **Diff-DoRA `g(·)` architecture** – Paper says "a lightweight controller modulates the magnitude component" without specifying the exact network depth or activation. We use a 2-layer MLP with sigmoid as a reasonable default.

3. **UrbanEV price column** – Need to confirm the exact column name and unit after downloading the dataset.

## Failed experiments

*(populate as experiments are run)*

## Results deviations

*(populate after P8)*
