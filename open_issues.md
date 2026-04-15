# Open Issues

## Implementation unknowns

1. **Routing label for ST-EVCDP** – The paper mentions CBD vs. Residential labels but does not describe the exact column name in the raw dataset. We use `zone_type` if available, else fall back to an occupancy-median proxy split.

2. **Diff-DoRA `g(·)` architecture** – Paper says "a lightweight controller modulates the magnitude component" without specifying the exact network depth or activation. We use a 2-layer MLP with sigmoid as a reasonable default.

3. **UrbanEV static attributes** – The local `poi.csv` is not keyed by station id, so POI context cannot yet be attached at the node level the same way ST-EVCDP capacity metadata can.

## Failed experiments

*(populate as experiments are run)*

## Results deviations

1. Several existing result files were produced with older quick-run defaults (`rank 8/16`, shorter `max_length`, small eval caps). Paper-aligned defaults now live in the training/config entry points, but the old artifacts remain in `outputs/`.
