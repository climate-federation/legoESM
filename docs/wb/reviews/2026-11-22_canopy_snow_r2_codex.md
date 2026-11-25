V1: A — refuse nonzero stores; preserve water without silently changing its location.  
V2: A — ≤1 preserves nonnegative transfers; ≤0.5 is unnecessary.  
V3: A — 1e-3 limits the departure from CLM; the boundary gradient still needs fixing.  
V4: A — an absent store can start empty without inventing mass.  
V5: A — reserve the snow contribution before limiting and partitioning soil demand.

1. **P2 — Empty-store gradient remains incorrect.** `packages/land/legoesm/land/canopy/interception.py:395` gates the quadratic ramp behind `snocan > 0`; line 407 therefore returns an autodiff derivative of **0 at an empty store**, although the right-hand derivative is **42.0772** with default settings and VAI=2.6. The claimed C1 continuation does not hold at initialization. `tests/land/unit/test_canopy_snow.py:176` explicitly exempts zero from its positive-gradient assertion. Evaluate the smooth continuation at zero and test its boundary derivative.

2. **P2 — Reserving snow corrupts the remaining soil/root partition.** `packages/land/legoesm/land/multilayer_land.py:1587` partitions the reduced soil demand using the original `LE_canopy`, which still includes the reserved snow contribution. For equal canopy and ground demands, with the canopy demand entirely snow-funded, reserving that demand leaves pure ground evaporation; the current partition nevertheless sends half through roots. Column mass can close while soil-layer withdrawals are wrong. Remove the reserved contribution from the canopy energy used to compute the partition, and test positive ground evaporation alongside canopy sublimation.

3. **P2 — Phase 2 can spend the snow reservation on liquid.** `packages/land/legoesm/land/multilayer_land.py:1615` caps total wet demand and then repeats the phase split, losing the source allocation established at line 1546. Reproduction with warm leaves, 0.10 kg/m² demand, 0.06 liquid, sufficient snow, and zero soil supply: the initial split reserves **0.04 snow**, but the second split draws **0.04 liquid and zero snow**. The snow exemption consequently funds a different store and suppresses supported sublimation. Preserve the reserved source allocation through the cap; add a warm mixed-store regression.

Focused reproductions confirmed these defects. The full test suite was stopped before completion; no repository files were edited.

**Verdict: NOT CLEAN.**
