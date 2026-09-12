# GYRE NEMO-testcase L2 round 62 preregistration

Date frozen: 2026-09-12. Model tip: `b6027cebae17`. CPU, fp64/libm.

This file is frozen before running the round-62 `--step-gap 60` and
`--decompose 30` measurements against the decision-36 after member. The
protocol, masks, weights, regions, depth bands, and cadence are those already
committed in `nemo_testcase_l2_gyre_year_owners.py`; none is changed here.

## Month re-ranking predictions

1. The first non-bit tracer state will be after two completed model steps
   (the `kt=3` entry row). CONFIRM if `kt=1` and `kt=2` T are bit-exact and
   `kt=3` T is non-bit; REFUTE otherwise.
2. At day 30, temperature will be the leading dimensionless field, the
   `0--100 m` band will carry more than half of total squared temperature
   error, and the west third will be the largest disjoint longitude region.
   CONFIRM only if all three hold; REFUTE if any does not. The empirical
   north/south cut is reported separately because it overlaps the thirds.
3. The temperature RMS after ten completed steps will be below 10% of the
   day-30 `1.239756827e-2 K` RMS; at least 90% of the day-30 RMS magnitude will
   therefore be acquired later. REFUTE if the ten-step fraction is at least
   10%. This is a magnitude comparison, not an additive error budget.

## Next measurements (not predictions)

The kt=2-entry equal-input tracer walk will substitute recorded NEMO ZDF
operands one at a time. The first arm at the exact bar identifies the earliest
candidate statement; it is not selected in advance. The kt=1 closure walk is
likewise an ordered statement comparison, not an attribution guessed here.

