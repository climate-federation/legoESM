# Preregistration: tracer-entry slow-forcing substitution, round 57

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 56 proves the literal Kmm tracer write is material but leaves subrow 8.3
outside the pointwise bar (`9758/9758`, maximum normalized column error
`1.9408056746e-5`). Round 53's few-ULP momentum-tail residual is not an unheld
counterexample: that arm replaced the split-explicit solver's slow forcing
with retained NEMO `zu_frc/zv_frc` and also held the after-level target.

Use the same current production step and row-8 streams as round 56, replacing
only `F_slow_u/F_slow_v` at the real barotropic solver call with the retained
`spg_dump_zu_frc/zv_frc`. Do not hold `mlf_baro_corr`, GM, FCT, thickness,
metrics, or tracer state. Re-score 8.3--8.10 in order with the unchanged bars.
Round 56 must reproduce as red, exactly one solver substitution must occur,
the production and instrument hooks must restore, and all previous controls
must fire. If all rows pass, classify the tracer-entry application locally
exact given the already-owned row-1.1 forcing and release Redi T/S. If 8.3
remains red, the transport output has an additional owner and the walk stops.
