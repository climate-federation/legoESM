# Preregistration: post-reconcile momentum tail, round 52

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 51 (`aaa492d7107ba00096d1d93c9240127856365dae71f56fbe65f41f611e1146c1`)
releases the remaining momentum tail. Existing day-180 streams score these
rows in NEMO order:

1. `finalize_lbc`, `stpmlf.F90:796-845`: post-`mlf_baro_corr` Naa U/V before
   and after the unconditional lateral-boundary link;
2. centered `mlf_baro_corr` Kmm rewrite, `stpmlf.F90:781-789`: the NOW U/V
   entering `dyn_atf_qco`, compared with the twin's production NOW operand;
3. U then V `dyn_atf_qco`, `dynatf_qco.F90:151-155`: plain velocity Robert-
   Asselin filter using restart Kbb, Kmm, and post-LBC Kaa.

Rows retain the POINTWISE `1e-15` normalized-RMS and maximum/RMS bars. The
oracle-Kmm arm must reproduce both filtered outputs at bar, while a stale
pre-`mlf_baro_corr` Kaa arm must fail. Identity must pass, point/roll plants
must fail, active populations and full-halo shapes must match, and all inputs,
sources, registration, and upstream receipt are hash-bound. The first U/V row
outside either bar is the ordered stop. Only all rows AT BAR releases the
tracer tail.
