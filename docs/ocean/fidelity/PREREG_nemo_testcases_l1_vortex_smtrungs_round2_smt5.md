# PREREG — SMT-RUNGS round 2: SMT-5 (T/S damping) record and ladder

Frozen before any SMT-5 NEMO record exists (the operator runs
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_smtrungs_round2_smt5/run.sh`
`--smoke` then `--run`). Scored next round against the SMT-5 card
(`VORTEX_SMT5_VEC-zps`, built from the files NEMO dumps). fp64, scalar libm.

| # | prediction | confirms | refutes |
|---|---|---|---|
| P1 | SMT-5 geometry = SMT-4 geometry: the 18-row geometry gate reports GEOMETRY IDENTICAL | 18/18 | any row differs (the damping hunk touches no geometry switch) |
| P2 | NEMO's dumped `votemper` record 1 equals the card's analytical initial T bit-for-bit on wet cells, `vosaline` = 35·tmask, all 12 records identical, `resto` = tmask/86400 exactly | 0 unequal cells | any unequal cell (SMT-4 proved T/S bit-exact at kt=0) |
| P3 | NEMO prints `fld_read: var votemper kt = 1 ( 0.0167 days)` with records b/a 0012/0001 at −15.5/+15.5 days (isecsbc = ndt05 = 1440 s) | that line | any other time or record pair (the card's model-time formula is wrong) |
| P4 | all five kt=1 rows stay AT-BAR (T, S bit-exact; U, V, SSH at SMT-4's kt=1 values) | 5/5 AT-BAR | any kt=1 row over the bar |
| P5 | first-over-bar is kt=2 on T/U/V/SSH (as SMT-4); S not over the bar before kt=4 | kt=2 / kt≥4 | first-over-bar earlier than SMT-4's |
| P6 | first non-bit statement in tra_dmp's operand chain: `pts(Kbb)` at kt=2 (SMT-4's inherited kt=2 T residual, 6.96e-10 K); `resto`, `zts_dta` and the time weight bit-exact at every kt=1..10 | Kbb operand first | `zts_dta`, `resto` or the weight non-bit at any kt (time interpolation or record selection is wrong) |
| P7 | day-100 T rms vs NEMO ≤ SMT-4's 2.55e-4 K: the restoring pulls both models to the same target | ≤ 2.55e-4 K | > 2.55e-4 K |

P5's reasoning (PLAUSIBLE): the damping contracts any T difference by
resto·dt = 2880/86400 = 1/30 per step, which cannot zero SMT-4's kt=2 residual.
P7 is PLAUSIBLE: a 1-day restoring dominates the 100-day trajectory.
