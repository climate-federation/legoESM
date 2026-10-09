# PREREG — SMT-RUNGS round 5: SMT-6 (BBL + geothermal) and SMT-6b (cold flank)

Frozen before any SMT-6 or SMT-6b NEMO record exists (the operator runs
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_smtrungs_round5_smt6/run.sh`
`--smoke` then `--run`). Scored next round against the cards
`VORTEX_SMT6_VEC-zps` and `VORTEX_SMT6B_VEC-zps`, each built from the files
its own NEMO run dumps. fp64, scalar libm. The before arm for every ladder
comparison is the admitted SMT-5 record (`smtrungs_rounds/round2/oracle_vortex_smt5/`).

## SMT-6 (gate closed by the stratification)

| # | prediction | confirms | refutes |
|---|---|---|---|
| P1 | SMT-6 geometry = SMT-5 geometry (same executables; the hunk touches no geometry switch): 18-row geometry gate GEOMETRY IDENTICAL; dumped target/resto byte-equal to SMT-5's | 18/18 and `cmp` silent | any row or byte differs |
| P2 | NEMO's BBL gate is closed on every bottom face at every kt = 1..10 step entry and at all 100 daily restarts: 0 open U, 0 open V of the 24 + 24 sloped faces; legoESM's `ahu_bbl`/`ahv_bbl` exactly 0 everywhere | 0 / 0 at every sample | any open face (then the closed-gate certification is void) |
| P3 | the geothermal tendency is nonzero on exactly the 3721 wet bottom cells and zero elsewhere; per-step increment `rn_Dt*qgh_trd0/e3t` spans 1.215e-7 K (500 m cells) to 1.199e-6 K (50.67 m cells) | 3721 cells, that range (±1e-3 relative for r3t) | any other cell set or a 1e-3 departure |
| P4 | offline replay of trabbc's statement on NEMO's own stage-3 operands (e3t_0, r3t(Kmm), rho0, rcp) equals the legoESM increment bit for bit | 0 unequal cells | any unequal cell (divisor association or constants wrong) |
| P5 | kt=1 rows as SMT-5: T and S bit-exact, U/V/SSH at SMT-5's kt=1 values | 5/5 AT-BAR | T over the bar at kt=1 (then the stage-3 accumulation order `((ldf+bbc)+bbl)+dmp` vs legoESM's `(rate+geo)+dmp` is the first non-bit statement) |
| P6 | first-over-bar stays SMT-4's inherited kt=2 statement; no earlier row | kt=2 | any kt=1 row over the bar |
| P7 | day-100 T rms vs NEMO within a factor 2 of SMT-5's 1.08e-5 K (both models carry the same 3.6e-6..3.6e-5 K damped geothermal equilibrium) | <= 2.2e-5 K | > 2.2e-5 K |

## SMT-6b (cold-flank anomaly, 1.7 K per level above jpkm1 on the bottom cell)

| # | prediction | confirms | refutes |
|---|---|---|---|
| Q1 | NEMO's dumped `votemper` record 1 equals the SMT-6b card's initial T on every cell (the card refuses otherwise): the anomaly formula is evaluated identically by NEMO's istate and the card | card builds from NEMO's dump | card refuses (`usr_def_istate identity`) |
| Q2 | at kt=1 NEMO's gate is open on exactly the 24 U + 24 V sloped faces (every sloped face; the closest SMT-6 margin is 1.68155 K < 1.7 K) | 24 / 24 | any other count |
| Q3 | the diffusive BBL trend is nonzero at kt=1 on exactly 64 bottom cells (36 shelf-side, 28 deep-side), T only: S = 35 everywhere makes every S flux exactly 0 | 64 T cells, 0 S cells | any other count, or a nonzero S trend |
| Q4 | gate operands bit-exact at kt=1 (bottom T/S at Kbb, S-EOS alpha/beta, zgdrho sign, `ahu_bbl`/`ahv_bbl`); the FIRST non-bit statement in the BBL chain, if any, is the bottom-cell divisor `e3t_3d*(1+r3t(Kmm))` and its Krhs association (trabbl lines 258-263; legoESM folds the trend into the stage-3 content as `h_stage*trend`) at ULP level, |dT| <= 1e-14 K | gate bit-exact; residual only on the 64 cells, ULP size | a gate operand non-bit, or a residual > 1e-14 K on those cells, or a residual off the 64 cells |
| Q5 | the gate stays open over the 100-day record on most sloped faces (damping restores toward the anomalous target, tau = 1 day) | >= 40 of 48 at day 100 | < 40 |

Q4 and Q5 are PLAUSIBLE (Q4 rests on reading, not a replay; Q5 assumes the
1-day restoring dominates the BBL's own erosion of the anomaly).
