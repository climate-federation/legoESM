# Split-explicit momentum chain round 38: ZAD inheritance peel

Status: **FROZEN BEFORE MEASUREMENT**. This registration binds the official
round-37 result (`053bafb547e66aa09a2f0a357c8bce3da5ba808689e3654b726e77d7f0ee0f23`),
whose first failing exact term is vertical advection, and reuses only existing
day-180 dumps and the admitted round-37 twin capture. No NEMO build, writer,
GPU run, or MPI run is permitted or required.

## Source contract and prior exclusions

DINO executes the stock `cfgs/DINO/WORK/dynzad.F90` path. It initializes the
surface carry at lines 83--84, visits `jk=1..jpk-2` at line 86, assembles
`e1e2t*ww(jk+1)` and the two horizontal neighbour sums at lines 93--98,
multiplies by the Kmm velocity difference at lines 100--101, divides by live
Kmm U/V thickness and adds the carried/new interface terms at lines 104--110,
then applies only the carried top-interface term in the bottom cell at lines
113--119. The first `wzv` call at `stpmlf.F90:275` precedes `dyn_adv` at
`:309`; its `ww` is therefore the vertical velocity consumed by ZAD. The
second `wzv` is post-split-explicit and is a timing falsifier, not the
structural ZAD operand.

The following candidates are already closed and are not re-opened: Kmm/current
velocity rather than Kbb/before (`dyn_zad_ldf_walk.py`); vectorized gather
versus the source recurrence (`zad_recurrence_walk.py`); live-thickness divisor
association (`zad_vertical_metric_walk.py`); and the faithful bottom-face mask
(`zad_level29_onset_walk.py` plus the landed bottom-mask fix). The campaign's
earlier year-5 direct substitution found that NEMO call-1 `ww` removed 119x of
the median ZAD error, but that scratch result is not allowed to own this
day-180 row. Round 38 repeats the causal arm at the exact round-37 state.

## Frozen arms and bars

For both native U and V staggering, the scorer must first reproduce the
round-37 production ZAD capture exactly and reproduce that diagnostic by a
direct call to legoESM's production `nemo_advective` kernel exactly. It then
scores these source-ordered arms against unchanged `zad_dump_du/dv.bin`:

1. Kmm/current velocity + production diagnosed `w` (baseline);
2. Kbb/before velocity + production `w` (time-level falsifier);
3. Kmm/current velocity + NEMO `wzv_dump_ww_call1.bin` (registered owner arm);
4. Kmm/current velocity + NEMO `wzv_dump_ww_call2.bin` (wrong-time operand).

Every ZAD arm uses the existing accumulating normalized-RMS bar of `1e-12` on
the active 3-D U/V masks. The direct/captured production identities are exact
(`max_abs == 0`). The call-1 `ww` substitution owns the row at bar only if
both U and V become AT BAR. It owns a bounded inherited majority if, for both
components, `1 - E(call1)/E(production) >= 0.90`; any remaining residual stays
open and cannot be called local to `wzv`. Otherwise `wzv` is refuted as the
majority owner. The Kbb and call-2 arms are descriptive falsifiers and cannot
win ownership merely by being the best post-hoc arm.

The scorer also scores production `w` against call-1 `ww` on the active
T-column population, and fails closed on a changed round-37 receipt, capture
hash, dump shape/hash, state recipe, time-level availability, hook restoration,
nonfinite wet value, or tracked worktree change. Sign reversal, longitude
roll, wet-NaN, a two-bar perturbation, a vertically shifted `ww`, and call-1/
call-2 source confusion are planted controls and must all fire.

## Architectural stop if inheritance is confirmed

NEMO's QCO `wzv` recurrence at `sshwzv.F90:218-227` uses the live Kmm
thickness and `r1_Dt*e3t_0*(r3t(Kaa)-r3t(Kbb))`. legoESM currently diagnoses
the `w` consumed by ZAD before the same step's split-explicit solve has produced
the Kaa free surface. A faithful default therefore requires an integrator
reorder, a two-pass/fixed-point coupling, or an equivalent state-contract
change; a local arithmetic selector cannot supply the missing future operand.
If the registered substitution confirms inheritance, round 38 records this as
a too-large design stop and does not ship a fake local fix.
