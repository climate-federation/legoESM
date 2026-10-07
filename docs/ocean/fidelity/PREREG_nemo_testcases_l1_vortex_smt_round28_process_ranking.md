# Preregistration — VORTEX_SMT round 28 (lane round 240): day-100 process ranking

Frozen before constructing or running a process-family arm. Base:
`ca822f33d` (round 239 / VORTEX_SMT round 27). Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round240/`.

## Question and instrument contract

The certified SMT-4 endpoint is day-100 wet three-dimensional temperature RMS
`2.552708052055443e-04 K` against the admitted NEMO run. This round ranks the
six process families named by operator note CI: HPG association, lateral
momentum diffusion, tracer LDF, implicit bottom drag, background vertical
mixing plus EVD, and the external-mode/barotropic replacement.

This is an **intervention-leverage ranking**, not yet an ownership claim. Each
arm changes one complete family in legoESM while retaining the same SMT-4 NEMO
trajectory as the target. The score is
`baseline T_rms - arm T_rms` at day 100; positive values remove part of the
gap and negative values worsen it. A family becomes an owner only after the
first non-bit statement inside it is named against NEMO's compiled operands.
The receipt must not relabel a large ablation response as source-exactness.

The committed probe extends the existing round-210 scorer and uses its field
reader, wet masks, fp64/libm policy, CPU production step, and day-100 metric.
It constructs these registered arms and no others:

1. `hpg_source_order`: the existing private source-order arm, which evaluates
   NEMO's HPG -> VOR -> KEG -> ZAD association without changing arithmetic.
2. `momentum_ldf_off`: set the SMT-4 level-Laplacian coefficient `A_h` to zero.
3. `tracer_ldf_off`: remove the SMT-4 GM/Redi tracer-LDF block.
4. `bottom_drag_off`: set the linear drag coefficient `rn_Cd0` equivalent to
   zero while keeping the compiled drag path and all other drag fields.
5. `vertical_mixing_evd_off`: set background `A_v`, background `K_v`, EVD
   tracer replacement, and EVD momentum replacement to zero as one registered
   family ablation.
6. `barotropic_replacement_off`: use the existing private hook that omits the
   per-stage external-mode replacement while leaving the external solve itself
   running.

The probe prints a side-by-side resolved-config diff for every arm, refuses an
unregistered changed field, stamps the clean worktree, and records days
1/2/5/10/20/30/60/100 for T/u/v/ssh. A registry-removal plant and a numerical
effect plant must both print `STATUS PLANT-FIRED` and exit nonzero.

## Frozen predictions and falsifiers

* **R28-P1 — baseline calibration.** A fresh baseline run reproduces
  `2.552708052055443e-04 K` at day 100 exactly and reproduces the certified
  kt=1..10 registry. Any mismatch refuses every family score.
* **R28-P2 — HPG control.** `hpg_source_order` changes day-100 T RMS by less
  than `2e-10 K`, reproducing round 239's floor-level result. A larger movement
  refutes the probe or the prior controlled arm and stops the ranking.
* **R28-P3 — magnitude prediction.** `tracer_ldf_off` has the largest positive
  day-100 removal and removes at least 50% of the baseline T RMS. Either a
  different winner or a smaller removal refutes this prediction; the measured
  ranking is retained.
* **R28-P4 — bounded execution.** Every arm reaches day 100 with finite scored
  fields. A non-finite arm is registered as `UNBOUNDED`, not omitted or ranked
  by a partial run.
* **R28-P5 — statement walk.** After ranking, walk only the winning positive-
  removal family in NEMO's compiled order under production JIT. Existing
  admitted records and instruments are reused. If they lack the winning
  family's direct operands at the relevant state, write one fail-closed,
  self-describing acquisition and stop `STOPPED_FOR_RECORD`; do not infer the
  statement. If no arm has positive removal, status is `HELD` and no family is
  called an owner.
* **R28-P6 — disposition.** This measurement round lands no physics unless a
  single already-recorded statement closes locally and passes the full shared-
  card trajectory gates within the round. Otherwise production is unchanged.

## Oracle source read before measurement

The exact compiled SMT-4 stage program calls HPG, VOR, and vector advection in
that order at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:328-344`, then
lateral momentum diffusion and vertical momentum diffusion at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:398-417`. It
calls tracer LDF and implicit tracer ZDF at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:542-554`.
The step-entry program forms drag before the split-explicit solve at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:194-267`.
These citations establish execution and order; the ablation magnitudes remain
measurements, not source claims.

## No hidden choices

All six arms and the 100-day window are ordered explicitly by note CI. No
production card, default, coefficient, timestep, run length, threshold,
stabiliser, carried state, NEMO source, or accepted trajectory changes. The
family ablations exist only in the diagnostic probe. Any additional scientific
choice stops with `DECISION_NEEDED`.

## Pre-measurement protocol amendment after a fail-closed refusal

The first execution reached the `vertical_mixing_evd_off` constructor only
after baseline, HPG, momentum-LDF, tracer-LDF, and bottom-drag arms had run.
It refused before taking a vertical-arm step: setting `K_conv=K_bg=0` under
`evd_composition="nemo_replace"` makes the fired-set readback ambiguous. That
attempt and its traceback remain evidence; none of its partial arm results is
the final ranking. Before re-running the ranking from its committed baseline,
the arm is expressed by the existing resolved family selector
`physics.convection.scheme="none"`, together with `A_v=K_v=0`. This disables
the same preregistered vertical-mixing-plus-EVD family without constructing an
invalid replacement configuration. R28-P1 through R28-P6, including the
frozen tracer-LDF prediction and all falsifiers, are unchanged. A regression
test proves that the rejected zero-replacement spelling still fails closed.

The corrected full execution then reached the last arm,
`barotropic_replacement_off`, where the production geometry guard reported a
non-positive `e3w_int`. This is the exact `UNBOUNDED` outcome already specified
by R28-P4, so it refutes R28-P4 but does not change the protocol. The first
emitter stopped instead of serialising that outcome. Before composing the
final report, its fail-closed path was corrected to retain the error and rank
the arm last. The six complete arms are reused only with the explicit clean
measurement commit `19a38791ee878118c4952ce4d0cb922fc58130c3`; their
snapshot digests and measurement commit
are written into the JSON, and the failed arm is rerun under the corrected
emitter. No arm definition or frozen prediction changed in this correction.
