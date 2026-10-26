# Preregistration: GYRE stage-ZAD cancellation walk, round 45

Date: 2026-09-11. Frozen at legoESM `718399bf8943` before any round-45
trajectory, ablation, or candidate execution.

## Question, candidate, and compiled source boundary

Round 44 established that GYRE's production stage-3 ZAD is only
`2.1820967613189157e-4` / `3.6207743606473086e-4` of NEMO U/V although the
stage-program `ww` is at bar.  The rejected source-literal candidate handed
the velocity-form `ww` already constructed by the WS-RK3 stage program to the
one shared momentum RHS.  It changed kt=2 U/V from
`2.7478404751243857e-12` / `3.305560306813421e-12` to
`7.632783294297951e-17` / `9.71445146547012e-17`, but the frozen Rule-12
comparison rejected 55 rows; the largest worsening was
`GYRE-zco.kt8.before.v`, `3.55306143251799e-05` or
`1.60015661435125e11` row-scale ulps.

The candidate will be reconstructed literally from rejected commit
`5976cfba94d3`: `_mom_pert_ws` passes the stage program's velocity-form
`nemo_qco_wzv_operands` result into
`latlon_cgrid_ocean_baroclinic_tendencies`, replacing only that call's
internally reconstructed ZAD geometry.  NEMO constructs `ww` from
`uu/vv(Kmm)` before stage 2/3 momentum
(`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90:327-333`) and
then calls vector `dyn_adv(...,Kmm,Kmm,...,Krhs)` (`:466-472`), whose active
package calls `dyn_zad` and consumes this `ww`.

The first-departure probe will compare shipped and candidate state after each
WS-RK3 stage at kt=1 and then each before-state through kt=10.  The frozen
prediction is: no tracer or ssh movement at the injecting stage; U/V first
move at the stage-2 or stage-3 ZAD/RHS boundary of kt=1; candidate-vs-NEMO
first-over-bar remains kt=3.  A different field, step, or earlier boundary is
REFUTED and becomes the next boundary to walk before attribution.

## Registered suspect ranking and one-variable ablations

With the candidate present, rescore kt=1..10 after each ablation against both
NEMO and the shipped trajectory.  An arm is the cancelling partner only if
(a) kt=2 U/V stay at or below `2e-16` absolute max, and (b) all 55 registered
Rule-12 worsening rows are at or below their shipped errors.  Each planted
violation, commit mismatch, or dirty stamp must exit nonzero.

1. **Out-of-RHS/remnant explicit ZAD**: ablate any old internally reconstructed
   explicit ZAD contribution outside the candidate handoff.  This is ranked
   first because it is the only suspect whose enclosing explicit arm executes
   when `adaptive_implicit_vertadv=False`.  Prediction: if a distinct remnant
   exists, this arm uniquely meets (a,b); if the override already replaces the
   only contribution, the arm is inert and this suspect is REFUTED.
2. **Stage-folded vertical term**: ablate `_stage_vertical_up3` at the shared
   stage update.  Prediction: inert on GYRE because that helper returns a term
   only for `adaptive_implicit_vertadv=True`; any movement confirms an
   execution-path error but is not enough without (a,b).
3. **Implicit/post-program arm**: ablate the adaptive-implicit carrier into
   vertical mixing and the once-per-step momentum remnant separately.
   Prediction: both inert because GYRE resolves `ln_zad_Aimp=F`, and the latter
   also excludes `rk3_ws`; any movement confirms an execution-path error but
   is not enough without (a,b).

**Falsifier:** if none of these one-variable arms meets both (a,b), the stated
partner set is REFUTED.  Do not land a compensating change.  Retain the
source-literal handoff only if the remaining worsening can be walked forward
to a separately source-cited, given-input-exact downstream statement; else
stop with the candidate rejected.

## Landing and Rule-12 contract

The only eligible placement is the shared WS-RK3 stage program: explicit
stage ZAD consumes velocity-form stage `ww`; an implicit partition executes
only when the namelist-resolved card flag is true.  No new public knob or
legacy default is allowed.

- GYRE: given-input ZAD from the round-41 split record must be `0` unequal
  cells; full kt=1..10 before/after, every moved row registered with owner and
  boundary, no first-over-bar earlier.
- LOCK_EXCHANGE and OVERFLOW: round-33 ZDF matrix/operand records must retain
  their resolved implicit contributions, round-25 external-mode obligations
  must remain exact, and kt=1..10 Rule 12 must not regress.
- DINO: no edit/run; report the shared-statement risk to its independent
  leap-frog branch.
- ORCA2: report `UNMEASURED` with the exact missing record specification.

The new GYRE first-over-bar is reported mechanically.  If kt=2 U/V clear the
bar, the next owner is named only after a source-order/given-input
discriminator identifies its first non-bit statement.

## Decision table

| Status | Item | Action this round |
|---|---|---|
| ASKED | Does NEMO's literal slope association apply to every card using the slope routine or only NEMO-identity cards? | Preserve identity-only scope; do not decide. |
| UNASKED | None. | No new configuration or scope choice is authorized. |

## Pre-code self-review

- Reuse the committed trajectory, Rule-12 comparator, stage exposure, and
  private static test hooks; add no production-visible diagnostic switch.
- Keep candidate, ablations, controls, and reports commit-stamped and
  fail-closed; expected failure plants must exit nonzero.
- Touch neither NEMO sources nor records and run CPU/fp64 only.
- `DISCHARGED` remains reserved for zero unequal wet cells.
