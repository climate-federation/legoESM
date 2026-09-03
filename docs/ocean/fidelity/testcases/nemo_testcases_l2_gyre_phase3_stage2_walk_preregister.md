# NEMO testcase lane 2 GYRE — post-subcycle RK stage walk preregistration

Date: 2026-09-01

Session: `ea650f83-28b8-4c68-b0cc-809a9fd417de`

Parent drag landing: `fe8af4f79b7dfeb5245ab637ee28d60e9a6aecfc`

This freezes the first walk after the complete 50-substep external-mode
program cleared the `1e-15` bar.  The accepted starting state has every
barotropic substep and kt=2 SSH AT-BAR.  Stage 1 U/V are AT-BAR.  The first
remaining momentum rows are stage 2 U/V at
`1.0381851039053744e-7` and `2.4701793455947666e-7` absolute; their owner is
UNMEASURED.  Stage-internal T/S remain UNMEASURED.

## Executing stage-2 program

The resolved GYRE run has `ln_tile=F` (`ocean.output:247`) and uses the HYB
barotropic update.  In `stprk3_stg.F90`, stage 2 sets `rDt=rn_Dt/2`
(`:178-182`), puts SSH at the N/N+1 midpoint but keeps `uu_b/vv_b(Kaa)` at the
full N+1 external-mode value (`:208-218`), and builds the Kmm transport at
`:261-279`.  The live vector-invariant arm diagnoses W from Kmm velocity
(`:290-298`), then calls HPG, vorticity, and advection in that order
(`:335-351`).  Stage 2 advances from Kbb with that RHS (`:377-386`).  Finally,
`stprk3_stg.F90:450-462` computes

`zub = uu_b(Kaa) - sum(e3u_0*uu_raw(Kaa))*r1_hu_0`

and adds the same column-uniform correction to every wet level (and likewise
for V).  The existing stage dump is written only after this correction at
`stprk3.F90:221-225,357-374`; it cannot distinguish raw integration from the
barotropic imposition.

The legoESM implementation follows the same collapsed identity at
`ocean_model_latlon_cgrid.py:4589-4669`: corrected stage 1 feeds the next
tendency, stage 2 restarts from the original state with `dt/2`, and the final
external-mode mean is imposed at every stage.  The pre-implementation search
found the existing corrected-stage hook, transport reconciliation, fixed-depth
mean utility, and native stage reader.  This round extends those private
WRITE-only/test-only seams; it does not introduce a public scheme switch.

## Frozen operand order

A separate fail-closed fp64 record will capture, for stage 2 only:

1. Kbb U/V and the already accepted corrected stage-1 Kmm U/V;
2. the existing source-built Kmm transports;
3. Krhs U/V after HPG + live ENE vorticity + vector-invariant advection;
4. raw Kaa U/V after `Kbb + (rn_Dt/2)*Krhs`;
5. external `uu_b/vv_b(Kaa)`, the raw fixed-depth mean, and `zub/zvb`;
6. corrected Kaa U/V after the column-uniform addition.

The candidate raw stage is exposed only after an otherwise ordinary step has
completed, so the diagnostic cannot feed later stages.  The candidate RHS is
reconstructed from the source identity `(raw-Kbb)/(rn_Dt/2)` before scoring.
Every row records absolute error, oracle magnitude, relative error, mask,
dtype, and exact-bar status before an owner label.  Existing oracle artifacts,
including the drag stream, must retain registered hashes bit-for-bit.

If the RHS and raw stage are AT-BAR while `zub/zvb` are not, a one-variable
oracle correction arm may replace only the stage-2 correction operand.  If the
RHS is first over bar, no composition owner is assigned: the next preregistered
subwalk is HPG -> vorticity -> advection, with each NEMO term dumped separately
before a causal arm.  If all stage-2 momentum rows clear, the walk advances to
stage-2 T/S and then stage 3 in source order.

## Controls, labels, and stopping rule

The new reader must reject corrupt magic/header/payload.  A planted `+1` in a
wet stage-2 RHS operand must be the first registered DEBT with magnitude at
least `1.0`.  A post-correction-minus-raw recurrence test must fail when its
literal coefficient is changed.  Unmodified step-entry, substep, stage, RHS,
and barotropic-frame hashes are bit-identity controls.

Scaling precedes every owner label.  An arm is `CONFIRMED_OWNER` only if its
movement matches the faithful residual and every downstream stage-2 U/V row
clears the exact bar.  Movement below one tenth of the residual is
`NEAR_NULL_NO_DISCRIMINATING_POWER`, not exoneration.  Partial improvement is
`CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER`.  After any supported landing, the CPU/fp64
gate restarts at kt=2 and walks through kt=10.  The loop stops only when all
kt=2 fields clear the bar or every remaining registered boundary has an honest
AT-BAR, DEBT, or UNMEASURED disposition.
