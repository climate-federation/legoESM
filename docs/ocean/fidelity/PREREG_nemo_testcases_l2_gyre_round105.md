# Preregistration: NEMO-testcases L2 GYRE round 105 shear free-surface routing split

Date: 2026-09-17. Frozen at incoming lane tip
`c9f6742f37d8ad566fdc2fdabb8dfb5a6efee05a`, before the round-105 baseline
or candidate is measured. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round105/`.

## The first non-bit statement and the source-required routing

Round 104 established that the first non-bit statement in the owned
`zdf_sh2` walk is the face-metric divisor in the record build's compiled
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:102` and its v-face
twin at `:107`. Its production output is non-bit in all 17,400 wet cells,
max `3.811744924985501e-14`. The arithmetic itself is already exact when fed
the correct free-surface time level.

The binding compiled call chain is:

1. `stprk3.f90:167-168` contains a commented `zdf_phy(kstp,Nbb,Nnn,Nrhs)`
   call followed by the executed `zdf_phy(kstp,Nbb,Nbb,Nrhs)` call.
2. `zdfphy.f90:317-320` forwards those two formals to
   `zdf_sh2(Kbb,Kmm,avm_k,sh2)`.
3. `zdfsh2.f90:99-108` consumes both `Kmm` and `Kbb` in the u/v face
   divisors. On this RK3 card both are therefore the step-entry free-surface
   slot.
4. That does NOT change the tracer solver's time level. Stage 3 declares
   `Kmm=N+1/2` at `stprk3_stg.f90:240` and writes the half-step free-surface
   ratio at `:256`; `trazdf.f90:461-477` consumes `r3t(...,Kmm)` in its
   implicit vertical-mixing divisor.

legoESM currently constructs one RK3 half-step free-surface field and passes
it both to the tracer solver, where it is correct, and into the shear metric,
where it is not NEMO's executed program. The candidate splits only this
routing: the shared shear helper gets an explicit optional shear-now field;
GYRE's RK3 final closure passes step-entry ssh to that input, while the
existing `eta_now` input remains the half-step field for `tra_zdf`. The
optional input defaults to the existing `eta_now`, so non-RK3 callers retain
their existing association. This is no new card or library configuration.

## Immutable comparison arm

The Rule-12 before arm is
`phase3/merge_main_2026-09-17/after/{ladder.json,day_gap.json}`. Its headline
rows are kt2 U/V `2.7377110452773967e-12` /
`3.2849219221489645e-12`, kt3 T/S `1.627497246303733e-4` /
`6.327735185607253e-6`, and day-30 T RMS
`1.2397011296352737e-2` K. It is the recorded baseline, never a scratch
toggle.

## Frozen predictions and falsifiers

**P1 — unmodified baseline and direct operand capture.** Before changing the
routing, the round-104 production walk will reproduce
`production_step_vs_recorded` as 17,400 of 20,416 unequal cells, max
`3.811744924985501e-14`, under the production JIT. A diagnostic-only trace
extension will capture the face metrics actually consumed by that same
production step; at least one live u/v face metric will differ from NEMO's
recorded-step-entry reconstruction. Running the same production closure with
JIT disabled will name any eager/JIT difference rather than silently treating
an eager row as production. REFUTED if the p_sh2 count or maximum differs, or
if the captured pre-split face metrics are bit-identical.

**P2 — source-required candidate, eager and production JIT.** With the split,
the captured GYRE shear face metrics and the production `p_sh2` are each BIT
against NEMO given NEMO's recorded stage entry: zero unequal cells and zero
max absolute difference in both the production-step JIT and production-closure
eager arms. The existing all-recorded replay remains BIT. REFUTED by one
unequal cell in either execution arm. An eager-only match is not a candidate.

**P3 — plant.** The existing `stage-shear-operand-ulp` plant advances one wet
cell of the exact p_sh2 reference by one ULP on the production-JIT path. It
must change the target row from 0 to exactly 1, print `STATUS PLANT-FIRED`,
and exit nonzero. REFUTED if it does not reach a wet cell, the row does not
move by exactly one cell, it prints PASS, or it exits zero.

**P4 — GYRE certified ladder.** Because the fix is in closure physics after
the kt2 prognostic velocity update, kt2 U/V are predicted to remain exactly
the before-arm values above and kt2 T/S remain AT-BAR. The first-over-bar row
must not move earlier; no AT-BAR row may leave the bar; every moved row will
be registered by the 954-row gate. The candidate is expected to reduce kt3 T
and S errors because it removes the measured magnitude owner feeding the next
closure, and not to increase either. REFUTED, and the candidate is HELD, if
the gate fails, kt2 T/S leave the bar, first-over-bar moves earlier, or either
kt3 tracer error increases.

**P5 — days 1-30.** The candidate is predicted to reduce the day-30 T RMS
below `1.2397011296352737e-2` K while preserving every Rule-12 prerequisite.
All days 1 through 30 are scored against the recorded year-owner arm.
REFUTED, and the candidate is HELD, if day-30 T RMS increases or the day-gap
gate refuses the run. A day-30 improvement cannot override a ladder failure.

**P6 — DINO shared-statement control.** DINO's compiled leapfrog program
passes distinct `Nbb,Nnn` levels at `DINO/BLD/ppsrc/nemo/stpmlf.f90:187-193`,
and `DINO/BLD/ppsrc/nemo/zdfphy.f90:316-319` forwards them to `zdf_sh2`.
Its tracer solver separately consumes NOW (`Kmm`) at
`DINO/BLD/ppsrc/nemo/trazdf.f90:219-235`. The new optional shear field is
not supplied by DINO's unchanged call, so the shear divisor continues to see
NOW through `eta_now` and BEFORE through carried `eta_before`. The DINO unit
suite and the cheapest committed DINO gate that executes this shear path are
predicted bit-identical before/after. REFUTED if any DINO output bit moves;
that movement is a registered result and the candidate is HELD pending a
source-level explanation.

**P7 — other tanks.** LOCK_EXCHANGE and OVERFLOW are predicted not to execute
the TKE shear statement under their instantiated cards. ORCA2 remains
UNMEASURED-WITH-SPEC: exercise the same optional/default shear input under its
card and compare a one-step state bitwise before relying on this routing
there. REFUTED for either tank if its instantiated configuration selects the
statement. No unmeasured tank result is called a pass.

**P8 — landing rule.** The candidate LANDS only if P2 and P3 confirm, the
Rule-12 ladder passes, the day arm is admissible, DINO is unchanged or its
movement is explicitly source-exact and accepted by the same gates, citation
mapping and its shifted-citation plant pass, focused tests pass apart from
named pre-existing failures, and the separate read-only Codex review does not
say `DO NOT SHIP`. Otherwise the physics is reverted or preserved as a held
manifest and this round is HELD. Local source exactness never overrides the
trajectory gate.

## Measurement order and controls

1. Commit this preregistration.
2. Add only the direct consumed-face-metric trace and gate rows, then run the
   unchanged routing under production JIT and eager execution. This is the
   immutable local before arm.
3. Run the DINO unit/gate controls before the routing change.
4. Add the routing split and its regression tests; rerun both local arms and
   the nonzero ULP plant.
5. Run GYRE kt=1..10 and days 1-30 against the immutable comparison arm, then
   the DINO after arm and the tank execution audit.
6. Run the separate review, citation gate plus shifted-citation plant, focused
   tests, and the full `tests/ocean/fidelity` plus `tests/ocean/unit` trees.

## Scope limits

CPU only, fp64, `JAX_ENABLE_X64=1`. No NEMO source is modified and neither
`makenemo` nor `mpirun` is run. No configuration field, default, scheme
selection, threshold, carried state, restart schema, year harness,
reconciliation gate, freshwater pair, #1484 guard, or held manifest changes.
No downstream TKE statement is changed. If the split cannot be expressed by
an explicit call input without a new configuration choice, work stops with
`DECISION_NEEDED` rather than inventing a default.
