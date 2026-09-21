# Preregistration — NEMO testcase L2 GYRE round 141

Date: 2026-09-21

Incoming lane tip: `245e5cfe98b50fa548daa36ea740e48e10d75872`

This document is frozen before reading any scientific value from the admitted
Round-140 developed three-dimensional momentum record and before running a
developed-state production comparison. Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round141/`.

Round 140 established, from NEMO's completed step-1080 state, that the first
non-bit external-forcing boundary is the incoming U slow forcing: 580 / 580
wet faces differ by at most `4.2854247978022983e-13 m s-2`; incoming V follows
at 570 / 570 and `4.433308633699682e-13 m s-2`. The operator reports that the
preregistered Round-140 three-dimensional record is now admitted at
`phase3/round140/oracle_developed_rhs`. No payload value from that record has
been read for this preregistration.

## P0 — fail-closed record readmission

The executing compiled target accumulates HPG, LDF, VOR, KEG, and ZAD into
`Krhs` at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:142-178`, records the
completed three-dimensional RHS at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:180-193`, depth-averages
it at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:210-228`, and applies
drag and wind at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:236-260`.

Before any comparison, the existing Round-83 gate must admit the exact
`1,486,548`-byte record, its producer/stamp, the closed ordered manifest,
literal depth and wind replays, all inherited restart/process/external/QCO/
Round-139 byte identities, `STOP 0`, `RUN_DONE`, and every header,
truncation, replay-ULP, stamp, and passive-admission plant. The same-run
post-wind boundary must be BIT against the admitted Round-139 incoming pair.
Any failure stops the round without a scientific claim.

Frozen prediction: P0 passes without changing a recorded digest or inherited
byte. A failed replay, ancestry check, completion marker, plant marker, or
same-run boundary identity REFUTES the prediction and stops the round.

## P1 — passive production-JIT three-dimensional boundary

Extend the existing Round-83 slow-forcing walk. Do not create another model
stepper and do not restore the retracted 27-field callback. Drive
`LatLonCGridOceanModel.step -> self._step_jitted` from the admitted completed
step-1080 NEMO state under CPU fp64/x64/libm. Add only the minimum write-only
RHS observer needed to capture the already-computed three-dimensional U/V
pair. Thickness, masks, and reference reciprocals come from the same admitted
stage-entry/model objects and are not materialized through the callback.

The instrumented step and an independently compiled ordinary step must return
bit-identical pytrees. Their actual external-call U/V forcing must also be BIT
on all wet faces. If either moves, the observer is non-passive: reject it,
leave the existing retraction in force, and stop without scoring its RHS.

Register, in compiled order, thickness U/V, completed `Krhs` U/V, masks U/V,
reference reciprocals U/V, and depth mean U/V. Signed zero is non-bit. Every
row reports scored cells, unequal cells, maximum absolute and RMS difference,
and the first unequal index. A missing-row plant and a one-ULP change to one
wet recorded RHS value must each exit nonzero with `STATUS PLANT-FIRED`.

Frozen scientific prediction inherited from Round 140: thickness, masks, and
reference reciprocals are BIT; completed `Krhs` U is the first non-bit row,
and completed `Krhs` V is the next non-bit row. At least one wet cell differs
in each RHS row and the maximum is nonzero. Exact RHS, an earlier geometry,
mask, or reciprocal mismatch, a non-bit same-run depth replay, a moved ordinary
state/external boundary, or a different first row REFUTES the prediction and
is retained in the receipt.

## P2 — conditional compiled-order term split

The admitted Round-140 stream has only the completed `Krhs`, not the cumulative
operator boundaries. Therefore no HPG, LDF, VOR, KEG, or ZAD owner may be
inferred from P1 alone.

If P1 confirms completed `Krhs` as first, extend the existing acquisition
pattern under one new target name with a WRITE-only step-1081 cumulative record
at these exact compiled boundaries:

1. after HPG;
2. after LDF;
3. after VOR;
4. after KEG; and
5. after ZAD, with the terminal identity to the completed Round-140 RHS.

The source card adds statements only, preserves every Round-140 inherited byte,
is syntax-proved before `makenemo`, prints a named `REFUSE` before every
nonzero exit, uses bash timing, and carries exact layout, stamp, truncation,
ULP, terminal-closure, and passive-admission plants. Run it in-round if the
NEMO configuration tree remains writable; otherwise report its `run.sh` as
`ACQUISITION_NEEDED`.

The model side reuses the existing source-order accumulator and production
operator components. Any observation path is claim-bearing only if its final
RHS, external forcing, and returned state are BIT against independently
compiled ordinary production runs. Rows are scored in the five-boundary order
above, U before V at each boundary.

Frozen conditional prediction: the first non-bit cumulative term is HPG U.
Any exact HPG pair makes LDF the next candidate; any earlier non-bit entry or
instrument movement REFUTES the prediction. If a later term is first, it is
named from the direct row, not inferred from residual subtraction. If the
new record is not acquired or the production observation is non-passive, the
round stops for that discriminator and names no term.

## P3 — day-240 magnitude and landing bar

The local day-180 one-step maximum is reported for the first non-bit term. It
is not called a day-240 carry. A day-240 magnitude requires a source-exact
candidate or a recorded year-long substitution family: run the same 360-day
member and Decision-45 scorer as the immutable before arm, with days
30/60/90/120/180/240/300/360 registered. A candidate lands only if all
Decision-43 and Decision-45 clauses pass, including no worse day-240 or
day-360 T3D RMS and measured DINO when the statement is shared.

Frozen prediction: this diagnostic walk does not by itself measure a
day-240 carry and lands no physics. If one single compiled statement is proven
wrong and can be transcribed without a configuration or carried-state choice,
its year arm is measured before a landing verdict. Otherwise day-240
sensitivity remains explicitly UNMEASURED and the receipt names the minimum
next discriminator; it does not assign the full `1.644674193e-2 K` gap to a
one-step term.

## P4 — review, citations, and scope

A separate read-only Codex pass must try to refute record ancestry, compiled
order, callback passivity, ordinary-state and external-boundary identity,
registry completeness, active masks, terminal closure, any conditional source
card, every plant, and the distinction between local day-180 magnitude and
day-240 carry. A `DO NOT SHIP` verdict blocks the diff. Every compiled-source
citation is mapped by the receipt citation gate; its shifted-citation plant
must exit nonzero.

No configuration, default, carried state, scheme, stabilizer, canonical NEMO
source, or immutable before arm changes in the diagnostic path. DINO,
LOCK_EXCHANGE, OVERFLOW, tanks, and ORCA2 are not rerun unless shared
production physics changes. ORCA2 remains `UNMEASURED-WITH-SPEC` for this
developed native momentum registry. `DECISION_NEEDED` is `NONE` unless a
configuration or carried-state choice is discovered.
