# Preregistration — NEMO testcase L2 GYRE round 152

Date: 2026-09-22

Incoming lane tip: `e4ba88549a26a189df9c97d5df8343b9e6cdd0cb`.
Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round152/`.  This document is
frozen before rerunning or extending the developed-state process walk.

## Fixed program and records

This round extends the existing
`nemo_testcase_l2_gyre_year_owners.py --developed-step-walk` instrument; it
does not write a second stepper.  The entry is the admitted NEMO restart after
step 1080, and the scored production-JIT step is step 1081.  The NEMO process
boundaries remain the admitted Round-123 record and the implicit-solve
internals remain the admitted Round-125 record.  The bridge, resolved GYRE-zco
card, forcing, fp64/libm policy, timestep, masks, and private `ssha` entry
override are unchanged from Round 136.

The compiled stage-3 order is frozen from the record-producing build.  It
clears tracer `Krhs`, calls advection, surface forcing, penetrative shortwave,
lateral diffusion, and implicit vertical diffusion in that order at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:861-970`.
The process record contains the entry tracer, all three T-point QCO ratios,
the cumulative `Krhs` after the first four calls, and the final tracer after
the implicit solve.  It does not contain FCT faces/coefficients or a NEMO FCT
activity map at step 1081; absent operands will remain `UNMEASURED` rather
than reconstructed from downstream differences.

No production physics, configuration, carried state, restart schema,
stabilizer, year harness, reconciliation gate, freshwater pair, or #1484
guard changes in this round.

## Frozen measurement and ranking

The existing gate must first readmit all three records and prove the bridged
entry temperature BIT on the complete 18,000-cell NEMO wet mask.  Its
write-only observer must change zero returned-state bytes relative to an
ordinary production step.  The entry-temperature ULP plant must move at least
one registered process boundary, print `STATUS PLANT-FIRED`, and exit nonzero.

For each compiled boundary, report both the cumulative state mismatch and the
isolated one-step temperature contribution mismatch: unequal cells, maximum
absolute difference, RMS, and effective RMS tendency error (contribution RMS
divided by `14400 s`).  Rank the six isolated contributions by RMS magnitude,
while separately naming the first non-bit row in compiled execution order.
Every one of the six registered rows is mandatory; a missing-row plant must
exit nonzero.

Frozen directional predictions from the prior admitted record are:

1. The entry tracer remains BIT and the production observer remains passive.
2. The inherited free-surface geometry row remains non-bit.  Among actual
   process calls, combined advection is the first non-bit contribution.
3. Vertical diffusion has the largest one-step temperature-contribution RMS,
   followed by lateral diffusion, shortwave, advection, geometry, and surface
   boundary.  The expected order may be refuted; the measured order is kept.
4. The model-side FCT nonosc limiter, EVD replacement, and TKE floors are all
   active.  NEMO FCT branch selection remains unmeasured by this record.
5. The first internally attributable FCT statement remains OPEN unless every
   preceding compiled operand is directly proven BIT on this same developed
   entry.  The cumulative `CALL tra_adv` boundary may be named; an internal
   limiter statement may not be inferred from spatial overlap.

The prediction is refuted by any different process ordering, a BIT geometry
row, a non-passive observer, an inactive model branch, or a different
magnitude order.  Failed predictions remain in the receipt.

## Candidate and campaign gates

Only a single source-exact statement proven through the production step on
this developed entry is eligible for a candidate.  If one exists, it must run
the complete Decision-43/45 ladder, month, and year gate against the current
Round-149/150 arms, register every moved row, preserve kt1 bar rows and the
first-over-bar boundary, and measure every recipe-derived shared card,
including DINO when it executes the statement.

If the available record stops at a non-bit call without directly recorded
internal operands, no candidate is fabricated or year-scored.  The expected
status is `HELD`, with the complete one-step ranking and the exact first
unresolved compiled statement.  ORCA2 remains `UNMEASURED-WITH-SPEC`: replay
the same developed-entry boundary table on its ocean-only card before
transferring the result.  No configuration or carried-state decision is made.
