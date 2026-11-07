# Preregistration — NEMO testcase L2 GYRE round 151

Date: 2026-09-22

Incoming lane tip: `b76a009de5ea07f658bfce64c82ea732eaa62c65`.
Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round151/`.  This document is
frozen before adding the amplitude control to the existing year harness and
before running or scoring any new member.

## Fixed program and initial perturbations

This round changes no production physics, configuration, carried state,
restart schema, stabilizer, year stepping program, forcing, grid, timestep, or
scored population.  It extends the existing certified from-rest member path so
the already transcribed seed-1 temperature pattern can be multiplied by one of
the three operator-ordered absolute amplitudes: `1e-8`, `1e-6`, or `1e-4 K`.
The unperturbed control is the landed Round-149/150 member-0 year trajectory.

The source pattern is the compiled NEMO ensemble statement at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/usrdef_istate.f90:101-105`:

`A * sin(NINT(pdept)*73 + NINT(gphit*1000)*179 + 1*997) * ptmask`.

NEMO's recorded Round-129 member used `A=1e-10 K` at line 103.  This round
uses the identical seed-1 argument, Fortran `NINT` transcription, temperature
field, and wet mask; only `A` changes.  The member manifest must record the
requested amplitude, measured wet-cell peak, complete initial-field delta
census, clean producer commit, fp64/libm policy, CPU platform, and unchanged
certified stepping-gate hash.  A plant that substitutes the wrong amplitude
must print `STATUS PLANT-FIRED` and exit nonzero.

## Fixed score

The three members run for 360 days at six steps per day with snapshots every
six steps.  They are scored against the unperturbed member 0 at exactly days
`30,60,90,120,180,240,300,360`.  The sole verdict row is the unweighted fp64
RMS of three-dimensional temperature over NEMO's 18,000-cell wet `tmask`.
Every member/day row must exist and be finite.  The scorer must verify member
0 reproduces the current immutable Round-149 arm at day 240
`1.644671864406711e-2 K` versus NEMO; this is an instrument control, not one
of the perturbation spreads.

The fixed amplification threshold is
`0.3 * 1.644671864406711e-2 = 4.934015593220133e-3 K`.  Equality passes.
The adjacent binary64 value below must fail and the value above must pass.
A missing registered day or amplitude must print `STATUS PLANT-FIRED` and
exit nonzero.

## Frozen predictions, falsifiers, and verdict

The directional prediction is a threshold between `1e-6` and `1e-4 K`:
the `1e-4 K` member reaches at least `4.934015593220133e-3 K` by day 240,
the `1e-6 K` and `1e-8 K` members remain below it, and the `1e-8 K` row stays
near the Round-129 deterministic sensitivity floor (`2.0891293703252062e-10
K` at day 240; the exact ratio is reported, not thresholded post hoc).

The campaign verdict is **THRESHOLD AMPLIFICATION** only if at least one of
the `1e-6` or `1e-4 K` members reaches the fixed threshold while the `1e-8 K`
member remains below it.  If all three day-240 spreads stay below `1e-6 K`,
the operator-registered alternative is **SYSTEMATIC TRACER-STEP OWNER** and
Round 152 walks the developed-state tracer step.  Any intermediate outcome is
reported as **INCONCLUSIVE UNDER THE REGISTERED DISCRIMINATOR**; no threshold
is invented after measurement.

The prediction is REFUTED if `1e-4 K` stays below the amplification threshold,
if either smaller arm crosses it, or if the `1e-8 K` row is not separated from
the amplified arm.  Failed predictions remain in the receipt.

No production code can land this round.  GYRE is measured; generic NEMO-GYRE,
DINO, LOCK_EXCHANGE, OVERFLOW, and ORCA2 execute no changed production
statement.  ORCA2 remains `UNMEASURED-WITH-SPEC`: repeat the same three
amplitudes on its own unperturbed ocean-only member before transferring this
amplification verdict.
