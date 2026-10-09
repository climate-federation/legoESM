# Preregistration — NEMO testcase L2 GYRE round 121

Date: 2026-09-19

Incoming lane tip: `02dd8aa84a0738146a0a8e8ca90b6a94f28c28c7`

This document is frozen before any Round-121 scientific measurement.  Evidence
will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round121/`.  Round 121 performs
the magnitude discriminator left OPEN by Round 120: one admitted NEMO `W`
operand is substituted at the current first/largest ZAD-input boundary, once,
and the otherwise ordinary production trajectory is followed through day 360.
The intervention is diagnostic only.  It does not authorize a production
physics change, configuration change, carried-state change, restart-schema
change, or new oracle acquisition.

## P1 — admitted source and exact intervention boundary

The source record is the already admitted Round-64/Round-46 kt=2 stage-1 record
used by Rounds 86 and 119.  Its SHA-256, binary layout, producer build, clean
source/worktree stamps, fp64 policy, seasonal clock and `kt=2` identity must be
rechecked before use; no scratch record is admissible.

The compiled GYRE C2 program orders HPG, LDF and VOR before computing `r3t(Kaa)`
and `ww`, then calls KEG and ZAD into the shared `Krhs` accumulator at
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/stp2d.f90:144-179`.  The QCO
`W` producer sets its bottom boundary, calls `div_hor`, and performs the
bottom-up recurrence at `sshwzv.f90:277-300`.  ZAD consumes `ww` in the
transport construction and updates U/V at `dynzad.f90:102-138`.

The diagnostic therefore replaces **only** the full stage-1 `W` array passed
to ZAD during model step `kt=2`, after live production W has been computed and
recorded.  All HPG/LDF/VOR/KEG results, Kmm velocities, thickness/free-surface
fields, masks, metrics and ZAD arithmetic remain live model values.  `kt=1`
and every step `kt>=3` use the ordinary model.  The existing Round-83 producer
gate will be extended in place; no second stage or trajectory harness will be
created.  The from-rest and ladder integrations will still be delegated to the
existing committed harnesses with their normal fp64, clock, cadence and seed.

Frozen reproduction: before substitution, production JIT must reproduce the
Round-119 W row exactly: `18,000 / 18,000` owned wet cells unequal, maximum
absolute difference `7.946658315637966e-7 m s-1`.  The W-only replay must also
reproduce the incremental ZAD boundary discrepancy
`1.9220297482797664e-9 m s-2` U and
`1.966061294804274e-9 m s-2` V.  A different count or maximum, an inadmissible
record, or a different first ZAD operand refutes P1 and stops the trajectory
experiment.

## P2 — local production-JIT discriminator and plant

Four separately labelled rows will be emitted from the same kt=2 stage entry:
ordinary production JIT, NEMO-W production JIT, ordinary production eager and
NEMO-W production eager.  Isolated arithmetic may be reported only as a fifth,
explicitly isolated control; it is never called production.

Frozen prediction for the NEMO-W production-JIT arm:

* the consumed W row becomes BIT on every owned active cell;
* every registered non-W ZAD operand and every pre-ZAD accumulator boundary is
  BIT-identical to the ordinary production-JIT arm;
* the approximately `1.9e-9 m s-2` W-owned ZAD increment is removed, leaving
  at most the already registered last-bit association floor; and
* the returned kt=2 state changes, proving the intervention reaches the
  production closure rather than an isolated copy.

The exact residual ceiling for “association floor” is frozen as
`8.470329472543003e-22 m s-2`, the Round-119 full accumulator floor.  Any non-W
operand movement, any pre-ZAD movement, a non-exact consumed W row, a residual
above that ceiling, or an unchanged returned state refutes P2.  Eager and JIT
outcomes are registered independently.

The production plant selects the first lexicographic active, finite, nonzero
NEMO W word whose one-ULP move toward `+infinity` propagates.  It changes that
one word only inside the full production step.  The planted run must report
one W cell unequal, leave every non-W input and pre-ZAD boundary BIT, move at
least one raw ZAD U/V word and at least one returned-state word, print
`STATUS PLANT-FIRED`, and exit nonzero.  Failure of any condition refutes the
control.

## P3 — ladder, month and registered year magnitude

If P1 and P2 pass, the same one-time kt=2 intervention is run from rest through
the existing kt=1..10 ladder and through one existing 360-day member run
(`member=0`, `days=360`, `snap-steps=6`, fp64, NEMO seasonal clock).  That one
member supplies both days 1--30 and the Decision-45 days
`30,60,90,120,180,240,300,360`.  A separate intervention provenance record
will name the hook, source-record SHA, incoming commit and exact affected step;
the ordinary harness manifest is not altered or passed off as an unperturbed
landing candidate.

The comparison arm is the immutable post-Round-110 trajectory.  For the year,
Decision 45 fixes `phase3/year_equivalence/gyre/` as the before arm, including:

| day | before T3D wet RMS [K] |
|---:|---:|
| 30 | `6.890484901489568e-5` |
| 60 | `1.9329973681936875e-4` |
| 90 | `1.8645021144913585e-3` |
| 120 | `1.0501256819510476e-3` |
| 180 | `3.580551011866709e-3` |
| 240 | `1.6446741930292448e-2` |
| 300 | `1.3597404177319843e-2` |
| 360 | `1.1223573910167267e-2` |

The frozen headline prediction is **no material trajectory ownership**: the
absolute before/after change remains below 10% at day 240, then day 360, then
day 30.  The corresponding fixed materiality thresholds are
`1.6446741930292449e-3 K`, `1.1223573910167267e-3 K`, and
`6.890484901489568e-6 K`.  A change at or above any threshold refutes that
prediction; day 240 has ranking priority.  This 10% classification is only an
attribution discriminator and does not relax Decisions 43 or 45.

For the ladder, every moved row is registered before/after.  Frozen prediction:
all kt=1 rows and kt=2 T/S remain BIT-identical to the before arm, the first
over-bar row is not earlier than kt=2 U/V, and kt=3 T/S may move but neither is
claimed to improve.  Any kt=1 movement, an earlier first-over-bar row, or an
unregistered moved row invalidates the trajectory measurement.  Days 1--30
and every fixed year row are emitted even when unchanged.  Day-240 T3D wet RMS
is the primary magnitude verdict, followed by day 360 and day 30.

If W has no material day-240 effect, Round 121 records the refutation and
returns the next round to the registered year-owner decomposition; it does not
optimize W.  If W has a material effect, no implementation change lands in
this round: a later preregistered walk must start at W's compiled producer,
name its first production-JIT non-bit statement, and independently satisfy the
Decision-43 ladder/month and Decision-45 year criteria.

## P4 — scope and blast radius

No shared candidate lands in Round 121.  The private hook defaults to absent,
is unreachable from every recipe, and is accepted only by the test-only model
constructor path.  A focused test must show default construction and every
resolved card remain unchanged.

GYRE and the generic NEMO-GYRE recipe execute the W/ZAD path.  DINO shares the
statements and is explicitly AT RISK for a future producer fix; any such
candidate must measure its committed gate and all recipe-derived executing
cards.  LOCK_EXCHANGE and OVERFLOW do not execute this private intervention.
ORCA2 remains `UNMEASURED-WITH-SPEC`: acquire its stage-entry W/divergence,
free-surface and ZAD-accumulator boundaries from its compiled card, prove the
production-JIT statement with a propagating plant, then run its certified
trajectory and all registered year rows before a shared landing.  No
stabilizer, freshwater-pair edit, year-harness edit, reconciliation-gate edit,
or `#1484` guard edit is permitted.
