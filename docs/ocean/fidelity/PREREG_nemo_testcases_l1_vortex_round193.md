# PREREGISTRATION — round 193 / VORTEX round 8 continuation: stage-term walk

Frozen before parsing or comparing the operator-produced stage-term values.
Lane tip at the start: `2532bd962`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round193`.

## Input and compiled order

The operator reports a successful round-192 acquisition at
`round192/oracle_stage23_terms`. Admission is accepted only if the committed
checker independently reproduces restart byte identity, parses both
self-describing stage files, and its extent plant exits nonzero.

The record's compiled build is
`tests/VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo`. For stages 2 and 3 it writes
the accumulator after WZV and then after HPG, VOR/EEN, KEG and ZAD in compiled
order (`stprk3_stg.f90:289-309`, `:323-341`; `dynadv.f90:135-141`). Stage 2
then writes the explicit update (`stprk3_stg.f90:367-388`). Stage 3 writes
after LDF, after the implicit ZDF solve, and after the common barotropic
correction (`stprk3_stg.f90:395-434`).

## Frozen predictions and falsifiers

1. **Admission.** The paired step-10 restarts are byte-identical, both stage
   records contain exactly their declared named groups, and the planted
   extent corruption exits nonzero. Any failure stops the walk.
2. **Production path.** Every legoESM comparison runs through `model.step`,
   hence `_step_jitted`, with fp64/libm and NEMO's recorded stage entry plus
   external-mode handoff. An eager-only comparison cannot name a statement.
3. **Calibration.** Re-exposing the stage output reproduces round 192's
   already measured stage-2 and stage-3 residuals to within 1% relative:
   stage 2 `u = 1.6701813777553198e-06`, stage 3
   `u = 3.3737007034684297e-06`. A disagreement stops attribution.
4. **Operator prediction.** HPG is predicted bit-exact at both stages. The
   first non-bit accumulator increment is predicted to be VOR/EEN or one of
   its live stage operands; if HPG is non-bit, that prediction is REFUTED and
   HPG is the first statement. If HPG and VOR/EEN are exact, the walk proceeds
   without skipping through KEG, ZAD, stage-2 update, stage-3 LDF/ZDF, and the
   common correction.
5. **Magnitude prediction.** The first non-bit operator or boundary carries
   at least half of the corresponding completed-stage velocity residual. If
   no single boundary does, the magnitude prediction is REFUTED and the
   receipt reports the ordered accumulation rather than inventing an owner.
6. **Plants.** Every exposed or substituted production arm receives a
   one-cell perturbation at its live input/output and must move its own scored
   row by more than both ten times baseline and `1e-12`. A plant that exits
   zero or moves only another arm makes that row UNMEASURED.

## Landing criterion

First name the earliest non-bit compiled statement. A model change is eligible
only if it makes that statement bit-exact given NEMO's recorded operands and
then passes the full Decision 43/45/55/59 scope: both VORTEX ladders, GYRE
ladder and day 30/240/360, generic card, private-workdir DINO month, LOCK and
OVERFLOW, complete moved-row registry, citations and plants. If the statement
requires a new record or cannot be closed within this round, land no physics
and report HELD or STOPPED_FOR_RECORD.

## Choices

No configuration, card, carried-state, timestep, resolution, stabilizer, or
default choice is made. ORCA2 remains unmeasured on its own lane.
