# ORCA2 round 52 preregistration — OVERFLOW stage-2 vorticity replay

Date frozen: 2026-09-27  
Base: `67669aa6ec39355eee076a711af16e07d7fffa49`  
ORCA2 claim label: **given NEMO's entry** (Decision 52; no ORCA2 trajectory
measurement is planned)  
OVERFLOW claim label: **given NEMO's recorded operands**

## Frozen question and scope

Round 51 established that the held QCO arm first moves independent OVERFLOW
stage-1 Kaa, while direct QCO, EOS, and active-u HPG replays are bit-exact on
NEMO operands.  This round advances exactly one statement in compiled order:
stage-2 vorticity.  It replays the production vorticity operator from the
recorded stage-2 tracer/ssh fields and the recorded stage-1 post-barotropic
velocity that NEMO swaps into stage-2 Kmm.

The producing configuration calls HPG, then vorticity, then advection at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:343-359`.
The stage-1 output becomes stage-2 Nnn/Kmm at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3.f90:200-207`.
The selected ENS dispatch is
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynvor.f90:242-249`, and its
active formula is
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynvor.f90:600-652`.

No configuration, selector, default, carried-state field, threshold, score
domain, stabiliser, or sea-ice field may change.  The held QCO package hunk is
not restored.  This is a statement read-out, not a landing round.

## Existing implementation and instrument

Repository search found the round-50 self-describing record parser, round-51
QCO/EOS/HPG replay, and the production operator-component return that already
materializes `after_vor_u/v`.  This round extends the round-51 gate; it does not
implement a second vorticity operator or add a NEMO acquisition.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R52-P1 | The admitted round-50 record and its exact HPG prerequisite reproduce round 51. | Admission is AT_BAR and `given.s2.hpg.u` remains bit-exact; the v domain remains `UNMEASURED_NO_ACTIVE_FACE`. | Any drift: stop and reconcile before scoring vorticity. |
| R52-P2 | Vorticity is structurally zero on the one-wet-row, zero-rotation OVERFLOW card, so the production replay is bit-exact on every active u cell. | `given.s2.vor.u` has 0 unequal active cells and NEMO's recorded `after_hpg_u` equals `after_vor_u` on the same mask; v remains explicitly unmeasured because it has no active face. | Any active-u unequal cell: vorticity is the first non-bit downstream statement; report its count, maximum and first location, keep QCO held, and do not continue to advection this round. |
| R52-P3 | The output comparison is non-vacuous. | Moving one active replayed u value by one representable fp64 step changes the exact row to one unequal cell and the plant exits nonzero. | A plant that does not fire: reject and repair the gate before quoting R52-P2. |
| R52-P4 | No scientific statement lands. | Final `packages/` diff equals the base; disposition is HELD with advection next if R52-P2 confirms, or vorticity named if it refutes. | Any package change: remove it unless a separately preregistered full shared-statement landing gate authorizes it. |

Failed predictions remain in the receipt as `REFUTED`; no score mask or bar
changes after measurement.

## Required verification

1. Re-run record admission and derive every array name and extent from the
   self-describing header.
2. Assert the stage-slot mapping from the record headers and compiled swap,
   then reconstruct stage-2 Kmm u/v from stage-1 `postbar_kaa_u/v` without
   filling an active value from legoESM.
3. Run the production JIT operator at CPU fp64/libm, score `after_hpg` and
   `after_vor` bitwise on the fixed masks, and run the active-u plant.
4. Run focused tests, Ruff/compile checks, citation gate plus planted shifted
   citation, shared-card battery, and the ocean-fidelity battery once.
5. Run separate read-only `codex exec` review and record its verdict or the
   mandated unavailable wording.

## Frozen OPEN

If vorticity is exact, replay stage-2 advection next from recorded
`after_vor`; if it is non-bit, walk the selected compiled ENS operands before
retrying the held QCO pair.  After this ordered OVERFLOW walk, return to
ORCA2's kt=1 stage-1 T owner, the independent Decision-52 initial state/year,
and round-20 slow forcing.  Sea ice remains out of scope at
`STOP_SELECTOR_GAP`.

ASKED: replay the next recorded statement in round 51's OPEN order.  
UNASKED: none.
