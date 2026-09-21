# Preregistration — NEMO testcase L2 GYRE round 136

Date: 2026-09-21

Incoming lane tip: `dda3f3257dad5a7f86e1177f934afc531e6567d9`

This document is frozen before the Round-136 developed-state production-step
run or scoring. Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round136/`.

Round 134 showed that daily coupled T/S replacement removes 99.683239% of the
day-240 temperature gap. Round 135 showed that temperature alone removes only
32.14%, salinity alone is harmful, and every sparser temperature cadence is
harmful. Those interventions locate a coupled tracer family, not a compiled
statement. Round 136 therefore drives the existing production-JIT tracer
process observer from NEMO's own developed state instead of walking the smooth
from-rest steps again. It lands no physics or configuration.

## P0 — actual oracle contract and availability

The Round-123 process record is first readmitted at producer commit
`af3f7215060fc17c71adc6794817c710df8ee471`. Its closed manifest must contain
exactly 360 frames, steps 1081 through 1440. The parser's stamp, truncation,
one-ULP raw-boundary, and effect-scale plants must all exit nonzero.

The record does **not** contain the advertised per-axis `ttrd_xad/yad/zad`
streams, salinity terms, limiter choices, or steps near days 30 and 90. Its
compiled writer contains only stage-3 temperature `Tbb`, the three `r3t`
ratios, cumulative `Krhs` after combined advection/SBC/QSR/LDF, and `Taa`
after ZDF. Thus the only exact equal-input developed step supported by the
available inputs is step 1081: the admitted Round-132 restart at completed
step 1080 supplies its entry, and the Round-123 frame supplies its output
boundaries. This round will not synthesize absent terms or call an independent
trajectory an equal-input walk.

The registered availability table is frozen as follows:

| requested day | exact entry | next-step process frame | verdict |
|---:|---|---|---|
| 30 | daily restart step 180 | absent | `UNAVAILABLE_BY_RECORD` |
| 90 | daily restart step 540 | absent | `UNAVAILABLE_BY_RECORD` |
| 180 | daily restart step 1080 | step 1081 | `MEASURED` |
| 240 | daily restart step 1440 | step 1441 absent | `UNAVAILABLE_BY_RECORD` |

Frame 1440 ends at day 240 but begins from step 1439, for which the daily
record has no full state; it is not relabeled as a day-240 equal-input step.
Because the explicit acquisition exception applies only if the day-180 trends
are missing, this limitation does not authorize a new NEMO acquisition.

## P1 — one production step from the admitted day-180 entry

Extend `nemo_testcase_l2_gyre_year_owners.py`; do not create a second stepper.
Readmit the Round-132 daily restart by its closed SHA manifest and reuse the
existing restart bridge and all four registered reset families to construct
the step-1081 entry: T/S, u/v and `uu_n/vv_n`, all four velocity histories,
SSH and both SSH histories, `ssha`, and TKE/en/avm/avt/dissl. Before stepping,
print and register every live state leaf. Every non-static, consumed leaf must
be either mapped from the restart or proven diagnostically reconstructed or
inert; an unclassified live leaf refuses the run.

Run exactly one `LatLonCGridOceanModel.step` call, which enters
`self._step_jitted`, with the existing write-only `tracer_process_trace` hook.
The separately compiled ordinary production result and the trace result's
`state_after` must be bit-identical over the full pytree. A one-ULP mutation of
a consumed wet entry temperature cell must move at least one registered
process boundary and must exit nonzero as `STATUS PLANT-FIRED`.

Frozen bridge predictions:

1. the bridged input `T` is bit-identical to the Round-123 writer's `Tbb` on
   all 17,400 wet cells;
2. the geometry-only temperature row is bit-identical between models;
3. diagnostic observation changes zero bytes of the ordinary returned state.

Any nonzero count refutes the corresponding prediction and prevents an owner
claim downstream of that boundary.

## P2 — compiled-order temperature boundaries and branch activity

Score, cell by cell on the certified wet mask, these cumulative stage-3
temperature boundaries in compiled order: geometry, combined advection,
surface boundary, shortwave, lateral diffusion, and vertical diffusion. For
each row register cells unequal, maximum absolute difference, RMS difference,
and the first differing `(j,i,k)`. “First non-bit” means the first cumulative
boundary whose float64 bit pattern differs after every earlier boundary is
BIT; signed-zero differences count.

Frozen scientific prediction: geometry is BIT and the first non-bit boundary
is combined advection, with at least one unequal wet cell. The model-side FCT
limiter is active in at least one wet cell and at least one advection-unequal
cell overlaps an active limiter cell. The developed EVD trigger and at least
one TKE floor are also active. Zero overlap, an earlier geometry difference,
or a later first boundary is `REFUTED` and remains in the receipt.

Branch activity must come from the production operands or an already admitted
writer, not from a threshold guessed from the final tendency. Report separate
maps/counts for the FCT nonosc limiter, EVD replacement, and TKE floors, and
label each map `NEMO`, `legoESM-on-NEMO-entry`, or `UNMEASURED`. No model map
may be presented as proof of NEMO's branch selection. Overlap is registered
against the first unequal process boundary only.

The available oracle resolves combined `CALL tra_adv` but not its internal
x/y/z or nonosc writes. Therefore this round may name that compiled call as
the first recorded non-bit statement if the prediction holds, but it may name
an internal FCT statement only if an existing admitted operand record proves
all preceding writes BIT on this same step-1081 entry. Otherwise the internal
statement remains OPEN rather than inferred from overlap.

## P3 — controls, citations, and outcome

The JSON registry must contain all four requested day rows, exactly one
`MEASURED` row, all six process rows, all three branch families, the complete
state-leaf classification, and the record/worktree digests. Plants for a
missing day, missing process row, one-ULP entry change, process-record stamp,
and shifted citation must exit nonzero; a plant path must print
`STATUS PLANT-FIRED`, never `STATUS PASS`.

A separate read-only Codex pass must try to refute record availability,
restart timing, complete state mapping, production-JIT entry, observer
passivity, first-boundary ordering, branch-map provenance, plants, and the
claim boundary. A `DO NOT SHIP` verdict blocks the receipt. Every source claim
must cite the compiled `ppsrc/nemo` branch of the record-producing build and
be mapped by the receipt citation gate.

No card, library default, carried production state, stabilizer, NEMO source,
year acceptance bar, DINO behavior, ORCA2 selector, LOCK_EXCHANGE, or OVERFLOW
path changes. Expected status is `HELD`; `ACQUISITION_NEEDED` and
`DECISION_NEEDED` are both `NONE`.
