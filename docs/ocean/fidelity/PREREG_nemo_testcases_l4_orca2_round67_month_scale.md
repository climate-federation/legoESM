# ORCA2 round 67 preregistration — independent month record acquisition

Date frozen: 2026-09-28

Base: `9427601555a2b903a2246f744479497aad15c091`

Claim label: **independent**.  The requested month comparison starts legoESM
from its own ORCA2 card state and compares it only with NEMO's own from-rest
trajectory.  No NEMO entry field may be loaded into legoESM.  This round
acquires the missing NEMO comparator; it does not score or rank a month.

Sea ice remains out of scope.  The card's six-entry `unmeasured_features`
tuple and every selector remain frozen.  The NEMO comparator nevertheless must
use the already-pinned one-category `VARIANT_ORACLE_ORCA1ICE` deck because that
deck defines the external ocean forcing protocol for this lane.

## Source-first boundary

The compiled pinned configuration starts from rest, reads T/S, initializes
u/v to zero, and copies the before level at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/istate.f90:93-140`.
It then applies the out-of-scope ice-mass SSH adjustment at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceistate.f90:440-465`.
The month must retain both statements unchanged.

The ocean restart writer stores SSH, u, v, T, and S at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/restart.f90:153-184`.
With `nn_stock=240`, its compiled frequency logic schedules the terminal
restart through
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/restart.f90:98-119`.

The two existing 30-day records are not admissible comparators.  Their deck
predates `VARIANT_ORACLE_ORCA1ICE`; relative to the pin, the deck manifest
differs at `namelist_ice_cfg` and resolves a different sea-ice program.  The
new acquisition therefore uses the pin's deck and the already-admitted
scalar-math uninstrumented executable.  A separate ten-step calibration run
must reproduce the pinned instrumented record before the month is admitted.

## Frozen predictions and falsifiers

1. **No admitted month.** No existing 240-step record combines the pinned
   `VARIANT_ORACLE_ORCA1ICE` deck hash with its input hash and the admitted
   scalar-math binary.  Finding such a complete record refutes the need for an
   acquisition; it must be admitted instead of launching another run.
2. **Executable calibration.** With the pinned deck at ten steps, the
   uninstrumented scalar-math executable predicts bit-identical payloads for
   every variable in both ocean restart shards and both ice restart shards
   against `variant_orca1ice_phase2x_a_10step_np2`.  Any missing variable,
   dtype/shape difference, or unequal bit is a refusal and the 240-step arm is
   not admitted.
3. **One controlled run-length change.** Calibration and month decks predict
   identical file inventories and bytes except for `namelist_cfg`; its parsed
   assignments may differ only at `nn_itend` and `nn_stock`, from `10` to
   `240`.  Any other assignment difference is a refusal.
4. **Complete month.** The month predicts NEMO `STOP 0`, terminal ocean and ice
   restart shards for step 240 on both ranks, a resolved from-rest start, the
   pinned one-category ice deck, and finite float64 payloads for `sshn`, `un`,
   `vn`, `tn`, and `sn`.  Missing/truncated files, a restart start, another
   step, non-finite values, or another dtype refutes admission.
5. **Disposition.** This round predicts `STOPPED_FOR_RECORD` with no model
   change.  The operator runs the committed acquisition; the next round admits
   it, advances legoESM from its own state under the same forcing, and ranks
   matched month errors by magnitude before any further bit-row walk.

Failed predictions remain **REFUTED** in the receipt.  Calibration identity
and month validity are separate predicates; a valid-looking month cannot
waive a failed calibration.

## Required controls and validation

- The admission checker parses NetCDF's self-describing variable names,
  dtypes, shapes, and payloads; it carries no predicted byte count.
- One-ULP calibration, missing terminal shard, and hidden deck-assignment
  plants must each refuse.
- The acquisition is fail-closed, uses a new target name, emits named
  `REFUSE:` lines, and does not use `/usr/bin/time`.
- Run shell syntax, preflight, focused tests, receipt citation audit with a
  firing rigid-shift plant, the default citation audit, and the separate
  read-only Codex review.  With no `packages/` change, trajectory and wide
  pytest batteries are not applicable to this record-acquisition round.

ASKED: acquire the missing pinned NEMO from-rest month required by round 66's
OPEN item.

UNASKED: configuration, selector, threshold, forcing, stabiliser, carried
state, model arithmetic, sea-ice implementation, and the held shared tracer
QCO/RK candidate.
