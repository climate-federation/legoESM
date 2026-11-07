# ORCA2 round 69 receipt — independent-month surface acquisition

Date: 2026-09-28

Base: `a7af362112aaf6aa72d1956768ff97f417057c47`

Disposition: **STOPPED_FOR_RECORD; a calibrated surface-input-only 240-step
acquisition is preflight-ready**

Claim label: **independent**.  This round produces no legoESM month error.
It prepares the exact NEMO ocean surface operands needed to advance legoESM
from its own card state under the admitted forcing protocol.

Sea ice remains out of scope.  The ORCA2 card, all selectors, and its six-entry
`unmeasured_features` tuple are unchanged.

## Answer

The admitted ten-step instrumented record contains 20 rank-step surface
frames.  A 240-step two-rank month needs 480, so steps 11 through 240 are a
460-frame record gap.  The admitted uninstrumented terminal month contains no
surface-operand stream.  Repeating step 10 or reconstructing another forcing
would be an unauthorized configuration choice, so the month ranking remains
unmeasured.

The committed acquisition creates a fresh scalar-math NEMO target from the
unmodified base source and the pinned ORCA2 compile keys.  It adds a separate
WRITE-only module and an additions-only two-line call-site patch.  NEMO's
compiled order completes `sbc` before exposing the ocean operands at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3.f90:151-152`.
The new call occupies that same boundary and writes no model field.

Each rank and step gets one self-describing record containing only the ten
production-ladder operands: `utau`, `vtau`, `taum`, `qsr`, `qns`, `emp`,
`sfx`, `rnf`, `fr_i`, and `rnf_tsc`.  Every field carries its own name, rank,
and three dimensions before its payload.  The checker walks that header to
physical EOF; it predicts no whole-file byte count or positional payload size.

Admission is deliberately two-sided.  All ten fields on both ranks for steps
1 through 10 must be raw-bit equal to the already-admitted surface record,
including signed zero.  Separately, every variable payload in all four
terminal ocean/ice restart shards must remain bit-exact against round 68's
uninstrumented month.  Thus a schema-correct but perturbing writer cannot pass.

## Frozen prediction ledger

| ID | verdict | deciding evidence |
|---|---|---|
| R69-P1 complete self-describing inventory | **UNMEASURED** | The parser and synthetic controls are green; 480 real frames require the operator run. |
| R69-P2 ten-step calibration | **UNMEASURED** | The gate requires 200 raw-bit field comparisons; no round-69 record exists yet. |
| R69-P3 write-only passivity | **UNMEASURED** | The gate requires four bit-exact terminal restart shards; no round-69 run exists yet. |
| R69-P4 surface-only scope | **CONFIRMED for source/preflight; UNMEASURED for output** | The patch removes zero lines, the module assigns none of the ten NEMO operands, and the output inventory gate permits only the 480 named surface files. |
| R69-P5 disposition | **CONFIRMED** | The sandbox does not launch MPI; operator acquisition is required. |

Failed predictions remain **REFUTED** after execution; calibration and terminal
passivity cannot waive one another.

## Fail-closed acquisition

The launcher:

- requires a clean committed tree and verifies every repo artifact is tracked;
- pins the base `stprk3`, compile-key file, scalar-math compiler card, month
  deck/input manifests, and ten-step calibration admission by SHA-256;
- creates the evidence directory before checking free space;
- applies the committed additions-only patch to a fresh target name and checks
  the compiled call, overwrite refusal, and absence of vector-math symbols;
- stages the exact admitted 240-step deck and input manifests, records the
  producer commit plus binary/compiled/source digests, and refuses existing
  outputs;
- uses no `/usr/bin/time`; and
- supports `--admit-existing` without rebuilding or rerunning NEMO.

The six admission plants are wrong field name, truncated payload, one-ULP
calibration mutation, missing frame, extra oracle stream, and one-ULP terminal
restart mutation.  Each has a focused synthetic control that refuses before a
real record is available.

## Validation

- acquisition preflight: `PREFLIGHT_PASS` and
  `ORCA2_ROUND69_SURFACE_PREFLIGHT_READY`;
- focused gate tests: `9 passed`;
- shell syntax and Python compilation pass;
- shared-card battery: `160 passed` with nine dtype warnings;
- tank battery: `10 passed`.

No `packages/` file changed.  Model trajectories cannot move; GYRE/DINO/tank
trajectory gates and the wide ocean battery are not applicable to this
acquisition-only round.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round69/`.

The required separate `codex exec --sandbox read-only` review was attempted.
It returned `Error: failed to initialize in-process app-server client:
Read-only file system (os error 30)`; independent review is therefore
unavailable in-sandbox.  The complete attempt is retained as
`codex_review.log` in the evidence directory.

## Scope ledger

ASKED: acquire the missing exact surface operands required for Decision 52's
independent ORCA2 month ranking.

UNASKED and unchanged: configuration, selector, threshold, forcing
reconstruction, stabiliser, carried state, model arithmetic, sea-ice
implementation, and the held QCO/RK change.

## OPEN

1. The operator runs
   `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round69_month_surface_acquisition/run.sh --run`.
2. The next round admits all 480 frames, runs legoESM's independent 240-step
   trajectory, and ranks terminal SSH/T/S/u/v errors by magnitude before any
   further bit-row walk.
3. The independent kt=1 SSH mismatch remains out of scope at the sea-ice
   `STOP_SELECTOR_GAP`; all six selectors stay frozen.
4. The source-ordered tracer QCO/RK statement remains locally exact but held
   by round 48's five OVERFLOW rows.
