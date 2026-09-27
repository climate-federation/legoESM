# ORCA2 round 51 receipt — OVERFLOW kt=3 QCO pair walk

Date: 2026-09-27  
Base: `932cbfa9ec2f2fbcbf51a03ca8e46e5b39b78c62`  
Preregistration: `0e5ad0e780c5a48d63180389c5a62c929015ce05`  
Disposition: **HELD**  
ORCA2 claim label: **given NEMO's entry** (Decision 52; no new ORCA2
trajectory was measured)  
OVERFLOW live-arm claim label: **independent**  
OVERFLOW statement-replay claim label: **given NEMO's recorded operands**

## Answer

The operator-produced round-50 record is admissible.  The held
source-ordered QCO arm first changes the independent OVERFLOW trajectory at
stage-1 Kaa: 52 of 17,000 active T cells move, maximum
`3.552713678800501e-15`, and 128 S cells move, maximum
`7.105427357601002e-15`.  The arm does not move the identical kt=3 entry or
the observed stage-2 HPG u row.

It does not land.  Relative to NEMO, the live stage-1 Kaa T unequal count
worsens 164 to 184 while S improves 153 to 101.  More importantly, direct
production replays on NEMO's recorded operands are bit-exact for the stage-1
QCO T/S assignment, stage-2 EOS density, and every active stage-2 SCO HPG u
cell.  The v-face domain has zero active cells on this one-row card.  Thus the
preregistered claim that HPG is the next non-bit statement is **REFUTED**.
The next unmeasured statement in compiled order is stage-2 vorticity.

The independent base is already non-bit at kt=3 entry: T 61/17,000
(`1.1451106729509775e-10` maximum), S 25/17,000
(`7.105427357601002e-15`), u 520/16,900
(`4.181444884787666e-09`), and ssh 17/200
(`1.4924866897914058e-13`).  Those are carried entry differences, not a new
kt=3 statement, and they are deliberately not mixed with the given-input
statement table below.

## Compiled source

The admitted record's own compiled configuration zeros and accumulates the
tracer RHS, then performs the stage-1/2 QCO assignment at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:492-541`.
For stages 2/3 it calls EOS, HPG, vorticity, then advection at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:327-363`.
The selected EOS polynomial is
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/eosbn2.f90:684-718`, and the
selected SCO HPG recurrence is
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynhpg.f90:341-419`.

These are the compiled files under the exact configuration that produced the
round-50 record.  No dead source arm or canonical-source line is cited.

## Record admission and controls

The round-50 self-describing schema admits all six stage files at producer
`932cbfa9e`.  Restart and mesh are exact.  The inherited-stream calibration
reproduces `exact=24/27`, `changed=3`, `admitted=16`.  The payload-digest and
producer-stamp plants both refuse.  The inherited-field plant changes the
classification to `exact=23/27`, `changed=4`, `admitted=16` and fails.

The direct EOS plant moves exactly one of 17,000 active `rhd` values by one
representable fp64 value (`1.0842021724855044e-19`) and makes the otherwise
bit-exact row refuse.  The parser derives names, ranks, extents, and payload
lengths from each record's header; no hand-predicted byte count is used.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round51/`:
`record_admission.log`, `record_payload_plant.log`,
`record_stamp_plant.log`, `inherited_admission.json`,
`inherited_plant.log`, `base.json`, `base_plant.json`, and `candidate.json`.
The large arrays remain outside git as hash-stamped npz sidecars.

## Frozen predictions

| ID | verdict | evidence |
|---|---|---|
| R51-P1 | **CONFIRMED** | Record/schema admission and all three inherited-record controls reproduce the frozen counts. |
| R51-P2 | **PARTLY CONFIRMED** | Base and candidate kt=3 T/S/u/ssh sidecars are array-equal.  The preregistered GYRE-wide observer requires absent GYRE-only TKE/slow-forcing bundles on OVERFLOW, so its separate `state_after` noninterference check is **UNMEASURED_WITH_SPEC**; the scorer used existing post-step write-only stage hooks rather than weakening that guard. |
| R51-P3 | **CONFIRMED within the scored pair boundary** | The identical entry is followed by the first arm movement at stage-1 Kaa T (52) and S (128).  The separate earlier RHS boundary readouts were omitted after their multi-compile instrument exceeded the round CPU envelope; compiled order places the candidate assignment after them. |
| R51-P4 | **REFUTED** | Given NEMO operands, QCO, EOS, and active-u HPG are all bit-exact.  HPG is not the next non-bit statement; VOR is next and remains unmeasured. |
| R51-P5 | **CONFIRMED** | The experimental package hunk was removed; the final `packages/` tree equals the round base. |

## Verification

- Focused round-50/51 parser/scorer and citation-gate controls: **29 passed**;
  round-51 Ruff and both gate `py_compile` checks are clean.
- Base and candidate are clean, CPU-only fp64/libm production-JIT runs.  The
  candidate comparison confirms identical entry and the stage-1 Kaa first
  movement.  An attempted persistent JAX cache was rejected after XLA warned
  that sandbox-worker target features differed; no cached result was used for
  the science comparison.
- Final package diff against `932cbfa9e`: empty.  Therefore this held round
  does not trigger the shared-model GYRE trajectory/year or DINO/tank landing
  gates.
- Shared-card battery: **170 passed** in 352.96 s.
- `tests/ocean/fidelity -n 12` collected 1,944 items, reached 99%, and then
  reproduced the documented xdist tail stall.  Its five red nodes were rerun
  serially and retain their pre-existing signatures: round-129 stale
  certification, round-51 private trace registry, SI3 scalar-math provenance,
  the worktree-stamp ratchet, and the `hires_lane_surface` case-board ratchet.
- Default receipt citation gate: **274 citations, PASS**, zero failures and
  zero unmapped.  This receipt: **4 citations, PASS**, zero failures and zero
  unmapped.  Shifting the QCO citation by two lines fires
  `SYMBOL-NOT-AT-LINE`.
- Separate read-only `codex exec` review: **independent review unavailable
  in-sandbox** (`failed to initialize in-process app-server client: Read-only
  file system`).
- Sea ice, all six selectors, and the ORCA2 card's `unmeasured_features` tuple
  are unchanged at `STOP_SELECTOR_GAP`.

## OPEN

1. Extend the admitted, self-describing kt=3 replay by one statement: execute
   stage-2 vorticity on the record's exact `after_hpg` and stage operands,
   compare with recorded `after_vor`, and plant one active u value.  The
   one-row card's v domain remains `UNMEASURED_NO_ACTIVE_FACE`.
2. Do not restore the source-ordered QCO arm alone: it worsens the independent
   live T row and no second non-bit statement has been established.
3. After the OVERFLOW ordered walk reaches a non-bit statement or exhausts
   the recorded sequence, return to ORCA2's whole-card kt=1 stage-1 T owner,
   the independent Decision-52 initial state/year, and round-20 slow forcing.
4. Sea ice remains out of scope at `STOP_SELECTOR_GAP`.

ASKED: admit the operator record and continue the held cancelling-pair walk.  
UNASKED: none.
