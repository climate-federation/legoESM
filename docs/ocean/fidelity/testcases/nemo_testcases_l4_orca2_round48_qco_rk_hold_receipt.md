# ORCA2 round 48 — source-ordered QCO/RK candidate held by OVERFLOW

**Date:** 2026-09-27  
**Base:** `7dd54b6f18d8f21307e5d2b2f5269b553dd6331f`  
**Preregistration:** `0a73d9af3`  
**Measured candidate:** `1722ef29a3c01b094af757520109e55694063d95`  
**Revert:** `bd5599b1f`  
**Disposition:** **HELD**

## Claim boundary

Every ORCA2 number below is **GIVEN NEMO'S ENTRY** under Decision 52 and the
pinned `VARIANT_ORACLE_ORCA1ICE` record.  No independent-initial-state result
is mixed into these tables.  Sea ice remains out of scope: the six selectors
and the card's `unmeasured_features` tuple are unchanged.

## Compiled statement and candidate

The executing preprocessed source separately multiplies the before-level
tracer by its stretch, multiplies the stage RHS by the stage interval and its
stretch and mask, adds the two rounded terms, and divides by the after-level
stretch at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:670-681`.

Candidate `1722ef29a3` preserved that operation order in the production WS-RK3
tracer assignment without adding a selector, stabiliser, or carried-state
field.  It was reverted after the mandatory tank comparison refused.

## Direct statement replay — GIVEN NEMO'S ENTRY

| row | base | candidate | outcome |
|---|---:|---:|---|
| stage-1 T | 57,141 / 228,641 unequal; max 3.552713678800501e-15 K | 0 / 228,641 | CONFIRMED |
| stage-1 S | 57,169 / 228,641 unequal; max 1.4210854715202004e-14 | 0 / 228,641 | CONFIRMED |
| fused-expression T control | 57,141 unequal | 57,141 unequal | control fires |
| fused-expression S control | 57,169 unequal | 57,169 unequal | control fires |
| Kaa input | 0 / 8,613 unequal | 0 / 8,613 | unchanged exact |
| interpolated-SSH Kaa control | 2 / 8,613 unequal | 2 / 8,613 unequal | control fires |

The committed eager/JIT test also proves a finite non-zero gradient.  The
record-backed one-cell plant exits non-zero with `REFUSE`.

## ORCA2 ten-step ladder — GIVEN NEMO'S ENTRY

All 40 checkpoints and 200 field rows completed.  Of 189 moved rows, 96 moved
toward NEMO by maximum, 62 away, and 31 retained the same maximum.  No AT-BAR
row left the bar.  The first whole-card non-bit checkpoint remains kt=1
stage-1 T: 233,341 / 399,600 unequal, maximum
0.0014770192519700243 K, still `UNATTRIBUTED`.

Thus the direct statement closes, but it does not own the first whole-card
divergence.

## Shared-card gates

The primary Decision 43/45/55/59 gate passed, but the separately mandatory
OVERFLOW trajectory non-regression gate failed.

| check | result |
|---|---|
| GYRE ten-step admission | 70 rows; 55 moved and all registered; first-over-bar remains kt=3; no kt=1 AT-BAR loss |
| GYRE strict 2-ULP diagnostic | FAIL, max 133,563.75 row-scale ULP; non-binding here because the standing gate explicitly registers all moved GYRE rows |
| GYRE day 30 T rms | 6.572572612618985e-05 -> 6.572572400342862e-05 K; improves by 0.0106138 floor units |
| GYRE day 240 T rms | 1.644836113029585e-02 -> 1.644836072695681e-02 K; improves by 2.016695 floor units |
| GYRE day 360 T rms | 1.122565978973451e-02 -> 1.122565989376804e-02 K; worsens by 0.520168 floor units, inside the strict ten-floor-unit allowance |
| LOCK_EXCHANGE-zco | PASS; first-over-bar remains kt=4 U; max worsening 1.009463 ULP |
| generic NEMO-GYRE | PASS; 0 moved rows through three steps |
| DINO recipes | both use Euler and do not execute this WS-RK3 statement |
| OVERFLOW-zps | **FAIL**; first-over-bar remains kt=2 T/U, but five later U rows exceed the 2-ULP non-regression bar |

The blocking OVERFLOW rows are kt6 U cell 668 (4.783 ULP), kt7 U cell 615
(2.180 ULP), kt8 U cell 668 (4.783 ULP), kt9 U cell 450 (57.812 ULP),
and kt10 U cell 450 (57.625 ULP).  No row changes status, but this gate has no
moved-row allowance.  The candidate therefore cannot land.  The GYRE
day-240 worsening plant fires as required.

## Preregistered predictions

| prediction | result |
|---|---|
| direct production stage-1 T/S become bit-exact | CONFIRMED |
| ORCA2 has no AT-BAR loss and first non-bit is not earlier | CONFIRMED |
| full GYRE Decision 43/45/55/59 gate passes | CONFIRMED |
| tanks and generic cards do not regress | **REFUTED** by the five OVERFLOW U rows above |
| candidate lands | **REFUTED**; candidate reverted |

The failed predictions remain frozen in the preregistration.

## Tests, citations, and review

- New source-order and recipe-census nodes: **2 passed**; the direct statement
  test had also passed independently before the combined battery stalled.
- Shared-card batteries: **160 passed**, then **10 passed**.
- `tests/ocean/fidelity -n 12` reached 99% and reproduced the documented xdist
  controller stall; it was interrupted and is not called green.  The same five
  pre-existing reds were rerun serially and retained their prior signatures:
  round-129 stale certification, round-51 private trace registry, SI3 scalar
  provenance, worktree-stamp ratchet, and the `hires_lane_surface` case-board
  ratchet.
- Default and round-48 receipt citation gates pass; the shifted-line plant
  fires.
- Separate read-only `codex exec` review: **independent review unavailable
  in-sandbox** (`failed to initialize in-process app-server client: Read-only
  file system`).

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round48/`.
The decisive files are `qco_base.json`, `qco_tip.json`, `orca2_compare.json`,
`gyre_ladder_compare.json`, `decision43_45_55_59.json`,
`overflow_compare.json`, `lock_compare.json`, and `generic_compare.json`.

## OPEN

1. The source-exact QCO/RK statement is locally correct but cannot land alone.
   Preregister a cancelling-pair analysis for the downstream OVERFLOW U change;
   identify the first shared statement that cancels those five row movements
   before retrying this candidate as a measured pair.
2. The ORCA2 first non-bit checkpoint remains kt=1 stage-1 T and
   `UNATTRIBUTED`; continue its compiled producer walk independently of the
   held pair.
3. The round-20 slow-forcing/barotropic walk remains open.
4. Build and bit-gate the **INDEPENDENT** ORCA2 initial T/S/SSH state required
   by Decision 52 before switching the year comparison away from NEMO's entry.
5. Sea ice remains exactly the card's current `unmeasured_features` tuple at
   `STOP_SELECTOR_GAP`.
