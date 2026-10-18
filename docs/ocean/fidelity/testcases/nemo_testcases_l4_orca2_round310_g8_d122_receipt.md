# ORCA2 round 310 — rung G8's lateral-momentum statements under Decision 122

Date: 2026-10-17 (Claude round; codex review: paused (quota)).

Status: **LANDED-pending** (operator review + push gate).

## Headline

- No ORCA2 card differs from its NEMO deck in lateral momentum diffusion, so
  there is no settings decision.  Rung 0, OMT-3, OMT-4 and rung 10 run
  ORCA2_OMIP_L4's laplacian, iso-level, file coefficient; OMT-0..2 run none,
  as their reduced decks say.
- Two of G8's four statements land on rung 0, OMT-3, OMT-4 and rung 10: the
  stored F area in the curl term and stage 3's interpolated F ratio.
- The fold-row wiring passes D122 but leans away on every card and grows
  rung 10's kt-10 sea-level error from 0.277 to 0.328 m.  It stays off;
  the user is asked.
- The stage-1 order stays off: on these vector cards its stage-1 block drops
  G1b's certified KE-gradient statement (it moves 117-165 rows on OMT-1/2,
  where no diffusion runs).

## Provenance

- Base `4b3442ff0` (round 309 landed).  Import `9313b333f`; preregistration
  + gate `6411d7e06`, addendum `bc679702c` (each frozen before the runs it
  covers); selection `1c4023e0f`; test `9f0e63900`.
- Import = package hunks of gridladder `50a49037fa..2b75714433` (rung G8,
  receipt `nemo_testcases_gridladder_g8_round1`, addendum 106), `git apply
  --3way`: one conflict in the config field list, ordering only (both kept).
  Not imported: the G-card recipe hunks (this branch has no G8 rung table).
  All four switches import default-off.
- Gate: `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round310_g8_d122_gate.py`
  (round-285 ladder, round-276 classifier, round-278 decision).  CPU, fp64,
  libm, production JIT.  Evidence `orca2_rounds/round310/` (`arms/`,
  `arms2/`, `final/`).

## Step 1 — card settings vs NEMO's deck

ORCA2_OMIP_L4's deck: laplacian, iso-level, coefficient from
eddy_viscosity_3D.nc (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:371-373`).

| card | deck | card operator / direction / coefficient | A_h | matches |
|---|---|---|---:|---|
| rung 0, OMT-3, OMT-4, rung 10 | lap, lev, -30 | NEMO div-curl laplacian, level, e3-weighted, file (148x180x30, max 4e4) | 1e5 (selector only) | yes |
| OMT-0, OMT-1, OMT-2 | ln_dynldf_off | none | 0 | yes (reduced deck) |

No biharmonic, Smagorinsky or Leith term on any card; free slip
everywhere.  Read from the resolved configs (`card_ldf.txt`) and each
card's own acquisition deck.

## Step 2 — one D122 row per G8 statement, seven pairs

(a) G8 receipt cited above.  (b) The same compiled lines run in
ORCA2_OMIP_L4: the curl term `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynldf_lev.f90:123`
reads `ORCA2_OMIP_L4/BLD/ppsrc/nemo/domhgr.f90:156`'s stored reciprocal and
the stage ratio from `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:245`;
stage 3 calls it at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:493`;
stage 1 orders `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:141`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:144`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:147`.  On OMT-0..2 NEMO never
reaches the diffusion routine, so (b) fails there.

Census = rows toward/away/equal; lost = exact rows lost; first over-bar row
unchanged on every pair (rung 0/OMT kt-1 stage-1 T; rung 10 independent
kt-1 entry ssh, given entry kt-1 stage-1 T).  Max |ssh err| = kt-10
stage-3 maximum |card − NEMO| sea level, control -> arm.

| statement | rung 0 | OMT-1 | OMT-2 | OMT-3 | OMT-4 | rung 10 indep | rung 10 given | lost | D122 (c)+(d) | lands |
|---|---|---|---|---|---|---|---|---:|---|---|
| stored F area | 144/31/0 | 0 moved | 0 moved | 134/41/0 | 152/23/0 | 159/16/0 | 149/26/0 | 0 | PASS | yes (not OMT-1/2: (b)) |
| stage-3 F ratio | 43/45/47 | 0 moved | 0 moved | 12/11/77 | 59/30/55 | 42/55/31 | 32/48/25 | 0 | PASS | yes (not OMT-1/2: (b)) |
| fold rows | 23/152/0 | 0 moved | 0 moved | 10/165/0 | 21/154/0 | 35/140/0 | 35/140/0 | 0 | PASS | no — asked |
| stage-1 order | 52/105/17 | 26/22/69 | 29/39/97 | 53/57/64 | 54/101/19 | 93/76/4 | 99/64/11 | 0 | PASS | no — drops G1b's KEG statement |
| all four | 59/116/0 | 26/22/69 | 29/39/97 | 48/127/0 | 64/111/0 | 76/99/0 | 75/100/0 | 0 | PASS | — |
| **SEL (area + ratio)** | 144/31/0 | — | — | 134/41/0 | 152/23/0 | 159/16/0 | 149/26/0 | 0 | PASS (per pair) | **landed** |

Max |ssh err| (m), control -> arm:

| statement | rung 0 | OMT-3 | OMT-4 | rung 10 indep | rung 10 given |
|---|---|---|---|---|---|
| stored F area | 0.394008 -> 0.394173 | 0.363164 -> 0.363341 | 0.393839 -> 0.394004 | 0.276969 -> 0.277238 | 0.261450 -> 0.261719 |
| stage-3 F ratio | unchanged | unchanged | unchanged | unchanged | unchanged |
| fold rows | 0.394008 -> 0.388496 | 0.363164 -> 0.357043 | 0.393839 -> 0.388327 | 0.276969 -> 0.327650 | 0.261450 -> 0.312126 |
| stage-1 order | unchanged | unchanged | unchanged | unchanged | unchanged |

Seven-pair D122 decision (round-278 classifier): PASS for each single
statement and for all four.  SEL ran on the five diffusion pairs (OMT-1/2
leave it off; each of its two statements moves 0 rows there): 175/200 rows
moved on each, max |ssh err| as the stored F area's row; its direction
census equals the area's, though 76-125 of its rows differ from the area
arm's (the stage-3 ratio's contribution).  SEL ran on `bc679702c`; its
controls were rerun there and reproduce the first controls row for row.  Rows moved: area and fold rows 175/200 on each
diffusion card; stage-3 ratio 100-144/200; OMT-1 has 160 rows (8 steps).

**Fold rows.**  Early rows (kt 1-3) lean away on every diffusion card
(3-5 toward / 30-32 away).  G8 certified them without tracer advection and
without G6's fold statements, which the ORCA2 cards still run off.

**Stage-1 order.**  Its stage-1 block adds the KE gradient as a plain sum;
every vector card selects G1b's contraction guard for that statement
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynkeg.f90:129-130`).  So on these cards it
changes the KE-gradient statement too, not only where the diffusion
enters: it moves 117/160 rows on OMT-1 and 165/200 on OMT-2, which run no
diffusion.  G8 has no momentum advection, so no rung certifies the
combination.  Keeping the guard inside that block would be new code, not an
import.

## Shipped cards (selection tree `1c4023e0f`)

- Resolved configs vs `bc679702c` (572 fields; rung 10 635): rung 0, OMT-3,
  OMT-4, rung 10 differ in the two landed fields only; OMT-0..2 identical.
- Shipped rung 0, OMT-3, OMT-4 and rung 10 (both labels) reproduce the SEL
  arm row for row; shipped OMT-1/OMT-2 reproduce their controls row for row
  (`final/instrument_check.txt`).  OMT-0's config is unchanged.

## GYRE, DINO, VORTEX

Base = round 309's identity tree `1c8615b64` (no package change to
`4b3442ff0`); tip `9313b333f`.

- Step jaxprs identical: DINO (123,236 lines), VORTEX zco flux/vector,
  VORTEX seamount flux/vector.
- GYRE ten-step residual archive identical (210 arrays, `np.array_equal`;
  sha256 `3b3d865cf5c82d38` both); day 30: 30/30 snapshot files
  byte-identical.

## Predictions

| ID | Verdict | Result |
|---|---|---|
| R310-P1 | **CONFIRMED** | area, fold rows, stage-3 ratio move 0 rows on OMT-1 and OMT-2. |
| R310-P2 | **CONFIRMED** | each statement and all four pass (c)+(d) on rung 0, OMT-3, OMT-4, rung 10 both labels; no refusal. |
| R310-P3 | **CONFIRMED** | area and fold rows move 175/200 rows on each diffusion card. |
| R310-P4 | **CONFIRMED** | GYRE ten-step and day 30, DINO and VORTEX identical. |
| R310-P5 | **CONFIRMED** | SEL passes (c)+(d) on rung 0, OMT-3, OMT-4, rung 10 both labels (0 exact rows lost, first over-bar unchanged). |

## Gates and tests

- New `tests/ocean/fidelity/test_orca2_round310_selection.py` (11 tests):
  reverting the recipe selection fails 6; tree restored, `git status`
  clean.  It also checks that each landed switch changes the traced rung-0
  step.
- Census plant (`--plant exact-loss`) fires, exit 2.
- Neighbouring ORCA2 selection/transfer, OMT-3 ladder, ORCA2 ldf and
  lateral-viscosity unit tests: 171 passed, 2 skipped.
- Receipt citation gate on this receipt: PASS, 9 citations, failures 0,
  0 map entries failing (81 entries shifted by the import and selection
  re-anchored; one range extent 64 -> 66 lines, as on the gridladder).
- Default gate (land.sh set, JAX_ENABLE_X64=1, in pieces under the
  10-minute cap): citation gate + private-import ratchet + TKE + recipe +
  freshwater + ORCA2 parallel receipt + barotropic state 143 passed;
  face-mask rank 2 passed.
- codex review: paused (quota).  The operator runs an independent Claude
  review before landing.

## Configuration choices

- ASKED (D122 + addendum 106, taken): the stored F area and stage-3 ratio on
  the cards whose deck runs the diffusion.
- UNASKED: none.  Leaving the fold rows and the stage-1 order off is
  holding back, not selecting; the fold rows go to the user below.  No
  runoff (D126), TKE switch, double diffusion or G6 fold statement changed.

## OPEN

1. DECISION: land G8's fold rows on rung 0, OMT-3, OMT-4, rung 10 (D122
   passes; leans away; rung 10 kt-10 sea-level error 0.277 -> 0.328 m), or
   keep them off until the G6 fold statements and the stage-1 V transport
   mask land?  Pick: keep off.
2. Stage-1 order with vector momentum advection needs a rung (G6 + G8) or
   a gridladder change that keeps G1b's KE-gradient guard in that block.
3. Unchanged: D126, D135, double diffusion (V4), fold statements (G9).
