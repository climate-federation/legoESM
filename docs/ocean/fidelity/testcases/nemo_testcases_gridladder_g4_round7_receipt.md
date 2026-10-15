# Grid ladder — G4 round 7 (gridladder_r25)

Status: **LANDED, DEBT. G4 is not certified.** Stage 1 is NEMO's in every
value after the source-ordered tracer update landed. GIVEN-NEMO-ENTRY is
AT-BAR through kt 10, but the INDEPENDENT chain first exceeds the exact bar
in u and v at kt 6; ssh follows at kt 8. The certification condition (every
field at the bar under both labels) is therefore not met.

Code commits: `20cbdf5e86` (housekeeping), `bb2cabcd17` (preregistration),
`d3bea91071` (tracer statement), `7bc7e0d09e` and `69435a4039` (G4 ladder
reader), `1f482bf9a` and `21d9f08a4` (citation gate), and `ce6eb35cc`
(mechanical historical-receipt reanchor). Record:
`gridladder_rounds/round21/oracle_g4_r21`. Evidence:
`gridladder_rounds/round25/`.

## 1. Housekeeping first

The exact ratchet initially failed 1 test and passed 5. Its only violation was
the production import of `_nemo_een_ratio_fold_exchange` from
`ocean_model_latlon_cgrid.py`. The symbol is now public at
`barotropic_latlon_cgrid.py:4144`, and
the production importer/call are public at
`ocean_model_latlon_cgrid.py:6508` and `ocean_model_latlon_cgrid.py:6517`.
The allowlist remains empty. Focused helper plus ratchet after promotion:
8 passed; the final combined focused battery below also includes all six
ratchet tests and passes.

`tests/test_dispatch_hardening.py` remains red only at the handed pre-campaign
`packages/land/legoesm/land/slab_land.py::step_land` row: 1 failed, 23 passed.
It was measured and left unchanged as instructed.

## 2. Registered stage-1 walk

The preregistration froze the round-23 candidate: the nonlinear-SBC divisor
at `ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/trasbc.f90:284` and
`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/trasbc.f90:285`. Its identical
Decision-122 line is
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:284` and
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:285`. Replacing the card's entry
top thickness by that written NEMO form moved **no output bit**: T remained
58 unequal values (max 3.552713678800501e-15), S remained 45 (max
7.105427357601002e-15). Prediction 3 in the preregistration is **REFUTED**;
the candidate was reverted and not landed.

Continuing in compiled order, the first non-bit statement is the stage-1/2
thickness-weighted tracer update at
`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/stprk3_stg.f90:503-505`. The compound
statement's products and sum were being contracted by XLA. Materializing its
written association closes T 58 -> 0 and S 45 -> 0; the implementation is
`ocean_model_latlon_cgrid.py:2519-2523`.
The deselected control restores exactly 58/45, so the option can fail.

| stage | ssh | uu_b / vv_b | u | v | T | S |
|---|---:|---:|---:|---:|---:|---:|
| 1 before | 0 | 0 / 0 | 56 signed-zero bits / 0 values | 0 | 58 / 3.552713678800501e-15 | 45 / 7.105427357601002e-15 |
| 1 after | 0 | 0 / 0 | 56 signed-zero bits / 0 values | 0 | 0 | 0 |
| 2 after | 0 | 0 / 0 | 56 signed-zero bits / 0 values | 0 | 0 | 0 |
| 3 after | 0 | 0 / 0 | 56 signed-zero bits / 0 values | 0 | 489 / 3.552713678800501e-15 | 0 |

Counts are bit-unequal cells / maximum absolute difference unless stated.
Every value at stages 1 and 2 is identical; the 56 u entries differ only in
the sign bit of zero. Stage-3 T is not bit-identical but is AT-BAR.

## 3. Selection and Decision 122

`rk3_stage_tracer_written_order` is default False
(`state.py:3232`) and only G4 selects it;
G1, G1b, G2 and G3 remain False
(`nemo_testcase_recipe.py:4015-4020`).
The card assignment is at
`nemo_testcase_recipe.py:3806`.

Decision 122 points to the same live statement in the full ORCA2 build:
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:678-680`. No exact ORCA2 row
is lost by the default-off option; the ORCA2 card remains deselected and is
not claimed measured by this round.

## 4. Ten-step ladders

The committed ladder injects NEMO's recorded freshwater and surface forcing
at each kt (`nemo_testcase_g2_orca2bathy_ladder.py:558-565`).
Bar = normalized maximum absolute error <= 1e-15. In the tables below each
cell is `status max_abs/unequal`; A = AT-BAR, D = DEBT. T/S surface fields at
kt 10 have no kt 11 entry frame, so the kt 10 restart supplies their all-level
depth rows instead.

### INDEPENDENT

| kt | ssh | uu_b | vv_b | u | v | T | S |
|---:|---|---|---|---|---|---|---|
| 1 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 3.553e-15/489 | A 0/0 |
| 2 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 7.105e-15/772 | A 0/0 |
| 3 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 1.421e-14/997 | A 0/0 |
| 4 | A 1.998e-15/280 | A 1.388e-16/277 | A 1.197e-16/276 | A 1.263e-15/176 | A 1.291e-15/190 | A 1.776e-14/1172 | A 0/0 |
| 5 | A 1.221e-15/1612 | A 4.441e-16/1783 | A 2.498e-16/1843 | A 1.110e-15/1310 | A 1.138e-15/1366 | A 1.776e-14/1335 | A 0/0 |
| 6 | A 1.998e-15/3852 | A 3.886e-16/4240 | A 4.441e-16/4317 | D 2.723e-14/3001 | D 1.089e-13/3244 | A 1.776e-14/1484 | A 7.105e-15/3 |
| 7 | A 2.054e-15/6139 | A 6.106e-16/6531 | A 6.939e-16/6529 | D 2.737e-14/4844 | D 1.089e-13/4972 | A 1.776e-14/1618 | A 7.105e-15/9 |
| 8 | D 2.454e-14/9656 | A 6.731e-16/10310 | A 8.882e-16/10252 | D 6.360e-14/7188 | D 1.089e-13/7322 | A 1.776e-14/1728 | A 7.105e-15/10 |
| 9 | D 2.420e-14/12384 | A 1.332e-15/13015 | A 1.308e-15/12934 | D 5.444e-14/9153 | D 1.089e-13/9471 | A 1.776e-14/1811 | A 1.066e-14/15 |
| 10 | D 5.729e-14/13089 | D 1.796e-14/13648 | D 2.345e-14/13633 | D 4.807e-14/10464 | D 1.089e-13/10727 | restart below | restart below |

Surface rows: 68 total, 53 AT-BAR, 15 DEBT. All-level depth rows: 36 total,
26 AT-BAR, 10 DEBT. First debt is kt 6 u/v. At kt 6 their whole-domain
normalized maxima are 5.727011341529318e-15 and 1.9802115519243914e-14.

### GIVEN-NEMO-ENTRY

Entries 1/3/5/7/9 plus each two-step successor cover kt 1..10.

| kt | ssh | uu_b | vv_b | u | v | T | S |
|---:|---|---|---|---|---|---|---|
| 1 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 3.553e-15/489 | A 0/0 |
| 2 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 7.105e-15/772 | A 0/0 |
| 3 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 7.105e-15/513 | A 0/0 |
| 4 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 7.105e-15/786 | A 0/0 |
| 5 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 7.105e-15/529 | A 0/0 |
| 6 | A 0/0 | A 0/0 | A 0/0 | A 5.551e-17/7 | A 5.551e-17/5 | A 1.066e-14/855 | A 0/0 |
| 7 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 7.105e-15/586 | A 0/0 |
| 8 | A 0/0 | A 0/0 | A 0/0 | A 5.551e-16/2 | A 7.216e-16/2 | A 1.421e-14/886 | A 0/0 |
| 9 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 7.105e-15/548 | A 0/0 |
| 10 | A 0/0 | A 0/0 | A 0/0 | A 2.220e-16/2 | A 4.441e-16/2 | restart below | restart below |

Surface rows: 68/68 AT-BAR. All-level depth rows: 20/20 AT-BAR.

## 5. Region and depth-class audit

Every surface row above is split by the listed region. `debt rows` counts
region/field/kt rows whose normalized maximum exceeds 1e-15; `worst` is the
largest normalized maximum, followed by its kt/field, unequal cells and raw
maximum absolute difference.

| label | region | rows | debt rows | worst normalized | at | unequal / max abs |
|---|---|---:|---:|---:|---|---|
| independent | fold | 68 | 0 | 7.428e-17 | kt9 T | 53 / 2.220e-15 |
| independent | seam | 68 | 0 | 4.026e-16 | kt8 ssh | 146 / 1.110e-15 |
| independent | south | 68 | 0 | 9.938e-16 | kt7 u | 37 / 4.594e-15 |
| independent | coast | 68 | 14 | 1.014e-14 | kt10 v | 897 / 5.359e-14 |
| independent | open | 68 | 15 | 2.631e-14 | kt7 v | 4563 / 1.089e-13 |
| independent | fold pivot | 68 | 0 | 5.942e-17 | kt9 T | 16 / 1.776e-15 |
| independent | fold row -1 | 68 | 0 | 5.942e-17 | kt9 T | 21 / 1.776e-15 |
| independent | fold row -2 | 68 | 0 | 7.428e-17 | kt9 T | 16 / 2.220e-15 |
| given | fold | 68 | 0 | 2.971e-17 | kt9 T | 10 / 8.882e-16 |
| given | seam | 68 | 0 | 1.188e-16 | kt9 T | 8 / 3.553e-15 |
| given | south | 68 | 0 | 7.396e-18 | kt4 T | 7 / 2.220e-16 |
| given | coast | 68 | 0 | 2.377e-16 | kt9 T | 90 / 7.105e-15 |
| given | open | 68 | 0 | 4.640e-16 | kt8 T | 699 / 1.421e-14 |
| given | fold pivot | 68 | 0 | 2.971e-17 | kt9 T | 4 / 8.882e-16 |
| given | fold row -1 | 68 | 0 | 2.971e-17 | kt9 T | 5 / 8.882e-16 |
| given | fold row -2 | 68 | 0 | 2.951e-17 | kt2 T | 8 / 8.882e-16 |

Depth classes are scored on restarts kt 2..10 (independent) and the five
two-step successors (given). `worst` is relative error; parentheses give
kt, unequal cells, worst level.

| label | field/class | rows | debt rows | worst relative (kt, unequal, level) |
|---|---|---:|---:|---|
| independent | u full | 9 | 5 | 2.105e-14 (10, 327206, 28) |
| independent | u partial | 9 | 5 | 2.113e-14 (10, 12567, 29) |
| independent | v full | 9 | 5 | 1.980e-14 (6, 101081, 1) |
| independent | v partial | 9 | 3 | 1.324e-14 (10, 12335, 28) |
| independent | T full | 9 | 0 | 5.942e-16 (9, 2410, 1) |
| independent | T partial | 9 | 0 | 1.188e-16 (9, 9, 15) |
| independent | S full | 9 | 0 | 6.999e-16 (9, 244, 4) |
| independent | S partial | 9 | 0 | 1.751e-16 (8, 8, 5) |
| given | u full | 5 | 0 | 1.296e-16 (8, 37, 1) |
| given | u partial | 5 | 0 | 1.621e-18 (8, 2, 28) |
| given | v full | 5 | 0 | 9.691e-17 (8, 36, 1) |
| given | v partial | 5 | 0 | 3.727e-18 (8, 2, 25) |
| given | T full | 5 | 0 | 4.640e-16 (8, 1003, 1) |
| given | T partial | 5 | 0 | 0 (10, 0, 1) |
| given | S full | 5 | 0 | 1.750e-16 (6, 1, 9) |
| given | S partial | 5 | 0 | 0 (10, 0, 1) |

The complete JSON artifacts retain every region metric, every per-level row,
cell count, location and raw value; the tables above are their lossless
verdict/worst-case index, not a replacement for them.

## 6. Verdict and next open

The stage-1 OPEN from round 23 is closed: the first non-bit statement was the
tracer stage update, not the plausible SBC divisor. G4 is **DEBT**, not
CERTIFIED, because INDEPENDENT u/v cross the bar at kt 6 and subsequently
ssh and both barotropic transports cross it. The locus is interior coast/open;
fold, seam and south remain at-bar. GIVEN-NEMO-ENTRY staying at-bar through
two steps while the independent chain crosses at step 6 localizes the next
open to compounding state error rather than a one-step forced-entry defect.
No new owner is claimed.

## 7. Tests and review

- Focused fast: 45 passed, 9 slow deselected (19.43 s).
- Slow G4 stage/control route: 1 passed (242.97 s). The control restores
  58 T and 45 S stage-1 mismatches.
- Receipt citation tests: 17 passed (5.01 s). The round-25 receipt gate passes
  all 14 citations with zero failures, zero unmapped citations and zero map
  audit failures; shifting `ocean_model_latlon_cgrid.py:2519-2523` exits 1
  with `SYMBOL-NOT-AT-LINE`.
- Dispatch hardening: 23 passed, 1 handed pre-existing failure
  (`slab_land.step_land`), unchanged.
- A larger G1 ladder attempt earlier in the round hit the documented JAX
  compiler-process abort after printing two failures; it was split rather
  than assigned a physics verdict.

Independent review: operator (Claude).

## 8. Process, evidence and limitations

ASKED: housekeeping ratchet; compiled-order stage-1 walk; G4-only cited
transcription; Decision-122 pointer; both kt 1..10 labels with region/depth
splits; focused tests; receipt.

UNASKED, offered for revert: the validator's G4 routing through its existing
inner-field reader and contiguous-frame counter. Both are fail-closed and
needed because G4 records inner NCAR fields and alternates bulk-frame sizes.

GitHub issue #1455 could not be read or updated: this clone's only remote is a
local filesystem path, not a GitHub repository. No external comment was
silently omitted.

The requested clone's `.git` directory was read-only in this sandbox, so the
commits were made in the writable clean clone `/tmp/gridladder_r25_codex`
created at the exact requested base `174f6966cf`. The original requested
working tree was not used for commits.

UNVERIFIED: the owner of the independent kt-6 compounding momentum debt; the
full ORCA2 card's response to this default-off option (Decision 122 pointer
only).
