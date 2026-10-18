# Grid ladder rung G9 — round 2 receipt (gridladder_r47)

Status: **LANDED, G9 NOT CERTIFIED.**  The stage-3 FCT now reads the T-pivot
fold north of the pivot row (G9-selected only).  kt-1 T and S on the fold
rows go from ~2.4e-4 off to within the bar.  One kt-1 row stays DEBT:
meridional velocity, 1.46e-15 on 59 interior cells, not on the fold.

- Record: round 46's (`gridladder_rounds/round46/oracle_g9_r46`), unchanged.
  No new acquisition: the record decided the FCT rows.
- Landing commit `5070601653`.  Scores: `gridladder_rounds/round47/score/final/`.
- codex review: paused (quota).  The operator runs an independent Claude
  review of this diff before landing.
- No preregistration commit this round: the walk used round 46's record and
  went straight to the stage-level replay.  Stated, not hidden.

## 1. NEMO's FCT as compiled, fold-sensitive statements in order

Source: `ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90` (`cmp`: byte-identical
to the record build `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1`).  The run has
`nn_hls = 2`, `ln_zad_Aimp = F`, `nn_fct_h = nn_fct_v = 2`, `nn_fct_imp = 1`
(the record's `ocean.output`).  `tra_adv` itself has no exchange.

| # | NEMO statement | what it does on the fold | card before this round |
|---|---|---|---|
| 1 | upstream fluxes, step 1 (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:503`) | loop reaches the halo row, so the face north of the pivot row is the fold image of the face below | wall: flux 0 |
| 2 | step-1 guess on the halo (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:538`) | guess above the pivot row = image | (follows 1) |
| 3 | averaged upstream fluxes (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:571`) | pivot face reads the halo guess: image | wall |
| 4 | antidiffusive flux (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:198`) | computed on the pivot face, then overwritten by 5 | wall placeholder |
| 5 | `lbc_lnk(ztFu 'U' -1, ztFv 'V' -1, zta_up1 'T' +1)` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:311`) | pivot V faces = image of the row below; second half of the pivot row's U flux and guess = mirror of the first half | none |
| 6 | `nonosc` bounds on the halo (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:817`) | pivot row's north neighbour in the 7-point bound = image of the row below | copy of the pivot row |
| 7 | `nonosc` face coefficient (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:910`) | pivot face reads the halo betas: equals the image face's coefficient | 1 (unlimited) |
| 8 | final trend (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:327`) | no exchange | — |

Measured on the card's own stage-3 inputs: the pivot-row second halves
required by statement 5 (U flux, guess) already equal the mirror of the
first halves, 0 cells differ.  So only the north-face parts are
transcribed; the half-row overwrite would be a no-op and its test vacuous.

## 2. First use of the existing record

Instrument: the card's stage-3 FCT inputs (private hook), replayed offline
through the shared FCT, against NEMO's stage-3 RHS after `tra_adv`
(record field 2/3).  Control: off the fold rows the replay agrees to
≤1e-13 normalised everywhere (only the trace's divisor form differs).

| arm | cells >1e-13 normalised | worst | fold rows 145–147 >1e-15 |
|---|---|---|---|
| card before | T 1,343 / S 1,344, rows 145–147 only | 3.3e-4 | 1,343 / 1,344 |
| with statements 1, 3, 5, 6, 7 | 0 / 0 | 1.9e-14 / 2.1e-14 | 0 / 0 |

Pivot V-face metric and transport are exact fold images on this grid
(`dx_v` top row = permuted row below; stage-3 transport top row = −image,
0 cells differ), so imaging the per-length flux is imaging NEMO's transport.

## 3. What changed (`5070601653`)

- Shared FCT gains `nemo_fold_exchange` (default False): five wiring lines on
  existing statements, using the existing fold-row helper.  Refuses grids
  without an active pivot-row fold.
- New config field `fct_fold_exchange`, default False, threaded to the stage-3
  FCT call.  Only G9 selects it, with selector row `FCT fold exchange`.
- Searched first: `fold`, `lbc_lnk`, `nonosc`, `fold_row`, `fold_ghost_source_T`,
  `upwind_cell_to_vface`, `nemo_een_ratio_fold_exchange`,
  `_rk3_fold_exchange_top_v` in `advection.py`, the grid operators and the
  model.  The FCT had no fold handling (wall north face); the fold-row helper
  was reused.
- Instrument: the ladder's stage-2 operator walk now also accepts G9.
- Other cards: traced step jaxpr, addresses masked, unchanged — G5
  `339a8016d262c4f6`, G8 `03ccdb94113eab61` (`round47/jaxpr/`).

## 4. Score at `5070601653` (both labels, kt 1..10)

Row = ssh, uu_b, vv_b, u, v, T, S (level 1 or 2-D); levels = all-level rows.
Flushed subnormals 0 everywhere; dry-cell mismatches 0 everywhere.

| label | kt | exact / AT-BAR / DEBT | worst row | T, S rows | round 46 worst |
|---|---|---|---|---|---|
| independent | 1 | 3 / 3 / 1 | v 1.46e-15 | 4.7e-16, 3.5e-16 | S 5.9e-6 |
| independent | 2..5 | 0 / 0 / 7 | v or ssh, 3.4e-14 → 8.2e-14 | ≤1.1e-14 | 2.9e-5 → 8.6e-5 |
| independent | 6..10 | 0 / 0 / 7 | v, 7.4e-14 → 3.5e-13 | — | 1.1e-4 → 3.2e-4 |
| given-NEMO-entry | 1, 3 | 3 / 3 / 1 | v 1.46e-15, 1.15e-15 | ≤7.0e-16 | T 5.3e-5 |
| given-NEMO-entry | 5, 7, 9 | 3 / 2 / 2 | u, v ≤ 8.7e-15 | ≤8.8e-16 | S ≤ 9.5e-5 |
| given-NEMO-entry | two-step 2..10 | 0 / 0 / 7 | ≤ 4.0e-14 | ≤ 9.6e-15 | ≤ 1.1e-4 |

kt-3 levels (given): u, T, S AT-BAR, v DEBT 1.1e-15.  Every T and S row
improved by 9–11 orders of magnitude; no row got worse.

Score SHA-256 (first 16): `indep_kt1_5.json` `0a6d6bf6e522c1e1`,
`indep_kt6_10.json` `da0e2712ec180b81`, `given_e135.json` `f7e380e8fcb31cae`,
`given_e79.json` `860e14d54dade69f`.

**Not certified**, so this does not certify the stage-1 V transport mask for
the ORCA2 lane.  What it shows: on G9's record, the fold-associated stage-1
mask, the two round-46 exchanges and these FCT statements together put every
kt-1 T and S cell, fold rows included, within the bar.  `traadv_fct.f90` is
byte-identical in ORCA2_OMIP_L4, so a Decision-122 import has the same lines.

## 5. The kt-1 meridional-velocity debt (review finding, located)

- 59 cells, all interior (fold 0, seam 0, south 0), first at (16, 47), worst
  at (17, 88).  Not downstream of the FCT debt.
- Stage walk (ladder `stages` arm): stage 1 u, v exact at every level;
  stage 2 u 394 / v 483 cells (22 at level 1), DEBT 1.3e-15.  So v first
  leaves NEMO at stage 2.
- Stage 1 T (582 cells) and S (115 cells) already differ in the last place,
  under the bar.  All 394 + 483 stage-2 velocity cells sit in or next to
  those tracer columns, in V-face pairs either side of one column — the
  pressure-gradient footprint.  Stage-2 HPG: 48,003 u cells differ, ≤2.7e-19.
  PLAUSIBLE (colocation, no perturbation test yet): the v debt is the
  stage-1 tracer last place carried through the stage-2 density.
- Next statement to walk: the stage-1 tracer update.  Round 46 found the
  card's stage-1 vertical transport off on ~271k cells (last place).
  This round's offline CEN2 replay on NEMO's own transports is also last-place
  off (122k cells, 4e-16), so its geometry inputs (`e3t`, `r1_e1e2t` from the
  mesh) must be validated before it can split the two.  The existing record
  holds what that walk needs (stage-1 transports and RHS), so no acquisition.

## 6. Tests and plants (CPU, fp64)

New value test `test_g9_stage3_fct_reads_the_fold_on_the_pivot_rows`: kt 1
of the jitted G9 step (state traced, x64) on NEMO's surface input.  After
stage 3, T and S on rows 145–147, every level, must be within 1e-15
normalised of NEMO's.

Plants: scratch worktrees at `1e16b68fb` + this diff with one wiring line
removed (`round47/plants/*.log`):

| copy | result (T, normalised) |
|---|---|
| restored | 1 passed |
| model wiring off | failed 5.3e-6 |
| step-1 upstream image off | failed 7.0e-9 |
| averaged upstream image off | failed 5.9e-5 |
| antidiffusive image off (`:311`) | failed 5.3e-6 |
| bound north neighbour off | failed 5.3e-6 |
| face coefficient image off | failed 7.1e-7 |

Gates (`JAX_ENABLE_X64=1`, `land.sh` PYTHONPATH):

| battery | result |
|---|---|
| G9 file | 15 passed (after the field-list update) |
| G2 ladder file + recipe unit tests | 37 passed |
| default gate | section 7 |

## 7. Default gate and receipt citation gate

Results in `gridladder_rounds/round47/final/`.

## Choices

ASKED: G9's definition and record (the brief); the bar and labels (D127).

UNASKED, offered for revert:
- the new field defaults False and only G9 selects it (no-silent-default rule);
  importing it to ORCA2 is a Decision 122 question;
- statement 5's pivot-row half-overwrite is not transcribed (measured no-op);
- the five north-face statements share one field;
- the ladder's operator-walk guard widened to G9 (instrument only).

UNVERIFIED: the cause of the stage-1 tracer last place (section 5);
the stage-2 v debt's link to it (colocation only).
