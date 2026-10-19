# Grid ladder rung G6 — round 5 receipt (gridladder_r40)

Status: **LANDED (pivot V-row ratio fold exchange, G6-selected); kt 1 is
AT-BAR on every field under both labels; G6 NOT CERTIFIED** — kt 2..6 stay
whole-domain debt whose owner waits on Decision 132 (unchanged by this round,
not touched).

Preregistration: `PREREG_nemo_testcases_gridladder_g6_round5.md` (commit
`85a13122c`, before the card was re-scored; it states that the offline replay
ran first).  Record: round 36 `gridladder_rounds/round36/oracle_g6_r36/`
(unchanged).  Evidence: `gridladder_rounds/round40/`.  Final runs stamped
`977112d8b0` (clean tree).  SHA-256 prefixes: independent `c852b6a201dd85b8`,
given-entry `bdf661c091eafb4f`, operators kt 1 `c3d10027af20f7bc`; probes
`probe_pivot_geom.py` `fdd1b14d776885be` (json `be030ee706f1222e`),
`probe_pivot_w_replay.py` `20c4e9474f3a3f0c` (json `063adddf1a5a1cf3`; resume
re-run with one added key: `79c35d37edd54a22`, json `e45a747a68910eac`),
`probe_pivot_v_zad.py` `998f167daccf85fd` (json `464e77965716a6c3`); passive
captures `capture_wzv_ops.py` `851bbd4b20dfd11d` (pre-fix card, `wzvcap/`)
and `capture_zad_ops.py` `64b29c679a7ac097` (post-fix card, `zadcap/`).

NEMO statements cited at their enclosing SUBROUTINE.  Every file below is
byte-identical between G6's build (G5 round 31,
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/`) and
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/` (`diff` clean: divhor, sbcrnf, sshwzv,
dynzad; stprk3_stg lines below grep-identical), so each line number holds in
both.

| statement | file:line | SUBROUTINE |
|---|---|---|
| velocity-form divergence | `divhor.f90:125-130` | `div_hor_RK3_t` :85 |
| runoff divergence (`ln_rnf_depth=F` arm) | `sbcrnf.f90:279-283` | `sbc_rnf_div` :238 |
| qco `w` recurrence | `sshwzv.f90:297-298` | `wzv_RK3_t` :236 |
| stage-2/3 momentum `wzv` call | `stprk3_stg.f90:327` | `stp_RK3_stg` :78 |
| stage ratios, then their fold exchange | `stprk3_stg.f90:167`, `:169` | `stp_RK3_stg` :78 |
| `dyn_zad` V update (divisor `e3v*(1+r3v)`) | `dynzad.f90:125-126` | `dyn_zad` :59 |

## 1. OPEN 2 — the fold-row ZAD residual: owner and fix

Card per-level horizontal divergence on the pivot T row (captured operands
of the stage-2 momentum `wzv` call, which feeds `dyn_zad`) against a literal
numpy replay of `div_hor_RK3_t` np_velocity plus the qco recurrence:

| check | result |
|---|---|
| card velocities vs NEMO's recorded stage-2 `uu/vv(Kmm)` | 0 / 0 unequal |
| replay calibration (wet cells off the pivot row, levels >= 2) | 2.17e-16 relative max (json key `..._levels_ge2`, added on resume; the original json held only the all-level figure 2.2e3, which includes the level-1 runoff cells) |
| level 1 | differs at 4 618 cells; the card adds runoff there and the replay does not (cells not matched to the runoff field) |
| NEMO's U-face `w` pair (inverted from its ZAD increment, round-39 method) vs card, pivot T row | median 4.3e-4 (reproduces round 39) |
| same, interior control rows 140 / 145 | 1.1e-12 / 2.0e-12 |

**First non-bit operand: the pivot V row's sea-level ratio `r3v(Kmm)`.**
NEMO computes the stage ratios (`:167`) and passes them through `lbc_lnk`
sign +1 (`:169`); on the T-pivot fold the pivot V row becomes the mirrored V
row below.  The card's shared face-ratio builder formed that row from its own
stencil, whose northern T neighbour wraps to row 0 (Antarctic land): on all
1 319 wet pivot-V cells the card's ratio is ~0.496 of NEMO's fold image.  It
reaches two consumers:

| consumer | card operand replaced by the fold image only | result |
|---|---|---|
| `div_hor` -> `ww` on the pivot T row (U pivot row of ZAD) | pivot V face thickness | pair median 4.3e-4 -> **1.3e-12** (= interior control) |
| `dyn_zad` V divisor on the pivot V row | ZAD `e3v` pivot row | max 4.9e-12 -> **4.2e-22** (last place) |

The `w` pair on the pivot row is exactly mirror-symmetric (0 unequal), and
the literal V-row `w` pair matches the card's, so `w` itself was not the V
row's owner; the divisor was.  The U pivot row's ratio is already the mirror
of its first half (0 unequal), so the U half of the exchange is a no-op here.

**Transcription defect -> landed** (`da61d1c10`, test `977112d8b`).  The
shared builder `nemo_qco_live_face_geometry_from_operands` takes an optional
`fold_grid`; with one it applies the existing `nemo_een_ratio_fold_exchange`
(the barotropic solver's, reused, not re-written).  Only G6 passes a grid:
`rk3_vector_raw_vmask` (selected only by the G6 card,
`nemo_testcase_recipe.py:3834`) feeds the stage faces
(`ocean_model_latlon_cgrid.py:2086`) and the three momentum/tracer `wzv`
calls (`:2146`, `:2166`, `ocean_pe_latlon_cgrid.py:5330`).  Caller grep: no
other `fold_grid=` / `fold_exchange_ratios=` site; every other card runs the
identical expression.  Edits were kept line-neutral in cited regions, so no
citation was re-anchored.

### Runoff (asked; not changed)

Runoff cannot own this residual: it enters only the surface level's
divergence, so only the surface `w`, which `dyn_zad` never reads (its
surface flux is set to zero, `dynzad.f90:102-103`).  Depth placement: G6's
deck and ORCA2's deck both set `ln_rnf=T`, `ln_rnf_mouth=T`
(`rn_hrnf=15 m` extra mixing), and leave `ln_rnf_depth` and
`ln_rnf_depth_ini` at the reference default `.false.`
(`ORCA2_OMIP_L4/EXP00/namelist_ref:474,478`), so NEMO adds all runoff to the
top level (`sbcrnf.f90:279-283`).  The card does the same: runoff is passed
for level 1 only (`ocean_pe_latlon_cgrid.py` `nemo_qco_wzv_operands`,
`jk == 0`).  The extra 15 m mixing at river mouths is a vertical-mixing
feature, not a divergence one; not examined here.

## 2. OPEN 3 — the interior ZAD last-place set: classified

After landing, kt 1 stage 2 after-ZAD, all levels: U 315 cells (303 full, 12
partial), V 600 cells (578 full, 22 partial), max 8.47e-22 absolute, 0
flushed subnormals.  Worst cell: NEMO -7.445317999987718e-06, card
-7.445317999987717e-06 — one unit in the last place (1.1e-16 relative).
Scorer verdict **AT-BAR** (normalised 8.5e-22 <= 1e-15), not exact.  Under
the bar, so no owner is named.

## 3. Operators at kt 1, stage 2 (after landing)

| boundary | U surface | V surface | U all levels | V all levels |
|---|---|---|---|---|
| after HPG | 0 | 0 | 0 | 0 |
| after VOR | 0 | 0 | 0 | 0 |
| after ZAD | 2.1e-22 / 16 | 4.2e-22 / 61 | 8.5e-22 / 315 | 8.5e-22 / 600 |
| after ZAD, round 39 | 3.5e-13 / 84 | 6.3e-13 / 195 | 5.2e-12 / 1 607 | 4.9e-12 / 3 191 |

Fold rows (pivot, row below, third row): 0 unequal, both components.

## 4. Both labels, kt 1..6 (`max_abs / unequal cells`, bar 1e-15)

| kt | label | ssh | uu_b | vv_b | u | v | T | S |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | independent = given | 0/0 | 0/0 | 0/0 | **0/0** (was 4.6e-9/135) | **1.4e-17/3** (was 2.6e-9/202) | 0/0 | 0/0 |
| 2 | independent = given two-step | 4.461e-03/16433 | 4.032e-03/15853 | 3.326e-03/15943 | 8.087e-03/15853 | 4.912e-03/15943 | 1.044e-03/16433 | 3.117e-03/16433 |
| 3 | independent | 9.152e-03 | 1.080e-02 | 5.253e-03 | 1.311e-02 | 6.716e-03 | 4.009e-03 | 7.847e-03 |
| 4 | independent | 1.855e-02 | 1.229e-02 | 9.786e-03 | 1.682e-02 | 9.214e-03 | 7.130e-03 | 1.096e-02 |
| 5 | independent | 2.883e-02 | 1.427e-02 | 7.508e-03 | 1.150e-02 | 1.078e-02 | 9.387e-03 | 1.736e-02 |
| 6 | independent | 5.074e-02 | 2.714e-02 | 2.234e-02 | 2.860e-02 | 2.236e-02 | 1.172e-02 | 1.771e-02 |
| 3 | given (restart 2) | 9.078e-03 | 8.415e-03 | 5.584e-03 | 1.273e-02 | 6.931e-03 | 2.609e-03 | 7.744e-03 |
| 4 | given two-step | 1.770e-02 | 1.030e-02 | 9.732e-03 | 1.579e-02 | 8.976e-03 | 5.848e-03 | 1.091e-02 |
| 5 | given (restart 4) | 2.671e-02 | 1.580e-02 | 7.635e-03 | 1.642e-02 | 8.693e-03 | 4.980e-03 | 6.989e-03 |
| 6 | given two-step | 5.330e-02 | 2.691e-02 | 2.714e-02 | 2.924e-02 | 3.072e-02 | 9.900e-03 | 1.496e-02 |

kt 2..6 cell counts are whole-domain (as round 39) and every number equals
round 39's to the printed precision.  kt 1's 3 V cells: interior, southern
rows (2 coastal, 1 open; first at row 10, column 77), normalised 1.0e-17,
AT-BAR.  Flushed subnormals (Decision 127): 0 in every run.

| prediction (preregistered) | measured |
|---|---|
| P1 fold-row ZAD residual removed; only the interior last-place set left | yes (fold rows 0; U 315 / V 600 left) |
| P2 kt-1 step-end unchanged or closer | closer: u exact, v 3 cells at 1.4e-17 |
| P3 kt 2..6 unchanged | unchanged |
| P4 no other card changes | yes (option off unless G6's flag; caller grep §1) |

**Certification:** G6 is at the bar at kt 1 under both labels, but NOT at kt
2..6 (sea level 4.461e-3 m at kt 2).  That debt's owner is the after-step
sea-level slot of Decision 132, which this round did not select or land on any
card.  G6 cannot be certified until Decision 132 is answered and kt 2 is
re-walked.

## D122 pointer (ORCA2 lane)

The same compiled lines run on ORCA2 (`stprk3_stg.f90:167/169`,
`divhor.f90:125-130`, `dynzad.f90:125-126`, identical line numbers in
`ORCA2_OMIP_L4`).  Any ORCA2 card that keeps NEMO's raw pivot V row wet must
take the fold-exchanged `r3v` there for both `ww` and the ZAD divisor, not
the wrapped stencil.  The ORCA2 card does not set `rk3_vector_raw_vmask`
today, so nothing changes there; not imported by this round.

## OPEN

1. kt 2+ whole domain: owner measured in round 39 (after-step sea-level
   slot); lands on the user's answer to Decision 132; then re-walk kt 2.

## Resume (after the independent Claude review: HOLD)

1. **Test gap closed.**  The round's test pinned the stage faces and the wzv
   builder, not the three wzv CALL SITES.  New
   `test_g6_every_wzv_call_site_takes_the_fold_exchange` steps G6 once and
   records, per call site, whether the exchange reached it (tracer and
   momentum solves in the stage transport, the tendency's first call).
   Planted `fold_exchange_ratios=False` at each site in turn: each failed
   (`assert {False, True} == {True}` naming that site), restored,
   `git status --porcelain` clean.  The old test's docstring overclaimed
   ("fails if ... the wzv call drops the exchange"); corrected.
2. **Finding (not changed).**  The step also reaches a FOURTH wzv call (the
   step body's leapfrog-style call 2, `wzv_call2_evaluation`) without the
   exchange.  Measured on G1b + G6 flag, random wet sea level, two steps,
   with the exchange forced at that one site: every prognostic identical;
   only the stored diagnostic `w` differs (35 cells, 6.3e-9).  Diagnostic
   only, so left alone and excluded from the new test by a commented filter
   (`resume/probe_call2_site.py` `4f86ca90a356b560`, json `b3b79c9ca5fce81f`).
3. **Citation.**  NEMO's qco `w` statement is `sshwzv.f90:297-298` (the
   assignment and its continuation) in both builds; the table is fixed.  The
   frozen preregistration keeps its `298-300` (not edited after the fact).
   The citation gate pins another receipt's legoESM lines, so nothing to
   re-anchor.
4. **Pre-existing red G2 test: the EXPECTATION was stale, not the code.**
   `test_the_option_is_off_by_default_and_selected_by_g2_only` checks the
   tail order of the config's fields with a fixed slice.  G6 round
   `9f8316fab` ("carry G6 vector fold operands") appended
   `nemo_hpg_sco_fold_geometry` and `rk3_vector_raw_vmask` last, both default
   off and selected only by G6 (`nemo_testcase_recipe.py:3833-3834`), without
   moving the slice.  No older field moved.  Slice widened by two, the two
   names appended, and their off defaults asserted.
5. Calibration figure: the receipt cites the BELOW-SURFACE figure, 2.17e-16.
6. **Second pre-existing red (G1 card), same class, also stale expectation.**
   `test_standin_card_builds_and_states_every_selector` (red at `85a13122c`
   too) planted the deck check `no_ice`; G5 round `888621bd3` renamed it
   `no_internal_ice_model` with an identical condition.  Plant key updated;
   no code changed.  UNASKED (test-only), offered for revert.

## Tests and review

- New `test_g6_pivot_v_ratio_is_nemos_fold_image` (G1b card with the G6 flag,
  non-zero initial sea level so the fold rows are live): the ZAD divisor's
  pivot V row equals the fold image of the row below with the flag, differs
  without; the `wzv` call's pivot face thickness equals the fold image and
  its pivot-row `w` moves while every other row is unchanged.  Shown to FAIL
  with the exchange disabled (`assert_array_equal(top[wet], below[perm][wet])`),
  then restored (`git status --porcelain` clean).
- `tests/ocean/unit/test_nemo_orca2g1b_card.py` +
  `tests/ocean/fidelity/test_nemo_testcase_g6_orca2vec.py`: 27 passed (CPU fp64).
- Receipt-citation gate + private-import ratchet: 23 passed on the clean
  tree (no citation re-anchored; the landing is line-neutral in cited code).
- Resume batteries (CPU fp64, clean tree except the receipt/G1-test edit
  committed with them): G1 card 53 passed; G1b card 20 passed; G1 tools 31
  passed; G1 ladder 86 passed (16 workers); G2-G6 77 passed, 13 deselected;
  citation gate + private-import ratchet 23 passed (stamped `5d9aac594819`).
- codex review: paused (quota).  The operator runs an independent Claude
  review of this diff before landing.

ASKED: owner of the kt-1 fold-row ZAD residual with literal replay and
citations in both builds; landing it G6-selected with a failing-when-reverted
test; runoff depth report; classification of the interior last-place set;
both labels re-scored kt 1..6.

UNASKED, offered for revert: none in the card or decks.  Lint-only lines in
the new test were reformatted by hand.  Two lines in cited regions exceed the
100-column limit to keep citations line-neutral.

UNVERIFIED: NEMO's stage-2 `r3v(Kmm)` is the HYB-interpolated pair of
exchanged ratios; the card forms the ratio of the interpolated sea level and
then exchanges it.  The two agree here to the bar (after-ZAD fold rows 0
unequal), but were not shown bitwise equal everywhere; that is a candidate
owner of the AT-BAR last-place set, not measured.
