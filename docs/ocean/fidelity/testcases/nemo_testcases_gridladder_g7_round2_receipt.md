# Grid ladder rung G7 — round 1 resume receipt (gridladder_r41, scoring)

Status: **LANDED. G7 CERTIFIED.** Every field is bit-identical to NEMO's
admitted record under both labels at kt 1..10: surface rows and all-level rows
0 unequal cells, INDEPENDENT and GIVEN-NEMO-ENTRY.  Worst row per label: none
above 0 (max abs 0.0, flushed subnormals 0, Decision 127 count 0 in every row).

- Record: `gridladder_rounds/round41/oracle_g7_r41` (operator run at
  `2691093f8`: `G7_PREFLIGHT_OK`, `PASS_G5_ADMISSION unequal_rows=None`,
  `G7_ACQUISITION_PASS steps=1..10 restarts=2..10 reference_completed=12`).
- Preregistration: `PREREG_nemo_testcases_gridladder_g7_round1.md` (`26e330e30`).
- Landing commits: `cba2e91be` (bottom residual), `9276a78c9` (dyn_zdf bottom
  RHS), `f3d7fcc79` (declared gaps), `e6dd6d1b6` (citation re-anchor).
- Final ladders stamped clean at `9276a78c9` (packages tree `07391c025df2`).
  The two later commits do not touch model code paths.
- Evidence: `gridladder_rounds/round41/` (`manifest_resume.sha256`
  `8f49ccfd18ec767b`).
- codex review: paused (quota).  The operator runs an independent Claude
  review of this diff before landing.
- Independent Claude review (operator): arithmetic and statements CONFIRMED
  (kt 1..10 independent, given-entry 1,3,5,7,9: 0 unequal, 0 flushed).  HOLD on
  tests only: (C) dropping the residual increment's outer rounding and (D)
  using the live depth for the bottom residual's `r1_h` both left every test
  green.  Second resume (`9355a5e82`, tests only, no model code) closes both;
  see section 6.

## 1. First scoring at the round-1 build (`2691093f8`)

| label | kt 1 | kt 2 | kt 3..5 |
|---|---|---|---|
| INDEPENDENT | AT-BAR: u 6 cells 1.39e-17, v 12 cells 1.11e-16, rest exact | DEBT every wet cell: ssh 5.78e-5 m, uu_b 9.10e-5, u 9.09e-5, v 3.58e-5 m/s, T 1.98e-5, S 5.88e-5 | DEBT, growing to ssh 1.33e-3 m, u 4.40e-3 m/s at kt 5 |

Stage walk, kt 1 (`g7_stages_kt1.json` 4639271d71ddba30): stages 1 and 2
exact on every field and level; stage 3 u/v at the bar (all-level u 181 full +
283 partial cells, v 239 + 256, max 8.2e-17).

Stage walk, kt 3 from NEMO's kt-2 restart (`g7_stages_kt3_given.json`
a5d2c59874245db1): already DEBT at stage 1, including the barotropic exit:
uu_b 3.68e-4, vv_b 1.00e-4 m/s, ssh 4.88e-5 m.  Worst face (115,144), Danish
straits, 3 wet levels, H = 30.0 m.

## 2. Walk to the first non-bit statement (kt ≥ 2, DEBT)

Ruled out first:

- Substep drag: exact at kt 1, where the substep velocities are nonzero, so
  the combined coefficient `pCdU` is right.
- Explicit 3-D drag double count: skipped under `zdf_drag_in_matrix`.
- Residual time level: the card uses the entry state.  NEMO calls
  `dyn_drg_init(Kbb, Kbb, …)` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:219`,
  SUBROUTINE `stp_2D` :63), and `ln_bt_fw = T`.
- Bottom-face level: the card's bottom face equals NEMO's `mbku` (wet-level
  count) on all 15853 U faces.  It also equals `mbkv` on 15875 of 15943 V
  faces.  The 68 that differ are the north-fold V row; they score exact.
- Rebuilt depth mean vs carried velocity, on NEMO's own kt-2 restart: the
  difference is at most 8.3e-17 m/s.  This is not the size of the error.
- Scale test of the bottom residual (probe `residual_scale.py`
  925de3b9404a6df7), kt 3 stage 1, uu_b max:

  | residual rate | error |
  |---|---|
  | ×0 | 1.84e-2 |
  | ×0.9 | 1.83e-3 |
  | ×1.0 | 3.68e-4 |
  | ×1.1 | 1.93e-3 |

  So the residual's size is right and its pattern is wrong.

**First non-bit statement: the bottom term in `dyn_drg_init`.**

```
zu_i = puu(ji,jj,ikbu,Kmm) - puu_b(ji,jj,Kmm)
pu_RHSi = pu_RHSi + (r1_hu_0/(1+r3u(Kmm))) * r1_2*(rCdU_bot(ji+1,jj)+rCdU_bot(ji,jj)) * zu_i
```

- Cite: `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1446-1450`,
  `:1461-1463`, SUBROUTINE `dyn_drg_init` (:1404).
- G5 build: the same lines in
  `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynspg_ts.f90`.
  The file is byte-identical between the two builds.

The shared path did three things differently:

1. it subtracted a depth mean rebuilt from min-rule live face thicknesses,
   instead of the carried `uu_b`;
2. it divided by the live depth, instead of NEMO's stored reciprocal;
3. it skipped NEMO's statement roundings.

From rest, at kt 1, the term is zero.  That is why kt 1 was clean.

Landed (`cba2e91be`): on the ice-drag family with substep drag (only G7
selects this), the bottom residual now runs NEMO's statement.  It uses the
same helper as G5's certified top residual,
`nemo_drg_residual_increment`, with:

- the carried `uu_b`/`vv_b`;
- the stored-depth reciprocal;
- the bottom pair of `rCdU_bot = -(rn_Cd0*ssmask)*rn_Uc0`
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:276`, SUBROUTINE `zdf_drg_lin`
  :258).

After the fix: kt 3 stage 1 from NEMO's restart has 0 unequal cells on every
field and level (`g7_stage1_kt3_candidate_dirty.json` 0c7bcb829d44fd25).

## 3. Second statement (kt 1 stage 3, AT-BAR seed)

After the first fix, the given-entry ladder was at the bar everywhere.  The
independent chain went over the bar at **kt 3, ssh, 9292 cells, max
1.021e-14** (`g7_independent_kt1_5_fix.json` 520df968a9bcf682).  The seed
was kt 1 stage 3, u 6 / v 12 level-1 cells.  Those differences run up whole
columns, which is the signature of one last-place change at the bottom of
the tridiagonal solve.

**Statement: the dyn_zdf bottom drag RHS term.**

```
puu(iku,Kaa) = puu(iku,Kaa) + zDt_2 * (rCdU_bot(ji+1,jj)+rCdU_bot(ji,jj)) * uu_b(Kaa) / e3u(iku)
```

- Cite: `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:166-169`, SUBROUTINE
  `dyn_zdf` (:67).
- The card formed `(dt*r_avg/e3)*uu_b`, a different association.
- The bottom diagonal (`dynzdf.f90:306`) was already bit-equal, because
  10800·4e-4 = 5400·8e-4 exactly.

Landed (`9276a78c9`): G7's bottom term now uses the same statement helper as
G5's certified top term, `nemo_dynzdf_drag_rhs`.  Result: kt 1 stage 3 and
the full kt 3 step from NEMO's restart have 0 unequal cells
(`g7_stage3_kt1_rhscand_dirty.json` 88e545f71212ebbb,
`g7_stage3_kt3_rhscand_dirty.json` 4d93d761dd85f59e).

## 4. Final ladders (clean, `9276a78c9`), kt 1..10, both labels

| label | surface rows | bit-identical | all-level row sets | unequal cells | flushed subnormal | worst |
|---|---:|---:|---:|---:|---:|---|
| INDEPENDENT (kt 1..10) | 70 | 70 | 36 | 0 | 0 | 0.0 |
| GIVEN-NEMO-ENTRY (entries 1,3,5,7,9; one and two steps) | 70 | 70 | 20 | 0 | 0 | 0.0 |

Files:

- `g7_independent_kt1_5_f2.json` e5ada0e55feda861
- `g7_independent_kt6_10_f2.json` d6003315f06d0fa9
- `g7_given_e135_f2.json` 2e9244dd1d402fe0
- `g7_given_e79_f2.json` 1d97cbd9a97f926f

Fields: ssh, uu_b, vv_b, u, v, T, S.  All-level rows cover u, v, T, S at
every kt that has a restart.  Every region is exact: fold, seam, south and
interior.

## 5. Predictions (frozen in round 1)

- **P1 CONFIRMED.** The drag log lines were checked by `run.sh`; the
  acquisition passed.
- **P2 CONFIRMED.** The admission passed, passivity included.
- **P3 CONFIRMED** (kt 1 not bit-identical: stage 3, at the bar).  Ranking:
  - (a) fold V-row rate: REFUTED as first; the fold scores exact.
  - (b) bottom residual form: CONFIRMED, as the DEBT statement.
  - (c) dyn_zdf association: CONFIRMED in its RHS term, not its diagonal.
- **P4 PARTLY REFUTED.** The wrong association was last-place, as predicted.
  The residual defect was not O(10%) of a face's velocity: it was 9e-5 m/s
  at kt 2, about 2% of the residual's own effect.

## 6. Tests, plants, batteries (CPU, fp64, `JAX_ENABLE_X64=1`)

New tests in `test_nemo_testcase_g7_orca2drg.py`:

- the residual increment equals NEMO's statement;
- G7's bottom residual reads the carried `uu_b`/`vv_b`;
- the dyn_zdf drag RHS equals NEMO's statement;
- G7's dyn_zdf bottom term runs it;
- (resume 2) `round(F + increment)` under jit with traced inputs equals
  NEMO's per-statement rounding (XLA would otherwise contract it to an FMA);
- (resume 2) with the stored depth selected and sea level off rest, the
  bottom residual's `r1_h` equals the certified top residual's stored
  `r1_hu_0/(1+r3u)` (dynspg_ts.f90:1462, :1487).

Plants, all red, each followed by a `cmp`-verified restore:

1. model file reverted to `2691093f8` → the residual test FAILS
   (`plant_revert.log`);
2. the bottom residual fed `0.5*uu_b` → FAILS (`plant_halfub.log`);
3. model file reverted to `cba2e91be` → the dyn_zdf test FAILS
   (`plant_zdf_revert.log`).
4. (C) outer rounding removed → the jit test FAILS, 781/4096 cells;
5. (D) live `1/H` for the bottom `r1_h` → the stored-depth test FAILS,
   4032/4032 cells (both on scratch copies under
   `round41/review_scratch/plant{C,D}`, `r41fix_plants.log`; the clone was
   never edited, both tests pass at HEAD).

| battery | result |
|---|---|
| G5 + G5 top-drag consumers + G6 + ORCA2 OMT-2 ladder + VORTEX SMT card | 53 passed |
| G7 + G2-family ladder + G4 + G3 + G1 tools | 114 passed, 13 deselected (slow) |
| G1 ladder + G1 card + G1b card + G7 (16 workers) | 177 passed |
| citation gate + private-import ratchet (clean, `e6dd6d1b6`) | 23 passed |
| G5 certified record, independent kt 1..3, final tree | 21/21 rows exact |
| (resume 2) citation gate + private-import ratchet, clean `9355a5e82` | 23 passed |
| (resume 2) G1-G7 card tests, 16 workers + 4 long G1 replays | 215 + 4 passed (13 slow deselected by default) |

G1-G6, SMT-2 and OMT-2 are unchanged.  Both new branches need
`implicit_ice_ocean_top_drag` together with bottom drag in the matrix and in
the substeps.  Only G7 selects that combination.  G5/G6's top terms now call
the two helpers, and their arithmetic is identical (G5's record re-scores
exact).

Decision 122: the full ORCA2 card does not run ice top drag, so it takes the
pre-existing bottom-residual branch.  These two statements do not reach it
as written.  The same compiled lines run in `ORCA2_OMIP_L4`.  Porting them to
the ORCA2 card's no-ice branch is a separate landing.

Decision 132 was not touched.

## Choices

ASKED: scoring both labels; landing cited transcription defects G7-selected.

UNASKED, offered for revert:

- `ORCA2G7_UNMEASURED` reduced to G5's input route (G5 round-4 precedent);
- the top residual and top dyn_zdf RHS (G5/G6) now go through the shared
  helpers (same arithmetic, G5's record re-scored exact).

UNVERIFIED: the north-fold V-row bottom level (68 faces) still differs from
`mbkv` in the card's face array.  It is not consumed by any scored value here.
It is registered and not fixed.
