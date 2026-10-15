# Grid ladder rung G1b — round 3 receipt (gridladder_r26)

Status: **LANDED** (two cited statements, G1b-card-selected). G1b is **NOT
certified**: GIVEN-NEMO-ENTRY is at the bar on every field over the whole
record; INDEPENDENT first leaves the 1e-15 bar at kt 20 (sea level, open
interior, worst 1.5e-14 over 100 steps).
Preregistration: docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g1b_round3.md
(0cd1e0900, before any ladder). Evidence:
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round26/.
Builds: `R4REF` = ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo (the record's binary);
`ORCA2` = ORCA2_OMIP_L4/BLD/ppsrc/nemo (D122 pointer).

## Take-aways

1. **Operator finding fixed (instrument, CONFIRMED).** `twin_dir` printed no
   newline, so the 13 twin directories became one nonexistent path. Fixed;
   the stride-10 plant now must miss exactly the r11..r13 steps.
   `run.sh --admit-existing` (no NEMO run): **ACQUISITION_DONE**, restarts
   for every step 2..100.
2. **np_CRV's top F row now reads NEMO's U fold halo** (`R4REF/dynvor.f90:744-748`).
   Round 24's attempt failed because `perm_u` indexes east faces and it was
   applied over the card's cyclic west column too (two columns off).
   kt-31 vorticity term: 68 U + 66 V unequal → 0 + 0.
3. **Vector-form Krhs order** (`R4REF/stp2d.f90:138`, `:144`, `:161`):
   HPG, then VOR, then KEG. kt-2 stage-1 right-hand side 227/230 → 4/0 faces.
4. **GIVEN-NEMO-ENTRY: every field at the bar**, 98 restart steps and kt 1-50;
   worst 2.8e-17 → 2.6e-18 (was 2.4e-6 at round 24, fold band).
5. **INDEPENDENT: not certified.** First over bar 27 → 20 (sea level, open
   interior); worst ~1.5e-14 unchanged. D96 registration below.
6. **Next statement, OPEN:** stage-1 right-hand side, 4 U faces at 1.7e-24
   at kt 2 (HPG or KEG not yet split); then the stage-1 velocity update
   (35 U / 49 V faces, 3.5e-18, on the card's own RHS).

## 0. Acquisition (operator finding)

`run.sh:211` `twin_dir() { printf '%s/r%02d' ...; }` — no newline, so
`$(for j ..; do twin_dir "$j"; done)` concatenated all directories into one
argument. Same python as `restart_cover`, record round22/oracle_g1b_r22
(restart_vs_frame and admit logs in round26/):

| twin dirs | before (concatenated) | after (one per line) |
|---|---|---|
| r01..r10 (stride-10 plant) | 1 arg, RESTART_FULL 0, MISSING 2..100 (fired vacuously) | 10 args, RESTART_FULL 82, MISSING 11,21,..,91,92..99 |
| r01..r13 (admission) | 1 arg, RESTART_FULL 0 → REFUSE | 13 args, RESTART_FULL 99, MISSING none |

The plant now refuses unless its MISSING list equals the union of the
r11..r13 lists (`want` in run.sh). `run.sh --admit-existing` at 172873b01:
PASS_G1_RECORD, seven record plants fired, twin frames identical kt 1..100,
`G1B_RESTART_PLANT_STRIDE10 RESTART_FULL 82 MISSING 11,..,99`,
`G1B_RESTART_EVERY_STEP 2..100`, `ACQUISITION_DONE ... N=100`.
The 17 new restarts equal NEMO's after-step frame on ssh, u, v, uu_b, vv_b
(0 unequal); plant: step-93 restart vs kt-92 frame, 16,433 unequal.
Ladder root round26/g1b_merged: r10's files + one full restart per step 2..100.

## 1. Statement A — top F row U fold halo (bb32968aa)

NEMO: `zwz = (ff_f + ((e2v*pv)(i+1) - (e2v*pv) - ((e1u*pu)(j+1) - (e1u*pu)))
* r1_e1e2f) * z1_e3f` on `jj = ntsj-1 .. ntej+1` (`R4REF/dynvor.f90:744-748`),
so the top F row reads `pu(ji, Nje0+1)`, NEMO's U fold halo.

Measured on NEMO's kt-31 entry frame: halo inner column i = −u(row below the
pivot, column 179 − i). The card's `perm_u` is exactly 179 − i, in east-face
(n_lon-column) indexing; card u column 0 is the cyclic copy of column n_lon.
Round 24's attempt applied `fold_row` over card columns 0..n−1 → two
columns off. Fix: the NEMO curl bracket takes `north_fold_u`; its north row
is `vector_sign_u * fold_ghost_source_T(zu)[1:][perm_u]` plus the cyclic
column; passed only on the `een_nemo_relative_vorticity` een_total path.

| kt 31, GIVEN-NEMO-ENTRY (rhs_terms) | before (card at 0cd1e0900, no A) | after (A) |
|---|---|---|
| replay vs NEMO (control) | 0 / 0 | 0 / 0 |
| card vorticity vs replay, U / V | 68 (pivot row, 8.6e-19) / 66 (row below, 4.6e-19) | **0 / 0** |
| card total vs NEMO, max | 8.6e-19 | 2.6e-23 (sum-order class, statement B) |

Unit test `test_nemo_curl_top_f_row_reads_u_fold_halo`: passes; with
round 24's column space planted, fails.

## 2. Statement B — vector-form Krhs order (b1a4fe915)

NEMO accumulates Krhs in place: `dyn_hpg` (`R4REF/stp2d.f90:138`), `dyn_vor`
(`:144`), `dyn_keg` (`:161`); stages 2-3 the same (`R4REF/stprk3_stg.f90:325`,
`:328`, `:332`). The card summed (KEG + HPG) with VOR. The association branch
already existed in the shared step behind a private test hook; measured with
the hook first, then exposed as model option `rk3_vector_rhs_nemo_order`
(default False; only the G1b card selects it; validator pins it; the plant
`{"rk3_vector_rhs_nemo_order": False}` is refused).

| GIVEN-NEMO-ENTRY stage-1 RHS (rhs arm), U / V unequal | card before | hook arm | card after |
|---|---|---|---|
| kt 2 | 227 / 230 (2.6e-23) | 4 / 0 (1.7e-24) | 4 / 0 |
| kt 3 | 367 / 322 (5.3e-23) | 9 / 0 (1.3e-23) | 9 / 0 |

## 3. Ladders (scored normalised by NEMO's field max; bar 1e-15)

| label, arm | card | first over bar | worst ssh / u / T |
|---|---|---|---|
| INDEPENDENT kt 1-100 | round 24 | ssh 27, u,v 30 (fold) | 4.9e-6 / 1.5e-5 / 1.6e-10 |
| INDEPENDENT kt 1-100 | + A | ssh 27, v 45, u 54, T 37 | 1.5e-14 / 2.2e-15 / 2.1e-14 |
| INDEPENDENT kt 1-100 | + A + B (**card**) | ssh **20**, u,v 36, T 35 | 1.5e-14 / 2.0e-15 / 2.5e-14 |
| GIVEN entry frames kt 1-50 | round 24 | 30 (fold) | 2.0e-9 |
| GIVEN entry frames kt 1-50 | + A | none | 1.4e-17 |
| GIVEN entry frames kt 1-50 | + A + B | none | 1.7e-18 / 2.6e-18 / 0 |
| GIVEN restarts, 98 steps (3-100) | round 24 (81 steps) | 30 (fold) | 2.4e-6 |
| GIVEN restarts, 98 steps | + A | none | 2.8e-17 |
| GIVEN restarts, 98 steps | + A + B | none | 2.6e-18 / 6.9e-18 / — |

S never over the bar. The kt-1 T/S row of the entry-frame arm (20, 30) is
exactly the 10,207 land cells (NEMO holds the initial value there; wet
cells exact) — not a wet-cell error, same as rounds 1-2.

B, row by row against + A: restart arm unequal faces 2,798,031 → 307,908,
254 rows newly exact, 0 lost; entry-frame arm 1,034,719 → 161,013, 130
newly exact, 0 lost. INDEPENDENT: 9 exact rows lost (T/S at kt 8, 16, ...)
and first over bar 27 → 20, worst unchanged. **D96 registration:** "G1b
INDEPENDENT sea level first over the bar 27 → 20" — near-zero lottery of a
statement that strictly improves every GIVEN row; B stays selected.

Where INDEPENDENT leaves: kt 20 open interior; kt 27 still 8.4e-15 in one
step there (before B as well). Given NEMO's entry, kt 20 and 27 are at the
bar: accumulated, not a statement at that step (PLAUSIBLE, as round 24).

## 4. OPEN — next first non-bit statement

kt 2, stage 1 (stages arm, GIVEN-NEMO-ENTRY): card RHS vs NEMO 4 U faces
(1.7e-24 of 2.1e-7); the card's fused KE+HPG term vs replay 5 faces 8e-25
(rhs_terms) — HPG or KEG not split (the term observer reports them fused).
Then, on the card's own RHS, the stage-1 velocity update differs from the
written statements on 35 U / 49 V faces (3.5e-18). Final kt-2 u/v: 69 / 96
faces at 1.7e-18. Next round: split the observer's KE_PGF bucket, cite.

## 5. ORCA2 pointer (D122)

| statement | ORCA2 build line | ORCA2 card |
|---|---|---|
| A: top F row U fold halo | `ORCA2/dynvor.f90:762-766` (vor_een np_CRV, `jj = ntsj-1 .. ntej+1`); deck `ln_dynadv_vec`, `ln_dynvor_een` (EXP00/namelist_cfg:346, :352) | does not select `een_nemo_relative_vorticity`; UNMEASURED |
| B: Krhs order | `ORCA2/stp2d.f90:141`, `:147`, `:164`; `ORCA2/stprk3_stg.f90:402`, `:407`, `:413` | not selected; UNMEASURED |

Correction (Rule 11): round 24's pointer `ORCA2/dynvor.f90:648` is in
`vor_ens`, not `vor_een`; the vor_een np_CRV statement is :762-766.

## 6. Tests and review

- G1b card file + NEMO curl test file: 13 + 7 passed.
- Focused battery (citation gate, ORCA2 receipt citations, card, curl,
  dispatch, private imports, validate_strict coverage, inline imports):
  24 reds at the base f22dd5f4e, the same 24 after (slab_land dispatch guard,
  inline-import budgets: pre-existing); the 2 new citation reds were the
  dirty-tree stamp and the shifted anchors → re-anchored with the lane's tool
  (79 spans: 59 gate map, 20 GYRE phase-3 round-8 receipt), then 193 passed.
- G1 ladder + tools test files: 112 passed.
- NOT RUN: G2/G3/G4 ladder suites, closed cards. Defaults read as unchanged
  (`north_fold_u` False; the option False → association value identical to
  the hook's), not measured.

REVIEW_PLACEHOLDER

Compliance without a gate: searched before building — the fold helpers
(`fold_ghost_source_T`, `fold_perm_u`, `apply_north_fold`), the existing
association branch and its hook, the restart-coverage python and the
re-anchor tool were found and reused; new code: one kwarg, one config field,
two call-site ORs, one ladder kwarg. One variable per before arm. Dual
review: NOT done (lane rule 6, single).

## 7. Predictions

| id | result |
|---|---|
| R3-V1 kt-31 vorticity 0 unequal on the two top rows | HOLDS (0 / 0) |
| R3-V2 nothing below the top F row moves | HOLDS (card vorticity vs replay 0 everywhere; unit test rows below identical) |
| R3-L1 GIVEN restart first over bar later than 30 | HOLDS (none in 98 steps) |
| R3-L2 INDEPENDENT first over bar not earlier than 27 | HOLDS after A (27); **REFUTED** after B (20, D96 lottery) |
| R3-C1 G1b not certified this round | HOLDS (INDEPENDENT) |

## 8. Choices

ASKED (brief): fix twin_dir and the stride-10 plant; admit with
--admit-existing; land the fold halo under the G1b selection; re-score; walk
on; tests; one review.

UNASKED, each offered for revert:
1. Landed a second statement (B) with a new model option
   `rk3_vector_rhs_nemo_order` (default False, G1b only).
2. The ladder's `rhs` arm now honours `--hook` (measurement only).
3. Re-anchor tool edited the GYRE phase-3 round-8 receipt (20 spans).

UNVERIFIED:
- the ORCA2 column (read, not run);
- closed cards and G2-G4 unchanged (defaults read, not run);
- the kt-20/27 INDEPENDENT events' cause (PLAUSIBLE, accumulated).

SHA-256 (round26/sha256_manifest.txt, first 16): admit_existing.log
4c4ca8a71e42192a; l1/rhs_terms_kt31_before.json a854efabf0c1500f;
l1/rhs_terms_kt31_dirty.json 3a16952519f5f044; l2/indep100.json
a6899f7e0045dc29; l2/restart_entry100.json 0af5a821b904004a;
l2/given_entry50.json 76437b5dc0471ebd; l3/indep100.json 823c743f2374549d;
l3/restart_entry100.json 89f4df30bb6e74f5; l3/given_entry50.json
74ff31cfe0aa2b0c.
