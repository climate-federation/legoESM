# Grid ladder rung G1b — round 2 receipt (gridladder_r24)

Status: **LANDED** (two cited statements, G1b-card-selected). G1b is **NOT
certified**: first step over the 1e-15 bar is 27 (INDEPENDENT, sea level)
and 30 (GIVEN-NEMO-ENTRY, fold band). A repaired acquisition is needed for
17 restart steps (ACQUISITION_NEEDED below); the analysis did not wait for it.
Preregistration: docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g1b_round2.md
(2893d075f, before any ladder). Evidence:
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round24/.
Builds cited: `R4REF` = ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo (the record's
reference binary); `ORCA2` = ORCA2_OMIP_L4/BLD/ppsrc/nemo (D122 pointer).

## Take-aways

1. **Housekeeping: the red G1 stand-in test was the test, not the code.**
   Introduced by 4c7c6cc8b (G3 round 1, not rounds 21/22): it renamed the
   deck check `uniform_T` to `initial_T` when the initial state became
   rung-keyed. The plant still fires, under the new name. Before: 1 failed
   ("Expected regex 'uniform_T'", message `['initial_T']`); after: 54 passed
   (1e4b74aae).
2. **The record is short of 18 restart steps, not 9.** r01's lone "step-1"
   file is an empty shell (no time record, kt = 0); r02..r09's restarts
   92..99 were re-opened and blanked at kt = 100. 82 of steps 2..100 hold a
   record. Step 1 cannot be written on this deck.
3. **First non-bit statement: np_CRV's relative vorticity divides by the
   wrong F area.** The card used the grid's `area_q`, which differs from
   NEMO's e1f·e2f by up to 65% on ORCA2's mesh (e1u, e2v exact). Landed:
   kt-2 vorticity term 914 U / 907 V faces unequal → 0 / 0.
4. **Second: the solver's fold unit was gated to flux-form decks.** NEMO's
   fold-row transport and exchange are outside its vector/flux branch. Landed:
   given NEMO's entry, fold-row error at kt 17-21 ≈ NEMO's own value →
   ≤ 3e-26 relative.
5. **Headline (INDEPENDENT, first over the bar):** step 2 → 19 → **27**
   (sea level) / **30** (velocities); worst normalised error 1.7e-1 →
   1.5e-5 over 100 steps.
6. **Next statement, CONFIRMED offline, not landed:** the top F row of the
   relative vorticity needs NEMO's U fold halo (dynvor.f90:744-748 at
   jj = Nje0). One card-side attempt with the shared fold helpers did not
   reproduce it and was reverted.

## 0. Acquisition repair (ACQUISITION_NEEDED)

NEMO's list logic (`R4REF/restart.f90:94-101` pointer starts at entry 1;
`:119-120` a file opens at kt = nitrst - 1, or at nitend when nothing is
open; `:188`, `:200-201` the pointer moves only after a write):

| twin list | what NEMO wrote (measured, round22/oracle_g1b_r22) |
|---|---|
| r01: 1, 11, .., 91 | one 215,144-byte shell named step 1, opened at kt = 100 (its log: "kt = 100"), time_counter length 0 |
| r02..r09: j, j+10, .., 90+j | 9 full restarts each; the 10th (92..99) re-opened at kt = 100 and blanked (215,144 bytes) |
| r10: 10, .., 100 | 10 full restarts |

Full restarts: 82 (steps 2-10, 12-20, .., 82-90, 100). Missing: 11, 21, ..,
91 and 92..99 (17 steps), plus step 1 (impossible; G2 round 3, G4 round 5).

run.sh (e5b7bb95e): three extra twins, each list ending on a 0 so nothing is
re-opened (`:120` needs nitrst > 0): r11 = 11,21,..,91 (as briefed);
r12 = 92,94,96,98; r13 = 93,95,97,99 (consecutive entries never open,
`:119`). `--resume` admits the smoke, reference and every finished twin in
place and runs only absent twins. Admission now requires, for every step
2..N, a restart with a time record and kt = step. Plants (on NEMO's own
files): r01's step-1 shell and r02's step-92 shell must not count, and the
ten stride-10 twins alone must leave steps uncovered. Measured on the record:
the check reports exactly the 17 missing steps (exit 1); each shell is
refused; a full restart (r02 step 82) passes. Preflight at e5b7bb95e:
`RESUME_READY ... run.sh --resume runs the absent twins of r01..r13`.

## 1. Admission of the existing record (no NEMO re-run)

- G1's check_records.py on r10 with r01 and the reference: PASS_G1_RECORD;
  its seven plants each exit 1 with STATUS PLANT-FIRED (admit/summary.log).
- Step frames of r01..r09 byte-identical to r10's at kt = 1..100.
- All 82 full restarts equal NEMO's after-step frame on ssh, u, v, uu_b,
  vv_b (0 unequal).
- NEMO's log: "vector form : keg + zad + vor is used", "vector form
  dynamics : total vorticity = Coriolis + relative vorticity".
- Ladder root round24/g1b_merged: r10's files plus the 72 full restarts of
  r01..r09 (symlinks).
- Geometry identity on G1b: 36/36 bit-identical.

## 2. Instrument

The G1 ladder now takes `--case ORCA2G1b-vec`; restart_entry scores every
step whose previous restart exists (on G1's record, every tenth: the same
steps as before). `rhs_term_replay(vector=True)` adds NEMO's np_CRV
(`R4REF/dynvor.f90:744-748`, r1_e1e2f from `R4REF/domhgr.f90:156`) and KEG
(`R4REF/dynkeg.f90:117-125`), subtracted after the vorticity trend
(`R4REF/stp2d.f90:142`, `:155`). Control, replay against NEMO's recorded
stp_2D right-hand side: 0 unequal at kt 1, 2, 3, 16, 17, 18 and (with the
e1u fold halo row added) 31.

## 3. Statement A — np_CRV F area (bf71392b9)

NEMO: `zwz = (ff_f + ((e2v*v)(i+1)-(e2v*v) - ((e1u*u)(j+1)-(e1u*u))) *
r1_e1e2f) * z1_e3f` (`R4REF/dynvor.f90:744-748`), `r1_e1e2f = 1/(e1f*e2f)`
(`R4REF/domhgr.f90:156`).

Walk at kt 2 (GIVEN-NEMO-ENTRY): first unequal solver operand = slow forcing
zu_frc, 895 faces, 4.95e-12 relative; term split: KE + pressure gradient
≤ 8e-25 on 5 faces, **vorticity 914 U / 907 V faces, max 4.9e-12 of 2.1e-7**.
Card relative part / NEMO's: 0.885 to 1.13 (median 0.995) — not a factor.

Retraction (Rule 11): my first hypothesis was the generic curl's form. A
probe refuted it: the generic and NEMO-literal curls agree to 8e-25 on 295
vertices; adding an option for the literal curl alone moved nothing. The
measured cause: `grid.area_q[1:, 1:]` vs e1f·e2f, max relative difference
0.646.

Option `een_nemo_relative_vorticity` (model config, default False):
een_total takes the existing NEMO-literal curl over a vertex area holding
e1f·e2f from the card's raw mesh bundle (west column its cyclic image, south
row the grid's). Result: kt-2 vorticity term 0 / 0 unequal.

## 4. Statement B — the solver's fold unit under vector form (a76d3a806)

NEMO, outside the `ln_dynadv_vec` branch (`R4REF/dynspg_ts.f90:556`): the
mid-step face depth with ssvmask on the fold-image V row (`:418-421`), the
transport `zhV = e1v*va_e*zhvp2_e` (`:436-438`), the end-of-substep exchange
(`:590-596`).

Walk: kt 17 substep record, fold-image V row transport: card 0, NEMO
2.44e-13 (error = NEMO's value, 68 faces); certified G1's card on G1's
record at kt 17: 0 unequal there. The unit's predicate required flux-form
momentum. Measurement before code (private hook): given NEMO's entry, kt
17-21, fold rows ≤ 3e-26 relative.

Option `barotropic_literal_fold_unit_vector_form` (BarotropicConfig, default
False) admits rk3_ws + vector_invariant to the predicate.

## 5. Ladders (both labels, all scored fields; normalised by NEMO's field max)

INDEPENDENT, kt 1..100 (l1/, l2/):

| card | first over bar (ssh / u,v,uu_b,vv_b) | worst ssh | worst u | worst T |
|---|---|---|---|---|
| round 22's card (before) | 2 / 2 | 1.2e-1 | 1.7e-1 | 1.1e-5 |
| + A | 19 / 19 | 1.1e-1 | 1.7e-1 | 1.1e-5 |
| + A + B (**the card**) | **27 / 30** | **4.9e-6** | **1.5e-5** | **8.0e-12** |
| card without B (control) | 19 / 19 | identical to "+ A" (565 non-exact rows) |

By region, the card: kt 1-26 every row AT-BAR (largest 5.5e-17); kt 27
open interior 8.4e-15 in one step (5e-17 at kt 26); fold band grows from
kt ~25 (7e-15 at kt 30). Seam and south wall stay below 1e-19 to kt 30.
T and S: never over the bar in 100 steps (land values excluded at kt 1).

GIVEN-NEMO-ENTRY: entry frames kt 1..50 (l2/given_entry50) and NEMO's restart
of kt-1 at 81 steps in 3..91 (l2/restart_entry100):

| arm | first over bar | region | worst |
|---|---|---|---|
| entry frames, kt 1-50 | 30 (ssh, u, v); T, S 34 | fold band only | 2.3e-9 |
| restarts, 81 steps | 30 (ssh, u, v) | fold band only | 2.4e-6 |

The kt-27 interior jump of the independent chain is absent one step from
NEMO's entry (kt 27 AT-BAR in both arms): accumulated, not a statement at
kt 27 (PLAUSIBLE: a last-place lottery like G1 round 7's).

Remaining last-place classes (AT-BAR, not exact): the stp_2D total sums
the card's fused KE-plus-pressure term then the vorticity, NEMO adds HPG,
VOR, then KEG (1,800-2,000 faces at 2.6e-23, kt 16-18); KE + pressure term
≤ 8e-25 on ≤ 35 faces; kt 1 stage 2-3 velocity 1.7e-18 on ~400 faces.

## 6. OPEN — next statement (CONFIRMED offline, not landed)

At kt 31 (GIVEN-NEMO-ENTRY) the first unequal solver operand is the pivot
row's U slow forcing (68 faces, 3e-7 of its own value); the right-hand side
replay with a zero north halo also fails there (68 pivot U, 66 V faces
below), and the card equals that replay. With e1u's fold-halo row (NEMO's
own u halo mirrors the row below the pivot, sign −1) the replay is 0
unequal everywhere, and the card differs on exactly those 134 faces.
Statement: `R4REF/dynvor.f90:744-748` on the top F row reads `e1u(ji,jj+1) *
pu(ji,jj+1)` from the fold halo. A card attempt through the shared fold
helpers (`fold_ghost_source_T`, `fold_perm_u`, `fold_row`) moved the error
(8.6e-19 → 8.3e-19) but did not close it; reverted. Next round: print the
card's fold descriptor (`pivot_row_stored`, `perm_u`) against the replay's
measured U mirror shift, then land under `een_nemo_relative_vorticity`.

Then: the kt-27 accumulated interior event; the three last-place classes.

## 7. ORCA2 pointer (D122)

| statement | G1b | ORCA2 build |
|---|---|---|
| A: np_CRV F area | landed, selected | `ORCA2/dynvor.f90:648` (vor_een np_CRV CASE), `ORCA2/domhgr.f90:156`. ORCA2 runs een_total; its grid `area_q` was not compared with e1f·e2f. UNMEASURED |
| B: fold unit, vector form | landed, selected | `ORCA2/dynspg_ts.f90:517` (zhvp2_e), `ORCA2/dynspg_ts.f90:535` (zhV), `ORCA2/dynspg_ts.f90:723` (exchange); vector branch `ORCA2/dynspg_ts.f90:666`. UNMEASURED |

## 8. Tests and review

- G1 + G1b card files, dispatch and private-import ratchets: 67 passed, then
  each ratchet 1 failed — both pre-existing at b403770eb (the private
  `_nemo_een_ratio_fold_exchange` import sits at line 6501 of the base
  ocean_model_latlon_cgrid.py; slab_land's lost guard is untouched here).
- G1b card file after the new tests: 13 passed. Plants inside: each new
  selection moved is refused by the deck validator; the predicate is False
  with the option off and True for G1 (flux form, unchanged).
- G1 ladder file (-n 12): 74 passed, 7 failed — the "only the G1 card
  selects" pins index fields from the end (4 already red at the base per
  round 22; my two appended fields broke 3). Expectation wrong, not code:
  now name-ordered checks, 8 passed; helper refuses a missing or reordered
  name.
- Citation tests: 3 failed after the edits (shifted anchors); re-anchored
  with the lane's tool from b403770eb (16 spans: 11 map, 5 in the GYRE
  phase-3 round-8 receipt): 195 passed.
- NOT RUN: G2/G3/G4 ladder suites; closed cards (GYRE, VORTEX, TSUNAMI,
  ORCA2, DINO). Default paths read as unchanged (both options default
  False; callers: their field, reader, the G1b card and validator, the
  ladder's before arms, tests), not measured.

Single review (codex), codex_review_r24.log: **"VERDICT: findings"**.
- "CONFIRMED — np_CRV is wrong on the north-fold F row ... production pads
  that operand with zero". True: section 6, OPEN, not landed. "The area
  offset and cyclic-west mapping are correct."
- "CONFIRMED — the G1b plants are vacuous as physics-path tests". True for
  the unit tests: they pin selection and the helpers, not the operator. The
  physics-path non-vacuity is the measured before arms (section 5: without
  A, first over bar 2; without B, 19), which need the record and are not in
  CI. Not coded this round.
- "Claims 2–5, defaults/callers, restart coverage, signs, and remaining
  staggering checks were otherwise confirmed."

Compliance without a gate: search-before-build (the NEMO-literal curl, the
raw e1f/e2f bundle, the atomic fold unit and its hook, the fold helpers, the
re-anchor tool were all found and reused; new code: two options, one area
helper, one predicate clause, the replay's two terms); one variable per
before arm; instruments' controls quoted before their numbers. Dual review:
NOT done (lane rule 6, single).

## 9. Predictions

| id | result |
|---|---|
| R1 S-P1 no stp_ctl stop | HOLDS (100 steps) |
| R1 S-P2 / R2-S1 |U|,|V| < 0.05 | NOT CHECKED this round |
| R1 V-P1 kt-1 U differs from G1 | NOT CHECKED (no G1-record comparison) |
| R1 F-P1 first non-bit row at kt 1 | HOLDS (u, v at kt 1, 5.2e-18 on ~400 faces, AT-BAR) |
| R1 F-P2 the velocity-form external update | **REFUTED**: kt-1 ssh and uu_b exact; the first large owner is the vorticity's F area |
| R1 F-P3 fallbacks; np_CRV last | HOLDS in the form "np_CRV vorticity" |
| R1 F-P4 no RVO/KEG before kt 1 stage 2 | HOLDS (kt-1 stage-0 vorticity term 0, rhs_terms) |
| R2-G1 geometry | HOLDS, 36/36 |
| R2-L1 a field leaves the bar before 100 | HOLDS (27) |
| R2-L2 first over-bar region is the open interior | HOLDS for INDEPENDENT (kt 27); **REFUTED** for GIVEN-NEMO-ENTRY (fold band, kt 30) |

## 10. Choices

ASKED (round brief): repair the acquisition with an extra twin from 11 and
--admit-existing; analyse the admitted record; ladders, walk, land cited
statements card-selected; housekeeping; tests; one review.

UNASKED, each offered for revert:
1. Two more extra twins (r12, r13) for the 8 blanked steps 92..99 the brief
   did not know of; the admission now checks restart content.
2. `--resume` now admits every finished twin in place and runs only absent
   ones (the brief named --admit-existing; that mode still re-admits all).
3. The ladder's restart_entry scores every step with a previous restart.
4. The G1 ladder's field pins rewritten by name (they fixed 4 pre-existing
   reds too).
5. The lane's re-anchor tool edited a GYRE receipt (5 spans) and the gate map.

UNVERIFIED:
- the ORCA2 column of section 7 (read, not run);
- that closed cards are unchanged (default paths read, not run);
- that run.sh's extra twins write what section 0 predicts (read off
  restart.f90, not run);
- the kt-27 event's cause (PLAUSIBLE).

SHA-256 prefixes (round24/): see sha256_manifest.txt in the evidence folder.
