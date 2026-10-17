# Grid ladder rung G6 — round 8 receipt (gridladder_r44)

**Superseded by "Resume 2" at the end: G6 CERTIFIED (every field at the
bar under both labels, kt 1..6).**

Status: **LANDED (one G6-selected statement); G6 NOT CERTIFIED;
STOPPED FOR RECORD.**  The given-entry label is now at the bar on every
row kt 1..6 (worst 1.5e-16 normalised; round 7: kt-6 v 1.023e-15).  The
independent label is still first over the bar at kt 4 ssh (2.15e-15
normalised, 2,447 cells; round 7: 2.25e-15, 7,529).  Its first departure
is kt 1, RK3 stage 2, inside `dyn_adv`; splitting KEG / ZAD / the stage
`ww` needs terms NEMO has not recorded: `ACQUISITION_NEEDED`.

Preregistration: none.  DEVIATION: this round measured before
preregistering (the stage walk found the statement, the re-score followed);
no prediction is claimed.  Records: round 42 `oracle_g6_r42/twin_s`
(stage frames, stp_2D terms, kt-1 stage-2 terms) and round 36
`oracle_g6_r36` (ladder).  Evidence: `gridladder_rounds/round44/`
(`final/`, `walk/`, `nonvacuity_*.log`, `stages_kt3_*`).  Re-score stamped
`23dfc33b06` (clean tree).  SHA-256 prefixes: given kt 3-6
`694ca049dedeabb4`, independent kt 1-3 `7f7db3ecf42930e1`, kt 4-6
`9b3d0b97881a37bc`.

## 1. Test hygiene (item 1)

The four kt-3 tests now assert 0 differing wet cells against NEMO's record;
no assertion bounds a host-measured count.  Each fails with its piece
removed (removed, run, restored with `git checkout --`, `git status
--porcelain` empty after each):

| piece removed | test that fails | its row (with -> without) |
|---|---|---|
| step-entry wzv stored ratios | step-entry wzv | `dyn_spg_ts` exit uu_b 0 -> 1,337 |
| stage-2 clock pair (this round) | stage-2 momentum wzv | stage-2 level-1 v 0 -> 54 |
| momentum-call stored ratios | all four (ERROR) | the clock pair raises without them |
| `e1e2t*ww` rounding | `e1e2t*ww` | step-end ssh 0 -> 568 |
| u-call `dyn_zad` association | u-call `dyn_zad` | after-ZAD u 0 -> 2,399 |

The old `<= 22` (ww) and `<= 2` (after-ZAD v) lines are deleted: the
exit and ssh rows carry those tests exactly.  The old `<= 38 / <= 46`
step-end lines are replaced by the stage-2/3 rows the momentum wzv owns.

## 2. Walk (item 2)

One-step substitution from NEMO's state: from restart 2 and 4 every row of
kt 3, 4, 5, 6 is at the bar (table §4), so no single later step owns the
independent kt-4 ssh departure.  The chain's first departure is kt 1
(1 v cell).  Stage arm on the r42 record, level 1 / all levels:

| kt | stage 1 | stage 2 u, v | stage 3 u, v |
|---|---|---|---|
| 1 (from rest) | exact | 1 / 22, 0 / 0 | 0 / 1, 1 / 32 |
| 3 (restart 2), before | exact | 61 / 557, 54 / 446 | 38 / 396, 46 / 460 |
| 3 (restart 2), after | exact | 1 / 2, 0 / 1 | 0 / 2, 1 / 17 |

At kt 3 stage 2 the card's momentum `wzv` used the full-step pair
(rn_Dt, r3ta).  NEMO's stage 2 uses rDt = rn_Dt/2 and the stored
half-sum `r3t(Kaa) = r1_2*(r3t(Kbb) + r3ta)` (np_HYB).  Equal only
algebraically; landed (§3).

kt 1, stage 2, source-ordered terms (`oracle_rkstage2_terms_kt00000001.bin`):
after HPG 0 / 0, after VOR 0 / 0, after `dyn_adv` (KEG + ZAD) u 3 cells
(levels 4, 5, 11; 4.2e-22 normalised), v 0.  NEMO records nothing between
KEG and ZAD and not the stage-2 `ww`, so the first non-bit statement is
one of: stage-2 `dyn_keg`, stage-2 `dyn_zad`, or the stage-2 `ww` it
reads.  **ACQUISITION_NEEDED**: `run_r44_stage_terms.sh` (§5).

An offline numpy replay of the stage-2 `ww` from the card's own stage
operands was tried and dropped: it needs NEMO's stage-2 `r3u(Kmm)` /
`r3v(Kmm)` (face ratios from the stage-1 sea level), which no record holds.
The new record writes them.

## 3. Landed (`23dfc33b06`)

| NEMO statement (SUBROUTINE `stp_RK3_stg`) | G5 r31 build / ORCA2_OMIP_L4 |
|---|---|
| `rDt = r1_2 * rn_Dt`; `r1_Dt = 1._wp / rDt` | `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stprk3_stg.f90:188-189` / `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:188-189` |
| `r3t(:,:,Kaa) = r1_2 * ( r3t(:,:,Kbb) + r3ta(:,:) )` | `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stprk3_stg.f90:221` / `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:221` |
| `CALL wzv( ..., uu(Kmm), vv(Kmm), ww, np_velocity )` | `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stprk3_stg.f90:327` / `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:327` |

Same text at the same lines in both builds.  `n_baro_upd` is the
module default np_HYB (no namelist control).

Code: new config bool `nemo_stage2_momentum_wzv_half_step` (default
False; G6 card True; deck check `stage-2 momentum wzv half step`).  The
ONE shared `nemo_qco_wzv_operands` gains `half_step_after_ratio`
(stored ratio half-sum, rounded; raises unless `stored_r3_ratios`); the
stage-2 momentum call passes `dt/2` with it.  Caller grep of the field:
the G6 branch of the card file and the stage-transport helper only.
The private round-160 arm `stage2_momentum_wzv_clock_pair` (heights
averaged, not ratios) is untouched; the new flag is ignored when that arm
supplies a clock.  Default False traces no new branch, so other cards'
programs are unchanged by construction (not diffed jaxpr by jaxpr).
Tracer-call `wzv` at stage 2 keeps the full-step pair: its T/S rows are
exact (registered, not changed).

## 4. Both labels, kt 1..6 (level 1 + 2-D; normalised max / differing cells)

| kt | label | ssh | uu_b | vv_b | u | v | T | S |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | indep. = given | 0/0 | 0/0 | 0/0 | 0/0 | 1.0e-17/1 | 0/0 | 0/0 |
| 2 | indep. = given two-step | 1.5e-16/78 | 2.8e-17/109 | 3.3e-17/103 | 1.0e-17/109 | 3.8e-17/103 | 0/0 | 0/0 |
| 3 | independent | 4.8e-16/1118 | 1.5e-16/1316 | 1.4e-16/1358 | 4.9e-17/1185 | 6.7e-17/1151 | 0/0 | 0/0 |
| 4 | independent | **2.15e-15**/2447 | 1.7e-16/2635 | 1.9e-16/2678 | 2.6e-16/2332 | 1.8e-16/2286 | 7.4e-18/2 | 1.8e-16/1 |
| 5 | independent | 6.4e-16/4521 | 1.9e-16/5073 | 2.1e-16/5039 | **3.55e-15**/4248 | **2.61e-15**/4316 | 5.9e-17/7 | 1.8e-16/6 |
| 6 | independent | **1.91e-14**/7345 | **2.01e-15**/7952 | **2.75e-15**/7973 | **5.67e-15**/6408 | **6.30e-15**/6660 | 1.5e-17/4 | 1.8e-16/7 |
| 3 | given (restart 2) | 0/0 | 0/0 | 0/0 | 0/0 | 8.3e-18/1 | 0/0 | 0/0 |
| 4 | given two-step | 0/0 | 0/0 | 0/0 | 2.8e-17/1 | 1.0e-17/4 | 0/0 | 0/0 |
| 5 | given (restart 4) | 0/0 | 0/0 | 0/0 | 1.4e-17/1 | 0/0 | 0/0 | 0/0 |
| 6 | given two-step | 0/0 | 0/0 | 0/0 | 2.7e-17/3 | 3.7e-18/3 | 0/0 | 0/0 |

Bold = DEBT (bar 1e-15).  All-level rows (full / partial): given two-step
kt 4 u 2.8e-17 / 0, v 1.0e-17 / 5.2e-18; kt 6 u 2.7e-17 / 3.3e-18, v
9.7e-18 / 2.4e-18; T, S exact.  Independent kt 2 u 6.4e-17, v 1.4e-16;
kt 4 u 3.5e-16 / 2.3e-16, v 2.4e-16 / 2.0e-16 (AT-BAR); kt 6 u 5.7e-15 /
5.2e-15, v 4.2e-15 / 3.1e-15 (DEBT).  T, S AT-BAR everywhere.  Flushed
subnormals (Decision 127): 0 in every run.

Versus round 7: every given-entry row equal or better (kt-6 two-step v
1.023e-15 / 2,183 cells -> 3.7e-18 / 3).  Independent: differing cells
fall on every row (kt 4 ssh 7,529 -> 2,447); kt 4 ssh and kt 5 ssh
improve (kt 5 ssh leaves DEBT); kt 5 u/v unchanged; the kt-6 maxima grow
(ssh 1.48e-14 -> 1.91e-14, uu_b 1.06e-15 -> 2.01e-15, vv_b 7.4e-16 ->
2.75e-15 — a new DEBT row, u 1.79e-15 -> 5.67e-15, v 3.14e-15 ->
6.30e-15).  The worst cells sit in a few interior Southern Ocean columns
(kt 4 ssh at the same cell in both rounds; kt 5 u at the same cell).
PLAUSIBLE: the kt-6 growth is the chain amplifying its unchanged kt-1/kt-2
seeds through a different set of cells, not this statement's own error
(every one-step row from NEMO is at the bar).  First over-bar row
unchanged (kt 4 ssh).  Registered under D96 as near-zero class; flagged
as the round's decision below because one new DEBT row appears.

**Certification: NOT certified.**  Given-entry at the bar kt 1..6;
independent over the bar from kt 4.

## 5. Acquisition (operator)

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_g6_orca2vec/run_r44_stage_terms.sh`
(default read-only preflight; `--run` builds `ORCA2G6_OMIP_L4_R44STGTRM`
from G6's pinned record binary and runs 6 steps).  Two additive WRITE-only
patches: `MY_SRC/stprk3_stg.F90` (stages 2-3, kt 1..6: the stage momentum
`ww`, `r3t(Kbb/Kmm/Kaa)`, `r3u/r3v(Kmm)`, `ssh(Kbb/Kmm/Kaa)`, then `Krhs`
before/after HPG, after VOR, after ZAD) and NEMO's `dynadv.F90` copied
into `MY_SRC` with one write after `dyn_keg` (unit set only around the
dumped call).  Admission: every FIX1 record file kt 1..6 byte-identical to
round 36, restarts 2/4/6 equal the writer-free reference, payload size and
finiteness per file; compiled sources additive, every other ppsrc file
identical.  Next round: admit, then walk kt-1 stage 2 KEG / ZAD / `ww`.

## D122 pointer (ORCA2 lane)

The landed statement runs on ORCA2 at the ORCA2_OMIP_L4 lines in §3.
Import waits for G6 certification.

## Tests and review

All CPU fp64, `JAX_ENABLE_X64=1`, PYTHONPATH as in `claude_rounds/land.sh`.
- Default gate (land.sh list): private imports, TKE terms, freshwater
  closure, stage face-mask rank, barotropic state, NEMO recipe 125 passed
  (run at `180f88919`; later commits touch one comment line and the gate
  map only); citation-gate test + private imports 23 passed (final tree).
- G2-G5, G5 top drag, G7 cards: 85 passed.  G6 card + G2/G6 ladder
  (incl. the four rewritten kt-3 tests and the acquisition patch test):
  27 passed.  G1/G1b cards, G1 tools, G1 ladder: 117 passed.  Card
  constructibility + `dyn_zad` association unit: 24 passed.
- Removal checks: five pieces, each removed alone, kt-3 tests FAIL (§1);
  logs `round44/nonvacuity_*.log`.  Acquisition patch test FAILS with the
  after-VOR write planted out.
- Receipt citation gate on THIS receipt (from its title): 6 citations,
  failures 0, unmapped 0, map audit 0; a planted shift of the G5-build
  stage-2 `r3t(Kaa)` citation (line 221) FAILS it (`SYMBOL-NOT-AT-LINE`).  Five
  round-44 map entries added; 73 gate pins and 21 lines of the GYRE
  phase-3 round-8 receipt re-anchored by the repo's tool (mechanical, from
  `eab4efd114`); one round-160 span the landing lengthened widened by hand.
- codex review: paused (quota).  The operator runs an independent Claude
  review of this diff before landing.

ASKED: test hygiene, the walk, land the cited statement G6-selected,
revert-failing tests, both labels kt 1..6, the acquisition.

UNASKED, offered for revert: the acquisition's patch test (a new test for
a new script); the field-order pins moved by one for the appended field;
the mechanical citation re-anchor; one stale code comment reworded.

UNVERIFIED: XLA's rewrite at the stage-2 momentum wzv is not read from
HLO (the statement is an algebraic-only difference, fixed by transcription);
"other cards unchanged" is by construction, not a jaxpr diff; the
acquisition script has not been run (dry-applied patches only).

## Resume: acquisition attempt 1 refused at admission (writer defect)

The operator ran `run_r44_stage_terms.sh --run` at `3d9fbf8a42`: build
additive, 6 steps, restarts 2/4/6 passive (27 variables), then admission
refused every stage-term file (9 of 11 3-D fields + 8 2-D).  From the
files: each holds the header, `ww` + 8 2-D fields, and FOUR `Krhs` pairs.
Against the same run's stage-2 term record (wet cells, kt 1 stage 2) they
are before-HPG, after-HPG (0 differing), after-VOR (0), after-ZAD (0);
the after-KEG pair is absent.  Cause: the copied `dynadv.F90` wrote only
when its unit was `>= 0`, and `OPEN(NEWUNIT=)` returns a negative unit, so
the write never fired.  The checker was right; the record cannot split KEG
from ZAD, so `--admit-existing` does not apply.

Tool fix: the after-KEG write is gated by a public logical, set and
cleared around the dump's `dyn_adv` call; new config
`ORCA2G6_OMIP_L4_R44STGTRM2`, new evidence `round44/oracle_g6_r44b`.  The
failed build `ORCA2G6_OMIP_L4_R44STGTRM` and `round44/oracle_g6_r44` are
left untouched as evidence.  The patch test now requires the logical gate
and refuses a sign test on the unit; planting the old `>= 0` guard back
FAILS it (restored with `cp -f`, `git diff` shows only the fix).

ACQUISITION_NEEDED: `run_r44_stage_terms.sh --run` (preflight first).
codex review: paused (quota).

ASKED: diagnose the refusal from the files, fix the tool, new config and
evidence names.  UNASKED: none.

UNVERIFIED: the patched Fortran is not compiled here (the operator's
build is the check); the logical's visibility in `stprk3_stg` follows the
same module chain that already exposed the unit in attempt 1.

Gates (resume, at `b8436bb464`): receipt citation gate on this receipt
6 citations, failures 0, unmapped 0; citation-gate test + private
imports 23 passed; patch test 1 passed (fails with the old guard
planted); preflight `G6R44_PREFLIGHT_OK` target
`ORCA2G6_OMIP_L4_R44STGTRM2`.  NOT RE-RUN to completion: the model-step
gate files (TKE terms, freshwater closure, stage face-mask rank,
barotropic state, NEMO recipe) each exceeded 10 minutes inside XLA
compilation under machine load 35-55; this resume changes no model code
(a shell script, one test, this receipt), and they passed at `180f88919`.

## Resume 2: record admitted, first departure walked, G6 CERTIFIED

Status: **LANDED (one G6-selected statement); G6 CERTIFIED.**  Every field
is at the bar under both labels, kt 1..6.  Independent kt 1..5 and
given-entry kt 3..5 are bit-identical on every level-1 and 2-D row; the
worst row anywhere is 3.65e-18 normalised.

Record: operator run of `run_r44_stage_terms.sh --run` at `b8436bb464`,
`G6R44_ACQUISITION_PASS steps=1..6 stage_terms=kt1..6,s2-3`, evidence
`gridladder_rounds/round44/oracle_g6_r44b` (build
`ORCA2G6_OMIP_L4_R44STGTRM2`, nemo.exe SHA-256 prefix `b3c3dc0f58d402bd`;
log `acquisition_r44_stage_terms_b.log`).  Attempt 1 (`oracle_g6_r44`)
stays as evidence.  Preregistration: none (DEVIATION, as in round 8).
Evidence of this resume: `round44/walk_b`, `round44/final_b`,
`round44/jaxpr`, `round44/probe_b`, `round44/resume2`.

### 1. Admission

- Passivity: compiled sources additive (both patched files), every other
  compiled file identical; every FIX1 record file kt 1..6 byte-identical to
  round 36; restarts 2, 4, 6 equal the writer-free reference on all 27
  variables.
- Headers: 12 files, kt 1..6 x stages 2-3, `NEMO_G6_STGTRM_1`, version 1,
  64-bit reals, 2-cell halo, 184 x 152 x 31; payload 11 3-D + 8 2-D fields,
  finite.  The five Krhs pairs are present (attempt 1 had four).
- Matched time levels: stage 2 slots (Kbb, Kmm, Kaa = Krhs) are (1, 3, 2)
  at odd kt and (3, 1, 2) at even kt; stage 3 (1, 2, 3) / (3, 2, 1).  These
  are NEMO's slot swaps around
  `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stprk3.f90:222` /
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:218`.
- Cross-record check: after HPG, after VOR and after ZAD equal the older
  stage-2 term record bit for bit on wet cells.  The raw dump also holds
  NEMO's dry-point and halo values; the older record is wet-canonicalised.

### 2. Walk: kt 1, RK3 stage 2 (from rest, both labels)

| boundary, in compiled order | u | v |
|---|---|---|
| stage momentum `ww` (T cells, all levels) | 137 of 430,552 differ, 16 columns, max 5.3e-20 (2.3e-17 normalised) | |
| Krhs after HPG | 0 | 0 |
| Krhs after VOR | 0 | 0 |
| Krhs after KEG | 0 | 0 |
| Krhs after ZAD | 3 (4.2e-22) | 0 |

The `ww` is computed first (the stage momentum `wzv` precedes HPG), so it is
the first non-bit operand.  KEG is exact, so the 3 after-ZAD cells come from
`ww`.  NEMO's statement: `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/sshwzv.f90:297` /
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sshwzv.f90:297` (same text).  Inside it, the
card's stage face thickness e3u(Kmm) differed from NEMO's
e3u_0*(1+r3u(Kmm)) on 21 u faces, each next to one of the 16 columns.

Cause (transcription, not an XLA reciprocal or FMA rewrite): at stage 2,
NEMO's Kmm ratios are not rebuilt from the stage sea level.  They are the
blends stage 1 stored,
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stprk3_stg.f90:177-179` /
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:177-179`
(r2_3 x ratio at Kbb + r1_3 x ratio of the after level), whose after-level
ratios come from
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stprk3_stg.f90:167` /
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:167` and
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/domqco.f90:266` /
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/domqco.f90:266`.  Stage 3 reads the half
blends `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stprk3_stg.f90:221-223` /
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:221-223` as its Kmm.
The card took the ratio of the blended sea level: equal as real numbers,
different in the last place on rare faces.

Instrument defect found and fixed on the way: the stage `dyn_zad` operand
observer published `w` BEFORE the production substitution of the stage
transport's `ww`, so it showed a stale field (2.3x NEMO's).  It now also
publishes the operands `dyn_zad` consumes; the old keys are unchanged.

An offline numpy replay of NEMO's `ww` from the record's own operands did
NOT reproduce NEMO's `ww` (4,649 cells differ, mostly on the southernmost
wet row, PLAUSIBLE: the runoff term is missing from the replay).  It is not
used as evidence; the attribution rests on the card arm below (137 -> 0).

### 3. Landed (`4054231a7a`)

New config bool `nemo_stage_momentum_wzv_stored_kmm_ratios` (default False;
G6 card True; deck check "stage momentum wzv stored Kmm ratios").  The ONE
shared `wzv` operand builder gains `stored_kmm_level` (1 = N+1/3, 2 =
N+1/2): Kmm's T, U and V ratios are the stored blends of the step-entry and
after-level ratios.  It raises unless the after level is handed in and the
stored-ratio option is on.  The blend helper moved into the shared module
(public name; the model module keeps its old name by import), so no second
copy exists.  Search: grepped `r2_3`, `_nemo_stage_ratio`, `stage_pair`,
`r3_native` under `packages/`; found the blend helper and the face-geometry
`r3_native` input, both reused.

Both stage-2 and stage-3 momentum `wzv` calls select it.  The stage-2 arm is
measured (137 -> 0, after-ZAD 3 -> 0); the stage-3 arm is carried by the
step-end rows below, not by its own stage-3 term row.

Other cards: jaxprs of one ordinary step for G5 and G7 are identical to
`1593038d2d` after stripping callback addresses (SHA-256 prefixes G5
`052f740ff7baa019`, G7 `327d43b08c23911b`, both trees); G6's changes.
Caller grep of the field: the G6 branch of the card file and the stage
helper only.

### 4. Both labels, kt 1..6 (level 1 + 2-D; normalised max / differing cells)

| kt | label | ssh | uu_b | vv_b | u | v | T | S |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1-3 | independent (kt 1-2 = given) | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| 4 | independent | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| 5 | independent | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| 6 | independent | 0/0 | 0/0 | 0/0 | 8.4e-19/2 | 3.65e-18/3 | 0/0 | 0/0 |
| 3 | given (restart 2) | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| 4 | given two-step | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| 5 | given (restart 4) | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| 6 | given two-step | 0/0 | 0/0 | 0/0 | 8.4e-19/2 | 9.1e-19/2 | 0/0 | 0/0 |

All levels (full / partial cells): kt 4 v 1.29e-18 (6) / 0 under both
labels; kt 6 independent u 1.34e-17 (47) / 3.34e-18 (1), v 9.65e-18 (54) /
4.83e-18 (4); kt 6 given u 6.69e-18 (33) / 3.34e-18 (1), v 9.65e-18 (39) /
2.41e-18 (3); T, S exact.  Flushed subnormals (Decision 127): 0 on every
row.  The kt-6 independent meridional barotropic velocity (round 8:
2.75e-15, 7,973 cells) is exact; so is round 8's first over-bar row (kt-4
ssh 2.15e-15).  Re-score stamped `4054231a7a` (clean tree; later commits
change comments, tests and the citation map only).  SHA-256 prefixes:
independent kt 1-3 `e099f70d62d72dbb`, kt 4-6 `5599fb33b7997911`, given
kt 3-6 `8c3a3ad96d1fb001`.

**Certification: CERTIFIED.**  Residual last-place cells at kt 4 and 6 are
AT-BAR and recorded above.

### D122 pointer (ORCA2 lane)

The landed statement runs on ORCA2 at the ORCA2_OMIP_L4 lines in §2
(same text, same lines as the G5 round-31 build).  G6 is now certified, so
its D122 import is open to the ORCA2 lane.

### Tests and review

All CPU fp64, `JAX_ENABLE_X64=1`, PYTHONPATH as in `claude_rounds/land.sh`.
- New test: kt-1 stage-2 momentum `ww` and after-ZAD bit-exact on the r44b
  record through the ordinary jitted step.  With the wiring removed it FAILS
  `137 == 0` (log `round44/nonvacuity_e_kmm_ratios.log`; restored with
  `git checkout --`, tree clean).  Card test pins the new field.
- Default gate (land.sh list): NEMO recipe 25 passed; stage face-mask rank +
  barotropic state 10 passed; citation-gate test + private-import ratchet +
  TKE terms + freshwater closure 107 passed (at `ebbfeabd21`).
- G2-G5, G5 top drag, G7 cards: 85 passed.  G6 card + G2/G6 ladder: 27
  passed, 1 failed (the config field-order pin; moved by one for the
  appended field, then 1 passed).  G1/G1b cards passed; G1 tools 31 passed;
  G1 ladder 86 passed (re-run with 8 workers after an XLA compile abort under
  machine load in the shared battery).
- Pre-existing, not touched: 3 failures in the `zad` qco-coupled unit file,
  identical at `1593038d2d`.
- Citation pins: 84 gate pins and 23 lines of the GYRE phase-3 round-8
  receipt re-anchored by the repo's tool from `1593038d2d` (mechanical); the
  momentum `wzv` span the landing lengthened (8 -> 12 lines) widened by hand.
- Receipt citation gate on THIS receipt (from its title): 18 citations,
  failures 0, unmapped 0, map audit 0, self-test fired; a planted shift of
  the G5-build `:177-179` blend citation FAILS it (`SYMBOL-NOT-AT-LINE`).
  Twelve map entries added for this resume.
- codex review: paused (quota).  The operator runs an independent Claude
  review of this diff before landing.

ASKED: admit the record, walk the kt-1 stage-2 departure, land the
transcription defect G6-selected, failing-when-removed test, jaxprs of other
cards, re-score both labels kt 1..6, gates, receipt.

UNASKED, offered for revert: the observer's extra consumed-operand keys
(needed to measure the stage `ww`); moving the blend helper into the shared
module; selecting the stage-3 call under the same switch (same statement
form, not separately measured); the mechanical citation re-anchor.

UNVERIFIED: the stage-3 arm has no stage-3 term row of its own; the numpy
replay of NEMO's `ww` is not validated; XLA's HLO at the changed call was not
read (the defect is an association choice, fixed by transcription).

## Resume 3: a test for the stage-3 stored-Kmm selection alone

Status: **LANDED (test only); G6 stays CERTIFIED.**  The independent
Claude review reproduced the certification and held one item: removing
ONLY the stage-3 selection failed no test (17 passed) while moving
independent kt-6 ssh (0 -> 864 cells, at the bar).

Added: the operators arm also reads NEMO's kt-1 stage-3 term record
(`oracle_g6_stage_terms_kt00000001_s3.bin`, slots checked = (1, 2, 3))
and captures the stage-3 momentum `ww` through the existing stage-3
`dyn_zad` operand observer.  The existing kt-1 r44b test (ordinary jitted
step, x64, traced inputs) now also asserts the stage-3 `ww` has 0 differing
wet cells against NEMO.  The stage-3 Kmm is the half blend NEMO stores,
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stprk3_stg.f90:221-223` /
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:221-223`.

- With the selection: 1 passed (`round44/resume3/stage3_ww_with.log`).
- Stage-3 selection alone removed (stage set `(1, 2)` -> `(1,)`): FAILS
  `121 == 0` (`round44/resume3/nonvacuity_stage3_selection.log`); restored
  with `git checkout --`, `git status --porcelain` shows only this resume's
  two files.

Search: grepped `stage3_zad_operand_observer`, `read_g6_stage_terms`,
`stage2_ww` in the ladder script and model; reused the observer, reader
and `bit_rows`.  No model code changed.

codex review: paused (quota).  The operator runs an independent Claude
review of this diff before landing.

ASKED: the stage-3 test, shown failing and restored, gates.  UNASKED: none.

UNVERIFIED: none beyond Resume 2.
