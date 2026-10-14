# Grid ladder rung G2 — round 3 receipt (gridladder_r12)

Status: **LANDED** (one cited NEMO statement, selected only by the G2 card).
**G2 is NOT certified**: the 100-step record leaves the bar at step 16
(sea level and U), where before it left at step 2.
Preregistration: docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g2_round3.md
(commit 39281221f, before any comparison of the card with the record).
Code: 82bd5d3c5 (the landing), 613b95451, 6edcd72a4, 6ab2b9519 (review
fixes). Every number below was regenerated from the clean commit 6ab2b9519
(stamp in each json; later commits touch citation pins and this file only).
Record: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round10/oracle_g2_r10
(twins a, b) and round11/oracle_g2_r11_consecutive (twin c).
Evidence: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round12/
(final numbers under final/). NEMO citations are to the compiled sources
under ORCA2G1_OMIP_L1_P3; the four files cited are byte-identical in the
record binary's build ORCA2G1_OMIP_L1_P5 (cmp, this round).
**Acquisition needed:** none.

## Take-aways

1. **The brief's premise about twin c was wrong, and so was the operator's
   reading of it.** The file named for step 1 is not NEMO's state after step
   1. NEMO created it at step 10 and never wrote into it: 0 time records.
   No all-level state exists at any odd step. The check's refusal stands;
   the check now states what NEMO wrote and admits the run on that basis.
2. **The 1e-13 term is a last-place difference in temperature and salinity,
   amplified by the pressure gradient of a 5 km column.** Measured four ways
   before any code (section 3). It was never in the velocities at depth.
3. **First non-bit statement: the diagonal of NEMO's vertical tracer solve.**
   NEMO rounds the layer thickness, then subtracts. The card's compiled code
   did both in one rounding. Same real number, one unit in the last place at
   19,386 wet cells of step 1.
4. **Landed, G2-card-selected.** After it T and S equal NEMO bit for bit on
   all 30 levels at steps 1 to 12, and the second step sits at 1e-17.
5. **100 steps.** Sea-level error at step 100: 1.55e-12 m before, 6.4e-14 m
   after. First step over the bar: 2 before, 16 after.
6. **What is left is one more last-place seed, located but not named.** The
   card's barotropic solver equals NEMO bit for bit through all 65 substeps
   when run alone. Inside the full step its result differs in 591 cells by
   one unit in the last place. With the solver's own result handed back,
   step 1 is NEMO's bit for bit on every scored field.
7. **G1 is unchanged.** Its 100-step certification test passes on this tree.

## 1. Admission: what twin c holds

- NEMO's log of twin c (ocean.output): finished, 10 steps, no error line.
  One restart file was opened, at kt = 10, named `ORCA2G1_00000001_restart`.
- That file: 215,260 bytes (a written restart is 30,265,184), time dimension
  of length 0, 31 variables defined, none written.
- Why, in NEMO's own statements. The list pointer starts at the first entry,
  so the next restart step is 1. A file opens one step before its restart
  step (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:119`); no step is
  0, so nothing opens. The same line has an end-of-run fallback, which fires
  at step 10 and creates the file named for step 1. The write is complete
  and the pointer moves only when the step equals the restart step
  (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:188`,
  `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:199-202`); 10 is not 1, so
  the variables are defined and left empty.
- Correction to the operator's note: "only a restart every step opens every
  step" is right, but the step-1 file is not a state, and consecutive list
  entries are not the cause: a list that starts at 1 opens nothing at all.
  A list of odd steps from 3 on would open and write (3, 5, 7, 9). The
  multiple-of-two rule applies only without a list
  (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/sbcmod.f90:346`). READ OFF THE CODE,
  not run.
- Re-admission, NEMO not re-run (g2_admit_existing_r12.log, commit
  6ab2b9519): twin c's ten step streams equal twin a's byte for byte and its
  two substep streams equal twin a's where read; its one restart file must
  hold no record; twin b's restarts of steps 2, 4, 6, 8, 10 each equal the
  recorded after-step state on the 2-D fields and level 1; twin b's step-10
  restart equals twin a's on all levels. Three plants refuse (a flipped bit
  in a step stream, a moved level-1 value in a restart, a record in the
  empty file).
- Limit, from the review: levels 2 to 30 of twin b's restarts at steps 2, 4,
  6, 8 have no second witness. Twin a wrote all levels only every tenth step.

## 2. Predictions

| prediction | outcome |
|---|---|
| R3-P1 nothing outside the restart set carries the term | HOLDS: rebuilt entry equals the chain bit for bit (0 difference on sea level, u, v, T) |
| R3-P2 the term does not enter through level 1 or the 2-D fields | HOLDS: with NEMO's exact level 1 and 2-D fields put in, step-2 U error 1.44e-13 (1.44e-13 without) |
| R3-P3 a last-place change in T is enough | HOLDS: one unit in the last place of T on every wet cell moves step-2 U by 8.6e-14; of S, by 1.6e-13 |
| R3-P4 level-1 T after step 1 differs by exactly one unit | HOLDS: 770 cells, largest 3.55e-15; S 768 cells, 7.1e-15 |
| R3-P5 not at the bar after this round | HOLDS |
| R3-P6 G1 unchanged | HOLDS: its 100-step certification test, 4 passed, final tree |

## 3. The second-step term, before any code

Entry = the exact initial state; scored at step 2 on all levels against twin
b's restart. One thing varied per arm in the card's state after step 1.
Exploratory runs (uncommitted probe); the committed arms of section 4
reproduce the result that matters.

| arm | sea level | U all levels | V all levels |
|---|---|---|---|
| nothing varied | 1.12e-13 | 1.44e-13 | 1.85e-13 |
| entry rebuilt from the restart set only | 1.12e-13 | 1.44e-13 | 1.85e-13 |
| NEMO's level 1 and 2-D fields put in | 1.11e-13 | 1.44e-13 | 1.85e-13 |
| T moved one unit in the last place (against the first row) | 1.16e-13 | 8.6e-14 | 1.88e-13 |
| S moved one unit in the last place (against the first row) | 1.71e-13 | 1.56e-13 | 4.01e-13 |

The U error is the same size at levels 1, 5, 10, 19 (1.4e-13): a
whole-column signal. Scale check: one unit in the last place of the density
anomaly (3.5e-18), times gravity, times 5,000 m, over a 50 km cell, times
one step of 10,800 s, is 4e-14 m/s per level touched.

## 4. The first non-bit statement, and the landing

Tracers have no advection and no lateral mixing on this deck. One step
changes T only by the thickness change and the implicit vertical solve.

**Offline replay** (rule 5; final/trazdf_replay_*.json). NEMO's statements
of that solve in numpy, one rounding per operation: the matrix
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/trazdf.f90:233` is its diagonal) and
the three recurrences
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/trazdf.f90:269-298`), with the
constant diffusivity times the interface mask
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/zdfphy.f90:227`). Inputs: the exact
initial T and S, NEMO's three recorded thickness ratios, NEMO's mesh file.

| quantity (step 1) | T | S |
|---|---|---|
| replay against NEMO, level 1, cells not bit-equal | 0 | 0 |
| card against NEMO, level 1, before | 770 (3.55e-15) | 768 (7.1e-15) |
| card against replay, all levels, before | 17,079 (7.1e-15) | 19,339 (1.07e-14) |
| the same, full cells / partial cells | 16,535 / 544 | 18,682 / 657 |
| fewest unequal cells on any one level, before | 294 | 369 |
| card against NEMO, level 1, after | 0 | 0 |
| card against replay, all levels, after | 0 | 0 |

The replay equals NEMO on every level-1 cell, so it is NEMO's solve; below
level 1 it is the same statements, not a recorded state.

**Which operand** (exploratory; the card's own solve trace against the
replay's arrays, wet cells):

| operand | cells not bit-equal, before |
|---|---|
| content, after thickness, interface thickness | 0, 0, 0 |
| lower and upper coefficients | 0, 0 |
| diagonal | 19,386 (largest 1.1e-13 on values up to 500) |
| forward sweep | 8 |
| solution | 17,079 |

**Why** (MEASURED on a sample): in 4,000 of 4,000 differing cells, and in
4,000 of 4,000 agreeing cells, the card's diagonal is the exactly rounded
value of thickness times (1 + ratio) minus the coefficient sum, taken as
ONE operation. NEMO takes two. The compiler fuses the multiply into the
subtraction; the existing helper that forbids this in the sweeps was not
applied to the diagonal.

**Control on the mechanism** (final/control_before_nofma_env_kt2.log): with
the backend told not to use fused instructions
(XLA_FLAGS=--xla_cpu_max_isa=AVX), the card WITHOUT the option gives T and
S bit-equal to NEMO at step 1 and on all levels at step 2. And the card
WITH the option gives the same 100-step numbers to every printed digit with
that flag and without it (final/independent_after_nofma_env.log, run at
82bd5d3c5). So this was the only fusion that reached a scored number.

**Landed.** Option zdf_tracer_nemo_rounded_thickness, default False,
selected by the G2 card only, through its own two-entry rung table. It
applies the helper the solver already had. Searched before building: the
solver file for an existing rounding guard (found the helper, used in the
two sweeps, not on the diagonal) and the model for an existing tracer-solve
option (found none).

- G1: not selected. One level takes the solver's early exit and builds no
  diagonal.
- Caller grep (committed test): one assignment in the whole package, the
  shared G1/G2 builder's. Library default and every other card unchanged.
  The new field is the last field of the configuration tuple, so no older
  field changes position.
- ORCA2 pointer: ORCA2 runs the same compiled statement. The ORCA2 lane
  does NOT have this. Not selected for ORCA2 here (another lane's card).
- The model refuses the option without NEMO's literal solver.

## 5. Ladders after the landing

Bar: max abs / max(max abs NEMO, 1) <= 1e-15. Numbers are max abs
differences; in brackets, cells not bit-equal. "Before" is this commit's
card without the option; it reproduces round 2's "after" column digit for
digit (1.121e-13, 1.436e-13 at step 2; 1.554e-12, 3.119e-12, 5.016e-12 at
step 100), so the two arms differ in the one option.

**INDEPENDENT** (the card's own chain from its own initial state):

| kt | ssh | U mean | V mean | u level 1 | v level 1 | T level 1 | ssh before | u level 1 before |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.13e-17 (612) | 1.18e-18 | 5.42e-19 | 2.09e-18 | 4.46e-18 | 0 (0) | 1.13e-17 | 2.09e-18 |
| 2 | 1.39e-17 (5502) | 1.52e-18 | 1.30e-18 | 2.49e-18 | 4.69e-18 | 0 (0) | 1.12e-13 | 1.44e-13 |
| 3 | 3.21e-16 | 5.20e-18 | 3.04e-18 | 8.67e-18 | 8.15e-18 | 0 (0) | 1.72e-13 | 1.41e-13 |
| 4 | 2.78e-17 | 7.81e-18 | 9.54e-18 | 1.04e-17 | 8.67e-18 | 0 (0) | 1.23e-13 | 1.43e-13 |
| 5 | 3.99e-17 | 1.06e-17 | 9.54e-18 | 3.59e-17 | 2.76e-17 | 0 (0) | 1.42e-13 | 2.09e-13 |
| 6 | 3.47e-17 | 1.56e-17 | 1.21e-17 | 3.32e-17 | 2.85e-17 | 0 (0) | 1.41e-13 | 2.22e-13 |
| 7 | 3.47e-17 | 1.39e-17 | 1.19e-17 | 1.06e-16 | 7.22e-17 | 0 (0) | 1.44e-13 | 1.93e-13 |
| 8 | 4.86e-17 | 1.91e-17 | 2.08e-17 | 8.86e-17 | 5.83e-17 | 0 (0) | 1.67e-13 | 2.22e-13 |
| 9 | 4.69e-17 | 3.47e-17 | 2.43e-17 | 6.68e-17 | 6.46e-17 | 0 (0) | 1.36e-13 | 2.32e-13 |
| 10 | 5.55e-17 | 2.91e-17 | 2.86e-17 | 2.66e-16 | 2.71e-16 | 0 (0) | 1.39e-13 | 2.58e-13 |
| 15 | 4.80e-16 | 2.45e-16 | 2.89e-16 | 4.26e-16 | 4.33e-16 | 3.55e-15 (3) | 1.86e-13 | 4.17e-13 |
| 16 | 2.04e-15 | 8.52e-16 | 2.96e-16 | 1.70e-15 | 7.75e-16 | 0 (0) | 2.08e-13 | 4.48e-13 |
| 20 | 2.73e-14 | 4.96e-16 | 1.21e-15 | 3.38e-15 | 3.37e-15 | 3.55e-15 (2) | 2.14e-13 | 5.19e-13 |
| 40 | 5.54e-14 | 9.88e-15 | 1.23e-14 | 3.67e-14 | 5.46e-14 | 7.11e-15 (48) | 4.59e-13 | 1.06e-12 |
| 60 | 3.80e-14 | 2.24e-14 | 2.54e-14 | 7.37e-14 | 1.40e-13 | not recorded | 6.16e-13 | 1.20e-12 |
| 80 | 5.04e-14 | 4.66e-14 | 4.75e-14 | 1.34e-13 | 4.94e-13 | not recorded | 8.78e-13 | 2.29e-12 |
| 100 | 6.41e-14 | 1.00e-13 | 5.59e-14 | 2.65e-13 | 1.01e-12 | not recorded | 1.55e-12 | 3.12e-12 |

Level-1 T and S frames exist only through step 49 of the record; at steps
50 to 100 T and S are scored on all levels from the restarts below. NEMO's
own maxima at step 100: sea level 0.0119 m, level-1 U 0.0124 m/s.

**Headline: first step over the bar, both labels.**

| | ssh | U mean | V mean | u level 1 | v level 1 | T level 1 | S level 1 |
|---|---|---|---|---|---|---|---|
| INDEPENDENT, before | 2 | 2 | 2 | 2 | 2 | 6 | 5 |
| INDEPENDENT, after | 16 | 23 | 20 | 16 | 20 | none through 49 | none through 49 |
| INDEPENDENT, first step not bit-equal, after | 1 | 1 | 1 | 1 | 1 | 13 | 13 |
| GIVEN-NEMO-ENTRY, one step and two steps, after | none | none | none | none | none | none | none |

At step 16 the sea-level cell over the bar is next to a coast (2.04e-15);
the open ocean is at 3.2e-16, the fold at 1.6e-19, the seam at 1.2e-17, the
south wall at 2.3e-17.

**By depth class** (INDEPENDENT, all levels, where NEMO wrote a restart;
FULL and PARTIAL as in round 2; no dry cell differs at any scored step):

| kt | u full | u partial | v full | v partial | T cells not bit-equal | S cells not bit-equal | u full before | u partial before |
|---|---|---|---|---|---|---|---|---|
| 2 | 1.64e-17 | 1.30e-18 | 6.63e-17 | 1.73e-18 | 0 | 0 | 1.44e-13 | 1.40e-13 |
| 4 | 3.35e-17 | 1.21e-17 | 6.61e-17 | 8.67e-18 | 0 | 0 | 1.43e-13 | 1.38e-13 |
| 6 | 1.66e-16 | 3.20e-17 | 1.78e-16 | 2.57e-17 | 0 | 0 | 2.28e-13 | 2.15e-13 |
| 8 | 2.15e-16 | 8.27e-17 | 2.61e-16 | 2.78e-17 | 0 | 0 | 2.27e-13 | 3.04e-13 |
| 10 | 8.75e-16 | 9.49e-17 | 8.88e-16 | 9.06e-17 | 0 | 0 | 2.68e-13 | 5.08e-13 |
| 20 | 3.41e-15 | 7.99e-16 | 3.40e-15 | 1.44e-15 | 40 | 50 | 5.30e-13 | 9.71e-13 |
| 30 | 2.49e-14 | 3.83e-14 | 3.32e-14 | 5.39e-14 | 301 | 388 | 8.84e-13 | 1.87e-12 |
| 50 | 6.84e-14 | 9.25e-14 | 1.29e-13 | 1.74e-13 | 2,325 | 3,144 | 1.48e-12 | 3.68e-12 |
| 100 | 2.70e-13 | 5.58e-13 | 1.02e-12 | 7.38e-13 | 14,715 | 18,953 | 3.28e-12 | 6.20e-12 |

First restart step over the bar by class, after (before in brackets): u
full 20 (2), u partial 30 (2), v full 20 (2), v partial 20 (2), T full 40
(4), T partial 60 (6), S full 40 (4), S partial 50 (8). Full-class T cells
not bit-equal at step 100: 372,986 before, 13,557 after.

The six barotropic histories against NEMO's restart at step 100: 5.6e-14 to
1.0e-13 (1.2e-12 to 1.6e-12 before).

**GIVEN-NEMO-ENTRY** (NEMO's restart of the step before; entries at kt = 1,
3, 5, 7, 9; final/given_entry_after.log): 90 of 90 rows at the bar. One
step: as round 2 (1.6e-17 or less on sea level). Two steps, all levels:

| entry kt | scored kt | u full | u partial | v full | v partial | T, S cells not bit-equal |
|---|---|---|---|---|---|---|
| 1 | 2 | 1.64e-17 | 1.30e-18 | 6.63e-17 | 1.73e-18 | 0, 0 |
| 3 | 4 | 3.44e-17 | 6.07e-18 | 3.91e-17 | 1.04e-17 | 0, 0 |
| 5 | 6 | 9.41e-17 | 9.97e-18 | 9.24e-17 | 1.05e-17 | 0, 0 |
| 7 | 8 | 8.73e-17 | 1.39e-17 | 2.63e-16 | 2.08e-17 | 0, 0 |
| 9 | 10 | 2.72e-17 | 1.21e-17 | 3.49e-17 | 1.39e-17 | 0, 0 |

Round 2's two-step rows were 9e-14 to 2e-13.

## 6. The open seed (located, not named)

What still differs at step 1: sea level in 612 cells, the barotropic
velocities in 591 and 642, level-1 U and V in 594 and 633, each by one unit
in the last place. From step 13 on that reaches T and S and the column
amplifies it.

Measured this round (final/substeps_kt1.json and exploratory runs):

- The solver's loop run alone, every recorded operand of 65 substeps against
  NEMO: all bit-equal except three rows. Two are NEMO's history arrays, which
  NEMO has not set at step 1. One is a single cell of the fold pivot row at
  substep 65, a number of 1.1e-112.
- The loop's exit run alone against NEMO's exit frame: 0 cells differ, for
  sea level and both velocities. The same with any subset of its operands
  exposed, and with none.
- The full step's barotropic velocities against that exit: 591 and 642
  cells. (The sea-level count printed beside them, 8,383, compares a stage
  value with a full-step value and is not a finding.)
- The full step with the loop's own exit handed back as constants: sea
  level, both barotropic velocities, level-1 U and V, T and S: 0 cells
  differ from NEMO at step 1.
- Not the cause, each measured: the loop driver (scan against indexed loop:
  identical); fused instructions (flag control above: identical).

So the same solver code gives two last-place answers depending on what else
is compiled with it. PLAUSIBLE, not measured: a compiler rewrite of a
statement just before or just after the loop that is allowed only when the
intermediate has a single reader. Handing the exit back does not show
whether step 2 onward would then be bit-equal: the hand-back needs a
second compile per step and was run for step 1 only.

## 7. Gates

| claim | gate | result |
|---|---|---|
| twin c is what NEMO wrote | acquisition check, three plants | admitted; plants refuse |
| the replay is NEMO's solve | level 1 against NEMO; dense-matrix unit test | 0 unequal; pass |
| the landing is needed | replay test and two-step test, each with its before arm | pass (before: over 500 level-1 cells, over 10,000 cells, U over 1e-14) |
| the option does something | solver unit test, compiled, default arm must differ | pass (over 100 of 24,576 cells differ) |
| the option is G2's only | package scan test; resolved-pair test | pass |
| the fusion flag acts inside the ladder | before arm under the flag | T, S bit-equal |
| the traced loop is NEMO's | substep test | pass |
| G1 certification | G1 100-step test | 4 passed |
| citations | citation gate on this receipt | see the last line of this file |
| dual review, claim review before code | NO GATE | single review only (lane rule 6); claim review NOT done |

Focused tests, synchronous, final code tree (tests_final_*.log): the solver
unit file, the two G2 files and the G1 tools file: 51 passed; the five slow
record tests: 5 passed; the G1 100-step certification: 4 passed. NOT RUN:
77 of the 81 tests of the G1 ladder file; any closed-card ladder. Three
repository ratchets fail on files this diff does not touch (a forcing test
file, the land restart module, the slab land dispatcher); not re-run on the
base commit.

## 8. Review

Single review (codex): **"VERDICT: findings"** (codex_review_r12.log; scope
52c1d1850..6edcd72a4). Acted on in 6ab2b9519, not re-reviewed:

- "CONFIRMED — the new `NamedTuple` field is inserted before existing
  fields, breaking positional construction". Fixed: the field is now the
  last field; a test pins that.
- "CONFIRMED — admission claims twin-b provides the all-level states but
  checks only SSH, barotropic velocities, and level-1 U/V". Partly fixable:
  the check now also requires twin b's step-10 restart to equal twin a's on
  all levels. For steps 2, 4, 6, 8 no second witness exists; stated in
  section 1.
- "PLAUSIBLE — two independent scientific choices are coupled through
  `_ORCA2G_ZDF_FACES`". Fixed: the tracer option has its own table.

## 9. Corrections to my own work, and choices

Corrections:
1. My first substep test asserted two unequal rows; there are three (the
   1.1e-112 cell). The test now pins all three.
2. The tracer replay arm first refused NEMO's own thickness array (nonzero
   on NEMO's level 31). Fixed before any number came from the arm.
3. I first reported the full step's sea level as "8,383 cells off the
   solver exit" in a printout; the two are different quantities.

ASKED (round brief): measure the second-step term, land a cited
transcription defect G2-card-selected, re-score 100 steps, fix the check,
one review.

NOT FOLLOWED, with reason: "admit the step-1 file as the state after step
1" (it holds no record); "from consecutive restarts" (none exist).

UNASKED, each offered for revert:
1. The four one-variable arms of section 3 and the one-unit perturbation.
2. Two new ladder arms (substep walk, tracer replay) and the before flag.
3. The acquisition check's new content (empty-file rule, step-10 witness).
4. The fusion-off flag as a measurement arm. It is not selected by any card.
5. The re-anchor tool rewrote twelve line numbers in the citation map and
   two in one older receipt (lines moved by this round's edits).

Compliance, stated because no gate enforces it: search-before-build
(section 4); controlled comparison (before and after arms differ in one
option, same record, same script, same commit; the before arm reproduces
round 2); instrument validated before quoted (the replay against NEMO's
level 1; the flag by its control); non-vacuity (section 7).

UNVERIFIED:
- levels 2 to 30 of T and S after step 1 against NEMO itself (no such
  record; they equal the replay, and NEMO at step 2);
- that the ORCA2 card would accept the option as built;
- that a list of odd steps from 3 writes restarts (read off the code);
- the three ratchet failures on the base commit;
- 77 G1 ladder tests; closed cards (defaults read as unchanged, one
  package scan, not run).

## 10. OPEN for round 4

1. Name the statement behind section 6. The tool is the hand-back, cut
   finer: hand back the loop's entry forcing, then its exit, then the
   after-loop values one at a time, and see which one restores bit equality.
   No NEMO run is needed: NEMO's side is already matched.
2. Then re-score 100 steps. If step 1 is bit-equal on every field, the
   record decides certification.
3. Still open from round 2: a validated exposure of the stage-1 right-hand
   side (R2-P3, G-P6).
4. An operator question, not blocking: the fusion-off flag removes this
   whole class at once for every card. Keeping per-statement options
   (recorded in the card) is my pick; the flag stays a diagnostic.
