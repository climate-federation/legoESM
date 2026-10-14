# Grid ladder rung G2 — round 2 receipt (gridladder_r11)

Status: **LANDED** (one cited NEMO statement, selected only by the G2 card;
the 100-step record is NOT at the bar: a second, much smaller term is open
and needs one more NEMO run, no build).
Preregistration: docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g2_round2.md
(commit e462ce809, before any comparison of the card with the record).
Code: 4783eb526 (twin check), 2bc857c4a (the landing), bd0bbdd1f (review
fixes). Every number below was regenerated from the clean commit bd0bbdd1f
(stamp in each json; later commits touch citation pins and this file only).
Record: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round10/oracle_g2_r10
(the brief's "oracle_g2_r1" is this directory). Domain file SHA-256
7125f7a54e8693ff8f1327b878d2258d258878bc7302ac123ac671f2339d2839.
Evidence: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round11/
(final numbers under final/). NEMO citations are to the G1 build's compiled
sources under ORCA2G1_OMIP_L1_P3; the five files cited are byte-identical in
the record binary's build ORCA2G1_OMIP_L1_P5 (cmp, this round).
**Acquisition needed:**
scripts/validate/ocean_fidelity/testcases/nemo_testcase_g2_orca2bathy/run_r2.sh
(operator: run_r2.sh --run; one 10-step run of the pinned record binary).

## Take-aways

1. **The one refusal was unused storage.** The 3,384 differing words of the
   kt = 2 substep twins all lie in the 2-cell halo. The twin check now
   compares what the reader consumes; the record is admitted. NEMO was not
   re-run.
2. **Geometry: 36 of 36 rows bit-identical on all 30 levels.**
3. **First statement over the bar: NEMO's vertical momentum solve.** At step
   1 the sea level, both barotropic velocities, T and S were at the bar, and
   level-1 U and V were off by 5.4e-10 and 3.8e-10. Stages 1 and 2 were at
   the bar; the error entered in stage 3.
4. **Landed, G2-card-selected.** NEMO's two thickness divisors in that solve.
   Step-1 level-1 U and V: 2.1e-18 and 4.5e-18. One step from NEMO's own
   state at steps 3, 5, 7, 9: every row at the bar (1.2e-17 or less on the
   dynamics).
5. **100 steps.** Sea-level error at step 100: 3.7e-5 m before, 1.6e-12 m
   after. First step over the bar: U, V step 1 before, step 2 after.
6. **Not finished.** The second step leaves the bar at about 1e-13. The six
   barotropic histories after step 1 match NEMO to 1e-18, so the term enters
   through levels 2 to 30 of the step-1 velocity, which this record does not
   hold. Named as a location, not as a statement.
7. **G1 is unchanged.** Its 100-step certification test passes on this tree.

## 1. Admission

- NEMO's log (all four runs, no error line): 31 levels; 65 substeps; Courant
  number 0.79801811104755460; bump centre (51,74) at longitude
  179.99992123688389, latitude 0. Round-1 G-P2 and G-P3 HOLD.
- The refusal, mapped through the stream's own header
  (twin_kt2_substep_diff_map.log): 14,219 bytes = 3,384 words, in 15 arrays
  of the entry frame and the 65 substep frames.

| where the differing words lie | words |
|---|---|
| halo rows 0-1 and halo columns 0-1, 182-183 | 3,384 |
| inner domain (land or sea) | 0 |

  The unused drag array differs on both whole halo rows and on the halo
  columns of rows 2-7 (390 words); the other 14 arrays differ at the first
  one to four words of row 0, in every substep frame. Values are storage
  garbage (about 6.9e-310; up to 1.0 in the drag array, which is zero on
  every wet face). The kt = 1 stream and every step stream kt = 1..10 are
  byte-identical.
- The one reader of these streams keeps the inner domain of each full plane.
  The check now does the same: full planes on the inner domain, every other
  array whole, and it reports the halo count (0 at kt = 1, 3,384 at kt = 2;
  g2_round10_admission.json). Round-1 G-P4 is REFUTED as worded (byte
  identity) and holds on the consumed support.
- Re-admission (g2_admit_existing.log, commit 4783eb526): PASS_G1_RECORD;
  60 fields per stage step, 6 per final-only step; twin b's step-10 restart
  equals twin a's. Seven plants refuse, the new inner-domain plant among them.
- Registered: before the fix the missing-substep plant "fired" on the real
  twin difference, not on its own plant. It now fires on the missing file.

## 2. Geometry identity (R2-P1 HOLDS)

Every mesh array the card carries against NEMO's mesh file, on all 30 card
levels (final/geometry.json): **36 of 36 rows bit-identical.** Plant: one
moved element of the card's vorticity mask turns that row red, 35 of 36,
exit 1.

- NEMO's levels below the card's 30: T, U, V masks hold nothing. The
  vorticity mask holds 7 nonzero values on level 31, at the two straits:
  NEMO writes the strait value on every level
  (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dommsk.f90:293`). The card has no level
  31. Counted, not judged.
- Not in NEMO's mesh file, so not compared: the bottom levels at U and V, the
  3-D interface thickness, the face depth sums.
- Instrument note: called directly, the gate ran in single precision and
  showed 19 of 36. It now sets double precision itself. Found by its own test
  before any number was quoted.

## 3. Ladders, kt = 1..10

Bar: max abs / max(max abs NEMO, 1) <= 1e-15 (shared scorer). All numbers are
max abs differences. "Before" is the card without this round's statement.

**INDEPENDENT** (the card's own chain from its own initial state):

| kt | ssh | U mean | V mean | u level 1 | v level 1 | T level 1 | cells not bit-equal (ssh) | ssh before | u level 1 before |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.13e-17 | 1.18e-18 | 5.42e-19 | 2.09e-18 | 4.46e-18 | 3.55e-15 | 612 | 1.13e-17 | 5.36e-10 |
| 2 | 1.12e-13 | 8.14e-15 | 6.79e-15 | 1.44e-13 | 1.52e-13 | 1.07e-14 | 11740 | 1.56e-10 | 8.25e-08 |
| 3 | 1.72e-13 | 2.45e-14 | 3.10e-14 | 1.41e-13 | 3.13e-13 | 1.42e-14 | 9315 | 2.59e-08 | 6.49e-07 |
| 4 | 1.23e-13 | 2.30e-14 | 2.57e-14 | 1.43e-13 | 3.60e-13 | 1.78e-14 | 15944 | 3.00e-07 | 1.69e-06 |
| 5 | 1.42e-13 | 3.25e-14 | 3.52e-14 | 2.09e-13 | 3.60e-13 | 1.78e-14 | 16245 | 1.28e-06 | 5.69e-06 |
| 6 | 1.41e-13 | 3.71e-14 | 4.06e-14 | 2.22e-13 | 3.74e-13 | 2.13e-14 | 16357 | 3.66e-06 | 1.02e-05 |
| 7 | 1.44e-13 | 4.80e-14 | 6.09e-14 | 1.93e-13 | 5.03e-13 | 3.20e-14 | 16393 | 3.83e-06 | 9.11e-06 |
| 8 | 1.67e-13 | 6.00e-14 | 6.32e-14 | 2.22e-13 | 4.54e-13 | 3.20e-14 | 16418 | 5.85e-06 | 5.83e-06 |
| 9 | 1.36e-13 | 6.69e-14 | 9.25e-14 | 2.32e-13 | 4.60e-13 | 5.33e-14 | 16423 | 6.67e-06 | 1.13e-05 |
| 10 | 1.39e-13 | 7.47e-14 | 1.27e-13 | 2.58e-13 | 5.96e-13 | 4.97e-14 | 16427 | 8.83e-06 | 1.26e-05 |

**GIVEN-NEMO-ENTRY, one step** (NEMO's restart of the step before: all
prognostics on all levels and the six barotropic histories; kt = 1 is the
initial state). Entries exist only at odd kt (preregistration, section 2).

| kt | ssh | U mean | V mean | u level 1 | v level 1 | T level 1 | u level 1 before | v level 1 before |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.13e-17 | 1.18e-18 | 5.42e-19 | 2.09e-18 | 4.46e-18 | 3.55e-15 | 5.36e-10 | 3.83e-10 |
| 3 | 1.56e-17 | 1.73e-18 | 1.73e-18 | 3.90e-18 | 7.48e-18 | 7.11e-15 | 7.37e-07 | 9.91e-07 |
| 5 | 1.39e-17 | 8.67e-18 | 8.67e-18 | 1.04e-17 | 9.11e-18 | 3.55e-15 | 4.38e-06 | 5.96e-06 |
| 7 | 1.21e-17 | 3.69e-18 | 5.20e-18 | 6.56e-18 | 6.94e-18 | 7.11e-15 | 5.00e-06 | 6.13e-06 |
| 9 | 1.39e-17 | 5.20e-18 | 3.47e-18 | 1.17e-17 | 1.04e-17 | 3.55e-15 | 5.04e-06 | 5.73e-06 |

After the landing every one-step row is at the bar; none is bit-identical.

**GIVEN-NEMO-ENTRY, two steps, all levels** (the only all-level score this
record allows; a different quantity from the table above):

| entry kt | scored kt | ssh | u full | u partial | v full | v partial | T all levels | u full before | u partial before |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 1.12e-13 | 1.44e-13 | 1.40e-13 | 1.58e-13 | 1.85e-13 | 1.42e-14 | 8.25e-08 | 4.20e-06 |
| 3 | 4 | 7.67e-14 | 1.13e-13 | 1.46e-13 | 1.71e-13 | 1.75e-13 | 1.42e-14 | 3.92e-06 | 8.71e-05 |
| 5 | 6 | 1.10e-13 | 1.26e-13 | 1.86e-13 | 1.47e-13 | 1.79e-13 | 1.42e-14 | 1.71e-05 | 3.18e-04 |
| 7 | 8 | 7.48e-14 | 9.98e-14 | 1.06e-13 | 1.37e-13 | 1.46e-13 | 1.78e-14 | 2.17e-05 | 2.31e-04 |
| 9 | 10 | 9.50e-14 | 9.27e-14 | 1.35e-13 | 2.09e-13 | 1.61e-13 | 1.42e-14 | 1.96e-05 | 2.68e-04 |

### Regions (independent, after the landing; count of cells not bit-equal)

| kt | field | fold | seam | south | coast | open |
|---|---|---|---|---|---|---|
| 1 | ssh | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 1.1e-17 (612) |
| 1 | u | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 2.1e-18 (594) |
| 1 | v | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 4.5e-18 (633) |
| 2 | ssh | 2.4e-65 (205) | 2.5e-50 (227) | 1.7e-25 (37) | 5.2e-14 (1235) | 1.1e-13 (10036) |
| 2 | u | 2.4e-67 (197) | 1.6e-50 (238) | 1.1e-26 (37) | 2.2e-14 (908) | 1.4e-13 (10286) |
| 5 | ssh | 6.9e-28 (202) | 3.8e-14 (280) | 1.7e-14 (46) | 7.2e-14 (2026) | 1.4e-13 (13691) |
| 5 | u | 3.5e-29 (184) | 3.1e-14 (270) | 2.0e-14 (42) | 7.7e-14 (1430) | 2.1e-13 (13422) |
| 10 | ssh | 4.1e-14 (205) | 8.0e-14 (281) | 4.0e-14 (46) | 1.4e-13 (2125) | 1.3e-13 (13770) |
| 10 | u | 1.2e-13 (198) | 1.3e-13 (270) | 3.9e-14 (42) | 1.3e-13 (1569) | 2.6e-13 (13769) |
| 10 | v | 9.9e-14 (200) | 1.7e-13 (274) | 6.3e-14 (46) | 2.1e-13 (1644) | 6.0e-13 (13767) |
| 100 | ssh | 5.2e-13 (205) | 4.3e-13 (281) | 3.3e-13 (46) | 1.6e-12 (2129) | 9.0e-13 (13772) |
| 100 | v | 9.3e-13 (200) | 4.7e-12 (274) | 4.1e-13 (46) | 3.1e-12 (1651) | 5.0e-12 (13772) |

Reading. At kt = 1 fold, seam and south hold no unequal cell (R2-P5 HOLDS).
With real depth the signal reaches every boundary by step 2 (5,300 m of water
against G1's 100 m). No region stands out: fold, seam and south stay at or
below the open-ocean error. The G1 fold, seam and wall statements carry over.

### Depth classes (independent, all levels, where NEMO wrote a restart)

FULL = wet cell at the 1-D ladder's thickness; PARTIAL = any other wet cell.
Cells: U 397,940 full and 15,090 partial; V 399,994 and 15,181; T 414,998 and
15,554. No dry cell differs at any scored step.

| kt | u full | u partial | v full | v partial | levels where partial / full is worse (u) | u full before | u partial before | before: partial / full worse |
|---|---|---|---|---|---|---|---|---|
| 2 | 1.44e-13 | 1.40e-13 | 1.58e-13 | 1.85e-13 | 1 / 25 | 8.25e-08 | 4.20e-06 | 14 / 12 |
| 4 | 1.43e-13 | 1.38e-13 | 4.16e-13 | 5.74e-13 | 6 / 20 | 3.92e-06 | 8.29e-05 | 22 / 4 |
| 6 | 2.28e-13 | 2.15e-13 | 4.17e-13 | 5.75e-13 | 4 / 22 | 3.01e-05 | 2.94e-04 | 25 / 1 |
| 8 | 2.27e-13 | 3.04e-13 | 4.68e-13 | 6.25e-13 | 5 / 21 | 6.60e-05 | 3.42e-04 | 26 / 0 |
| 10 | 2.68e-13 | 5.08e-13 | 5.96e-13 | 7.29e-13 | 7 / 19 | 6.39e-05 | 4.17e-04 | 23 / 3 |
| 20 | 5.30e-13 | 9.71e-13 | 1.08e-12 | 1.86e-12 | 8 / 18 | 1.19e-04 | 4.29e-04 | 25 / 1 |
| 50 | 1.48e-12 | 3.68e-12 | 3.43e-12 | 3.55e-12 | 10 / 16 | 1.17e-04 | 3.34e-04 | 25 / 1 |
| 100 | 3.28e-12 | 6.20e-12 | 5.41e-12 | 8.55e-12 | 9 / 17 | 2.19e-04 | 4.27e-04 | 26 / 0 |

By level at kt = 2, U:

| level | full cells | partial cells | before full | before partial | after full | after partial |
|---|---|---|---|---|---|---|
| 1 | 15853 | 0 | 8.2e-08 | - | 1.4e-13 | - |
| 5 | 15547 | 81 | 8.2e-08 | 1.7e-12 | 1.4e-13 | 2.5e-18 |
| 10 | 15229 | 67 | 4.4e-08 | 4.2e-06 | 1.4e-13 | 7.2e-16 |
| 13 | 15086 | 52 | 5.7e-09 | 7.2e-07 | 1.4e-13 | 2.1e-16 |
| 19 | 14282 | 237 | 1.0e-09 | 7.4e-08 | 1.3e-13 | 2.8e-16 |
| 25 | 11751 | 652 | 3.1e-10 | 1.7e-08 | 3.2e-14 | 2.8e-14 |
| 28 | 6803 | 2057 | 1.9e-10 | 2.2e-08 | 8.4e-14 | 4.3e-14 |
| 30 | 0 | 4087 | - | 7.3e-09 | - | 1.4e-13 |

Reading. Before the landing the partial cells carried the error: 50 times the
full-cell error at kt = 2, worse at 22 to 26 of 26 levels from kt = 4 on
(R2-P4 HOLDS on the card it was frozen for). After the landing the two
classes are the same size and the full class is worse at most levels: what
remains is not a partial-cell term.

## 4. The 100-step record (independent)

| kt | ssh after | u level 1 after | v level 1 after | ssh before | u level 1 before |
|---|---|---|---|---|---|
| 1 | 1.13e-17 | 2.09e-18 | 4.46e-18 | 1.13e-17 | 5.36e-10 |
| 2 | 1.12e-13 | 1.44e-13 | 1.52e-13 | 1.56e-10 | 8.25e-08 |
| 10 | 1.39e-13 | 2.58e-13 | 5.96e-13 | 8.83e-06 | 1.26e-05 |
| 20 | 2.14e-13 | 5.19e-13 | 9.85e-13 | 1.54e-05 | 1.60e-05 |
| 40 | 4.59e-13 | 1.06e-12 | 1.89e-12 | 2.21e-05 | 2.08e-05 |
| 60 | 6.16e-13 | 1.20e-12 | 3.32e-12 | 2.35e-05 | 3.90e-05 |
| 80 | 8.78e-13 | 2.29e-12 | 4.18e-12 | 3.15e-05 | 6.37e-05 |
| 100 | 1.55e-12 | 3.12e-12 | 5.02e-12 | 3.71e-05 | 1.04e-04 |

NEMO's own maxima at step 100: sea level 0.0119 m, level-1 U 0.0124 m/s.

| first step over the bar | ssh | U mean | V mean | u | v | T | S |
|---|---|---|---|---|---|---|---|
| before | 2 | 2 | 2 | 1 | 1 | 2 | 2 |
| after | 2 | 2 | 2 | 2 | 2 | 6 | 5 |

No row leaves the bar earlier; no row at the bar before leaves it (R2-P6
HOLDS: the record is not at the bar). The six barotropic histories against
NEMO's restart: 7.4e-14 to 1.4e-13 at step 10 (5.5e-6 to 9.0e-6 before).

## 5. The first statement over the bar, and the landing

**Walk, step 1** (each stage exposed by the model's private hook, level 1
against NEMO's stage frames; stages_kt1_*.json, before the landing):

| stage | u level 1 | v level 1 | ssh |
|---|---|---|---|
| 1 | 1.18e-18 | 5.42e-19 | 6.94e-18 |
| 2 | 1.57e-18 | 4.20e-18 | 6.94e-18 |
| 3 | 5.36e-10 | 3.83e-10 | 1.13e-17 |

Stage 3 is the only stage that runs NEMO's implicit vertical solve.

**Offline replay** (rule 5; final/zdf_replay_kt1_before.json). The card's own
stage-3 inputs, taken by its write-only observer, are pushed through NEMO's
statements in the written order: remove the barotropic velocity
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynzdf.f90:160-161`), build the matrix
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynzdf.f90:192-198` for U,
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynzdf.f90:361-367` for V), run the three
recurrences (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynzdf.f90:333-349`), put the
barotropic velocity back
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:413-414`,
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:417-418`). The observer is
passive: the step with it equals the step without it, bit for bit (gated).

| quantity (step 1) | U | V |
|---|---|---|
| replay against NEMO, level 1 | 2.09e-18 | 4.39e-18 |
| card against NEMO, level 1, before | 5.36e-10 | 3.83e-10 |
| card's solve against the replay's, full cells, before | 2.85e-13 | 3.43e-13 |
| card's solve against the replay's, partial cells, before | 1.27e-08 | 1.19e-08 |
| replay with the interface stretched by the after-level ratio, against NEMO | 3.84e-13 | 4.02e-13 |
| replay with no interface stretch, against NEMO | 3.84e-13 | 4.02e-13 |
| card's solve against the replay's, all cells, after | 1.19e-18 | 6.51e-19 |
| card against NEMO, level 1, after | 2.09e-18 | 4.46e-18 |

- MEASURED: NEMO's statements on the card's inputs give NEMO's level 1; the
  card's solve did not; the two variants show which interface stretch NEMO
  uses (the now-level ratio).
- READ OFF THE CODE: why. The shared solve divided by the mean of the two
  T-cell thicknesses on the diagonal (NEMO: the mesh's face thickness, the
  smaller neighbour at a partial cell, times one plus the after-level U
  ratio) and by a face-mapped T-point interface thickness (NEMO: the 1-D
  ladder times one plus the now-level U ratio).
- The replay's V top row is excluded (its north T row is a fold halo).

**Landed.** Option zdf_momentum_nemo_face_thickness, default False, selected
by the G2 card only. Both divisors come from assemblers the model already
had (the drag block's face thickness; the stage face ratio), so there is no
second implementation. Searched before building: the model file for the
solve's divisors and for an existing NEMO face-thickness option; found the
two assemblers and no option.

- G1: not selected. One level has no interface. Its 100-step certification
  test passes on the final tree (tests_g1_100step.log: 4 passed).
- Caller grep (committed test): the option is assigned on one line of the
  whole package, the shared G1/G2 builder's, keyed by a two-entry rung table.
  Library default and every other card unchanged.
- ORCA2 pointer: ORCA2 runs the same compiled statement (same keys, same
  partial cells). The ORCA2 lane does NOT have it: its card does not select
  this option and no equivalent existed before this round. Not selected for
  ORCA2 here (another lane's card).
- The model refuses the option without NEMO's literal solver, without the
  carried barotropic pair, or with an interface ladder that varies by column.

## 6. The open term (not named as a statement)

After the landing one step from NEMO's state is at the bar and the second
step is not (1e-13). Measured this round:

- Step 2 from the card's own step-1 state: level-1 U is already 2.8e-14 off
  after stage 1, the barotropic velocity 8.1e-15 (after/stages_kt2_*.json).
  Step 3 from NEMO's restart: stage 1 at 1.7e-18.
- The card's six barotropic histories, sea level and barotropic velocities
  after step 1 against NEMO's own solver entry frame at step 2: 5e-19 to
  1.3e-17 (after/kt2_entry_histories.log).
- T and S on all levels at step 2: 1.4e-14 or less.

So what differs at the entry of step 2 is in levels 2 to 30 of U and V after
step 1 (or the carried vertical velocity): PLAUSIBLE by elimination, not
measured, because the record holds those levels only every second step.
Candidates, unranked and unmeasured: the pressure gradient's downward
recursion, the vorticity term at depth, the density's depth at 30 levels.

## 7. Predictions

| prediction | outcome |
|---|---|
| R2-P1 geometry 36 of 36 | HOLDS |
| R2-P2 step 1: no row bit-identical, every row at the bar | REFUTED on the card as frozen (U, V at 5e-10); holds after the landing |
| R2-P3 first non-bit row = the solver's entry forcing | NOT MEASURED: no card-side exposure of that frame was built; the first row OVER THE BAR was the stage-3 velocity |
| R2-P4 partial class worse than full | HOLDS before the landing; reversed after it |
| R2-P5 kt = 1: no unequal cell at fold, seam, south | HOLDS |
| R2-P6 not at the bar through 100 steps | HOLDS |
| R2-P7 tiny numbers reappear | REFUTED: no number below the smallest normal double, in NEMO or in the card, in any scored row of the 100 steps (level 1 and 2-D fields) |
| R2-P8 G1 unchanged | HOLDS by its 100-step certification test; the other 77 tests of that file NOT RUN |
| round-1 G-P5 T leaves 20 after step 1 | HOLDS (NEMO level-1 T reaches 20.00042) |
| round-1 G-P6 stage-1 right-hand side bit-identical | NOT ESTABLISHED: my right-hand-side exposure differed from NEMO's frame by 5e-6 while the stage-1 velocity built from it is at 1e-18, so the two are not the same quantity. Instrument unvalidated; no finding drawn |

## 8. Gates

| claim | gate | result |
|---|---|---|
| twin streams equal where consumed | record checker; inner-domain plant | pass; plant refuses |
| geometry on every level | G2 ladder, geometry arm; plant | 36 of 36; plant 35 of 36, exit 1 |
| the scorer can fail | score plant; level plant | both exit 1 (level plant: partial class 1.0) |
| the landing is needed | two-step test with its before arm | pass (after below 1e-12 on all levels; before above 1e-7 at partial cells) |
| the option is G2's only | package scan test; resolved-pair test | pass |
| the replay is NEMO's solve | dense-matrix unit test; NEMO level 1 | pass; 2.1e-18 |
| the observer is passive | bit comparison inside the replay arm | pass |
| G1 certification | G1 100-step test | 4 passed |
| next acquisition script | bash syntax; preflight; committed test | PREFLIGHT_OK; its run and admission branches are UNEXECUTED |
| dual review, claim review before code | NO GATE | single review only (lane rule 6); claim review NOT done |
| citations | citation gate on this receipt (commit c8c6cf693) | PASS: 7 citations, 0 unmapped, audit clean; shifted-citation plant exits 1 |

Focused tests, synchronous, at commit c8c6cf693 (tests_focused.log): the two
G2 files, the G1 tools file and the citation gate file, slow tests included:
55 passed. NOT RUN: 77 of the 81 tests of the G1 ladder file; the ratchet
files; any closed-card ladder.

## 9. Review

Single review (codex): **"VERDICT: findings"** (codex_review_r11.log; scope
9e0cebd1b..c7aedcf7c+1). Acted on in bd0bbdd1f, not re-reviewed:

- "PLAUSIBLE — Enabling the option also replaces `dz_u/dz_v` before the
  fallback barotropic `depth_mean`". Fixed: the option refuses that state.
- "CONFIRMED — Dry-cell differences never affect `class_summary.status`".
  Fixed; a dry-only test added. No dry cell differs in this record.
- "CONFIRMED — The production non-vacuity test scores only level 1". Fixed:
  the test runs two steps and scores all levels by depth class, both arms.
- "CONFIRMED — `test_the_option_is_off_by_default_and_selected_by_g2_only`
  checks only the private selector dictionary". Fixed: package scan added.
- "CONFIRMED — `run_r2.sh` fully compares only even restarts." Fixed: every
  restart is judged against the verified step stream of its own step, with a
  plant; proved on twin b's existing restarts (5 equal; plant caught).

The first review attempt hung waiting on standard input and was ended by its
own timeout; nothing was killed by name.

## 10. Corrections to my own work, and choices

Corrections:
1. I expected the 100-step chain to cost 90 minutes (round 1's 54 s per
   step). Almost all of that is compilation: 98 s once, then 0.7 s per step.
   Three early chain chunks were sized for the wrong cost.
2. One probe run from another directory imported an older installed copy of
   the package and failed loudly; rerun with absolute paths. No number came
   from it.
3. The first chain state could not be reloaded (the field class forbids
   assignment); fixed in the script before any resumed number was used. The
   final tables come from unbroken 100-step runs.

ASKED (round brief): everything in the preregistration's ASKED list; landing
a cited statement selected by the G2 card.

UNASKED, each offered for revert:
1. "Consumed support" = the inner domain of full planes.
2. Given-entry at odd kt only, with two-step all-level rows.
3. The depth-class definition.
4. A new G2 ladder script beside G1's; G1's geometry row builder gained an
   all-levels switch (default unchanged).
5. The G2 card's unmeasured list drops "record not admitted" (it is admitted).
6. The next acquisition's design (section 11).
7. The re-anchor tool rewrote two line numbers in one older receipt and
   twelve in the citation map (lines moved by this round's edits).

Compliance, stated because no gate enforces it: search-before-build (section
5); controlled comparison (before and after arms differ in one option, same
record, same script, same commit); instrument validated before quoted (the
replay against NEMO's level 1; the geometry gate's precision); non-vacuity
(section 8).

UNVERIFIED:
- that levels 2 to 30 after ONE step match NEMO (the open term says some do
  not);
- that the ORCA2 card would accept the option as built (its interface ladder
  and its histories were not inspected);
- the right-hand-side exposure (section 7);
- the 77 G1 ladder tests not run; closed cards (defaults read as unchanged,
  one package scan, not run);
- run_r2.sh's run and admission branches on a real record.

## 11. OPEN for round 3

1. **Acquisition (operator):** run_r2.sh --run. Twin c: the pinned record
   binary, 10 steps, NEMO's restart after every step. Admitted only if its
   step streams equal twin a's byte for byte, its even restarts equal twin
   b's, and every restart equals its own step's verified stream on the 2-D
   fields and level 1.
2. With it: one-step given-entry rows on ALL levels at every kt = 2..10, by
   level and depth class. That names which levels carry the 1e-13 term after
   one step.
3. Then the walk at depth: the same offline-replay pattern, operator by
   operator, on NEMO's all-level entry and exit states.
4. Still open from round 1: a stage writer for all levels is a new build;
   ask for it only if item 3 cannot name the statement.
5. R2-P3 and G-P6 need a validated exposure of the solver's entry forcing
   and of the stage-1 right-hand side before they can be scored.

SHA-256 prefixes, round11/: final/independent_after.json=7c5764b261f077ae,
final/independent_before.json=bcfa0da1184a02e6,
final/given_entry_after.json=aa4e84f16e3b3f20,
final/given_entry_before.json=6f3d671b14ec650f,
final/geometry.json=3324486d67b65d6b, final/geometry_plant.json=0bbc08948b7aa33b,
final/zdf_replay_kt1_before.json=3fdc3410d65c05c6,
final/zdf_replay_kt1_after.json=00d49ad81ea027e1,
codex_review_r11.log=b0321fe4f3615f77, g2_admit_existing.log=d6895432d5303785,
twin_kt2_substep_diff_map.log=999305f77782f47f,
tests_g1_100step.log=ce49ceac4e534e89, preflight_run_r2.log=df79053b728b1dab;
round10/oracle_g2_r10/g2_round10_admission.json=e895c2154665677d,
citation_receipt.json=2124a213d80de5af, citation_receipt_plant.json=a77ee382c3e58b43,
tests_focused.log=f86f25d6a21b3f14.
