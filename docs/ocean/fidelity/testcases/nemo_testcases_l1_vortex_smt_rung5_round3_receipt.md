# Receipt — SMT-RUNGS round 3: SMT-5 (T/S damping) record admission, identity, ladders, 100 days

**Status: HELD (no model change).** The SMT-5 record is admitted, geometry and
NEMO's own damping inputs are bit-identical to what the card reads, and the
damping arm shows no detectable defect at the first step where it is live. The
first non-bit statement of the SMT-5 ladder is SMT-4's inherited kt=2
statement (already named in the SMT-4 scoring receipt), not a damping
statement, so there is no cited one-variable fix to land. Base `1af3fae7d`;
evidence `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/smtrungs_rounds/round3/`;
preregistration `docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_smtrungs_round2_smt5.md`
(committed before the record existed).

**ORCA2 pointer.** ORCA2 rung 1 runs exactly this program: the SMT-5 build's
compiled `tradmp.f90`, `dtatsd.f90`, `fldread.f90`, `daymod.f90` and
`stprk3_stg.f90` are byte-identical to SMT-4's (`cmp` silent on all five), which
round 2 showed identical to ORCA2_OMIP_L4's. Only the resto field differs (rung
1: marginal seas; SMT-5: every wet cell). What carries to the ORCA2 rung-1 walk
unchanged: (1) the card's model-time/record weights equal NEMO's printed
fld_read lines; (2) a `resto` read from the file is bit-exact; (3) the
equal-entry increment test below (NEMO rung-1 entry minus NEMO rung-0 entry at
an entry both share, compared with the legoESM ON-minus-OFF step) is the
instrument for the first live damping step on ORCA2.

## Round 3 — 1. Record admission

The lane's own checker (the shared VORTEX `check_records.py`) was re-run on
both acquired arms; header, field-name and truncation plants all exit nonzero.

| arm | status | parsed records | restart vs plain build | plants |
|---|---|---:|---|---|
| kt=1..10 (+ stage-flux terms) | ADMITTED | 27 | byte-identical | 3/3 refused |
| 100-day daily restarts | ADMITTED | 3067 | byte-identical | 3/3 refused |

Both `run.user.log` end `STOP 0`. Admission JSON `adm/kt_admission.json`
(sha256 `fedb2e19…`), `adm/day100_admission.json` (`7c2c97d2…`). The 100-day
arm's step-entry writer stops at kt=60 (60 files); the daily record is the
restarts.

Damping target files' provenance (NEMO's own dump, Decision 107e; identical
across smoke, kt and 100-day arms):

| file | sha256 |
|---|---|
| data_1m_potential_temperature_nomask.nc | `2972e475429eb690926efce6e7d3bb95a59586d3390f00ae12b8e9bb119698db` |
| data_1m_salinity_nomask.nc | `6734075c074474558c834527731f2f19fe60103ecda942eecb3b4a79e6aee4b8` |
| resto.nc | `024743b23a2d6620d5b32616eb17eb55f45a7501310caae7261028cc9160f20c` |

## Round 3 — 2. Geometry and input identity (P1, P2, P3)

- **Geometry gate vs NEMO's `mesh_mask.nc`: GEOMETRY IDENTICAL, 18/18 rows
  EXACT** (zero ULP; the non-vacuity row counts 1164 faces where
  `e3u_0 != e3t_0`). `smt5_geometry.json` `6c2f05f7…`.
- **SMT-5 mesh == SMT-4 mesh**: all 39 variables of the two `mesh_mask.nc` are
  `array_equal` (0 differing).
- **Card arrays == the files NEMO read**: target T, target S and resto, as the
  card loaded them, equal an independent netCDF4 read of the run-directory
  files (the three file hashes above are the ones NEMO opened with
  `cn_dir = './'`).
- **P2**: record 1 of `votemper` equals the card's analytical initial T on all
  37144 wet cells (0 unequal); `vosaline` = 35 and the initial S = 35 on wet
  cells (0 unequal); all 12 records identical; `resto` equals `tmask/86400`
  on all 39690 cells (0 unequal).
- **P3**: NEMO's `ocean.output` prints `fld_read: var votemper kt = 1 (0.0167
  days) ... records b/a: 0012/0001 (days -15.5/15.5)`; kt=2 prints 0.0500 days.
  The card uses elapsed `(kt-1)*2880` s plus 1440 s, i.e. 0.0167 and 0.0500
  days. The damping switches resolved by NEMO: `ln_tsd_dmp = T`,
  `ln_tradmp = T`, `nn_zdmp = 0`, `cn_resto = resto.nc`.

## Round 3 — 3. The kt=1..10 ladder, SMT-4 as control (P4, P5, P6)

Compiled program under test (SMT-5 build): the damping call sits in stage 3
only, after lateral tracer diffusion and before the vertical solve
(`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:526`,
`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:529`,
`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:538`); it reads the
target at the step (`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/tradmp.f90:181`,
`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/dtatsd.f90:212`, interpolation
`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/fldread.f90:246`) and adds
`resto*(zts_dta - ts(Kbb))` for `nn_zdmp = 0`
(`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/tradmp.f90:190`,
`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/tradmp.f90:193-194`).

### 3a. INDEPENDENT label (legoESM's own trajectory vs NEMO's entry kt)

Normalized max-abs, bar = 1e-15, shown SMT-4 / SMT-5 (same harness; SMT-4
reproduces the certified ladder row for row). Full per-field first unequal
cell, count, max abs and rms are in `measure.json` (`07cabb0f…`); the
certified-format ladder is `smt5_ladder.json` (`b2dc3019…`).

| kt | T | S | u | v | ssh |
|---:|---|---|---|---|---|
| 1 | 0 / 0 | 0 / 0 | 2.22e-16 / 2.22e-16 | 2.22e-16 / 2.22e-16 | 1.36e-20 / 1.36e-20 |
| 2 | 6.96e-10 / 6.96e-10 | 6.09e-16 / 6.09e-16 | 3.22e-09 / 3.22e-09 | 6.83e-10 / 6.83e-10 | 3.94e-10 / 3.94e-10 |
| 3 | 2.97e-09 / 2.94e-09 | 8.12e-16 / 8.12e-16 | 1.02e-08 / 1.02e-08 | 1.66e-09 / 1.66e-09 | 6.24e-10 / 6.24e-10 |
| 4 | 2.42e-09 / 2.37e-09 | 1.02e-15 / 1.02e-15 | 2.31e-08 / 2.31e-08 | 3.49e-09 / 3.49e-09 | 8.11e-10 / 8.06e-10 |
| 5 | 4.95e-09 / 4.74e-09 | 1.02e-15 / 1.22e-15 | 3.64e-08 / 3.62e-08 | 6.09e-09 / 6.10e-09 | 1.11e-09 / 1.10e-09 |
| 7 | 1.22e-08 / 1.12e-08 | 1.22e-15 / 1.22e-15 | 1.44e-07 / 1.33e-07 | 2.18e-08 / 2.08e-08 | 3.38e-09 / 3.19e-09 |
| 10 | 3.46e-08 / 3.09e-08 | 1.62e-15 / 1.42e-15 | 8.31e-07 / 7.30e-07 | 2.10e-07 / 1.85e-07 | 1.30e-08 / 1.19e-08 |

- **First over the bar**: kt=2 on T, u, v, ssh; S first over at kt=4 (1.02e-15,
  just above 1e-15). Same rows and same kt as SMT-4.
- **kt=1 and kt=2 are bit-for-bit SMT-4's**: every statistic in
  `measure.json` (first unequal cell, count, max abs, rms) is identical for
  all five fields. Damping first changes a number at kt=3.
- From kt=3 SMT-5 sits 1-12 % below SMT-4 on T/u/v/ssh by kt=10 (T 0.89x,
  u 0.88x, v 0.88x, ssh 0.92x); S is within ±20 % of SMT-4 at a 1e-15 level.

### 3b. GIVEN-NEMO-ENTRY label (one legoESM step from NEMO's entry kt, vs NEMO's entry kt+1)

Never mixed with 3a (D52). Max abs (rms), SMT-4 / SMT-5, step kt -> kt+1:

| kt -> kt+1 | T | u | ssh |
|---:|---|---|---|
| 1 -> 2 | 1.43e-08 (1.41e-10) / same | 3.22e-09 (4.24e-11) / same | 3.94e-10 (2.62e-11) / same |
| 2 -> 3 | 4.66e-08 (2.87e-10) / 4.66e-08 (2.87e-10) | 6.45e-09 / 6.45e-09 | 3.50e-10 / 3.50e-10 |
| 5 -> 6 | 9.26e-08 (1.07e-09) / 8.88e-08 (1.03e-09) | 1.24e-07 / 1.15e-07 | 3.08e-09 / 2.88e-09 |
| 9 -> 10 | 2.22e-07 (2.48e-09) / 2.06e-07 (2.23e-09) | 3.55e-07 / 3.07e-07 | 1.61e-08 / 1.43e-08 |

Instrument retraction: the first draft of this label re-seeded only the card's
INITIAL state, which drops the model's barotropic/time-filter history, and gave
residuals of ~1e-4 K / 1e-3 m s-1 from kt=2 for BOTH cards. A well-posed step
cannot do that (the independent label at the same step is 1e-8), so it was an
instrument defect, not a result; the label now re-seeds the prognostic fields
on the carried (stepped) state each kt. Only the corrected numbers are quoted.

### 3c. The damping chain at the first live step (P6)

Instrument: where NEMO's SMT-5 and SMT-4 entries are bit-equal (kt=1 and kt=2
are, all five fields), NEMO's SMT-5 entry minus NEMO's SMT-4 entry at the next
kt is NEMO's damping increment (the only difference between the two NEMO
runs), and legoESM ON minus OFF from the same seed (one variable: the card's
damping field) is legoESM's increment. Both are observed through the rounded
after-tracer field, so equality is only visible to its ulp.

| step | field | NEMO increment: cells nonzero / max abs | legoESM increment: cells nonzero / max abs | max abs difference (ulps of the field) | cells differing |
|---|---|---|---|---|---:|
| 1 -> 2 | T | 0 / 0 | 0 / 0 | 0 | 0 |
| 1 -> 2 | S | 0 / 0 | 0 / 0 | 0 | 0 |
| 2 -> 3 | T | 36298 / 4.0370602e-04 K | 36264 / 4.0370602e-04 K | 7.1e-15 K (3 ulp, 1 cell; 37143 cells <= 2 ulp) | 17520 |
| 2 -> 3 | S | 527 / 1.4e-14 | 0 / 0 | 1.4e-14 (2 ulp) | 527 |

- At kt=1 the target equals the state exactly, so the increment is exactly
  zero in both models (nothing to attribute; the per-stage records exist only
  at kt=1, where the damping increment is zero in both).
- At the first live step the T increment (up to 4.04e-4 K per step) agrees
  with NEMO's to the rounding floor of the after-tracer field: 37143/37144
  cells within 2 ulp, one cell 3 ulp, no binade crossing. Inherited non-bit
  residual (~1e-10 K, 3a/3b) alone makes two correctly-rounded sums differ by
  an ulp, so this is **not distinguishable from bit-exact** at this
  observation level (PLAUSIBLE that all four operands are bit-exact; MEASURED
  that nothing above the ulp floor exists). No NEMO record of `zts_dta` or of
  the stage-3 Krhs exists, so "zts_dta/weight bit-exact at every kt" is not
  directly measurable (see P6).
- **S, sub-ulp**: the S increment is about 2e-16 per ulp of offset from 35,
  below half an ulp of 35 (3.6e-15). NEMO moves 527 cells (<= 2 ulp, 4e-16 normalized) and
  legoESM moves none. Scaling legoESM's resto by 1/10/100 changes
  0/1183/17437 of the 17437 non-35 cells, i.e. legoESM absorbs sub-ulp S
  increments more than the naive 3 % per-ulp expectation (NEMO's 527 = 3.0 %
  of 17437 matches it). PLAUSIBLE owner: the stage-3 S rate sum or the
  vertical solve rounding; unlocated, below the bar, not landed.

## Round 3 — 4. The 100-day comparison (P7)

Same harness for both cards (SMT-4 control reproduced: day-100 T rms
2.552708e-04 K = the certified 2.5527e-04). Each card is scored against its
own NEMO run (different physical decks: the pair is not a candidate/control
claim on a shared oracle). rms / max over wet cells, SMT-4 -> SMT-5:

| day | T rms (K) | S rms (psu) | u rms (m/s) | v rms (m/s) | ssh rms (m) | T max (K) |
|---:|---|---|---|---|---|---|
| 1 | 1.41e-07 -> 9.68e-08 | 2.7e-14 -> 2.4e-14 | 6.3e-08 -> 4.3e-08 | 7.4e-08 -> 4.7e-08 | 1.10e-08 -> 7.4e-09 | 1.66e-05 -> 1.06e-05 |
| 5 | 4.42e-06 -> 1.14e-06 | 5.7e-14 -> 5.4e-14 | 1.78e-06 -> 5.2e-07 | 1.56e-06 -> 5.9e-07 | 1.79e-07 -> 3.8e-08 | 4.72e-04 -> 1.49e-04 |
| 10 | 2.45e-05 -> 1.97e-06 | 7.8e-14 -> 6.3e-14 | 7.3e-06 -> 8.5e-07 | 7.9e-06 -> 1.03e-06 | 7.7e-07 -> 1.55e-07 | 2.36e-03 -> 2.01e-04 |
| 30 | 1.45e-04 -> 5.29e-06 | 1.4e-13 -> 6.8e-14 | 2.67e-05 -> 2.72e-06 | 2.99e-05 -> 2.67e-06 | 7.4e-06 -> 4.1e-07 | 1.19e-02 -> 5.75e-04 |
| 60 | 2.00e-04 -> 9.47e-06 | 1.9e-13 -> 6.9e-14 | 2.95e-05 -> 5.0e-06 | 2.98e-05 -> 4.5e-06 | 1.25e-05 -> 7.8e-07 | 9.38e-03 -> 1.20e-03 |
| 100 | 2.55e-04 -> 1.08e-05 | 2.4e-13 -> 6.9e-14 | 2.83e-05 -> 5.5e-06 | 2.72e-05 -> 5.1e-06 | 1.17e-05 -> 9.2e-07 | 9.83e-03 -> 1.33e-03 |

- **Residual shrinks**: day-100 T rms 2.55e-04 -> 1.08e-05 K (0.042x); u 0.19x,
  v 0.19x, ssh 0.078x, S 0.29x. All five rows move down, all days.
- **Leaves the floor**: first day T rms > 10x its day-1 value: day 4 (SMT-4)
  vs day 5 (SMT-5); first day above 1e-6 K: day 3 vs day 5. SMT-5's T rms
  peaks at day 70 (1.12e-05 K) and then flattens (1.08e-05 at day 100); SMT-4's
  is still rising at day 100.
- **Caveat that matters for ORCA2**: a one-day restoring to a fixed target
  pulls both trajectories to the same point, so the shrinkage measures reduced
  sensitivity to every other statement, not a fidelity gain of those
  statements. Damped residuals mask, they do not repair; ORCA2 rung 1's
  residual should be read against rung 0's with that in mind.

## Round 3 — 5. Prediction scoring (round 2's frozen table)

| # | prediction | result | label |
|---|---|---|---|
| P1 | geometry identical, 18 rows | 18/18 EXACT; mesh 39/39 variables equal SMT-4's | CONFIRMED |
| P2 | dumped inputs = card's analytical state; resto = tmask/86400 | 0 unequal cells in every comparison | CONFIRMED |
| P3 | kt=1 at 0.0167 d, records 0012/0001, -15.5/+15.5 d | exact line printed (plus kt=2 at 0.0500 d) | CONFIRMED |
| P4 | five kt=1 rows AT-BAR at SMT-4's values | 5/5 AT-BAR, equal to SMT-4 bit for bit | CONFIRMED |
| P5 | first over bar kt=2 (T/U/V/SSH), S not before kt>=4 | kt=2 on T/u/v/ssh; S first over at kt=4 | CONFIRMED |
| P6 | first non-bit operand is `pts(Kbb)` at kt=2 (6.96e-10 K); resto, zts_dta, weight bit-exact at every kt | kt=2 T = 6.958e-10 K identical to SMT-4's; resto bit-exact (file == card == tmask/86400); increment at the first live step within the ulp floor; zts_dta/weight not directly observable | CONFIRMED for the Kbb-first part and resto; PLAUSIBLE for zts_dta/weight |
| P7 | day-100 T rms <= 2.55e-4 K | 1.08e-05 K | CONFIRMED (caveat above) |

P5's reasoning ("damping cannot zero SMT-4's kt=2 residual") held: kt=2 is
identical to SMT-4 because the target equals the state at kt=1, a stronger
statement than predicted.

## Round 3 — 6. First non-bit statement, and what is not landed

First non-bit statement of the SMT-5 ladder: the one SMT-4 already carries at
kt=2 (T first unequal cell [1,5,0], 19518 cells, 1.43e-8 K max, the HPG
accumulator association named in the SMT-4 scoring receipt), because the
damping increment is exactly zero through kt=2 and the kt=1/kt=2 rows are
bit-identical to SMT-4. Inside the damping arm no operand above the ulp floor
was found. So: no cited NEMO statement with a one-variable fix in the damping
arm exists to land. Held items, named:

1. S sub-ulp increment (NEMO 527 cells flip, legoESM 0): below the bar;
   resolving it needs a NEMO stage/tracer-term record at kt=2 (this acquisition's
   stage records exist at kt=1 only, where the increment is zero). Not
   requested: the amplitude is 4e-16 normalized and S is uniform.
2. One T cell at 3 ulp at the first live step (bound for four roundings is 2):
   unexplained, PLAUSIBLE implicit-solve rounding; not chased.

Landing gates (SMT-1..4 registries, GYRE certified year, DINO, tanks, Decision
96 census) apply to model changes. This round changes NO model or physics
file: the diff over the base touches harness scripts, one test, the citation
map and this receipt (`git diff 1af3fae7d..HEAD --stat -- packages src`
empty). The three harness edits are additive and default-off for every
existing card: the trajectory gate and the 100-day script pass model time only
for the card that carries the damping field, the geometry gate gains one
case, and the 100-day score also reports S.

## Tests, instruments, review

- New `tests/ocean/fidelity/test_nemo_testcase_l1_vortex_smtrungs_round3_measure.py`:
  5 passed (statistic, one-ULP plant found at its cell, masked-out difference
  invisible, refusals, and the real SMT-5 input-identity gate). Non-vacuity:
  reversing the first-unequal-cell order makes the C-order test FAIL; the
  script's `--plant` moves one wet T cell one ULP and exits 1 with exactly 1
  unequal cell where the unplanted row has 0.
- Focused existing suites on the edited harnesses: 25 passed (SMT-5 damping
  10, round-239 source order, round-240 process ranking, round-241/245
  100-day).
- Citation gate (from heading "## Round 3 —"): PASS, 8 citations, 0 unmapped,
  0 failures, 0 map entries failing audit; planting a 2-line shift on the
  `tradmp` damping-statement citation makes it FAIL (exit 1, unmapped). The
  default cumulative run is PASS (274 citations).

UNVERIFIED: the NEMO-side damping increment is inferred from two NEMO runs'
rounded entries (no per-term record); NEMO's `zts_dta` is not observed; the
given-entry label keeps legoESM's own time-filter history rather than NEMO's
(NEMO's is not in the entry record); day-100 comparison is one realization
per card at fp64/libm on CPU.

## Choices made this round

| choice | status |
|---|---|
| passing model time (elapsed seconds at step start) from the harnesses to the damped card only | ASKED (round 2's OPEN item 2; the model refuses without it) |
| scoring S in the daily comparison | UNASKED, additive output only (the five fields the task names); revert = drop "S" from the score tuple |
| "floor" defined as 10x the day-1 T rms and as 1e-6 K | UNASKED, reporting convention; both numbers given |
| given-entry label re-seeds on the carried state | UNASKED instrument choice, forced by the history finding above |
