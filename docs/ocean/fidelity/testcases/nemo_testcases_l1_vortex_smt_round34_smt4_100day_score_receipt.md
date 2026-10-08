# Receipt — VORTEX_SMT round 34 (lane round 246): SMT-4 100-day score

**Status: STOPPED_FOR_DECISION (measurement complete).** The admitted SMT-4
trajectory reproduces the certified kt=1..10 registry and all 100 previously
measured daily score rows exactly. Its day-100 wet three-dimensional
temperature RMS is `2.5527080520554426e-04 K`, 1.43979 times SMT-3. The four
authorised seamount mini-ladder rungs are now scored and rendered. Production
is unchanged; the batch stops for the operator to choose PR preparation or a
new source-faithful SMT-4 owner experiment.

Base: `e2a05d99a` (round 245). Preregistration commit: `b505feb72`.
Measurement-gate commit: `9837bbb39`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round246/`.
Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_smt_round34_smt4_100day_score.md`.

## 1. Admission and calibration

The round-237 self-describing admission contains 3,067 records and reports
`ADMITTED`; the reference restart is byte-identical. Its header, field-name,
and truncation plants each report `REFUSED`. The round-246 gate independently
counts exactly 100 NEMO restarts at steps 30, 60, ..., 3000 and 100 legoESM
daily snapshots.

The compiled executing restart program schedules frequency-based output at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:104-119` and writes
the scored `sshn/un/vn/tn/sn` state at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:176-180`. The score
therefore uses NEMO's own daily state without interpolation or reconstruction.

Before accepting any long-run row, the shared trajectory gate replayed the
record's first ten entries against `round238/smt4_ladder.json`. It reported
`REPRODUCED` with no mismatch. R34-P1 and R34-P2 are **CONFIRMED**.

## 2. Registered 100-day checkpoints

All values use the shared round-210 wet masks and definitions. RMS and maxima
are legoESM minus NEMO; temperature is K, velocity is m/s, and sea-surface
height is m. The final column compares temperature RMS with the current SMT-3
score through the identical instrument.

| day | T rms | T max | u rms | u max | v rms | v max | ssh rms | ssh max | T rms / SMT-3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `1.405693975e-07` | `1.656655910e-05` | `6.286792087e-08` | `6.134416567e-06` | `7.431603295e-08` | `6.945482528e-06` | `1.104148579e-08` | `1.825341884e-07` | `0.934651` |
| 2 | `5.100120075e-07` | `2.762692694e-05` | `1.065039598e-07` | `5.349402916e-06` | `1.162550943e-07` | `1.398670522e-05` | `2.408124881e-08` | `2.041301283e-07` | `1.273986` |
| 5 | `4.424519998e-06` | `4.717998696e-04` | `1.775484908e-06` | `2.066524131e-04` | `1.557389852e-06` | `1.884112806e-04` | `1.787936677e-07` | `1.112697741e-06` | `1.020806` |
| 10 | `2.454180680e-05` | `2.356847707e-03` | `7.312516990e-06` | `8.627041680e-04` | `7.914167801e-06` | `6.414736349e-04` | `7.704382891e-07` | `4.035443991e-06` | `1.125215` |
| 20 | `7.226803498e-05` | `6.097387029e-03` | `1.608965554e-05` | `1.382646873e-03` | `1.741920380e-05` | `1.154321811e-03` | `3.463151482e-06` | `1.438721865e-05` | `1.007015` |
| 30 | `1.450215284e-04` | `1.185073149e-02` | `2.673795366e-05` | `1.377274492e-03` | `2.985170460e-05` | `1.903269130e-03` | `7.380713703e-06` | `3.743847510e-05` | `0.840542` |
| 60 | `2.003482609e-04` | `9.383780316e-03` | `2.947599702e-05` | `1.120145107e-03` | `2.982349828e-05` | `2.086656129e-03` | `1.247144123e-05` | `8.324561119e-05` | `0.980894` |
| 100 | `2.552708052e-04` | `9.830831628e-03` | `2.834815879e-05` | `9.356145245e-04` | `2.715247914e-05` | `9.952453748e-04` | `1.171711158e-05` | `7.429504203e-05` | `1.439791` |

Every one of the 800 scalar daily score values equals the round-238 score
JSON exactly; only the snapshot-directory provenance and the newly recorded
ladder-reference path differ. Day-100 T RMS exactly reproduces
`2.5527080520554426e-04 K` and remains above SMT-3's
`1.7729713625071864e-04 K`. R34-P3 and R34-P4 are **CONFIRMED**. This curve is
a controlled trajectory measurement, not a causal attribution.

## 3. Four-rung endpoint and open owner

| rung | added ORCA2-facing module | day-100 T rms [K] | ratio to previous rung |
|---|---|---:|---:|
| SMT-1 | background mixing + EVD | `4.3321114781972461e-05` | — |
| SMT-2 | linear bottom drag | `8.1037591477894766e-06` | `0.187062572` |
| SMT-3 | lateral tracer diffusion | `1.7729713625071864e-04` | `21.878381751` |
| SMT-4 | lateral momentum diffusion | `2.5527080520554426e-04` | `1.439790910` |

SMT-4's day-100 owner remains open. Round 240's complete destructive
family-off ranking found no positive-removal family: disabling every tested
family worsened the error, and disabling the barotropic replacement was
unbounded. The standing stage-3 first non-bit boundary remains the HPG
accumulator over partial cells; NEMO executes HPG/VOR/ADV in that order at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:328-344` and
lateral momentum diffusion later at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:398-404`.
Round 239 made HPG source order locally bit-exact but trajectory-inert and
slightly worse at day 100. Thus neither HPG nor lateral momentum diffusion is
a measured day-100 owner. A further round needs a new source-faithful
substitution ranking; another family-off ablation would repeat a refuted
instrument.

## 4. Visuals, controls, and disposition

The shared renderer produced a 100-frame MP4 (`1,209,739` bytes), a 100-frame
GIF (`2,211,437` bytes), and the day-1/30/60/100 montage (`320,954` bytes).
The panels use SMT-4's resolved seamount bathymetry. The SSH-difference scale
is fixed to the measured day-100 maximum `7.429504202654657e-05 m`; the
velocity-arrow scale is fixed from day 1. R34-P5 is **CONFIRMED**.

The committed gate stamps a clean worktree and reports:

> `STATUS PASS: short=REPRODUCED day100_T_rms=2.55270805205544263e-04 K T_ratio=1.439790910 restarts=100 frames=100`

Deleting day 60, changing day-100 T by one ULP, and moving it below SMT-3 each
print `STATUS PLANT-FIRED` and exit 1. This is a measurement-only round: no
file under `packages/` or `src/` changes, and no physics battery or DINO month
integration is required. R34-P6 is **CONFIRMED**. **UNASKED list: EMPTY.**

## 5. Independent review, citations, tests, and decision

Independent review and the final citation/focused-test results are appended
after the complete round diff is committed.

**DECISION_NEEDED:** Proceed to PR preparation and retain SMT-4's non-unique
day-100 owner as registered debt (**recommended**), or authorise a new
source-faithful SMT-4 substitution-ranking round. No new rung or physics
change is authorised by this receipt.

**ACQUISITION_NEEDED: NONE.**
