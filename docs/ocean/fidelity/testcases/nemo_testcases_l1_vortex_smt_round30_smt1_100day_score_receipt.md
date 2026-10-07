# Receipt — VORTEX_SMT round 30 (lane round 242): SMT-1 100-day score

**Status: HELD (measurement complete).** The admitted SMT-1 vector seamount
record reproduces the certified kt=1..10 registry exactly. Its independent
100-day legoESM run remains bounded: wet three-dimensional temperature RMS is
`4.3321114781972461e-05 K` at day 100, below the preregistered `1e-3 K`
falsifier and `0.9952715502` times the already-measured SMT-0 vector value.
This round changes no model, card, option, carried state, or certified row.

Base: `2c6007d46` (round 241). Measurement commit:
`8e0065ed08e63ae285061dd50f2190070033eb4e`. Final evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round242/`.
Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_smt_round30_smt1_100day_score.md`.

## 1. Admission and calibration

The operator-run round-241 acquisition ended `STATUS ADMITTED`, with its
restart control byte-identical, its self-describing parser complete, and its
header plant printing `PLANT_FIRED`. The round-242 gate independently counted
exactly 100 NEMO restarts at steps 30, 60, ..., 3000 and 100 legoESM daily
snapshots.

The compiled restart program schedules frequency-based output at
`VORTEX_SMT1_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:104-119` and writes
the scored `sshn/un/vn/tn/sn` state at
`VORTEX_SMT1_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:176-180`. These are
the executing statements in the admitted build; no reconstructed or
interpolated NEMO state enters the score.

Before the 100-day score was accepted, the shared trajectory gate replayed
the record's first ten entries against the round-237 certified SMT-1 registry.
It reported `REPRODUCED`, no mismatches, and the same first-over-bar row. Thus
R29-P1 and R29-P2 are **CONFIRMED**.

## 2. Registered 100-day checkpoints

All values below use the shared round-210 wet masks and definitions. RMS and
maxima are against NEMO; temperature is K, velocity is m/s, and sea-surface
height is m. The last column compares the SMT-1 temperature RMS with the
already-measured SMT-0 vector seamount curve, not the flat VORTEX card.

| day | T rms | T max | u rms | u max | v rms | v max | ssh rms | ssh max | T rms / SMT-0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `3.208268711e-08` | `1.586049362e-06` | `3.396454890e-08` | `1.923101845e-06` | `2.845805347e-08` | `2.229387539e-06` | `1.235771620e-08` | `1.994134334e-07` | `1.001310` |
| 2 | `5.149665118e-08` | `1.943375429e-06` | `5.161625762e-08` | `4.266583323e-06` | `4.523526041e-08` | `2.487404215e-06` | `1.587449517e-08` | `1.990008644e-07` | `1.000373` |
| 5 | `1.065935488e-07` | `5.161593717e-06` | `1.141144980e-07` | `6.064795307e-06` | `1.006529851e-07` | `5.431408091e-06` | `1.925762006e-08` | `2.103999163e-07` | `1.023170` |
| 10 | `2.610840895e-07` | `1.394122439e-05` | `2.678671186e-07` | `1.206855193e-05` | `2.666832203e-07` | `1.384748956e-05` | `4.231066921e-08` | `4.325172427e-07` | `1.009613` |
| 20 | `5.292694208e-07` | `2.842435492e-05` | `5.982149525e-07` | `3.282488738e-05` | `5.894932441e-07` | `2.635057687e-05` | `1.186724338e-07` | `1.268828174e-06` | `0.991747` |
| 30 | `1.170184670e-06` | `7.243091793e-05` | `1.381808378e-06` | `7.207404284e-05` | `1.363536219e-06` | `6.973607718e-05` | `3.268413663e-07` | `1.653514793e-06` | `0.995381` |
| 60 | `6.012214170e-06` | `3.146825988e-04` | `5.349548536e-06` | `3.498934942e-04` | `5.534237980e-06` | `3.002766821e-04` | `1.950227158e-06` | `2.070221129e-05` | `1.025080` |
| 100 | `4.332111478e-05` | `2.175281224e-03` | `2.696790485e-05` | `1.079193858e-03` | `2.616956023e-05` | `8.628720817e-04` | `6.895902500e-06` | `1.130918587e-04` | `0.995272` |

At day 100 the SMT-1/SMT-0 ratios are `0.9952715502` for T RMS,
`0.9819625726` for u RMS, `1.0000988759` for v RMS, and `0.9979111600` for
ssh RMS. This is a curve comparison only. It does not name a causal owner or
claim that background mixing is responsible for the difference. All 100 days
and all eight RMS/max metrics are finite; R29-P3 and R29-P4 are
**CONFIRMED**.

The first version of the post-run comparison table incorrectly used the flat
VORTEX vector curve when the preregistration named SMT-0 vector. That table was
not cited or retained. The gate and its control now require the seamount card
`VORTEX_SMT_VEC-zps`; the SMT-1 score itself never changed. This explicitly
retracts the wrong-reference comparison rather than silently replacing it.

## 3. Visuals, controls, and disposition

The shared renderer produced a 100-frame MP4 (`1,289,131` bytes), a 100-frame
GIF (`1,727,176` bytes), and the day-1/30/60/100 montage (`328,728` bytes).
Every panel uses the SMT-1 card's resolved seamount bathymetry. The
SSH-difference scale is fixed to the measured day-100 maximum
`1.1309185865338023e-04 m`; the velocity-arrow scale is fixed from day 1.
R29-P5 is **CONFIRMED**.

The final gate stamps a clean worktree and reports:

> `STATUS PASS: short=REPRODUCED day100_T_rms=4.33211147819724610e-05 K restarts=100 frames=100`

Deleting day 60 from the registry prints
`STATUS PLANT-FIRED: daily row registry is incomplete` and exits 1. Moving the
day-100 value to the frozen bound prints
`STATUS PLANT-FIRED: day-100 T RMS 0.001 exceeds frozen bound` and exits 1.

The required separate read-only Codex review was attempted on the committed
round diff. It did not initialize, so no SHIP verdict is inferred. Its output
is quoted verbatim:

> `WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)`
> `Reading additional input from stdin...`
> `Error: failed to initialize in-process app-server client: Read-only file system (os error 30)`

**Independent review unavailable in-sandbox.** This is measurement only; no
physics or accepted trajectory depends on an inferred review verdict.

The focused round-242 controls report `4 passed`. The final citation and
focused-suite summaries are recorded in the final validation commit.
No full physics tree or DINO month integration is required because no file
under `packages/` or `src/` changes. R29-P6 is **CONFIRMED**.

### OPEN — next round

Proceed to the deferred SMT-2 100-day comparison and movie. Reuse the committed
`smt2vec100d` acquisition arm and certified SMT-2 registry; if its daily NEMO
record is still absent, preregister the acquisition and stop
`STOPPED_FOR_RECORD`. Do not infer an owner from the SMT-1/SMT-0 curve.

**DECISION_NEEDED: NONE. ACQUISITION_NEEDED: NONE.**
