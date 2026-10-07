# Receipt — VORTEX_SMT round 32 (lane round 244): SMT-2 100-day score

**Status: HELD (measurement complete).** The admitted SMT-2 vector seamount
record reproduces the certified kt=1..10 registry exactly. Its independent
100-day legoESM run remains bounded: wet three-dimensional temperature RMS is
`8.1037591477894766e-06 K` at day 100, `0.1870625719` times SMT-1. The frozen
temperature prediction is confirmed; the frozen raw-NEMO U-maximum prediction
is refuted. No model, card, option, carried state, or certified row changes.

Base: `b937f824e` (round 243). Preregistration commit: `409e0fc46`.
Measurement-gate commit: `80551c09f`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round244/`.
Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_smt_round32_smt2_100day_score.md`.

## 1. Admission and calibration

The operator-run round-243 acquisition ended `status: ADMITTED`; its reference
restart is byte-identical, its self-describing parser completed, and its header
plant printed `PLANT_FIRED`. The final log markers are
`VORTEX_round222_smt2_vec_100d_KT1_10_ORACLE_READY` and
`ROUND243_SMT2_RECORD_READY`. The round-244 gate independently counts exactly
100 NEMO restarts at steps 30, 60, ..., 3000 and 100 legoESM daily snapshots.

The compiled executing restart program schedules frequency-based output at
`VORTEX_SMT2_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:104-119` and writes
the scored `sshn/un/vn/tn/sn` state at
`VORTEX_SMT2_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:176-180`. No
interpolation or reconstructed NEMO state enters the score.

Before any long-run row was accepted, the shared trajectory gate replayed the
record's first ten entries against the round-237 certified SMT-2 registry. It
reported `REPRODUCED` with no mismatch. R31-P1 and R31-P2 are **CONFIRMED**.

## 2. Registered 100-day checkpoints

All values use the shared round-210 wet masks and definitions. RMS and maxima
are legoESM minus NEMO; temperature is K, velocity is m/s, and sea-surface
height is m. The ratio compares SMT-2 temperature RMS with SMT-1 through the
same scorer.

| day | T rms | T max | u rms | u max | v rms | v max | ssh rms | ssh max | T rms / SMT-1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `3.009006254e-08` | `1.953681823e-06` | `3.130114835e-08` | `1.769572090e-06` | `2.708432374e-08` | `2.060270591e-06` | `1.230737048e-08` | `2.092418477e-07` | `0.937891` |
| 2 | `4.835205938e-08` | `1.702269054e-06` | `4.967169369e-08` | `4.433974903e-06` | `4.229685730e-08` | `2.766035351e-06` | `1.521374821e-08` | `2.023699659e-07` | `0.938936` |
| 5 | `8.673023344e-08` | `4.287731894e-06` | `9.293992035e-08` | `4.585228546e-06` | `8.373746326e-08` | `4.719703421e-06` | `1.710844485e-08` | `1.473387115e-07` | `0.813654` |
| 10 | `2.071836645e-07` | `1.156461406e-05` | `2.003535480e-07` | `8.501554233e-06` | `1.947568272e-07` | `1.017036663e-05` | `3.639925735e-08` | `3.011485135e-07` | `0.793551` |
| 20 | `3.959757166e-07` | `1.733356480e-05` | `3.886548833e-07` | `2.049738178e-05` | `3.735964174e-07` | `2.298455928e-05` | `5.864778999e-08` | `6.277690340e-07` | `0.748155` |
| 30 | `7.656779707e-07` | `3.392053290e-05` | `7.562899611e-07` | `4.503659482e-05` | `6.842185984e-07` | `3.297182824e-05` | `1.015612127e-07` | `1.188107948e-06` | `0.654322` |
| 60 | `1.870252921e-06` | `4.802040463e-05` | `1.196338850e-06` | `4.337386957e-05` | `1.309719318e-06` | `5.505260730e-05` | `4.951966669e-07` | `6.605859332e-06` | `0.311076` |
| 100 | `8.103759148e-06` | `1.739061947e-04` | `5.584274557e-06` | `2.157627635e-04` | `5.912739345e-06` | `1.597966848e-04` | `2.563488936e-06` | `3.960025435e-05` | `0.187063` |

At day 100 the SMT-2/SMT-1 ratios are `0.1870625719` for T RMS,
`0.2070711310` for u RMS, `0.2259395760` for v RMS, and `0.3717408904`
for ssh RMS. The temperature prediction was “within 2x of SMT-1”; it is
**CONFIRMED**. R31-P4 is **CONFIRMED**. These curve ratios do not name a causal
owner.

The second half of R31-P3 predicted that NEMO's raw day-100 maximum absolute U
would be lower on SMT-2 than SMT-1. The common restart reader and each card's
own wet U mask measure `1.1348825563438096 m/s` for SMT-2 versus
`0.8300632333203959 m/s` for SMT-1. That prediction is **REFUTED** and retained;
the smaller legoESM-minus-NEMO error does not imply a smaller oracle speed.

## 3. Visuals, controls, and disposition

The shared renderer produced a 100-frame MP4 (`1,334,678` bytes), a 100-frame
GIF (`1,673,056` bytes), and the day-1/30/60/100 montage (`332,489` bytes).
The panels use SMT-2's resolved seamount bathymetry. The SSH-difference scale
is fixed to the measured day-100 maximum `3.9600254352178954e-05 m`; the
velocity-arrow scale is fixed from day 1. R31-P5 is **CONFIRMED**.

The committed gate stamps a clean worktree and reports:

> `STATUS PASS: short=REPRODUCED day100_T_rms=8.10375914778947659e-06 K T_ratio=0.187062572 restarts=100 frames=100`

Deleting day 60, changing the T result beyond the frozen 2x prediction, and
reversing the NEMO-U ordering each print `STATUS PLANT-FIRED` and exit 1. The
gate records the real U prediction as `REFUTED` rather than weakening or
removing it.

The initial focused suite reports `9 passed in 3.57s`. Final focused and
citation results are recorded below after review. No full physics tree or DINO
month integration is required because no file under `packages/` or `src/`
changes. R31-P6 is **CONFIRMED**. This round names no first non-bit statement:
the ordered program is measurement-only by operator note CI, and production is
unchanged.

## 4. Independent review and citation gate

Independent review: **PENDING**.

Citation gate: **PENDING**.

## 5. OPEN — next round

Proceed to SMT-3's deferred 100-day comparison and movie, one rung only. First
inventory the existing round-224 SMT-3 oracle directory for exactly 100 daily
restarts and its admission record. If complete, preregister scoring and reuse
the shared scorer, renderer, certified SMT-3 ladder, and round-244 gate pattern;
if not, preregister the existing `smt3vec100d` acquisition arm and stop for the
operator record. Register days 1/2/5/10/20/30/60/100 and do not infer a causal
owner from the curve. SMT-4 remains the following round. Production changes
are outside this deferred measurement sequence unless a cited Decision-96 net
improvement is independently named.

**DECISION_NEEDED: NONE. ACQUISITION_NEEDED: NONE.**
