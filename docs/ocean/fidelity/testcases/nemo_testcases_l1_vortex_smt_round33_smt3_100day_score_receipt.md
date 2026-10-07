# Receipt — VORTEX_SMT round 33 (lane round 245): SMT-3 100-day score

**Status: HELD (measurement complete).** The admitted SMT-3 trajectory
reproduces the current post-landing certified kt=1..10 registry exactly. Its
day-100 wet three-dimensional temperature RMS is
`1.7729713625071864e-04 K`, 3.8431 times smaller than the pre-landing SMT-3
measurement and 21.8784 times SMT-2. Both frozen magnitude predictions are
confirmed. No model, card, option, carried state, or certified row changes.

Base: `cb2a145b795c` (round 244). Preregistration commit: `ba179a805`.
Measurement-gate commit: `123b9b36c`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round245/`.
Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_smt_round33_smt3_100day_score.md`.

## 1. Admission and calibration

The round-224 self-describing admission contains 3,067 records and reports
`ADMITTED`; the reference restart is byte-identical. Its header, field-name,
and truncation plants each report `REFUSED`. The round-245 gate independently
counts exactly 100 NEMO restarts at steps 30, 60, ..., 3000 and 100 legoESM
daily snapshots.

The compiled executing restart program schedules frequency-based output at
`VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:104-119` and writes
the scored `sshn/un/vn/tn/sn` state at
`VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:176-180`. Thus the
score uses NEMO's own daily state without interpolation or reconstruction.

Before accepting any long-run row, the shared trajectory gate replayed the
record's first ten entries against `round237/landing_smt3.json`, the certified
registry after the SMT-3 mask-plus-live-divisor landing. It reported
`REPRODUCED` with no mismatch. The previous scorer path to round 226 was stale
for the current production tree and is replaced, not re-baselined. R33-P1 and
R33-P2 are **CONFIRMED**.

## 2. Registered 100-day checkpoints

All values use the shared round-210 wet masks and definitions. RMS and maxima
are legoESM minus NEMO; temperature is K, velocity is m/s, and sea-surface
height is m. The final column compares temperature RMS with the current SMT-2
score through the identical instrument.

| day | T rms | T max | u rms | u max | v rms | v max | ssh rms | ssh max | T rms / SMT-2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `1.503976648e-07` | `1.684692980e-05` | `7.004846967e-08` | `7.365742518e-06` | `9.036945710e-08` | `7.570893772e-06` | `1.213541790e-08` | `2.070552911e-07` | `4.998250` |
| 2 | `4.003278676e-07` | `3.409043641e-05` | `1.350373764e-07` | `7.785225217e-06` | `1.751540526e-07` | `2.211934720e-05` | `1.875155014e-08` | `1.819588056e-07` | `8.279438` |
| 5 | `4.334339785e-06` | `5.372494668e-04` | `2.035339384e-06` | `2.526142377e-04` | `1.595872902e-06` | `1.485257624e-04` | `1.519149722e-07` | `1.186719908e-06` | `49.974958` |
| 10 | `2.181077155e-05` | `2.146908676e-03` | `7.018137437e-06` | `7.143809524e-04` | `7.810591861e-06` | `5.744696024e-04` | `5.699176733e-07` | `3.693022547e-06` | `105.272641` |
| 20 | `7.176464070e-05` | `5.279472787e-03` | `1.943784271e-05` | `1.322577956e-03` | `1.923925176e-05` | `8.697613252e-04` | `2.773651899e-06` | `2.239087003e-05` | `181.234954` |
| 30 | `1.725333875e-04` | `1.388417129e-02` | `3.976730749e-05` | `1.640902194e-03` | `4.310334301e-05` | `2.136183313e-03` | `8.075498857e-06` | `4.728042816e-05` | `225.334141` |
| 60 | `2.042506718e-04` | `1.034854866e-02` | `7.128521155e-05` | `2.051270629e-03` | `7.184987317e-05` | `3.360332300e-03` | `3.081463271e-05` | `4.330449154e-04` | `109.210187` |
| 100 | `1.772971363e-04` | `9.779769286e-03` | `7.589695144e-05` | `2.238703054e-03` | `7.228177385e-05` | `1.420688413e-03` | `4.103403297e-05` | `4.425164875e-04` | `21.878382` |

The pre-landing round-226 day-100 T RMS was
`6.813785267886451e-04 K`; the current result removes 73.9796% of that RMS.
The current SMT-2 reference is `8.1037591477894766e-06 K`. Therefore R33-P3
is **CONFIRMED** on both inequalities. The curve is a controlled result of the
already-landed production state, but it does not name a new causal owner.
R33-P4 is **CONFIRMED**.

## 3. Visuals, controls, and disposition

The shared renderer produced a 100-frame MP4 (`1,160,309` bytes), a 100-frame
GIF (`1,842,734` bytes), and the day-1/30/60/100 montage (`324,245` bytes).
The panels use SMT-3's resolved seamount bathymetry. The SSH-difference scale
is fixed to the measured day-100 maximum `4.4251648745360916e-04 m`; the
velocity-arrow scale is fixed from day 1. R33-P5 is **CONFIRMED**.

The committed gate stamps clean commit `123b9b36c` and reports:

> `STATUS PASS: short=REPRODUCED day100_T_rms=1.77297136250718641e-04 K T_ratio=21.878381751 restarts=100 frames=100`

Deleting day 60, setting day-100 T above the frozen pre-landing bound, and
setting it below SMT-2 each print `STATUS PLANT-FIRED` and exit 1. This is a
measurement-only round: no file under `packages/` or `src/` changes, and no
physics battery or DINO month integration is required. R33-P6 is
**CONFIRMED**. No first non-bit statement is named because the operator's
ordered work item is this deferred trajectory comparison, not a statement
walk.

## 4. Independent review, citations, and tests

The separate read-only Codex review, citation gates, shifted-citation plant,
and final focused test summaries are recorded here after their committed-diff
passes.

## 5. Choices and OPEN

**UNASKED list: EMPTY.** The 100-day length, daily cadence, checkpoint fields,
masks, and movie form are inherited from Decisions 87/93 and the operator's
one-rung order. No scientific option, threshold, stabiliser, or source changes.

OPEN: proceed to SMT-4's deferred 100-day comparison and movie, one rung only.
First inventory the admitted round-237 SMT-4 daily record for exactly 100
daily restarts and its admission report. If complete, preregister and score it
with the current certified SMT-4 registry and this gate pattern; otherwise
preregister the existing `smt4vec100d` acquisition arm and stop for the
operator record. Register days 1/2/5/10/20/30/60/100 and infer no causal owner
from the curve. After SMT-4 the ordered batch stops with `DECISION_NEEDED` for
the operator's PR work, as note CI requires.

**DECISION_NEEDED: NONE. ACQUISITION_NEEDED: NONE.**
