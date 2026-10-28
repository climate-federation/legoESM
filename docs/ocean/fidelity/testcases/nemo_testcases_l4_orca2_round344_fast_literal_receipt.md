# ORCA2 round 344 — native-transcendental fast-literal K0 receipt

- **Round**: `orca2_r344`
- **Author**: Codex
- **Disposition**: **LAND**; the repaired-code CPU and GPU arms are complete
- **Preregistration**: `docs/ocean/fidelity/testcases/PREREG_nemo_testcases_l4_orca2_round344_fast_literal.md`
- **Repaired CPU artifact**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r344/arm_fast_literal_cpu_fp64.json` (SHA-256 `3881e7f0cd5575f2c09be004053c9e10dcb5ae5f4864967dd7faf62465f5478c`)
- **Exact control artifact**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r344/arm_exact_cpu_fp64.json` (SHA-256 `dfc177f2893fb206583e220be61e1c724aceaabac0c7227163980e19f81edaf5`)
- **Round-343 comparison report**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r343/orca2_round343_speed_benchmark_report.json` (SHA-256 `e1e8c4a241fb201ac3d8f72a35e3964eb8ad78fa95e309a9da09d2da58f710e1`)
- **Repaired GPU artifact**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r344/arm_fast_literal_gpu_fp64.json` (SHA-256 `60b80c87fd3b0b99e697ae32571291024c298ed4a16c424a8c5d05eea05bc5fa`)
- **Repaired two-arm report**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r344/orca2_round344_fast_literal_benchmark_report.json` (SHA-256 `25ffbfb5182a53b50bd952890bf7734667f323ae21f5a1922656f754f98f4035`)
- independent review: Gemini Flash **LAND**; Argon re-check verified the IWM
  and DDM repairs, and its artifact-digest hold is resolved here

## Verdict

The CPU fast-literal card changes only `ocean.transcendentals` from `libm` to
`native`; the exact recipe object and all 44 retained inventory entries remain
unchanged.  The builder and fail-closed verifier are at
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round343_speed_benchmark.py:943-990`.
The literal DDM, IWM, and TKE htau/etau implementations retain their written
formulas and operation-order barriers at
`packages/ocean/legoesm/ocean/physics/vertical_mixing/double_diffusion.py:309-358`,
`packages/ocean/legoesm/ocean/physics/vertical_mixing/internal_wave_mixing.py:388-482`,
and `packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:2708-2811`.
Only the precision policy dispatches EXP/LOG10/TANH/SIN to native XLA.

The repaired CPU result is the intended outcome: **1.5395 s/step**, a **14.00x**
speedup over round 343 exact CPU, with day-8 wet SST RMS **6.35e-11 C** instead of the
normal-mode **5.29e-3 C**.  The native result therefore preserves the exact
physics to round-off growth rather than switching formulas.

## Speed and one-year projection

All CPU values use fp64, 64 steps, two warm-up steps, and the round-343 timing
protocol.  The normal/exact columns are the frozen round-343 measurements.

| Metric | Exact CPU | Normal CPU | Repaired fast-literal CPU | Normal GPU | Repaired fast-literal GPU |
|---|---:|---:|---:|---:|---:|
| Steady median (s/step) | 21.5587 | 1.3185 | **1.5395** | 0.4680 | **0.5283** |
| Odd-kt median (s/step) | 22.5248 | 1.3837 | **1.7780** | 0.7576 | **0.9169** |
| Even-kt median (s/step) | 19.5232 | 1.2021 | **1.3943** | 0.1898 | **0.2230** |
| Compile + warm-up (s) | 239.44 | 175.42 | **246.58** | 263.73 | **354.62** |
| Projected 2,920-step year, steady (min) | 1049.19 | 64.16 | **74.92** | 22.77 | **25.71** |
| Projected year including compile (min) | 1052.46 | 67.04 | **78.98** | 27.15 | **31.60** |
| Host peak RSS (MiB) | 11290.5 | 9847.1 | **10015.3** | 6739.1 | **6740.8** |

Repaired fast-literal CPU is 14.00x faster than exact CPU and 1.17x slower than the
formula-switching normal CPU arm.  The latter comparison is descriptive, not
a changed acceptance bar.  The repaired fast-literal GPU is 40.81x faster
than exact CPU and 1.13x slower than the formula-switching normal GPU arm.

## Eight-day accuracy against saved exact states

Values are wet-cell RMS and absolute maximum in degrees C.  `Fast` is the
round-344 CPU arm; `Normal` is round-343 `normal_cpu_fp64` beside it.

| Day | SST RMS Fast | SST RMS Normal | SST max Fast | SST max Normal | 3-D T RMS Fast | 3-D T RMS Normal | 3-D T max Fast | 3-D T max Normal |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.229e-11 | 1.748e-3 | 3.351e-9 | 8.255e-2 | 4.515e-11 | 1.754e-3 | 2.357e-8 | 2.451e-1 |
| 2 | 3.763e-11 | 2.820e-3 | 3.180e-9 | 2.028e-1 | 2.898e-11 | 3.295e-3 | 6.731e-9 | 5.977e-1 |
| 3 | 2.190e-10 | 4.390e-3 | 2.753e-8 | 2.998e-1 | 1.483e-10 | 4.742e-3 | 6.347e-8 | 6.654e-1 |
| 4 | 7.038e-11 | 4.337e-3 | 5.317e-9 | 1.895e-1 | 7.081e-11 | 5.289e-3 | 2.957e-8 | 5.840e-1 |
| 5 | 4.584e-11 | 4.464e-3 | 2.236e-9 | 1.508e-1 | 4.419e-11 | 5.517e-3 | 1.520e-8 | 6.699e-1 |
| 6 | 5.284e-11 | 5.319e-3 | 2.888e-9 | 3.494e-1 | 5.191e-11 | 5.793e-3 | 1.493e-8 | 7.020e-1 |
| 7 | 5.696e-11 | 4.836e-3 | 3.152e-9 | 1.744e-1 | 5.121e-11 | 6.251e-3 | 1.378e-8 | 7.040e-1 |
| 8 | **6.349e-11** | **5.293e-3** | **2.806e-9** | **1.604e-1** | **6.432e-11** | **6.891e-3** | **1.101e-8** | **8.072e-1** |

Fast CPU is closer to exact than normal CPU for every preregistered day and
metric.  The exact/libm control reproduced the saved exact states bit for bit
at days 1 and 2: all four SST/3-D T RMS and maximum scores are zero.

## Predictions and controls

1. **Native reachability: confirmed exactly.** DDM, IWM, htau, and etau all
   execute with native fp64 under truly eager whole calls and JIT.  Rounding the
   LOG10 result before NEMO's following subtraction restores bit-identical IWM
   output.  A plant that strips only this barrier reproduces 7 unequal words in
   the 116-word salinity output; the exact eager/JIT assertion then fails.
2. **Exact/libm preservation: confirmed.** The scalar-libm unit outputs are
   bit-identical; rung-10 and standalone S4 K0 resolve identically; class
   defaults and all non-fast callers are unchanged.  The preservation battery
   passed 87 tests, with one operator-only gradient test deselected.
3. **Fast-card isolation: confirmed.** The real card reports only
   `ocean.transcendentals` changed and 44 retained inventory entries.  The
   planted TKE-selector violation fires.
4. **Differentiability: confirmed.** The real one-step K0 `N=1` primal, JVP,
   and VJP are finite, the derivative is nonzero, and reverse equals forward
   under the registered `rtol=atol=2e-12` comparator.

The repaired focused battery passes 13 tests with one operator-only test
deselected.  The clean-tree receipt gate finds 4 citations, 0 failures, 0
unmapped citations, and all 9 self-test plants fire.  The default ORCA2 landing
battery passes 155/155 in 1321.90 s, including the 20-test citation suite and
six-test private-import ratchet.  No NEMO executable was run and no process was
killed.

## GPU acquisition status

The operator completed the repaired-code GPU arm and replacement two-arm report
at `015b97ec56c2ef03d85dc9e0251ae41170ee2a82`.  It measured **0.5283
s/step**, **25.71 min/year**, and day-8 wet SST RMS **1.098e-10 C** with
maximum **1.100e-8 C**.  Day-8 wet 3-D T RMS was **1.321e-10 C** with maximum
**6.394e-8 C**.  Every artifact SHA-256 at the top of this receipt was
recomputed directly from the current file with `sha256sum`.

The controlled two-backend benchmark is complete.  No NEMO executable was run
and no process was killed during receipt completion.
