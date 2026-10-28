# ORCA2 round 344 — native-transcendental fast-literal K0 receipt

- **Round**: `orca2_r344`
- **Author**: Codex
- **Disposition**: **HELD for GPU acquisition**; CPU implementation and evidence complete
- **Preregistration**: `docs/ocean/fidelity/testcases/PREREG_nemo_testcases_l4_orca2_round344_fast_literal.md`
- **CPU artifact**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r344/arm_fast_literal_cpu_fp64.json` (SHA-256 `5f490521693c16da8609eca5b389520d406d0636d1759aad20f11d15d5e631aa`)
- **Exact control artifact**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r344/arm_exact_cpu_fp64.json` (SHA-256 `dfc177f2893fb206583e220be61e1c724aceaabac0c7227163980e19f81edaf5`)
- **Round-343 comparison report**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r343/orca2_round343_speed_benchmark_report.json` (SHA-256 `e1e8c4a241fb201ac3d8f72a35e3964eb8ad78fa95e309a9da09d2da58f710e1`)
- independent review: operator (Claude)

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

The CPU result is the intended outcome: **1.5802 s/step**, a **13.64x** speedup
over round 343 exact CPU, with day-8 wet SST RMS **7.16e-11 C** instead of the
normal-mode **5.29e-3 C**.  The native result therefore preserves the exact
physics to round-off growth rather than switching formulas.

## Speed and one-year projection

All CPU values use fp64, 64 steps, two warm-up steps, and the round-343 timing
protocol.  The normal/exact columns are the frozen round-343 measurements.

| Metric | Exact CPU | Normal CPU | Fast-literal CPU | Normal GPU | Fast-literal GPU |
|---|---:|---:|---:|---:|---:|
| Steady median (s/step) | 21.5587 | 1.3185 | **1.5802** | 0.4680 | **UNMEASURED** |
| Odd-kt median (s/step) | 22.5248 | 1.3837 | **1.7528** | 0.7576 | **UNMEASURED** |
| Even-kt median (s/step) | 19.5232 | 1.2021 | **1.2829** | 0.1898 | **UNMEASURED** |
| Compile + warm-up (s) | 239.44 | 175.42 | **231.09** | 263.73 | **UNMEASURED** |
| Projected 2,920-step year, steady (min) | 1049.19 | 64.16 | **76.90** | 22.77 | **UNMEASURED** |
| Projected year including compile (min) | 1052.46 | 67.04 | **80.70** | 27.15 | **UNMEASURED** |
| Host peak RSS (MiB) | 11290.5 | 9847.1 | **10472.7** | 6739.1 | **UNMEASURED** |

Fast-literal CPU is 13.64x faster than exact CPU and 1.20x slower than the
formula-switching normal CPU arm.  The latter comparison is descriptive, not
a changed acceptance bar.

## Eight-day accuracy against saved exact states

Values are wet-cell RMS and absolute maximum in degrees C.  `Fast` is the
round-344 CPU arm; `Normal` is round-343 `normal_cpu_fp64` beside it.

| Day | SST RMS Fast | SST RMS Normal | SST max Fast | SST max Normal | 3-D T RMS Fast | 3-D T RMS Normal | 3-D T max Fast | 3-D T max Normal |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.351e-11 | 1.748e-3 | 3.313e-9 | 8.255e-2 | 4.366e-11 | 1.754e-3 | 2.325e-8 | 2.451e-1 |
| 2 | 3.578e-11 | 2.820e-3 | 2.084e-9 | 2.028e-1 | 2.882e-11 | 3.295e-3 | 8.234e-9 | 5.977e-1 |
| 3 | 1.860e-10 | 4.390e-3 | 2.332e-8 | 2.998e-1 | 1.290e-10 | 4.742e-3 | 5.556e-8 | 6.654e-1 |
| 4 | 9.188e-11 | 4.337e-3 | 8.411e-9 | 1.895e-1 | 8.898e-11 | 5.289e-3 | 3.040e-8 | 5.840e-1 |
| 5 | 5.207e-11 | 4.464e-3 | 2.447e-9 | 1.508e-1 | 5.123e-11 | 5.517e-3 | 1.835e-8 | 6.699e-1 |
| 6 | 5.959e-11 | 5.319e-3 | 4.957e-9 | 3.494e-1 | 5.201e-11 | 5.793e-3 | 1.339e-8 | 7.020e-1 |
| 7 | 6.646e-11 | 4.836e-3 | 5.123e-9 | 1.744e-1 | 6.089e-11 | 6.251e-3 | 2.372e-8 | 7.040e-1 |
| 8 | **7.163e-11** | **5.293e-3** | **3.414e-9** | **1.604e-1** | **7.772e-11** | **6.891e-3** | **1.652e-8** | **8.072e-1** |

Fast CPU is closer to exact than normal CPU for every preregistered day and
metric.  The exact/libm control reproduced the saved exact states bit for bit
at days 1 and 2: all four SST/3-D T RMS and maximum scores are zero.

## Predictions and controls

1. **Native reachability: confirmed with one qualified refutation.** DDM, IWM,
   htau, and etau all execute with native fp64 under eager and JIT.  The frozen
   prediction of bit-identical eager/whole-call-JIT IWM was refuted: 7 of 116
   output words differ by exactly one ULP.  This is XLA-context round-off; the
   formula, selector, and operation order are unchanged.  The gate now records
   the observed maximum of one ULP instead of hiding the refutation.
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

Focused gates: 87 passed/1 deselected; benchmark/report gates: 7 passed/1
deselected; real full-card gradient: 1 passed in 1269.75 s.  No NEMO executable
was run and no process was killed.

## GPU acquisition hold

`nvidia-smi` could not communicate with the NVIDIA driver, so no GPU could be
proven idle at 0 MiB.  In accordance with the preregistration and user kill
rule, the GPU arm was not launched.  The committed completion runner refuses
unless the selected device reports exactly 0 MiB used and 0% utilization, then
runs only `fast_literal_gpu_fp64` and assembles the comparison report:

`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round344_fast_literal_gpu/run.sh --run GPU_INDEX`

This receipt is therefore **HELD**, not a completed two-backend benchmark.
No scientific decision is needed; only the missing idle-GPU acquisition and
the designated operator review remain.
