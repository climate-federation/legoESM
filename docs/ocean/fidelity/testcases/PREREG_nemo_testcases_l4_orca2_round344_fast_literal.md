# ORCA2 round 344 preregistration — native-transcendental fast literal mode

- **Round**: `orca2_r344`
- **Author**: Codex
- **Branch**: `fidelity/nemo-testcases-orca2-rounds`
- **Base implementation**: round-343 benchmark series ending at `d4c8ef938`
- **Date**: 2026-10-28

## Objective and fixed configuration

Add and measure a standalone ORCA2 K0 **fast-literal** configuration.  It keeps
the exact card's NEMO-literal DDM, IWM, TKE htau/etau, source-order, and
rounding selectors, and changes only the card and precision policy
transcendental backend from scalar CPU `libm` to native XLA.  No class default,
physical formula, coefficient, timestep, forcing, mask, score, or saved exact
reference changes.

The literal statements are those executed by the compiled ORCA2 source:

- DDM nested EXP and written update order: ORCA2 `zdfddm.f90:159-172`.
- IWM EXP/TANH/LOG10 and written level/update order: ORCA2
  `zdfiwm.f90:232-275,277-316`.
- TKE latitude htau and below-mixed-layer EXP: ORCA2
  `zdftke.f90:858-865,490-502`.

The scalar-libm path remains the exact path.  Under the native policy the same
literal arithmetic calls XLA transcendental primitives; therefore any
difference from exact mode is classified as arithmetic round-off, not a
formula or physics switch.

## Frozen implementation predictions and falsifiers

1. **Native literal reachability.**  With `PrecisionPolicy.fp64(
   transcendentals="native")`, the existing DDM/IWM NEMO-literal functions and
   the TKE scalar-literal htau/etau selections return finite fp64 arrays under
   eager and JIT execution.  The parent refuses all four at their policy
   guards.  **Refute** if any literal path still refuses native, changes its
   formula/operation-order statements, or eager/JIT differs beyond identical
   native-XLA execution.
2. **Exact/libm preservation.**  Parent and candidate produce bit-identical
   outputs for DDM, IWM, TKE htau and TKE etau under scalar `libm`; the resolved
   rung-10 ORCA2 card and standalone S4 K0 card are field-for-field unchanged.
   Every non-fast selector/default remains unchanged.  **Refute** on one
   unequal output word or one changed resolved field/default.
3. **Fast-card isolation.**  The fast card differs from exact K0 only in
   `ocean.transcendentals: libm -> native`; all literal physics selectors,
   including DDM enabled/form, IWM form/tsdiff and both TKE evaluations, remain
   exact-card values.  A plant that changes any retained selector must fail.
   **Refute** on any additional resolved-card difference.
4. **Differentiability.**  A real one-step fast K0 loss at `N=1` has finite
   forward and reverse directional derivatives, with reverse equal to forward
   under the existing gradient comparator.  **Refute** on non-finite values or
   comparator failure.

## Frozen benchmark protocol

Reuse round 343's script, inputs, daily exact snapshots, warm-up definition,
temperature scores, timing statistics, and one-year projection.  Run only:

| Arm | Backend | Precision | Steps | Steady window | Accuracy snapshots |
|---|---|---:|---:|---|---|
| `fast_literal_cpu_fp64` | CPU | fp64/native | 64 | kt 3..64 | kt 8,16,...,64 |
| `fast_literal_gpu_fp64` | one GPU proven idle at 0 MiB | fp64/native | 64 | kt 3..64 | kt 8,16,...,64 |

The fixed references are
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/legoesm_year1_2920step/candidate_step*.npz`.
Report median seconds/step, odd/even medians, compile warm-up, projected 2,920
step year, and daily SST/3-D T all-cell/wet RMS and maximum beside round 343's
normal fp64 arms.  The accuracy values are measurements, not a newly chosen
pass/fail bar.  The directional prediction is that both fast-literal arms are
closer to exact than the matching round-343 normal fp64 arm at every reported
day/metric; **refute** on any metric that is not closer.  CPU/GPU differences
are reported directly rather than assigned a post-hoc tolerance.

## Required gates

- Focused native reachability, isolation, libm preservation, and planted
  violation tests.
- Real K0 `N=1` forward/reverse gradient check.
- Rung-10 and S4 K0 resolved-card parent/current controls; all shared callers
  parent/current controls.
- Default gate, ratchets, round-343 tests, receipt citation gate, and
  `tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py` immediately
  before the final commit.
- Independent review: operator (Claude).

