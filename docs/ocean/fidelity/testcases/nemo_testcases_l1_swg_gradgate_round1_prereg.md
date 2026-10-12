# Certified SWG reverse-mode gradient gate — round 1 preregistration

Date: 2026-10-12

Status: **FROZEN BEFORE MEASUREMENT**. This document fixes the SWG input,
objective, random direction, comparison bars, checkpoint rows, memory
extrapolation, and confirm/refute rules before any SWG gradient number is run.
No failed result will be repaired in this round: a non-differentiable statement
is reported with its executed source line as `DECISION_NEEDED`.

## Imported gate and unchanged-card control

The shared GYRE gate lineage was cherry-picked from read-only
`fidelity/gyre-gradient-gate` rather than duplicated. It includes the approved
TKE inactive-operand treatment and the active-cell CEN2 denominator treatment.
Before and after that import, the committed SWG ladder produced byte-identical
result rows: given-entry 338/338, chained 338/338, independent 50/50,
carried-state 29/29, and thirty-day 690/690. The existing committed card-digest
helper also remained identical for TSUNAMI (`8bb5ed6c1bc031bd`), VORTEX
(`f7489467bc156303`), VORTEX_VEC (`5041b91c67bd1443`), and GYRE
(`3b2b358145ced147`). Evidence lives under `swggg_rounds/round1/`.

## Frozen production path and initial boundary

Every row uses CPU fp64 with `PrecisionPolicy.fp64(transcendentals="libm")`
and enters every modeled step through production
`LatLonCGridOceanModel._step_jitted`. The gate builds `build_swg_card()` and
refuses any change from the certified `swe_rk3`, one-layer, total-ENS,
constant-viscosity, no-tracer, no-TKE program. State and geometry must print
`float64`.

The start is the content-pinned admitted kt=2 stage-1 record under
`swg_rounds/round1/oracle_swg_r1/run`, not rest. Its SSH, native U/V, before
stress, and current stress rebuild the public production state and surface
forcing. Before any rollout, the gate reconstructs all thirteen stage-entry
fields and requires array equality with the admitted record.

The compiled NEMO program executes three RK stages at
`tests/SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:138-312`; stage 3 includes
viscosity, centered wind, and drag at `:250-303`. The live total-ENS thickness
division is `dynvor.f90:639-660`. NEMO first forms masked F thickness at
`dynvor.f90:898-926` and replaces every zero denominator at `:927-928`.
Free-slip is selected at `dommsk.f90:171-178`, and its U/V/F masks are fixed
from the static T mask at `:206-214`.

## Frozen directional objective

The zero-perturbation production rollout supplies the after-state truth for
each window. The scalar loss is the mean of three wet-domain normalized MSEs:
SSH on wet T cells, U on wet U faces, and V on wet V faces. Each component is
divided by that truth field's masked spatial variance. Static dry cells and
closed faces have exactly zero weight.

The differentiated input is a three-field `(SSH, U, V)` perturbation at the
admitted boundary. PRNG key 0 supplies its independent standard-normal leaves;
after masking, the combined pytree L2 norm is `1e-3`. PRNG key 42 supplies an
independent masked direction with combined L2 norm one. Reverse mode
differentiates the scalar line in `alpha`; this equals the full input VJP dotted
with the registered direction without materializing the full VJP. Forward mode
uses `jax.jvp` on the same line. Central finite differences use
`alpha = +/-1e-3`.

Agreement uses the shared `checkpointed_loop` with schedule `uniform`; its body
calls production `_step_jitted`. Before scoring, its zero-perturbation primal
must be bitwise identical to the direct per-step production rollout.

## Predictions and exact bars

Rows are one step (1 hour), 24 steps (1 day), and 720 steps (30 days), each in
its own process.

- **Prediction P1:** reverse, forward, and central FD are finite and nonzero for
  every window. Reverse/forward relative error is at most `1e-9`; reverse/FD
  relative error is at most `0.05`. All conditions CONFIRM; any violation,
  missing row, exception, zero, or NaN REFUTES.
- **Prediction P2:** NaNs are absent. SWG has neither tracer advection nor TKE.
  ENS divides by a live F thickness whose compiled zero replacement is cited
  above; the free-slip masks are static multiplicative operands, not traced
  state decisions. A reverse-only NaN REFUTES and its first executed owner is
  reported rather than guarded.
- **Prediction P3:** on the one-step production row, the broken-VJP plant
  doubles only reverse mode, leaves forward/FD unchanged, and exits nonzero.
  Failure to turn the row red REFUTES instrument non-vacuity.

## Resource rows and 60-day extrapolation

For 24 and 720 steps, each of `none`, `uniform`, and `binomial` runs in its own
process through the shared schedule implementation in
`packages/ml/legoesm/training/checkpoint_schedule.py`. A cold compiled reverse
pass is followed by one blocked warm reverse pass. Host RSS is sampled every
0.05 s during the warm pass; the row records baseline, peak, increment, sample
count, cold compile-plus-reverse wall time, and warm reverse wall time.

Each resource row CONFIRMS only if the scheduled primal is bitwise production,
loss is finite, and the directional reverse derivative is finite and nonzero.
Peak RSS must be at least baseline, wall times must be positive, and at least
one sample beyond baseline must exist. Otherwise it REFUTES; failed compilation
or allocation is `UNMEASURED`, never a memory peak.

The 1440-step (60-day) estimates are computed only from successful 720-step
rows. With measured increment `I = peak720 - baseline720`:

- `none`: `baseline720 + 2 I` (linear tape storage),
- `uniform`: `peak720` (constant extra tape),
- `binomial`: `baseline720 + (38/27) I`, using
  `ceil(sqrt(1440))/ceil(sqrt(720))`,
- warm reverse wall time: `2 * warm720` for every schedule; this is a labeled
  schedule-complexity extrapolation, not a measured 1440-step run.

## GPU operator rows

The existing shared operator script is extended, not duplicated. It runs the
same one/24/720 agreement rows and 24/720 resource rows serially on physical
GPU 1 with `CUDA_VISIBLE_DEVICES=1` and
`PrecisionPolicy.fp64(transcendentals="native")`. Every JSON must label that
policy as GPU/XLA native and not certified bit-exact. GPU rows are operator work
because this sandbox reports `CUDA_ERROR_NO_DEVICE`.

## Landing rule

The round can be `LANDED` only if the one/24/720 CPU agreement rows pass, the
one-step broken-VJP plant binds, all six CPU resource rows complete and pass,
the step-gradient matrix contains a fail-closed isolated SWG one-step row,
direct tests pass, the SWG ladder/card controls remain unchanged, the receipt
citation gate passes, and the GPU operator script is committed. Any executed
non-differentiable owner stops the round for a decision; no guard, clipping,
stabilizer, `stop_gradient`, physics, card, or tolerance change is authorized.

## Observed control result and protocol amendment

After the frozen protocol was committed at `79f086a51`, the one-step row and
broken-VJP plant behaved as predicted. The 24-step run stopped before any
derivative: the `checkpointed_loop:uniform` zero-perturbation primal was not
byte-identical to a Python sequence of production `_step_jitted` calls. The
differences were finite CPU roundoff in 70 U values (maximum
`1.3877787807814457e-17`), 55 V values (`6.071532165918825e-18`), and 74 SSH
values (`8.881784197001252e-16`). Thus the preregistered bitwise composition
prediction is **REFUTED** and remains so; it is not relabeled as confirmation.

Before measuring the 24- or 720-step derivatives or any resource row, the
protocol is amended as follows. Directional agreement uses the same
Decision-49 composition as GYRE: an explicit Python window with a rematerialized
production `_step_jitted` call at every step, whose primal must remain bitwise
production. Resource rows necessarily exercise `checkpointed_loop`; their
primal is gated by the schedule implementation's already-committed value
contract (`rtol=1e-9`, `atol=1e-11`) from
`tests/unit/test_checkpoint_schedule.py:94-102`, while recording whether it was
also bitwise and its maximum absolute difference. No physical input, card,
loss, random perturbation, derivative bar, FD step, or memory protocol changes.
