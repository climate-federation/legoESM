# Preregistration — GYRE identical-twin gradient parameter recovery

Date: 2026-09-20

Incoming lane tip: `4cac617cd`

This document is frozen before any GYRE gradient-twin model execution or
measurement. Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradient_twin/`. The experiment
uses no NEMO run and changes no model/config default.

## Fixed experiment

The model is the certified `GYRE-zco` NEMO-identity card built by
`legoesm.ocean.fidelity.nemo_testcase_recipe`, with the initial state and
time-dependent forcing used by
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_year_fromrest.py`.
The timestep is 14,400 s. All execution is CPU, JAX fp64, and the model precision
policy is fp64/libm.

Truth is one noise-free run at the card values:

* `TKEConfig.c_k = 0.1`;
* `LateralViscosityConfig.A_h = 1.0e5 m2 s-1`.

The recovered value starts at twice truth and is represented as `log(p)`, so
positivity is structural. There are three arms in this order: `A_h`, `c_k`, and
`both`. Each arm uses the identical initial state, forcing, timestep, objective,
observation days, and optimization protocol; only its trainable leaf set differs.

The observations are the full 3-D `T`, `S`, `u`, and `v` fields after steps 30
and 60 (days 5 and 10). At each observation time and for each field, the score is
the wet-masked, live-thickness volume-weighted MSE divided by that truth field's
volume-weighted variance. The loss is the arithmetic mean of the eight normalized
scores (four fields at two times). T/S use live T-cell volume; u/v use their live
face-control volumes and native wet masks. The shared `legoesm.ml.loss` weighted
MSE implementation is the only squared-error reducer.

Parameter leaves are spliced with
`legoesm.core.param_overrides.apply_param_overrides` inside the loss. The 60-step
rollout is one `jax.lax.scan`; every step is wrapped in `jax.checkpoint`; the
complete value-and-gradient function is jitted once. Every selected leaf must
have a finite, exactly nonzero start gradient or the arm aborts as inert.

## Adjoint-first decision

Before any optimizer update, each selected coordinate is checked at `log(2 p*)`
against two central finite differences:

`[L(x+h)-L(x-h)]/(2h)`, with `h = 1e-3` and `1e-4` in log-parameter space.

For each coordinate and each step size, the recorded ratio is
`jax.grad / finite_difference`. A finite ratio in `[0.9999, 1.0001]` CONFIRMS
the reverse-mode adjoint. Any non-finite value, zero start gradient, or ratio
outside that closed interval REFUTES it. A refutation stops the affected arm
before optimization; the first deviating reverse-mode operation is then
localized and recorded. The known absence of forward mode through the
custom-VJP Thomas solver is not a refutation or a substitute for this check.

## Optimizer and recovery verdict

The initial optimizer is the shared
`legoesm.ml.training.create_optimizer` with `TrainingConfig`: Adam, peak learning
rate `0.1`, five warmup steps, cosine decay over the requested number of updates,
global-norm clip `1.0`, and zero weight decay. At most 100 updates are permitted.
No update occurs unless both finite-difference resolutions pass for every leaf in
the arm.

An arm succeeds only when, on the same recorded step, every selected parameter
has `abs(p-p*)/p* < 1e-2` and the loss is below `1e-6` times its step-0 value.
Every update records loss, parameter value(s), wall time, and process peak RSS.
If a finite-difference-confirmed arm does not succeed, an 11-point one-dimensional
scan in log parameter over `[log(0.25 p*), log(4 p*)]` is run for each failed
coordinate with the other coordinate fixed at truth. The optimizer may be tuned
once after that scan; the changed setting and both attempts must be reported.

## Frozen predictions P1–P5

### P1 — reverse-mode finite differences

Prediction: for both `A_h` and `c_k`, at relative log steps `1e-3` and `1e-4`,
`jax.grad / central_FD` lies in `[0.9999, 1.0001]`. This must hold in the
single-parameter arms and coordinate-wise in the joint arm. Inside confirms;
outside refutes and stops optimization.

### P2 — lateral viscosity recovery

Prediction: the `A_h`-only arm reaches `abs(A_h-1.0e5)/1.0e5 < 1e-2` in at most
100 updates and its loss at that same update is below `1e-6` of its initial
loss. Both clauses confirm; either missing refutes.

### P3 — TKE coefficient recovery

Prediction: the `c_k`-only arm reaches `abs(c_k-0.1)/0.1 < 1e-2` in at most 100
updates and its loss at that same update is below `1e-6` of its initial loss.
It is predicted to converge more slowly than `A_h`. Both clauses confirm;
either missing refutes. The joint arm is reported separately and succeeds only
if both P2/P3 parameter tolerances and the same loss-reduction clause hold.

### P4 — JIT parity

Prediction: the eager and jitted start loss agree to relative error at most
`1e-12`, using `abs(jit-eager)/abs(eager)`; the nonzero start loss is required.
At or below confirms; above, zero, or non-finite refutes and stops the arm.

### P5 — campaign-program identity

Prediction: every float64 bit of the truth run's day-10 3-D temperature field
equals the day-10 snapshot emitted by the existing campaign year harness for
the same card, initial state, forcing, timestep, CPU backend, and fp64/libm
policy. Zero unequal wet or dry cells confirms; any unequal cell refutes the
twin setup and stops all scientific claims.

## Controls, provenance, and reporting

The direct two-step unit test must cover the loss reducer, both override routes,
the non-inert assertion, and both finite-difference resolutions. Its non-vacuity
control severs each selected leaf from the forward path in turn; each severed
case must fail the non-inert/FD gate rather than merely change a number.

JSON summaries record the git SHA, dirty state, backend, dtype/policy, CLI,
resolved card values, observation steps, loss definition, optimizer settings,
FD values/ratios, full recovery curve, per-gradient wall time, and peak RSS.
Only those summaries are committed; state snapshots remain in the evidence
root. The receipt labels P1–P5 individually as CONFIRMED, REFUTED, or
UNMEASURED and carries an OPEN section. No result is claimed from a first output
until the direct controls and campaign-identity check pass.
