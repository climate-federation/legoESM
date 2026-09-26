# OVERFLOW-zps stage-composition arms — preregistration

Date: 2026-09-01.  Scope: the certified OVERFLOW-zps NEMO WS-RK3 card,
step 1 stages and the resulting `kt=2` entry.  Drafted by the codex session
before its first Arm-A run; committed unchanged (predictions frozen as
drafted) by the takeover session after that run had already been made and
REFUTED, so the commit does not precede the first measurement.  The
reconciliation of that refutation is recorded in
`nemo_testcases_l1_phase3_receipt.md`, not here.

## Pre-implementation search and source verification

Searched the canonical WS-RK3 implementation and tests for a live-stage EOS/
HPG path and per-stage vertical-UP3 path: `ocean_model_latlon_cgrid.py`,
`ocean_pe_latlon_cgrid.py`, `vertical.py`, and
`tests/ocean/unit/test_nemo_ws_tracer_rk3.py`.  Found the shared NEMO EOS,
`nemo_sco` HPG, FCT tracer stages, and vertical-UP3 kernels; found no existing
interleaved live-stage operand path.  These existing kernels will be reused.
No new public selector is permitted: both changes belong to the unbranched
NEMO WS-RK3 scheme identity, while legacy schemes retain their existing path.

The independent report was read first from
`hunt/overflow-kt2-independent:docs/ocean/fidelity/testcases/overflow_kt2_independent_hunt.md`.
Its two source readings were then checked independently:

* NEMO stage 1 calls flux-form `dyn_adv` at `Kmm`
  (`src/OCE/stprk3_stg.F90:309-315`).  Stages 2 and 3 recompute
  `eos(ts,Kmm)`, call `dyn_hpg(...,Kmm)`, then call flux-form `dyn_adv` with
  the `Kmm` velocity and transports (`:317-334`).  The qco stage update uses
  `Kbb`, `Kmm`, and `Kaa` thickness factors (`:360-388`); barotropic
  correction writes the stage `Kaa` velocity (`:433-446`); only then does the
  tracer call advance the stage tracer with the same transports (`:452-468`).
  Thus stage 2 consumes stage-1 `Kaa` T/S/ssh through the rotated `Kmm`, and
  stage 3 consumes stage-2 `Kaa` T/S/ssh.
* legoESM computes `_geom_density` once from the step-entry state and passes it
  into the initial tendency (`ocean_model_latlon_cgrid.py:3832-3847`).  Its
  `_mom_pert_ws` replaces only u/v and reuses that bundle at every stage
  (`:4158-4183`); `tendencies` also reads eta from that unchanged state before
  accepting the precomputed bundle (`ocean_pe_latlon_cgrid.py:4370-4389`).
  This confirms frozen start-of-step T/S/eta in stages 2 and 3.
* With adaptive-implicit vertical advection active, the canonical tendency
  path deliberately omits explicit vertical momentum advection
  (`ocean_pe_latlon_cgrid.py:2679-2684,2700-2737`).  The WS path instead adds
  one stage-3 vertical increment after the momentum program
  (`ocean_model_latlon_cgrid.py:5027-5064`).  NEMO calls the full `dyn_adv`
  package in every stage at the sites above.  This confirms the second source
  difference.

The report's replay numbers remain **UNMEASURED** until reproduced by the
committed phase-3 gate.  They are used here only to freeze predictions.

## Fixed arms and decision rules

All comparisons use the existing gate's halo stripping, wet U-face
intersection, instantaneous frame, `e3u_0`-weighted baroclinic reduction, and
fp64 policy.  Scaling is printed before any ownership label.

### Arm A — live stage EOS/HPG operands

One variable: interleave tracer and momentum stage construction so momentum
stages 2 and 3 recompute density, pressure anomaly, and eta geometry from the
stage-advanced T/S/eta that represents NEMO `Kmm`.  Momentum advection remains
otherwise unchanged.

* Baseline: `kt=2` baroclinic U L-inf `3.31e-6 m s-1`; stage-2 baroclinic U
  L-inf `7.8e-7 m s-1`.
* **CONFIRM:** `kt=2` baroclinic U falls to `<=7e-7 m s-1` and stage 2 falls
  to approximately `1.65e-7 m s-1` (within a factor of two).
* **REFUTE:** either metric fails to move in the predicted direction by at
  least half the predicted reduction.

### Arm B — vertical UP3 in every momentum-stage RHS

One variable after Arm A: send the existing NEMO vertical-UP3 explicit share
through each WS-RK3 momentum-stage RHS at that stage's `Kmm` velocity and
transport.  Remove the former once-per-step explicit rewrite so the term is
not double counted; keep the existing stage-3 adaptive implicit share in the
implicit solve.

* Arm-A baseline prediction: stage-2 residual approximately
  `1.65e-7 m s-1`, stage-3 residual approximately `5.0e-7 m s-1` before its
  remaining surface-localized term.
* **CONFIRM:** stage-2 residual falls to approximately `1e-10 m s-1` and
  stage-3 falls to approximately `2.6e-7 m s-1` (each within a factor of two,
  with the same registered frame and reduction).
* **REFUTE:** either residual fails to move in the predicted direction by at
  least half the predicted reduction.

Neither arm is allowed an owner label from a source argument alone.  A
CONFIRMED label requires the measured movement above in the committed gate;
otherwise the result is PLAUSIBLE or REFUTED with its actual movement printed.

