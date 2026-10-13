# SWG online-learning design

Date: 2026-10-13

Status: round-1 data/model interface frozen; neural implementation begins in
round 2. The learned term closes only the 6.25-to-50 km resolution gap. It does
not compensate a legoESM/NEMO mismatch.

## Resolved model and forcing seam

The coarse deck is the parameterized NEMO-faithful SWG card with `nn_SWG=2`,
50 km spacing, `rn_Dt=400 s`, and all fine-deck physics unchanged. The fine
and coarse gravity-wave Courant number is
`sqrt(9.80665 * 500) * dt / dx = 0.56019` per axis. NEMO's executed SWE RK3
uses `rn_Dt/3`, `rn_Dt/2`, and `rn_Dt` and the live pressure-gradient
acceleration `-grav * delta(ssh) / e1,e2`
(`SWG_DG/BLD/ppsrc/nemo/stprk3.f90:140-171`,
`SWG_DG/BLD/ppsrc/nemo/stprk3.f90:196-227`, and
`SWG_DG/BLD/ppsrc/nemo/stprk3.f90:252-290`),
so scaling `dt` with `dx` preserves its explicit gravity-wave stability
number. NEMO's velocity-Courant diagnostic independently defines
`Cu=abs(u)*rDt/e1u`, `Cv=abs(v)*rDt/e2v`
(`SWG_DG/BLD/ppsrc/nemo/diacfl.f90:74-97`).

Round 2 will add an optional pure-pytree momentum-acceleration argument to the
certified `swe_rk3_step`/stage path. Native U acceleration is added to the U
RHS and native V acceleration to the V RHS at every RK stage, after the
resolved physical terms and before the velocity update. `None` selects the
current expression without adding or reassociating any operation; the
unchanged-card digest and complete NEMO ladder must therefore remain
bit-identical. The closure is passed as an explicit per-window argument, not
captured in a JIT closure, following the repository's `SegmentForcing`
doctrine. The steady analytic wind remains a separate explicit forcing input.

## Closure pytree

The deterministic closure is an Equinox module. Inputs are coarse native
T-point SSH, U-face velocity, V-face velocity, their wet masks, fixed metric
coordinates, and an optional fixed-shape noise field. Outputs are U- and
V-face accelerations in m s-2, multiplied by their native masks. Deterministic
training supplies a zero noise field; the same call signature admits sampled
noise in phase 6 without changing the solver seam. Static masks/metrics are not
trainable leaves. Every trainable leaf must have a finite, nonzero loss
gradient under `legoesm.training.inert_params.assert_no_inert_over`; an inert
leaf aborts before optimization.

## Swappable objective and curriculum

The deterministic objective is the equal mean of three variance-normalized,
native-stagger, wet-area MSEs for SSH, U, and V. Targets are filtered daily
snapshots; the loss scores every daily endpoint within the trajectory window,
including the final endpoint. Normalization variances come only from the
training split and are frozen in the run manifest. The loss is a callable
argument, so phase 6 can replace it with an ensemble energy score without
changing closure or rollout APIs.

The fixed curriculum is 1, 2, 4, 8, 16, 32, and 60 days. Each stage begins
from the preceding stage's selected checkpoint; validation selects by the same
trajectory loss on held-out start dates. Optimizer construction must use
`legoesm.ml.training.create_optimizer`; learning rate, warmup, batch/start-date
sampling, and split dates remain round-2 preregistration items rather than
hidden defaults.

Uniform per-production-step checkpointing is the default: the gradient gate
measured a 1440-step GPU reverse pass at 2.45 s and about 0.5 GB, versus 4.2 s
/ 15.5 GB without checkpointing and 6.2 s / 0.37 GB binomial. The training
rollout must preserve the production step boundary used by that gate and may
not substitute the last-bit-different compiled-day scan.

## Pre-registered phase-5 gates

Each trained window gets a 20-year free run from the same initial state and
steady wind. Stability requires every saved field finite, positive total layer
thickness, and native advective `Cu,Cv < 1` at every daily save. A failed run is
reported as failed, never clipped, damped, or shortened.

Climate scoring uses identical windows and the round-1 instrument. For mean
transport, the learned run must lie within the closed interval joining the
no-closure result and filtered truth. For the basin kinetic-energy spectrum,
its area under absolute log-spectrum error over mutually resolved nonzero
wavenumbers must be smaller than the no-closure error. EKE maps, basin EKE,
jet position, time-mean SSH, learned energy forcing, offline-versus-online
skill, and the rotated/stronger/mirrored cases are reported without moving
these two preregistered gates.

## Stochastic hooks

The closure call always accepts noise, the rollout accepts a per-step/per-member
noise pytree as an explicit argument, and the objective interface accepts one
or more trajectories. No deterministic parameter is duplicated for the
stochastic branch. Phase 6 will preregister the noise law, ensemble size, and
energy-score weighting before any stochastic training.
