# Preregistration — certified GYRE reverse-mode gradient gate, round 1

Date: 2026-10-11

Incoming lane tip: `31bd4575b828195455deedcd5a382da286532424`.

This document is frozen before any GYRE gradient, finite-difference, timing, or
memory measurement. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round1/`.
No NEMO executable will be run and no physics, configuration, stabilizer,
clipping, guard, or `stop_gradient` change is permitted in this lane.

## Fixed program and objective

The program is the certified card returned by
`build_nemo_testcase_card("GYRE-zco")`, starting from the card's day-0 state.
The instantiated card at the incoming tip resolves to a 22 x 32 x 30 active
cell array (31 W levels), `dt=14400 s`, forward-Euler outer dispatch containing
the NEMO RK3 Wicker-Skamarock identity, 50 split-explicit barotropic substeps,
NEMO AB3/AM4 filtering, NEMO-literal TKE, and Redi diffusivity 1000 m2/s. State
and vertical geometry both print as float64 after selecting the fp64/libm
precision policy. The production call is
`LatLonCGridOceanModel._step_jitted` at
`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:13209-13361`.
Forcing is the existing GYRE campaign forcing builder used by
`nemo_testcase_l2_gyre_year_fromrest.py`; it is not reimplemented here.

The differentiated input is an additive temperature perturbation on the wet
day-0 T cells. A fixed PRNG-0 unit vector times `1e-3 K` is the evaluation
point; an independent fixed PRNG-42 wet-cell unit vector is the tested
direction. The scalar after-state objective reuses the gradient-twin
four-field definition from `origin/feat/gyre-gradient-twin-2` commit
`901cfe0d8`: the arithmetic mean of live-volume-weighted MSE for T, S, u, and v
against the unperturbed after-state, each divided by that truth field's
live-volume-weighted variance. One-step and six-step (one model day) programs
use the campaign's Python per-step composition, not `lax.scan`; this retains
decision 49's compilation boundary and avoids the branch's measured scan versus
per-step last-bit gap. Every individual step enters through `_step_jitted`.

## Registered predictions and falsifiers

The agreement bars are the existing matrix bars: relative reverse-versus-
forward directional error at most `1e-9`, and relative reverse-versus-central-
finite-difference error at most `0.05` with `h=1e-3` in the unit direction.
Both one-step and six-step rows must also have finite nonzero reverse, forward,
and finite-difference derivatives. Any non-finite or zero derivative, any
relative error above its bar, a forward-mode trace failure, a non-fp64 state or
geometry operand, or a call path bypassing `_step_jitted` REFUTES the row.

Prediction: both rows are **REFUTED by a present reverse-mode NaN**. The lane
tip evaluates NEMO's live `zsqen=SQRT(en); dissl=zsqen/zmxld` statement
(`cfgs/DINO/MY_SRC/zdftke.F90:832-837`) as the unguarded
`jnp.sqrt(tke_curr) / l_eps_final` at
`packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:3512-3516`.
The earlier twin localized a zero-TKE transpose NaN there and decision 48's
primal-bit-identical double-`where` guard only exists on
`origin/feat/gyre-gradient-twin-2` (`e7f3797de`); that commit is not an ancestor
of this lane tip. A finite reverse derivative in both rows, with both registered
agreement bars passing, falsifies the predicted NaN. A NaN whose first owner is
not the cited dissipation statement also falsifies that localization and must
be recorded rather than reassigned by inference.

The deliberately broken-VJP plant doubles the cotangent of an otherwise
primal-identity input. The unplanted synthetic control must pass the same
reverse/forward classifier and the planted control must exit nonzero. Failure
of either behavior invalidates the instrument. Because the real card is
predicted already red, a plant that merely leaves that row red is not accepted
as non-vacuity evidence.

## Memory and timing protocol

For `none`, `uniform`, and `binomial`, one-step and six-step reverse passes are
measured in separate processes after one warm reverse call. CPU rows use host
RSS sampled during the warm reverse execution; GPU rows use physical GPU 1
only and sample `nvidia-smi` memory during that same interval. Wall time is the
blocked warm reverse execution, excluding the first compile. The script stamps
commit, dirty state, backend, device, x64 flag, precision policy, schedule,
step count, loss definition, and sampled baseline/peak. A non-clean tracked
tree, wrong commit stamp, wrong backend/device, unavailable sampler, non-fp64
operand, or schedule primal differing bitwise from the per-step production
program makes the row UNMEASURED and exits nonzero.

The 360-step estimates are declared before measurement. Let `B` be sampled
memory immediately before reverse and `P6` the six-step sampled peak. For
`none`, estimate `B + 60*(P6-B)` (linear tape). For `uniform`, estimate `P6`
(constant extra tape). For `binomial`, whose checkpoint budget is
`ceil(sqrt(N))`, estimate
`B + (P6-B)*ceil(sqrt(360))/ceil(sqrt(6)) = B + (P6-B)*19/3`.
These are schedule-complexity extrapolations, not allocations or successful
60-day runs. A non-finite adjoint does not erase measured resource use, but it
forbids calling any schedule trainable or usable for online learning.

## Required verdict

No gradient row may be called passing until the plant has demonstrated a
nonzero exit and both one-step and six-step rows clear both numerical bars.
The predicted NaN is reported with `DECISION_NEEDED`; it is not repaired in
this round. Failed predictions remain in the round receipt.
