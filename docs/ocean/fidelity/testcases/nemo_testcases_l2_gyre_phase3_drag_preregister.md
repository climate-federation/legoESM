# NEMO testcase lane 2 GYRE — bottom-drag boundary preregistration

Date: 2026-09-03  
Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`  
Reconciled baseline: `57429ecf5f377ce2bf220f36bc05313cd29e0dfd`  
Baseline gate SHA256: `f467f08979ee0215d3ca5ed18ff8a6bb92bebbb8923f38be0e36b8fe274ddb`

This freezes the next measurement before the GYRE card, private control, or
gate is changed.  The discarded WIP branch and its absent ENE-coefficient dump
are not evidence for this round.

## Re-pinned boundary

CPU/fp64 on the reconciled tree keeps every kt=1 step-entry field bit-exact;
u, v, and SSH are UNINFORMATIVE because the initial state is at rest.  All five
fields first exceed the `1e-15` whole-step bar at kt=2.  The internal momentum
stage register first exceeds it at stage 1 (`u=3.501175472095776e-8 m/s`,
`v=2.951292088959525e-8 m/s`).  Inside the external-mode loop, all substep-1
frames and all substep-2 frames through PGF are AT-BAR.  The first remaining
boundary is substep-2 `trd_u`, `6.731208045740773e-14 m/s2` absolute against
oracle magnitude `5.409032045634095e-10 m/s2`; `trd_v` differs by
`6.755504590908956e-14 m/s2`.

## Oracle source and resolved program

The runtime namelist resolves `ln_non_lin=T`, `ln_drgimp=T`
(`output.namelist.dyn:328-334`) and `rn_Cd0=1e-3`, `rn_ke0=2.5e-3`,
`rn_Cdmax=0.1`, `rn_z0=0.003` (`:336-343`).  `rn_Uc0=0.4` is present but
inactive because the nonlinear arm is selected.

NEMO builds `rCdU_bot=-Cd0*sqrt(0.25*(zut^2+zvt^2)+ke0)` from the bottom
Kmm velocity at T points (`zdfdrg.F90:138-190`).  `dyn_drg_init` freezes its
two-face average once for the whole external-mode window and adds the Kmm
bottom baroclinic residual to the slow RHS (`dynspg_ts.F90:1584-1643`).  In
the substep loop, the no-wetting/drying arm adds
`zCdU_u*un_e/hur_e` and `zCdU_v*vn_e/hvr_e` to the live Coriolis tendency,
using the substep-entry velocity and depth (`dynspg_ts.F90:699-705`), before
the velocity update (`:719-732`); the histories rotate at `:804-816`.

The same stored Kmm coefficient enters NEMO's implicit 3-D momentum solve.
With split-explicit free surface, `dyn_zdf` first removes the Kaa barotropic
velocity, adds its bottom stress at the bottom cell (`dynzdf.F90:148-160`),
and adds the coefficient to the implicit diagonal (`:293-305`, with the v
sibling at `:466-479`).

## Pre-implementation search

The DINO campaign already provides the complete canonical transcription:
`nemo_effective_bottom_drag_r` / `nemo_bottom_drag_rate_faces`,
`zdf_drag_in_matrix`, `zdf_baroclinic_only`, and
`barotropic_drag_substep`, including Kmm-rate threading, the bottom-cell
implicit diagonal, baroclinic residual correction, and explicit substep drag.
The GYRE card selects `bottom_drag_scheme="nemo_quadratic"` but currently
leaves those three composition flags false.  No new drag operator or public
scheme switch is authorized.  The resolved NEMO identity will select the
existing three-part composition as one inseparable program; a private
`_NEMOWSRK3TestHooks` ablation will restore the omitted-drag baseline for the
one-variable control.  The isomorphism tripwire must retain the existing
single NEMO routine owners.

## Frozen scaling and decisions

At the at-rest kt=1 Kmm state, NEMO's frozen nonlinear rate is exactly
`Cd0*sqrt(ke0)=1e-3*0.05=5e-5 m/s` on wet T points.  Therefore:

- substep 1 drag must be exactly zero because its entry velocity is zero;
- substep 2 drag must scale as `-5e-5 * U_entry/H_u` and
  `-5e-5 * V_entry/H_v`, approximately `6.7e-14 m/s2`, the measured combined
  tendency remainder;
- the drag operand is CONFIRMED only if the production identity clears the
  substep-2 combined `trd_u/v` rows at `1e-15`, and the private omission arm
  reproduces the baseline rows while changing no other operator;
- a residual-scale movement without clearance is
  `CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER`; a movement below 0.1 residual is
  `NEAR-NULL_NO_DISCRIMINATING_POWER`, never exoneration.

After any landing, kt=1 and kt=2..10 are rerun.  The walk then advances to the
next first-over-bar registered boundary.  HPG operands are revisited only if
they become that boundary; the prior `6.7e-15` movement against a
`4.92e-13` term remains a refuted arm, not a generic HPG exoneration.

## Controls and honesty

The existing planted state, registry, coverage, arm-manifest, barotropic, and
bit-identity controls remain mandatory.  The new drag control must perturb a
nonzero rate and fail the drag row.  Direct ENE coefficients and metric
transports remain UNMEASURED because the current oracle artifact set does not
contain them.  No external adversarial review has occurred for this round.
