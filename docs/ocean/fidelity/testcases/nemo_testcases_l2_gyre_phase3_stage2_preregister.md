# NEMO testcase lane 2 GYRE — stage-2 boundary preregistration

Date: 2026-09-03  
Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`  
Reconciled baseline: `57429ecf5f377ce2bf220f36bc05313cd29e0dfd`  
Bottom-drag preregistration commit: `239511e3216fe69f905e6b1a40156cdc6421d013`

This freezes the next ordered measurement after the bottom-drag landing.  It
is derived again from the NEMO source and the reconciled implementation; no
round-7 WIP receipt or its invalid preregistration SHA is evidence here.

## Re-pinned boundary after the drag landing

The CPU/fp64 production run clears every one of the 800 registered
barotropic-substep rows and clears stage 1 (`u` and `v` each
`2.710505431213761e-19 m/s`).  The first remaining internal momentum debt is
therefore stage 2: `u=1.057858031242796e-5 m/s` and
`v=1.0572800989805292e-5 m/s`.  Dividing by the stage interval
`rn_Dt/2=7200 s` gives a residual tendency scale of about `1.47e-9 m/s2`.
The whole-step first debt remains kt=2; SSH alone now clears kt=2 at
`6.5052130349130266e-18 m`.

## Source order and first operands

GYRE uses the module-default hybrid external-mode update (`n_baro_upd=np_HYB`;
`stprk3_stg.F90:40-44`).  Stage 2 sets `Kbb=N`, `Kmm=N+1/3`,
`Kaa=N+1/2`, and `rDt=rn_Dt/2` (`:174-215`).  It then constructs the Kmm
transport (`:257-304`), and accumulates the momentum RHS in the fixed order
EOS/HPG, vorticity, advection (`:317-336`) before restarting the velocity from
Kbb and applying the stage interval (`:360-388`).  Finally it replaces the
reference-thickness depth mean by the hybrid barotropic Kaa target
(`:433-446`).

The critical interleave is source-mandated: after each momentum Kaa update,
NEMO advances tracers on the same Kmm transport (`stprk3_stg.F90:452-521`).
Stages 1 and 2 start their tracer RHS at zero, apply advection and
`tra_sbc_RK3`, and restart from Kbb with the QCO weights (`:508-565`).  For a
nonlinear free surface, `tra_sbc_RK3` adds the water-exchange heat and salt
terms at stages 1 and 2 using the Kbb surface tracer and Kmm top-cell
thickness (`trasbc.F90:224-292`).  The stage-2 EOS/HPG therefore consumes the
stage-1 T/S/SSH bundle, not a full post-physics Euler tracer state.

The reconciled legoESM identity already interleaves tracer and momentum
stages, but `_stage_tracers` currently seeds its helper with
`state_new.T/S` (the full pre-applied physics Euler state) and supplies no
stage-local `tra_sbc_RK3` source.  GYRE has live TKE/EVD, lateral tracer
diffusion, penetrative shortwave, and seasonal EMP, so the old comment that
this ordering is inert does not apply to this case.

## Frozen measurements and decisions

The gate will first score, in source order:

1. stage-2 Kbb velocity, stage-1 Kmm velocity, and hybrid barotropic target;
2. the stage-1 T, S, and SSH values actually handed to stage-2 EOS/HPG;
3. the NEMO source-ordered stage-2 HPG, vorticity, and advection increments;
4. the corrected stage-2 Kaa velocity.

The first over-bar row in that order owns the next boundary.  Raw Kaa and its
barotropic correction are gauge-dependent because legoESM removes the depth
mean before integration while NEMO removes it afterward; only the
baroclinic RHS and corrected sum are owner-capable.

A private one-variable arm may replace only the stage-2 EOS/HPG
thermodynamic bundle with the oracle stage-1 T/S/SSH.  It is
`CONFIRMED_CAUSAL_OWNER` only if the direct HPG rows and corrected stage-2
rows all clear `1e-15`.  Movement at least 0.9 of the faithful stage-2
residual without clearance is `CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER`; movement
below 0.1 residual is `NEAR_NULL_NO_DISCRIMINATING_POWER`.  Scaling is printed
before any label.  If this bundle is causal, the production fix must fold the
source-exact tracer interleave into the single WS-RK3 identity, not expose a
public mix-and-match switch.

## Controls and honesty

The stage-2 operand and term dumps are accepted only with their exact headers,
payload sizes, fp64 dtype, finite owned values, and WRITE-only MY_SRC
placement.  Their uninstrumented state output must retain its pinned SHA256.
A planted nonzero stage-2 HPG/RHS violation must become DEBT at the registered
row.  Internal tracer-stage agreement and all downstream owner labels remain
UNMEASURED until these checks actually run.  No external adversarial review
has occurred for this round.
