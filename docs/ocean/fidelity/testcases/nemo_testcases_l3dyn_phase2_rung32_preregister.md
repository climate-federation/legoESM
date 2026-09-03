# SI3 lane 3 — phase 2 rung 3.2 preregistration

Issue: climate-federation/legoESM #1699

Branch: `fidelity/nemo-testcases-l3-si3dyn-codex`

Oracle root: `/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv2d/final`

This file is committed before the legoESM ICE_ADV2D implementation or any
candidate/oracle score.  The shipped NEMO run already contains 485 ordered
`ice_stp` entry frames and its final restart; those files are immutable inputs.

## Fixed card and source program

The card is the shipped ICE_ADV2D case with only the already-recorded ORCA1
deck choices: `jpl=1`, `nlay_i=nlay_s=3`, `nn_icesal=2`, Prather on/UMx off,
thermodynamics off, ponds off, and landfast off.  It retains the shipped
ICE_ADV2D case values: 99 x 99 T cells, `dx=dy=3000 m`,
`dt=1200 s`, 485 steps, and prescribed `u_ice=v_ice=0.5 m s-1`
(`EXPREF/namelist_cfg:17-20,25-37`, `EXPREF/namelist_ice_cfg:32-39`, ORCA1
`namelist_ice_cfg:24-26,75-76,108`).  No default outside this card changes.

Initial thickness, snow, concentration, salinity, and temperature will be
transcribed through the shipped float32 input arithmetic in
`ICE_ADV2D/EXPREF/make_INITICE.py:48-54,82-114`; candidate initial prognostics
will not be loaded from an oracle entry frame.  Geometry will be constructed
from `usrdef_nam.F90:82-87`, `usrdef_hgr.F90:68-105`, and the pinned namelist,
then scored against every phase-1 mesh array.

The two-dimensional transport program is fixed by
`src/ICE/icedyn_adv_pra.F90`:

* `adv_x` is `:499-719`; `adv_y`, including the y limiter and all cross-moment
  updates, is `:722-943`;
* at `icycle=1`, odd `kt` executes x then y (`:253-301`) and even `kt`
  executes y then x (`:303-351`); the parity expression is
  `MOD((kt-1)/nn_fsbc,2)` versus `MOD((jt-1),2)`, with `nn_fsbc=1`;
* the case CFL is 0.2, so `icedyn_adv_pra.F90:116-132` resolves one cycle;
* `Hbig_pra`, `Hsnow_pra`, and `ice_var_zapneg` remain in the executed call
  graph (`:405-421`).  Each will be ported or receive a measured rung-local
  inactive waiver; no assumed waiver is permitted.

All five Prather moments (`sx`, `sy`, `sxx`, `syy`, `sxy`) are prognostic and
restart-carried for every one of the card's 11 packed tracers.  The card restart
must reject a missing, extra, retyped, or perturbed moment and must reproduce a
split continuation bitwise.  The oracle's option-4 layer-salinity moments have
no option-2 candidate equivalent and will remain explicitly UNMEASURED; the
50 common oracle moment endpoints are scored.

## Preregistered measurements and verdicts

All scored arrays are fp64 on CPU after
`set_policy(PrecisionPolicy.fp64())`.  The immutable normalized pointwise bar
is `1e-15`; it is encoded in the gate and is not a judgment call.

The gate will emit exactly these registered groups unless a coverage failure
aborts first:

| group | preregistered rows | AT-BAR condition |
|---|---:|---|
| geometry: 18 coordinates/metrics/Coriolis + 4 masks | 22 | every row `<=1e-15` (masks exact) |
| `kt=1` entry: 11 packed tracers + surface temperature + U/V | 14 | every row `<=1e-15` |
| ordinary trajectory: 13 fields after each of 485 completed steps | 6305 | every row `<=1e-15` |
| common final oracle Prather moments | 50 | every row `<=1e-15` |
| **total** | **6391** | every scored row satisfies its condition |

The first-divergence sweep is lexically fixed by time, then by the card tracer
order, then surface temperature, U, V; final common moments follow in
`(sx,sy,sxx,syy,sxy)` and tracer order.  It reports the first row above the
bar.  Overall status is AT-BAR only if all 6391 rows are AT-BAR; any numeric
row above the bar is DEBT, and any missing comparison is UNMEASURED with a
nonzero exit.

## Coverage and planted violations

The phase-1 mesh/restart/namelist manifests and exact 19-array frame registry
remain discovery-driven inputs.  Every supplied array must be VERIFIED,
WAIVED with a sourced reason, or loudly UNMEASURED.  Separate exact-set checks
cover the card frame boundary and all 55 candidate moment leaves.

Controls are preregistered to go red for: an unaccounted mesh array, an
unaccounted frame array, a geometry perturbation, an ordinary-state
perturbation, reversed `kt` sweep parity, a perturbed `sxy` carry, a missing
moment, a retyped moment, and a selector/clock/velocity change.  The focused
test suite must also JIT the complete card step and return a finite nonzero
reverse-mode gradient.  Both coefficient ratchets are mandatory for every
touched source file.

## Loudly outside this rung

Rungs 3.3 and above; rheology, ridging/rafting, thermodynamics, ponds,
landfast L16, coupled ice--ocean behavior, option-4 salinity evolution and
layer-salinity moments, and production integration of this fidelity-card
restart state are UNMEASURED.  No result from this rung classifies them.
