# NEMO testcase lane 1 — phase-2 preregistration

Status: **PREREGISTERED BEFORE ANY LEGOESM/ORACLE COMPARISON.**  Phase 2 is
limited to legoESM card construction, exhaustive geometry, initial condition,
and the `kt=1` RK3 step-entry state.  It makes no trajectory claim.

## Pre-implementation search: searched and found

The required search covered `packages/ocean/legoesm/ocean`,
`tests/ocean`, `scripts/validate/ocean_fidelity`, and the phase-1 documents for
`lock_exchange`, `overflow`, `gravity current`, `x-z`, `partial cell`, `zps`,
`recipe`, `NEMO`, `DINO`, `Kamm`, `bridge`, and `time level`.

Found and dispositioned:

* `ocean/experiments/{lock_exchange,overflow}.py` are reusable conceptual
  experiments, but their global/latitude-based domains do not reproduce the
  certified Cartesian three-row meshes.
* `ocean/fidelity/veros_configs/{lock_exchange,overflow}.py` are Cartesian
  precedents, but their dimensions, resolutions, bathymetry and ICs are Veros
  cases rather than the pinned NEMO cases.
* `ocean/recipes.py` and `ocean/fidelity/nemo_recipe.py` already provide the
  pure-config NEMO scheme bundle and shared `NEMORecipe` container.
* `create_beta_plane_cgrid_geometry`, `create_partial_cell_coordinate`,
  `create_full_step_coordinate`, `min_cell_to_uface`, and
  `min_cell_to_vface` are the canonical Cartesian/vertical/face operators.
* `nemo_io.py`, `nemo_state_bridge.py`, and `time_levels.py` provide the
  established mesh, C-grid staggering, and fail-closed time-label machinery.
* `dino_1226/{kamm_twin_90d,nemo_geometry_gate,fidelity_bar_gate}.py` provide
  the fp64 harness, coverage-first gate, and arithmetic-class bars.

Therefore the implementation is a pure test-case recipe over canonical blocks
plus comparison glue under `ocean/fidelity/` and
`scripts/validate/ocean_fidelity/testcases/`.  No bespoke solver is authorized.

## Exact cards

Both cards call `set_policy(PrecisionPolicy.fp64())` before allocating an
array, print every compared array dtype, and hard-fail unless all are
`float64` (integer masks/indices excepted).  The shared numerical selections
are `eos="veros_gsw"`, `tracer_advection="fct2"`,
`momentum_advection="flux_form"`, `momentum_flux_scheme="upwind3"`,
`momentum_time_integrator="rk3_ws"`,
`adaptive_implicit_vertadv=True`, and `implicit_vertical_mixing=True`.
These resolve the certified TEOS-10/FCT2/`ln_dynadv_up3`/`key_RK3`/
`ln_zad_Aimp` arms.  Unexercised trajectory physics remains unclaimed.

| card | pinned domain and IC |
|---|---|
| `LOCK_EXCHANGE-zco` | closed `130 x 3`, `dx=dy=500 m`, no rotation; 20 active 1 m layers plus NEMO's permanently dry `jpk=21` record; flat 20 m bathymetry; rest, `S=35`, `T=5 C` where `glamt<=32 km`, otherwise `30 C` |
| `OVERFLOW-zps` | closed `202 x 3`, `dx=dy=1000 m`, no rotation; 100 active 20 m reference layers plus dry `jpk=101`; `H=500+750[1+tanh((glamt-40)/7)] m`; rest, `S=35`, `T=10 C` where `glamt<=20 km`, otherwise `20 C` |

The NEMO horizontal definitions are
`tests/{LOCK_EXCHANGE,OVERFLOW}/MY_SRC/usrdef_hgr.F90:71-104`; dimensions are
`usrdef_nam.F90:71-76`; ICs are `usrdef_istate.F90:65-75`.  OVERFLOW active
zps T-thickness construction is `usrdef_zgr.F90:157-186`.  The general NEMO
partial-cell face rule is the neighboring minimum at
`tools/DOMAINcfg/src/domzgr.F90:1163-1169`; those U/V comparisons are the
D2.5 minimum-rule payoff, not an averaged-face surrogate.

## Time-level registry

`oracle_step_entry_kt00000001.bin` is registered as **before**.  The write-only
instrument executes at `tests/*_OMIP_L1/MY_SRC/stprk3.F90:88-100` and writes
`ts/uu/vv/ssh(...,Nbb)` before forcing, physics, `stp_2D`, or RK stages.  The
same source calls stage 1/2/3 at `stprk3.F90:202-225`.  Consequently this
dispatch compares the native legoESM IC to NEMO's first-step **entry**, not to
an after-step field.  Relabelling it now/after is a hard failure.

## Gate and preregistered outcomes

The DINO arithmetic classes are immutable: pointwise geometry, masks, IC and
step-entry fields use normalized error `<=1e-15`; only an explicitly
accumulating statistic may use `<=1e-12`.  Exact integer/mask rows require
identity.  Normalization is `max(max(abs(oracle)), 1)` for dimensional fields;
zero oracle fields therefore cannot make the check vacuous.

Every oracle mesh variable must occur exactly once in the phase-2 registry as
`VERIFIED`, `WAIVED`, or `UNMEASURED`, with a nonempty reason.  A file-side
planted unknown variable, a geometry perturbation, and a wet-cell IC gross
excursion must each turn the gate red.  Missing/extra registry entries, dtype
drift, empty comparison masks, nonfinite values, shape mismatch, or a measured
row outside its bar also fail.

Preregistered classifications:

1. **CONFIRM geometry** iff all registered horizontal metrics/coordinates,
   Coriolis, vertical ladders, wet masks, bathymetry, T thicknesses and face
   minimums clear their bars after explicitly disposing NEMO's dummy bottom
   record.  Otherwise **REFUTE** and do not interpret ICs.
2. **CONFIRM IC** iff wet T/S, all native C-grid u/v values and SSH match the
   source-built legoESM IC at the pointwise bar.  Otherwise **REFUTE**.
3. **CONFIRM kt=1 entry** iff the registered before-level dump header is fp64,
   halo removal yields the exact global mesh, and all five fields match the
   same source-built IC at the pointwise bar.  Otherwise **REFUTE**.

No `kt>1` entry, tendency, density/rab/BN2, adaptive-vertical-advection term,
BBL transport, FCT tendency, RK stage, or trajectory statistic is measured in
this dispatch.  Those rows must remain loudly **UNMEASURED**.
