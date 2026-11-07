# NEMO testcase lane 1 — phase-2 review correction preregistration

Date: 2026-08-31  
Scope: review findings 1--8 only; no trajectory measurement or claim.

## Read-the-source resolution

The resolved oracle cards select `ln_hpg_sco=T` and `ln_hpg_djc=F`
(`lock_exchange_zco/output.namelist.dyn:430-437`,
`overflow_zps/output.namelist.dyn:431-438`).  NEMO 5.0.2 dispatches those
flags to distinct routines: `src/OCE/DYN/dynhpg.F90:117-123` maps SCO to
`hpg_sco` and DJC to `hpg_djc`, while `:188-197` enforces exactly one choice.
The selected SCO stencil is the surface recurrence and slope correction at
`:340-360` plus the interior recurrence and correction at `:367-390`.

The pre-implementation search found that the shared canonical lat-lon module
already contains a selectable `pgf_scheme="nemo_sco"`, including qco stretch,
`gdept_z0` correction, validation, and direct oracle-transliteration tests.
It is used by DINO but was not selected by these two new cards.  Therefore the
correction will reuse and source-refresh that canonical option, select it on
both cards, and add testcase-card coverage; no fidelity-only solver is needed.
The global default remains `adcroft`.

## Pre-registered gates

1. Both testcase cards must select `nemo_sco` with `nemo_trapezoid`.
2. The existing direct `hpg_sco` test must retain a level-isopycnal qco rest
   control at zero PGF to roundoff and a nonzero synthetic free-surface/step
   violation whose analytic force is detected.  The fully nonuniform-density
   transliteration must agree with the NEMO 5.0.2 recurrence to `rtol=1e-12`,
   `atol=1e-19` on wet faces.
3. A planted replacement of `nemo_sco` by `smc03` in card expectations must
   fail the direct card test.
4. `bottom_index_rule="nemo_tpoint"` must raise if `t_depth_ref` is absent.
   A stretched ladder where the T-point rule and midpoint fallback choose
   different bottom levels is the non-vacuous control.  Both cards must supply
   their explicit NEMO T-point ladder.
5. Every `kt=1` row is scored with exact equality.  Only T is an informative
   alignment measurement.  S, u, v, and SSH remain exact controls but are
   reported as `UNMEASURED` for staggering/alignment because their registered
   fields are spatially uniform or zero.
6. fp64 policy mutation occurs only at the gate/harness entry point.  Calling
   a card builder under fp32 must leave the policy unchanged; the gate must
   construct fp64 arrays after setting fp64.
7. Unknown-case dispatch is checked before invoking a builder.  A monkeypatched
   known builder that raises `KeyError` must propagate that internal error.

## Selector source ledger

| legoESM selector | oracle evidence | preregistered disposition |
|---|---|---|
| `pgf_quadrature=nemo_trapezoid` | `dynhpg.F90:343-380` uses the surface half-weight and interior two-level density sums on `e3w` | VERIFIED as part of `nemo_sco` direct test |
| `barotropic_solver=explicit_substep` | both resolved namelists set `ln_dynspg_ts=T`; `dynspg.F90:208-233` maps that flag to `np_TS` | VERIFIED selector identity only; dynamics UNMEASURED |
| barotropic time filter | LOCK resolves `nn_bt_flt=3`, `rn_bt_alpha=.07` (`:439-447`); OVERFLOW resolves `nn_bt_flt=1`, alpha zero (`:440-448`); meanings are in `dynspg_ts.F90:1068-1094,1265-1273` | WAIVER: the shared card's `cosine` is not either oracle filter; remove the false mapping and record the missing selectable parity as trajectory debt |
| `vertical_momentum_scheme=nemo_advective` | both resolved namelists set `ln_dynadv_up3=T`; `dynadv.F90:78-90` dispatches that active arm to `dyn_adv_up3`, whose vertical flux is at `dynadv_up3.F90:250-358`. `dynzad.F90` is called only by the inactive vector arm (`dynadv.F90:79-83`). `ln_zad_aimp=T` is resolved at `LOCK:280-300`, `OVERFLOW:281-301`. | WAIVER: the legoESM selector transcribes `dynzad`, which is not the oracle's active UP3 vertical stencil; adaptive/UP3 vertical momentum parity remains UNMEASURED |
| tracer time integration | `stprk3.F90:50-64,194-207` advances active tracers and momentum through three RK stages; `stprk3_stg.F90:519,586,598` calls tracer advection/diffusion within a stage | WAIVER: legoESM's inner `tracer_time_integrator=euler` is fidelity plumbing under the outer RK3 momentum card, not a demonstrated NEMO tracer-RK3 match; trajectory row remains UNMEASURED |
| vorticity absence | both `usrdef_hgr.F90:100-104` set `f=0`; resolved `ln_dynadv_up3=T` and `dynadv.F90:78-90` select flux-form advection, while `dynvor.F90:879-913` reduces the flux-form extra to Coriolis plus Cartesian metric terms | VERIFIED absence for these flat Cartesian, nonrotating domains; no independent vorticity selector is required |

## Receipt correction

The phase-2 receipt will distinguish exact byte alignment from informative
coverage, name all selector waivers, and report test counts by file.  The prior
combined count will not be presented as the direct-lane count.
