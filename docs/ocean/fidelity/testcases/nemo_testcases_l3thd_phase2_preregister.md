# NEMO testcase fidelity preregistration — lane 3b, SI3 thermodynamics phase 2

Tracker: `climate-federation/legoESM#1699`

## Claim and immutable inputs

This phase measures only the existing legoESM single-column ice path against
the accepted NEMO 5.0.2 `C1D_OMIP_L3/EXP_SASICE` oracle.  It makes no coupled
ocean or trajectory claim beyond the first measured divergent sub-call.  The
oracle root is
`/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_sasice_scope_gate2`;
its Phase-1 receipt identifies the complete 8,760-step run and its planted
controls.  The forcing member is
`ERA5_NorthGreenland_surface_84N_-36E_1h_y2018.nc`, SHA-256
`e5ec49445d2569019c45dec24255b9c7daf050079444b0e6e6d86a5b82317afe`.

The comparison runs on CPU after
`set_policy(PrecisionPolicy.fp64())`.  The gate records every oracle and
legoesm dtype and rejects any legoESM floating array that is not `float64`.

## Pre-implementation search

The required search covered every Python file under
`packages/ice/legoesm/ice/`, the Phase-1 gate and frame registry, the dossier
§4 reuse-vs-new table and Appendix B, the shared tridiagonal solver, and the
ocean `ConstantsConfig` pattern.  Results:

- `sea_ice.py:988` and `:1694` contain the active zero-layer thermodynamics;
  `scm.py:77-184` is the already-shipped standalone column driver.  This work
  extends that path and does not create a second ice model.
- `state.py:23-56` has bulk ice/snow/salinity state but no layer enthalpy.
  `snow.py:104,316` supplies reusable combined conductance and snow-ice
  flooding; `brine.py:67` supplies a bulk salt budget but not SI3 option 2.
- `_future/bitz_lipscomb.py:1-19,147-226` is explicitly parked and unwired.
  It has a shared-Thomas-solver ice-only step, U64 conductivity at `:126-134`,
  no snow coupling, and no SI3 `ice_thd_dh`, option-2 salinity, or open-water
  sequence.  It will be promoted and rewired, not duplicated as another model.
- `packages/core/legoesm/timestepping/tridiagonal.py:30-199` is the reusable
  pure-JAX Thomas solve.  `packages/ocean/legoesm/ocean/constants_config.py:
  31-98` is the required named-config precedent; no module monkey-patching is
  permitted.
- Dossier Appendix B was rechecked: legoESM/NEMO differ at least in `c_pi`
  2106/2096.7 J kg-1 K-1, `L_f` 333700/333360.1 J kg-1, ice conductivity
  2.04/2.034396 W m-1 K-1, snow conductivity 0.31/0.5 W m-1 K-1, and lead
  albedo 0.06/0.066.  The NEMO selection must therefore be a first-class
  constants config.

## Constructible identity

The only new selectable thermodynamic identity is the resolved ORCA1 subset:
single-category HFN (`jpl=1`), BL99 with three ice and three snow layers,
Pringle-2007 conductivity, option-2 bulk salinity with drainage and flushing
and `rn_sinew=0.75`, `ln_pnd=.false.`, and `ln_icedA=.false.`.  The selectors
are sourced to `icethd.F90:148-190`, `icethd_zdf_bl99.F90:34-590`,
`icethd_dh.F90:91-475`, `icethd_sal.F90:204-249`, and
`icethd_do.F90:130-307`.  Validation must reject altered layer counts,
conductivity/salinity choices, ponds, lateral melt, or multiple categories;
no unsupported Frankenstein combination may execute.

## Card, state bridge, and time levels

The column card uses the official ERA5 member for provenance and consumes the
oracle exchange stream as prescribed per-step boundary input.  It maps NEMO
category volume to legoESM thickness by `h_i=v_i/a_i`, snow by `h_s=v_s/a_i`,
bulk salinity by `s_i=sv_i/v_i`, and maps NEMO volumetric layer enthalpies
without rescaling.  Geometry/IC checks cover `nlay_i=nlay_s=3` and the case
initial `hti`, `hts`, `ati`, `tsu`, `smi` values before any tendency.

Every thermodynamics frame has the Phase-1 Rule-1d registry:

| id | frame | time level | source |
|---:|---|---|---|
| 0 | ENTRY | now: global pre-thermodynamics | `icethd.F90:112` |
| 1 | POST_ZDF | now: selected-category 1-D | `icethd.F90:151-152` |
| 2 | POST_DH | now: selected-category 1-D | `icethd.F90:154-155` |
| 3 | POST_TEMP1 | now: selected-category 1-D | `icethd.F90:157-158` |
| 4 | POST_SAL | now: selected-category 1-D | `icethd.F90:160-161` |
| 5 | POST_TEMP2 | now: selected-category 1-D | `icethd.F90:163-164` |
| 6 | POST_DO | now: global post-open-water growth, pre-correction | `icethd.F90:189-190` |
| 7 | EXIT | now: global post-correction/LBC | `icethd.F90:221-225` |

## Bars, controls, and predeclared outcomes

For each scalar/vector field at each registered boundary, the score is
`max(abs(legoesm-oracle))/max(1,max(abs(oracle)))`.  The pointwise DINO bar is
`1e-15`; exact integers/selectors/layer counts use equality.  The primary
number is the lexicographically first `(kt, stage, field)` over the bar while
sweeping `kt=1..8760`, stages in the table order, and the Phase-1 write order.
The sweep stops reporting ownership at that row.  Later rows may be printed as
diagnostic debt but cannot support a trajectory claim.

Predeclared hypothesis: geometry, IC, and kt=1 ENTRY are AT-BAR; the first
divergence is kt=1 POST_ZDF because the parked core lacks SI3 snow, P07, and
Picard/dqns linearisation.  This is CONFIRMED only if all prior rows are
AT-BAR and the first over-bar row is POST_ZDF.  It is REFUTED if geometry, IC,
or ENTRY exceeds its bar, or if POST_ZDF is entirely AT-BAR.  Ownership is not
assigned from magnitude alone: the gate first reports dimensional scale,
absolute and normalized error, then reruns private one-variable arms (surface
flux, derivative, snow conductivity, P07 ice conductivity, and layer state).

Three independent planted violations must exit nonzero: a geometry/IC value,
a registered stage value, and an unregistered/invalid SI3 selector.  Direct
tests cover eager, `jax.jit`, and finite reverse-mode gradients of the promoted
kernel.  Any later fix remains inside the declared SI3 identity and is
accepted only when the relevant planted control remains red.

## End-of-task choice register

- ASKED — use only the ORCA1-resolved single-category HFN, BL99 3+3/P07,
  salinity option 2 (`rn_sinew=0.75`), no ponds, and no lateral melt identity.
- ASKED — reuse the existing ice model, parked BL99 work, shared Thomas solver,
  and shared snow-ice flooding implementation; do not construct a second model.
- ASKED — use the pinned ERA5/exchange inputs, one-hour step, CPU, and fp64.
- ASKED — use the `1e-15` DINO bar, required time-level registry, first-
  divergence stopping rule, one-variable private arms, and planted controls.
- ASKED — keep all shipped NEMO files read-only; use copy-only instrumentation.
- ASKED — correct the three review findings before continuing ZDF ownership.
- ASKED — do not push.
- UNASKED — none.
