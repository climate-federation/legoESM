# NEMO testcase fidelity receipt — lane 3b, SI3 thermodynamics phase 2

Tracker: `climate-federation/legoESM#1699`

## Verdict

**DEBT.**  The scope-exact legoESM column is implemented as a selectable path
inside the existing sea-ice module.  Geometry, initial conditions, and every
`kt=1` ENTRY field are at the pre-registered bar.  The first over-bar row is
`kt1.POST_ZDF.t_su`: absolute error `6.51945413210342e-08 K`, oracle scale
`255.4688501439127 K`, normalized error `2.55195658039359e-10` against the
`1e-15` bar.  This confirms the pre-registered first-boundary hypothesis but
does not establish a trajectory result.  No `kt>1` ownership or trajectory
claim is made.  The ordinary CLI therefore exits 1 after writing its complete
JSON result; a scientific debt is not a green gate.

All comparisons ran on CPU after `set_policy(PrecisionPolicy.fp64())`.
Storage, compute, accumulation, and control policy dtypes printed `float64`;
every compared legoESM array printed `float64`.

## Phase-1 citation corrections landed first

Commit `8d3a14c348` makes the five review-only corrections before Phase 2:

- `ln_ECMWF=.false.` is inherited from `SHARED/namelist_ref:236`; ORCA1
  `EXPREF/namelist_cfg:129-142` never sets it explicitly.
- ice-reference citations are `ln_cat_usr:43`, `rn_snwblow:141`,
  `nn_flxdist:143`, and `ln_frazil:191`.

The Phase-2 preregistration is commit `bb227d2b71`.

## Pre-implementation search

Before implementation, the search covered every Python file below
`packages/ice/legoesm/ice/`, dossier §4 and Appendix B, the Phase-1 frame
registry/gate, the shared Thomas solver, and the ocean `ConstantsConfig`
pattern.  Findings, recorded before implementation in the preregistration:

- Active thermodynamics was the zero-layer path in `sea_ice.py`; `scm.py`
  already supplied the standalone column driver.  The SI3 choice therefore
  belongs inside those modules, not in a second ice model.
- `state.py` had bulk thickness, temperature, snow, and salinity but no
  three-layer ice/snow enthalpies or ice age.  `snow.py` and `brine.py` had
  reusable bulk operations but not the ordered SI3 sequence or option-2
  salinity profile.
- `_future/bitz_lipscomb.py` was explicitly parked and unwired.  It had an
  ice-only shared-Thomas solve and Untersteiner conductivity, but no snow
  layers, P07 conductivity, `dqns` Picard iteration, `ice_thd_dh`, option-2
  salinity, or `ice_thd_do`.
- `packages/core/legoesm/timestepping/tridiagonal.py` is the reusable JAX
  Thomas solve.  `packages/ocean/legoesm/ocean/constants_config.py` supplies
  the named-config precedent.

The parked implementation is now promoted to
`packages/ice/legoesm/ice/bitz_lipscomb.py`; its former `_future` path is only
a compatibility import.  The old public ice-only API remains tested.  The
selected production dispatch is `SeaIceConfig.thermo_scheme="si3_bl99"` in
the existing `step_sea_ice`; the private `_si3_step_with_trace` hook exposes
the registered sub-call states to the fidelity instrument.

## Constructible identity and source map

`SI3ThermoConfig` has exactly one executable identity.  Validation rejects
any changed layer count, conductivity, salinity choice, drainage/flushing
choice, ponds, lateral melt, category count, dynamics/transport activation,
or non-NEMO constants set before a tendency can run.

| resolved choice | source |
|---|---|
| one category, HFN | `iceitd.F90:129-180`; accepted `ocean.output:615-627` |
| BL99, 3 ice + 3 snow layers | `icethd.F90:148-152`; `icethd_zdf_bl99.F90:34-590` |
| P07 conductivity | `icethd_zdf_bl99.F90:261-275` |
| option-2 salinity, drainage/flushing, `rn_sinew=.75` | `icethd_sal.F90:204-249`; `icethd_dh.F90:328-362` |
| vertical thickness change/open-water growth on; lateral melt off | `icethd.F90:154-183`; `icethd_dh.F90:91-533`; `icethd_do.F90:130-307` |
| ponds off; `ln_icedA=.false.` | `icethd.F90:161,176-177`; accepted `ocean.output:635-727` |
| aEVP, Prather, ridging/rafting selected but C1D-inert | accepted `ocean.output:820-846,861-862`; `ln_c1d` skips `ice_dyn` at `icestp.F90:167-171` |

The apparent `rn_sinew` inconsistency remains a **FINDING**, not a repair:
ORCA1 resolves `nn_icesal=2` with `rn_sinew=.75` even though its adjacent
comment says `.30` is required for option 2.  The card keeps `.75`.

## Immutable column card and inputs

Card: `C1D_OMIP_L3/EXP_SASICE`, `dt=3600 s`, documented `nsteps=8760`.
Accepted oracle root:
`/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_sasice_scope_gate2`.
The card consumes the oracle exchange-frame stream and the same official ERA5
member; the sweep stops at its first divergent step, so only the `kt=1` inputs
are consumed in this dispatch.

| artifact | digest |
|---|---|
| official `C1D_v5.0.0.tar.gz` | MD5 `9456e6a0a84d40630ad1804fd4061caf`; SHA-256 `54a2ceefd9126e180676e68eaa28ded85cc3b93ea3f0dda0fb964a035a4fc382` |
| `ERA5_NorthGreenland_surface_84N_-36E_1h_y2018.nc` | SHA-256 `e5ec49445d2569019c45dec24255b9c7daf050079444b0e6e6d86a5b82317afe` |
| `oracle_si3_thd_frames.bin` | SHA-256 `7fc9df2707a85581075e3c69b26784155151a55640a5693eb32c34fb710ea49b` |
| `oracle_si3_exchange_frames.bin` | SHA-256 `998f4790a8c832962ed8437b9b48fc514555d55c8770108889b30636fb968f51` |

Archive provenance and URL are the Phase-1 pinned NEMO `sette_inputs` r5.0.0
listing.  No analytic or synthetic forcing is used.

The case does **not** read an `init_*` file: the accepted control print has
`nn_iceini_file=0`.  The actual resolved analytic initial values printed in
`ocean.output:747-759` and verified exactly by the gate are
`nlay_i=nlay_s=3`, `hti=2`, `hts=.2`, `ati=.9`, `tsu=270`, and `smi=6.3`.
The bridge uses `h_i=v_i/a_i`, `h_s=v_s/a_i`, `S_i=sv_i/v_i`, and converts
NEMO layer energy content to volumetric enthalpy using the category volume.

## Appendix-B constants recheck

NEMO values were re-read from `phycst.F90:39,57-66`,
`eosbn2.F90:1898-1899`, and the accepted run's actual `ocean.output:138-149,
615-727`.  They are selected through `IceConstantsConfig`; no global constant
is mutated.

| quantity | legoESM default | selected NEMO value |
|---|---:|---:|
| ice heat capacity [J kg-1 K-1] | 2106 | 2096.7 |
| fusion latent heat [J kg-1] | 333700 | 333360.1 |
| sublimation latent heat [J kg-1] | 2834000 | 2834400 |
| pure-ice conductivity [W m-1 K-1] | 2.04 | 2.034396 |
| snow conductivity [W m-1 K-1] | .31 | .5 |
| ice density [kg m-3] | 917 | 917 |
| snow density [kg m-3] | 330 | 330 |
| ocean reference density [kg m-3] | 1025 | 1026 |
| ocean heat capacity [J kg-1 K-1] | 3994 | 3991.86795711963 |
| liquidus slope [K PSU-1] | .054 | .054 |
| lead albedo | .06 | .066 |

## Rule-1d frame registry

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

SI3 has no leapfrog temperature level inside this chain.  ENTRY is global;
stages 1-5 are selected-category one-dimensional arrays; POST_DO and EXIT are
global again.  The gate fails closed on magic, version, step, stage, shape,
layer counts, bit width, payload size, non-finite values, hashes, and dtype.

## `kt=1` gate and first-divergence ownership

Metric: `max(abs(legoesm-oracle))/max(1,max(abs(oracle)))`; bar `1e-15`.
Exact geometry/count checks use equality.  All 78 field rows are emitted by
`nemo_si3thd_phase2_gate.py`; the stage summary is:

| boundary | rows | over-bar fields | largest normalized error |
|---|---:|---|---:|
| geometry/IC | 7 | none | 0 |
| ENTRY | 12 | none | `1.7782396272275097e-16` (`e_i`) |
| POST_ZDF | 7 | `t_su,e_i,e_s` | `2.55195658039359e-10` (`t_su`) |
| POST_DH | 7 | `t_su,e_i,e_s` | `2.55195658039359e-10` (`t_su`) |
| POST_TEMP1 | 7 | `t_su,e_i,e_s` | `2.55195658039359e-10` (`t_su`) |
| POST_SAL | 7 | `t_su,e_i,e_s` | `2.55195658039359e-10` (`t_su`) |
| POST_TEMP2 | 7 | `t_su,e_i,e_s` | `2.55195658039359e-10` (`t_su`) |
| POST_DO | 12 | `t_su,e_i,e_s` | `2.55195658039359e-10` (`t_su`) |
| EXIT | 12 | `t_su,e_i,e_s` | `2.55195658039359e-10` (`t_su`) |

Thus the first divergent subsystem owner is the BL99 snow/surface vertical
diffusion solve (`icethd_zdf_bl99.F90:307-590`).  That boundary ownership is
measured; the remaining arithmetic mechanism is unresolved and is not called
physical-model error.  Later at-bar rows are printed diagnostics only and do
not erase the earlier debt.

The private ZDF boundary hook perturbed one input at a time by `1e-6` of its
dimensional scale.  Normalized surface-temperature responses were:

| arm | response |
|---|---:|
| snow layer state | `1.2098898482456677e-07` |
| `qns_ice` | `5.688031903022738e-08` |
| `dqns_ice` | `4.1556596477045266e-08` |
| ice layer state | `4.783916891639648e-10` |
| bottom temperature | `1.1125313088910906e-16` |

These are scaling/sensitivity arms, not causal attribution.  They hold all
other card inputs fixed and do not tune any selector.

## Controls and tests

All three independent CLI plants exited 1:

- `--plant-geometry`: `GateError: geometry/IC gate`;
- `--plant-stage`: `GateError: ENTRY gate`;
- `--plant-selector`: `ValueError` naming the rejected ORCA1 identity.

CPU/fp64 tests: 92 passed across `tests/test_param_specs.py`, the existing
parked-BL99 compatibility tests, the existing ice-column tests, and the new
Phase-2 tests.  The new tests directly exercise eager execution,
`jax.jit`, finite reverse-mode gradient, existing-column dispatch, exact card
hashes/geometry, first-divergence order, and all three red controls.  A further
41 configuration/complexity tests passed.  `compileall`, `git diff --check`,
and Ruff E/F/I checks on all new files passed.

The lane-owned inline-coefficient ratchet nodes pass after moving every fixed
SI3 value to a source-cited module-level oracle block.  The broader five-ratchet
run still has seven pre-existing failures in atmosphere/core/coupler/ocean
files and the `grid_type` baseline, none touched by this lane; they are not
silently reported as a green repository-wide run.

## Review and tree integrity

The required Claude review was attempted read-only twice after the implementation
and tests; its API endpoint failed DNS (`ENOTFOUND`) both times.  The required GLM-5.2
review tool is not available in this environment.  Therefore the Phase-2 diff
is **UNREVIEWED by the mandated external pair**; test/gate results do not waive
that status.

No file in `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2` or any shipped NEMO
configuration was modified or deleted.  No stale item was removed.  There is
nothing newly identified for `FLAGGED FOR FUTURE DELETION`; the `_future`
module remains as a compatibility shim so existing imports are not broken.
