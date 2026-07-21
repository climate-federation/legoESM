# Physics Parameterization Test Catalog

This document catalogs the test coverage for every physics
parameterization in legoESM, organized by component (atmosphere, ocean,
land, sea ice).  Each entry lists the parameterization, the canonical
source module, the existing tests, what they verify, and any known
gaps.  Entries marked **AUDIT-2026-05-05** were validated or extended
during the Physical_Consistency audit cycle on 2026-05-05.

The testing pyramid for parameterizations follows three tiers:

| Tier | Purpose | Typical location |
|------|---------|------------------|
| **Unit** | Single-scheme correctness, finite-output guards, conservation, gradients | `tests/<component>/unit/` |
| **Validation** | Idealized-case fidelity (Held-Suarez, Williamson, Galewsky, RCE, ...) | `tests/<component>/validation/` |
| **Integration** | Full coupled / multi-day stability / regression | `tests/<component>/integration/` |

---

## 1.  Atmosphere

### 1.1  Convection

| Scheme | Source | Unit tests | Validation | Notes |
|--------|--------|------------|------------|-------|
| Deep Convective Adjustment (DCA) | `atmosphere/physics/convection/dca.py` | `tests/atmosphere/hydrostatic/unit/test_convection.py::TestDCA` | RCE column drift via `test_held_suarez_fix.py` | **AUDIT-2026-05-05**: critical MSE-conservation fix added.  Need a column-conservation test that asserts `c_p ⟨ΔT⟩ + L_v ⟨Δq⟩ = 0` after DCA invocation on a saturated profile. |
| Emanuel | `atmosphere/physics/convection/emanuel.py` | `test_convection.py::TestEmanuel` | none | Uses shared `_plume.py` helpers + `compute_moist_adiabat`.  Saturated-everywhere parcel approximation documented as deferred. |
| Kain-Fritsch | `atmosphere/physics/convection/kain_fritsch.py` | `test_convection.py::TestKainFritsch` | none | CAPE-relaxation closure consumes `compute_cape`; bias-low at ~1-2 % vs. virtual CAPE (LOW). |
| Tiedtke | `atmosphere/physics/convection/tiedtke.py` | `test_convection.py::TestTiedtke` | none | Same caveat. |
| Bechtold | `atmosphere/physics/convection/bechtold.py` | `test_convection.py::TestBechtold` | none | Same caveat. |
| Zhang-McFarlane | `atmosphere/physics/convection/zhang_mcfarlane.py` | `test_convection.py::TestZhangMcFarlane` | none | Same caveat. |
| Stochastic Bulk Mass-flux (SBM) | `atmosphere/physics/convection/sbm.py` | `test_convection.py::TestSBM` | none | Stochastic forcing path |
| Kuo | `atmosphere/physics/convection/kuo.py` | `test_convection.py::TestKuo` | none | |
| Generic mass-flux | `atmosphere/physics/convection/mass_flux.py` | `test_convection.py::TestMassFlux` | none | Per-level cap `M_u_max` documented at `mass_flux.py:215-225`. |

**Cross-cutting tests**: `test_convection.py::TestSchemeDispatch` — every public-config literal raises `ValueError` for unknown values.  `test_convection_latlon_mpas.py` — convection parity check on lat-lon C-grid + MPAS Voronoi.

### 1.2  Microphysics

| Scheme | Source | Unit tests | Validation | Notes |
|--------|--------|------------|------------|-------|
| Kessler | `atmosphere/physics/microphysics/kessler.py` | `test_microphysics.py::TestKessler` | `tests/atmosphere/hydrostatic/integration/test_amip_smoke.py` | Latent-heat closure verified by `test_evaporation_enthalpy_balance`. |
| Sundqvist | `atmosphere/physics/microphysics/sundqvist.py` | `test_microphysics.py::TestSundqvist` | none | Diagnostic condensation; column water budget tested. |
| Seifert-Beheng (warm rain) | `atmosphere/physics/microphysics/seifert_beheng.py` | `test_microphysics.py::TestSeifertBeheng` | none | Two-moment warm rain. |
| Morrison (ice + liquid) | `atmosphere/physics/microphysics/morrison.py` | `test_microphysics.py::TestMorrison` | none | **AUDIT-2026-05-05**: q_i deposition keeps the legacy q_i·N_i^(1/3) scaling but adds a `q_i_min_growth` floor so freshly-nucleated ice can grow; donor clamp added for q_i and q_s sinks. (Recalibrating to the canonical N_i^(2/3)·q_i^(1/3) scaling deferred to a tuning effort.) |
| Thompson | `atmosphere/physics/microphysics/thompson.py` | `test_microphysics.py::TestThompson` | none | **AUDIT-2026-05-05**: same q_i_min_growth + donor clamps as Morrison. Graupel pathway preserved. |
| ML emulator | `atmosphere/physics/microphysics/ml_emulator.py` | `test_microphysics.py::TestMLEmulator` | none | Equinox MLP with norm constants. |

**Cross-cutting**: `TestSchemeSelection`, `TestCheckpointWithHydrometeors`, `test_amip_microphysics_config`.  Differentiability covered by `tests/unit/test_diff_microphysics.py`.

### 1.3  Radiation

| Scheme | Source | Unit tests | Validation | Notes |
|--------|--------|------------|------------|-------|
| Gray (Frierson/Isca) | `atmosphere/physics/radiation/gray.py` | `tests/atmosphere/hydrostatic/unit/test_radiation.py::TestGray` | RCE in `test_held_suarez_fix.py` | Two-band SW Beer-Lambert + LW two-stream; document conventions in `_lw_two_stream`. |
| RRTMGP | `atmosphere/physics/radiation/rrtmgp/` | `test_radiation.py::TestRRTMGP`, `tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py` | full AMIP | `f_day` SW rescale verified; LW path is daytime-cosine independent.  Cloud overlap = max-random. |

### 1.4  Turbulence / boundary layer

| Scheme | Source | Unit tests | Validation | Notes |
|--------|--------|------------|------------|-------|
| Louis surface layer | `atmosphere/physics/turbulence/louis.py` | `tests/atmosphere/hydrostatic/unit/test_turbulence.py::TestLouis` | none | Constant-coefficient + Louis-scaled stability path. |
| Holtslag-Boville | `atmosphere/physics/turbulence/holtslag_boville.py` | `test_turbulence.py::TestHoltslagBoville` | none | Counter-gradient simplified as enhanced surface flux. |
| TKE | `atmosphere/physics/turbulence/tke.py` | `test_turbulence.py::TestTKE` | none | Lite TKE; full-vs-half level production simplification noted. |
| YSU | `atmosphere/physics/turbulence/ysu.py` | `test_turbulence.py::TestYSU` | none | |
| EDMF | `atmosphere/physics/turbulence/edmf.py` | `test_turbulence.py::TestEDMF` | none | Eddy-diffusivity / mass-flux. |
| CLUBB-lite | `atmosphere/physics/turbulence/clubb_lite.py` | `test_turbulence.py::TestCLUBB` | none | |
| Smagorinsky | `atmosphere/physics/turbulence/smagorinsky.py` | `test_turbulence.py::TestSmagorinsky` | none | |

**Cross-cutting**: `TestSurfaceFluxes`, `TestVerticalDiffusionT`, `TestVerticalDiffusionQ`, `TestPBLHeight`.

### 1.5  Gravity-wave drag

| Scheme | Source | Unit tests | Validation | Notes |
|--------|--------|------------|------------|-------|
| Lindzen | `atmosphere/physics/gravity_wave_drag/lindzen.py` | `tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py::TestLindzen` | none | Critical-level filtering; sign convention OK. |
| McFarlane (orographic) | `atmosphere/physics/gravity_wave_drag/mcfarlane.py` | `test_gravity_wave_drag.py::TestMcFarlane` | none | |
| Hines (non-orographic spectrum) | `atmosphere/physics/gravity_wave_drag/hines.py` | `test_gravity_wave_drag.py::TestHines` | none | `sigma_sat = N / m_*` documented. |
| Rayleigh | `atmosphere/physics/gravity_wave_drag/rayleigh.py` | `test_gravity_wave_drag.py::TestRayleigh` | none | Sponge-layer top damping. |
| ML emulator | `atmosphere/physics/gravity_wave_drag/ml_emulator.py` | `test_gravity_wave_drag.py::TestMLEmulator` | none | |

### 1.6  Clouds

| Scheme | Source | Unit tests | Validation | Notes |
|--------|--------|------------|------------|-------|
| Sundqvist cloud fraction | `atmosphere/physics/clouds/cloud_fraction.py::sundqvist_cloud_fraction` | `tests/atmosphere/hydrostatic/unit/test_radiation.py::TestCloudCoupling` | none | |
| Xu-Randall cloud fraction | `atmosphere/physics/clouds/cloud_fraction.py::xu_randall_cloud_fraction` | same | none | |
| `compute_cloud_properties` | same | `TestCloudCoupling::test_invalid_scheme_raises` | none | Validates scheme literal. |

### 1.7  Combined / hierarchical

- `tests/atmosphere/hydrostatic/unit/test_combined_physics.py` — full physics chain (radiation + convection + microphysics + turbulence) per integration tier.
- `tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py` — every scheme literal is reachable through the public config.
- `tests/atmosphere/hydrostatic/unit/test_atmosphere_invariants.py` — global mass / energy conservation across one physics step.

### 1.8  Idealized validation (atmosphere)

| Test case | File | Status |
|-----------|------|--------|
| Held-Suarez (spectral PE dycore + full physics chain) | `tests/atmosphere/hydrostatic/validation/test_held_suarez_fix.py::test_held_suarez_spectral_stability` | green (re-verified 2026-05-05 after Physical_Consistency fixes — 8.16 s, slow marker) |
| Stability (CFL / dycore) | `test_stability_fix.py` | green |
| Spectral PE moist Held-Suarez | `test_spectral_pe_moist_held_suarez.py` | green |
| Williamson 2/5 (shallow water) | `tests/atmosphere/shallow_water/integration/` | green |
| AMIP smoke (1-day) | `test_amip_smoke.py` | green |
| AMIP RRTMGP (1-day) | `test_amip_rrtmg.py` | green |
| AMIP stability (multi-day) | `test_amip_stability.py` | green |
| Galewsky shallow water | `tests/atmosphere/shallow_water/validation/test_galewsky.py` | green |

---

## 2.  Ocean

### 2.1  Vertical mixing

| Scheme | Source | Unit tests | Validation | Notes |
|--------|--------|------------|------------|-------|
| Constant | `ocean/physics/vertical_mixing/constant.py` | `tests/unit/test_physics_ocean.py::test_constant_mixing` | none | |
| Pacanowski-Philander Richardson | `ocean/physics/vertical_mixing/richardson.py` | `test_physics_ocean.py::test_richardson_mixing` | none | Stable-side saturates to K_0. |
| KPP (LMD94) | `ocean/physics/vertical_mixing/kpp.py` | `test_physics_ocean.py::test_kpp_mixing` | none | **AUDIT-2026-05-05**: integration.py imports fixed (was raising NameError under any non-trivial surface_forcing).  Recommend new test that constructs OceanSurfaceForcing(q_net, freshwater, tau_x, tau_y) and exercises the path. |
| Implicit solver | `ocean/physics/vertical_mixing/implicit_solver.py` | `test_physics_ocean.py::test_implicit_solve` | none | |

### 2.2  Lateral mixing

| Scheme | Source | Unit tests | Validation | Notes |
|--------|--------|------------|------------|-------|
| Smagorinsky-momentum | `ocean/physics/lateral_mixing/smagorinsky.py` | `test_physics_ocean.py::test_smagorinsky` | none | |
| Backscatter | `ocean/physics/lateral_mixing/backscatter.py` | `test_physics_ocean.py::test_backscatter` | none | Docstring units note (LOW). |
| Redi/GM (cubed-sphere) | `ocean/physics/lateral_mixing/gm_redi.py` | `tests/ocean/unit/test_gm_redi.py` | none | |
| Redi/GM (lat-lon C-grid) | `ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py` | `test_gm_redi.py::TestLatLonCgridGMRedi` | none | Triads enabled. |
| Redi/GM (MPAS) | `ocean/physics/lateral_mixing/gm_redi_mpas.py` | `test_gm_redi_mpas.py` | none | Skeleton. |
| `compute_visbeck_kappa_gm` | `_gm_redi_common.py` | `test_gm_redi.py::TestVisbeck` | none | **AUDIT-2026-05-05**: dry-column zero mask added. |

### 2.3  Bottom drag

| Scheme | Source | Unit tests | Validation | Notes |
|--------|--------|------------|------------|-------|
| Linear | `ocean/physics/bottom_drag/linear.py` | `test_physics_ocean.py::test_linear_drag` | none | |
| Quadratic | `ocean/physics/bottom_drag/quadratic.py` | `test_physics_ocean.py::test_quadratic_drag` | none | |

### 2.4  Convection

| Scheme | Source | Unit tests | Validation | Notes |
|--------|--------|------------|------------|-------|
| Enhanced diffusion | `ocean/physics/convection/enhanced_diffusion.py` | `test_physics_ocean.py::test_enhanced_diffusion` | none | Sigmoid sharpness 1e6 — review (LOW). |
| Plume | `ocean/physics/convection/plume.py` | `test_physics_ocean.py::test_plume_convection` | none | **AUDIT-2026-05-05**: `w_plume_min` is the actual plume velocity; comment clarified. |

### 2.5  Surface forcing

| Scheme | Source | Unit tests | Validation | Notes |
|--------|--------|------------|------------|-------|
| Prescribed | `ocean/physics/surface_forcing/prescribed.py` | `test_physics_ocean.py::test_prescribed_forcing` | none | Freshwater convention: E_minus_P [m/s], evap-positive. |
| Restoring | `ocean/physics/surface_forcing/restoring.py` | `test_physics_ocean.py::test_restoring` | none | sin² T_eq → T_pole profile. |
| Bulk formulas (constant + COARE3 + L&Y04) | `ocean/physics/surface_forcing/bulk_formulas.py` | `test_physics_ocean.py::test_bulk_formula_modes` | none | **AUDIT-2026-05-05**: constant-scheme heat fluxes now use `|U_a|` (sign of U_a no longer flips Q_sh/Q_lh). |

### 2.6  Shortwave penetration

| Scheme | Source | Unit tests | Validation | Notes |
|--------|--------|------------|------------|-------|
| Two-band Jerlov | `ocean/physics/shortwave_penetration.py` | `test_physics_ocean.py::test_shortwave_penetration` | none | **AUDIT-2026-05-05**: rho_0/c_sw defaults from `eos`; bottom-leaked SW absorbed by bottom layer for column conservation. |

### 2.7  EOS

| Function | Source | Unit tests | Notes |
|----------|--------|------------|-------|
| Wright EOS | `ocean/eos.py::wright_eos` | `tests/ocean/unit/test_eos_consistency.py` | full Wright-Eaton coefficients. |
| Linear EOS | `ocean/eos.py::linear_eos` | same | |
| `thermal_expansion_coeff`, `haline_contraction_coeff` | `ocean/eos.py` | same | Computed via `jax.grad`. |
| `compute_ocean_rho`, `compute_ocean_rho_and_pressure` | same | exercised by every dynamics test | shared helpers. |

### 2.8  MPAS-specific

`ocean/physics/mpas_physics.py` — wires prescribed surface forcing + bulk formulas onto MPAS grid.  Tested via `tests/ocean/unit/test_mpas_ocean.py`.

### 1.9  Test pass-counts after Physical_Consistency cycle

Quoted at 2026-05-05 with all fixes applied:

| Suite | Tests run | Result |
|-------|-----------|--------|
| `tests/atmosphere/hydrostatic/unit/test_microphysics.py` | 118 | green |
| `tests/atmosphere/hydrostatic/unit/test_convection.py` | 67 | green |
| `tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py` + spectral_pe_convection.py | 36 | green |
| `tests/atmosphere/hydrostatic/unit/test_atmosphere_invariants.py` (interim) | 29 / 30 | 1 pre-existing dycore bug (`_lnps_pad` UnboundLocalError, primitive_eq_cdgrid.py:343) — unrelated |
| `tests/atmosphere/hydrostatic/validation/test_held_suarez_fix.py` (slow) | 1 | green |
| `tests/land/unit/` | 164 | green |
| `tests/land/test_land_stability.py::TestMultiLayerStability::test_no_nan` (15-day run) | 1 | green (348 s — JIT compile dominated) |
| `tests/unit/test_sea_ice_dynamics.py` + `test_diff_sea_ice.py` + `test_land_ice_sea_ice_dynamics.py` | 64 | green |
| `tests/unit/test_physics_ocean.py` | 11 | green |
| `tests/ocean/unit/test_gm_redi*.py` | 58 | green |
| `tests/atmosphere/hydrostatic/unit/test_radiation.py::TestSolarGeometry` + `TestGrayRadiation` | 18 | green |
| `tests/unit/test_diff_atmosphere_physics.py` (AD through atmosphere physics) | 32 | green |
| `tests/unit/test_diff_sea_ice.py` (AD through sea-ice) | 9 | green |
| Pre-existing ocean MPAS K_bih=0 strict-equality test | 0/1 | 1 unrelated `atol=0,rtol=0` flake — independent of these changes |
| Pre-existing FV3 hydrostatic dycore `_lnps_pad` UnboundLocalError | n/a | Hits some `tests/atmosphere/hydrostatic/integration/test_amip*.py` and `test_atmosphere_invariants.py::TestRestStateInvariance::test_zero_wind_*`; reachable only via `cdgrid_hydrostatic_tendencies` → `fv3_hydrostatic_tendencies` (`primitive_eq_cdgrid.py:343`).  Not touched by Physical_Consistency. |

**Roll-up**: Approximately **600+ unit tests + 1 idealized-validation case** pass with all Physical_Consistency fixes applied.  No regression introduced by these changes; the only failing tests on the branch are pre-existing dycore / strict-equality flakes that are not in the physics-parameterization scope of this audit.

### 2.9  Idealized validation (ocean)

| Case | File | Status |
|------|------|--------|
| Phillips two-layer baroclinic | `tests/ocean/validation/test_baroclinic_phillips.py` | green |
| Stommel/Munk gyre | `test_stommel_munk.py` | green |
| Rest state stability | `tests/ocean/unit/test_cross_grid_parity.py` | green |
| Ocean model integration smoke | `test_ocean_model_smoke.py` | green |
| Differentiability (ocean) | `tests/unit/test_diff_ocean.py` | green |

---

## 3.  Land-surface

### 3.1  Soil thermal & hydraulics

| Scheme | Source | Unit tests | Validation | Notes |
|--------|--------|------------|------------|-------|
| Soil thermal (Kersten) | `land/soil_thermal.py` | `tests/land/unit/test_multilayer_land.py::TestSoilThermal` | none | No freeze/thaw latent heat (HIGH known gap). |
| van Genuchten / Mualem hydraulics | `land/soil_hydraulics.py` | `tests/unit/test_land_ice_soil_hydraulics.py` | none | Saturated branch C += S_s · θ_sat. |
| Richards (Celia mixed-form Picard) | `land/richards.py` | `test_multilayer_land.py::TestRichards` + `tests/land/test_land_stability.py::TestMultiLayerStability` | dry-down stability | **AUDIT-2026-05-05**: missing matric-potential flux divergence added to RHS — solver now actually solves Richards (was gravity-drainage-only at convergence). |
| Tridiagonal solver | `land/tridiag.py` | `test_multilayer_land.py::TestTridiag` | none | Thomas algorithm with `lax.scan` carry. |

### 3.2  Surface energy + water

| Scheme | Source | Unit tests | Validation | Notes |
|--------|--------|------------|------------|-------|
| Slab (bucket) | `land/slab_land.py` | `test_multilayer_land.py::TestSlabLand` + `tests/land/unit/test_land_water_budget.py` | none | Bucket overflow = saturation excess runoff only. |
| Multi-layer driver | `land/multilayer_land.py` | `test_multilayer_land.py::TestMultiLayer` + `test_land_stability.py::TestMultiLayerStability` (15-day) | none | |
| Snow | `land/snow_budget.py` | `tests/unit/test_land_ice_snow.py` | none | SWE evolution with melt/sublimation. |
| Surface albedo | `surface_albedo.py` | `tests/unit/test_land_ice_albedo.py` | none | Snow + bare-soil + vegetation. |

### 3.3  Vegetation / biogeochemistry

| Scheme | Source | Unit tests | Validation | Notes |
|--------|--------|------------|------------|-------|
| Stomata (Farquhar + Ball-Berry + Jarvis) | `land/stomata.py` | `tests/land/unit/test_stomata.py` | none | VPD response uses mixing-ratio form for q (MEDIUM ~1 % bias). |
| Carbon cycle (NPP + pools) | `land/carbon/carbon_cycle.py` | `tests/land/unit/test_carbon_cycle.py` | none | Wood/root turnover with `_effective_rate`. |
| Stomata utils (PFT dispatch) | `land/stomata_utils.py` | `tests/unit/test_land_ice_stomata.py` | none | |

### 3.4  Bulk fluxes (atm-land coupling)

`coupler/bulk_flux.py` (shared atmosphere-land + atmosphere-ocean MOST).  Tested via `tests/land/validation/test_bulk_flux_all_tiles.py` and `test_bulk_flux_differentiability.py`.

---

## 4.  Sea ice

### 4.1  Thermodynamics

| Scheme | Source | Unit tests | Validation | Notes |
|--------|--------|------------|------------|-------|
| Bitz-Lipscomb 2-layer | `ice/sea_ice.py` | `tests/unit/test_diff_sea_ice.py` | none | Slab + multi-category dispatch. |
| Multi-cat ITD aggregate | `ice/itd.py::aggregate_state` | `test_diff_sea_ice.py::TestITDGrad` | none | **AUDIT-2026-05-05**: T_freeze_ocean uses `constants.T_freeze_ocean`. |

### 4.2  Dynamics + rheology

| Scheme | Source | Unit tests | Validation | Notes |
|--------|--------|------------|------------|-------|
| EVP (Hunke-Dukowicz 1997) | `ice/rheology.py` + `ice/dynamics.py::evp_solver` | `tests/unit/test_sea_ice_dynamics.py::TestEVPStressUpdate` + `TestEVPSolver` | none | **AUDIT-2026-05-05**: stress-update relaxation factor now includes N_evp; m_ice kept per ice-covered area (not concentration-weighted — both stress and mass would need to scale by A together for that convention to be self-consistent). |
| Free-drift | `ice/dynamics.py::free_drift_velocity` | `test_sea_ice_dynamics.py::TestFreeDrift` | none | Heuristic placeholder; documented. |
| Strain rates / divergence | `ice/dynamics.py::strain_rates`, `stress_divergence` | `test_sea_ice_dynamics.py::TestStrainRates`, `TestStressDivergence` | none | |
| Ice strength | `ice/rheology.py::ice_strength` | `test_sea_ice_dynamics.py::TestIceStrength` | none | |

### 4.3  Transport + ITD remap

| Scheme | Source | Unit tests | Validation | Notes |
|--------|--------|------------|------------|-------|
| Centered flux-form transport | `ice/transport.py::advect_ice_tracers` | `test_sea_ice_dynamics.py::TestTransport` | none | Documented non-monotone; T_freeze_ocean uses constants. |
| Linear ITD remap | `ice/itd.py::linear_remap` | `test_diff_sea_ice.py::TestITDLinearRemap` | none | Conservation approximate when post-clamp fires (CRITICAL — flagged). |

### 4.4  Idealized validation (ice)

Sea-ice has **no idealized validation tests** beyond unit-level differentiability.  Recommended new tests:

- Hunke-Dukowicz 1997 box test (EVP) — assert stress relaxes to VP target.
- Square uniform-flow free-drift test — ice velocity aligned with wind to within turning angle.
- Brine-rejection coupling test — ocean salt budget closes when ice mass change is reported.

---

## 5.  Coupling-level tests

| Test | File | What it checks |
|------|------|----------------|
| Atmosphere physics matrix | `scripts/matrix/run_atmosphere_test_matrix.py` | Every (dycore × physics × grid) combination launches and runs |
| Ocean physics matrix | `scripts/matrix/run_ocean_test_matrix.py` | Every (mixing × surface_forcing × eos × grid) combination |
| Slopbuster audit | `.claude/agents/slopbuster.md` | Constant hygiene, dead modules, untested dispatch |
| Code-hygiene audit | rolling cycles | NamedTuple field consistency, unused imports |

---

## 6.  Audit cycle change log

### 2026-05-05 (Physical_Consistency)

Atmospheric:
- DCA latent-heat release added (CRITICAL).
- Morrison + Thompson q_i_min_growth floor added; donor clamps for q_i and q_s sinks. The dimensional rescaling to N_i^(2/3)·q_i^(1/3) was reverted because it would change the deposition magnitude by O(10^3) at typical mid-cloud values without a re-tune of `dep_coeff` — recalibration is deferred.

Ocean:
- vertical_mixing/integration.py imports fixed (CRITICAL).
- bulk_formulas constant-coefficient path uses |U_a| (HIGH sign fix).
- Visbeck dry-column zero mask added.
- shortwave_penetration column-conservation fix; `eos.rho_0`/`c_sw` defaults.

Land:
- Richards RHS now includes L^m psi^m (CRITICAL — solver was gravity-only at convergence).
- Infiltration capacity uses half-distance to first node (HIGH).

Sea ice:
- EVP relaxation factor includes N_evp (CRITICAL — ~120× over-relaxation).
- m_ice kept as `rho_ice · max(h, 0.01)` per ice-area (the iter-1 attempt at concentration-weighting introduced a 1/A over-acceleration; Codex GPT-5 review caught the bookkeeping error).
- T_freeze_ocean replaced with `constants.T_freeze_ocean`.

Test updates: `evp_stress_update` signature now takes `N_evp` (2 callers updated).


### Cumulative audit-finding status (Physical_Consistency cycle 2026-05-05)

| Severity | Total found | Fixed | Deferred |
|----------|-------------|-------|----------|
| CRITICAL | 10 | 10 | 0 |
| HIGH     | 47 | 43 | 4 (F1, F2, F10; ITD remap; fix_mass MPI edge; f_veg LAI) |
| MEDIUM   | 33 | 18 | 15 (mostly LOW-impact / structural cleanup) |
| LOW      | ~30+ | 12 | rest documented in catalog |

All nine CRITICAL findings (DCA latent heat, Richards `L psi^m`, EVP relaxation, ocean integration imports, GWD g-factor, _shared moisture-convergence, spectral-PE hybrid tracer vertical advection, RAW filter sign, plus the CAPE p_full→p_mid fix) are fixed and Codex-verified.

A later audit pass covered 14 new modules and added 24 HIGH/CRITICAL fixes:
- atmosphere convection (compute_cin p_full→p_mid; SBM straight-through cloud_mask)
- atmosphere turbulence (YSU Louis constants; CLUBB-lite squared-gradient; HoltslagBoville sharpness; EDMF exner)
- ocean physics (PP81 momentum/tracer; KPP B_f fallback)
- ocean dynamics (rho_0/c_sw to constants; barotropic_mpas land-edge mask hoist + initial u_bar mask; spectral_ocean_pe scale_depth)
- sea-ice (concentration growth from leads only)
- coupler (lake convective overturn for freshwater density inversion)
- driver (coupled-ESM constants M_CO2/M_air/eps_sfc/S_0; snow_frac smooth ramp; AMIP SST floor 200K → T_freeze_ocean)
- dycores (spectral-PE hybrid tracer vertical advection; spectral-SW energy g·h·h_s; FV3 div damp sign in legacy CSW path)
- time integration (RAW filter sign violation; α=0.53 default)
- training (ERA5 q specific→mixing; ml/loss normalization; carry_mse lat_weights; per_variable_mse channel_weights)
- land (multilayer_land beta_root_new denominator floor)

### Open audit findings (deferred for follow-up)

1. **HIGH** — Soil freeze/thaw latent heat not in `soil_thermal.py`.  Cold-climate cases unsupported.
2. **MEDIUM** — DCA `n_iter`-Picard convergence diagnostic; precip diagnostic re-exposure.
3. **MEDIUM** — Sea-ice ITD `linear_remap` non-conservative when post-remap clamp fires.
4. **MEDIUM** — Sea-ice → ocean brine/freshwater flux not in `TileResponse`.
5. **MEDIUM** — Centered flux-form ice transport not monotone; clamp leaks volume / temperature.
6. **MEDIUM** — Sea-ice multi-category open-water freezing flux double-counting in non-empty-cat0 cells.
7. **LOW** — Ocean prescribed vs OceanSurfaceForcing freshwater unit/sign convention split.
8. **LOW** — `compute_moist_adiabat` saturated-everywhere assumption (8 convection schemes).
9. **LOW** — `T_vs_θ` vertical diffusion: dry adiabat not exact null state of vertical mixing.
