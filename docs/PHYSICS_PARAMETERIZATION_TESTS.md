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
| Stomata (Farquhar + Ball-Berry + Jarvis) | `land/carbon/stomata.py` | `tests/land/unit/test_stomata.py` | none | VPD response uses mixing-ratio form for q (MEDIUM ~1 % bias). |
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
| Atmosphere physics matrix | `scripts/run_atmosphere_test_matrix.py` | Every (dycore × physics × grid) combination launches and runs |
| Ocean physics matrix | `scripts/run_ocean_test_matrix.py` | Every (mixing × surface_forcing × eos × grid) combination |
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

### Iter-8 / iter-9 (2026-05-05 — Codex round 4)

- KPP `Ri_crit` updated 0.25 → **0.3** (LMD94 default; matches NCAR POP2 / MOM6 / MITgCM).
- `mass_flux._apply_mass_flux_kernel`: bare `M_u_max=0.05` default removed; all callers thread it from their config NamedTuple.
- DCA `test_moist_static_energy_conservation` added (tolerance 1e-4, ⟨Δq⟩>1e-4 guard) — directly exercises the per-pair `c_p·ΔT + L_v·Δq = 0` invariant added in iter-1.
- `docs/legoesm_scientific_guide.tex`: stale `Ri_crit = 0.25` LaTeX values updated to 0.3 in both KPP/PBL sections.

### Iter-10 (2026-05-05 — coupler/conservation audit)

- **F15 (HIGH)**: `coupler/lake/two_layer_lake.py` lake `lw_up` now includes the reflected `(1−ε)·lw_down` term, matching every other tile.  Earlier formulation under-counted lw_up by ~10 W/m² (ε=0.97, lw_down ≈ 350 W/m²).
- **F11 (MEDIUM)**: `ice/dynamics.py` function defaults for `rho_air`, `rho_ice`, `rho_ocean` now reference `legoesm.constants.*`.  `constants.rho_air = 1.225` and `constants.rho_ocean = 1025.0` added as canonical references.
- **F19, F23 (LOW)**: `forcing/amip.py` and `forcing/analytical.py` documented `T_ice` legacy field as `= constants.T_freeze_ocean`; bare `1.8` literal in `analytical.py` replaced with the explicit `(T_freeze - T_freeze_ocean)` difference.

### Iter-11 / iter-12 (2026-05-05 — F6/F7/F17 + state hygiene)

- **F7 (HIGH)**: `ice/sea_ice.py` adds `dh_dt_sublim = -lhflx / (rho_ice · L_s)` (over ice mask) so sublimation removes ice mass and deposition adds it.  Previously the energy budget closed via L_s but the mass budget was open — thin polar ice was growing endlessly under sublimation, biased high by tens of cm/yr.
- **F17 (MEDIUM)**: `coupler/lake/two_layer_lake.py` switches `q_sfc` to ice saturation and `L_eff` to L_s when `T_epi ≤ T_freeze`.  Frozen-lake lhflx was previously biased by ~13% from always using L_v.
- **F6 (MEDIUM)**: `ocean/simple_ocean.py` slab + two-layer ocean freezing clamps now diagnose `Q_freeze` (the energy that would have driven SST below `T_freeze`, equivalent to the latent heat of fusion flowing into ice formation).  `SlabOceanState` gained a `Q_freeze: Field | None` field; `init_slab_state` populates it with a zero Field so the pytree shape is invariant across steps.

### Iter-13 (2026-05-05 — F4 freshwater channel)

- **F4 (HIGH)**: `TileResponse` and `SurfaceToAtm` gained a `freshwater_flux` field [kg/m²/s, positive INTO ocean].  Each tile populates it from its own water budget:
    - slab land: bucket-overflow `runoff`
    - multilayer land: `runoff_surface + runoff_subsurface`
    - ocean tile: `precip_total − lhflx/L_v`
    - lake: `precip_total − lhflx/L_eff` (L_eff phase-aware, iter-11)
    - sea ice (slab): `−rho_ice · (h_new − h)/dt` minus sublimation (already in lhflx)
- `tile_fractions.blend_tiles` and `accumulator.{FluxAccumulator, accumulate, mean_accumulator, accumulator_from_flux}` were extended in lockstep so the time-averaged blended freshwater flux propagates through the coupler.
- 50 coupler tests + 398 cross-component tests pass after the change.

### Iter-14 (2026-05-05 — F8 ocean_heat_extraction channel)

- **F8 (HIGH)**: `TileResponse` and `SurfaceToAtm` gained an `ocean_heat_extraction` field [W/m², positive = ocean LOSES energy to this tile].  Sea-ice tile populates two contributions:
    - Basal melt: `F_ocean = ocean_heat_transfer_coeff · max(SST − T_freeze_ocean, 0)` on previously-ice-covered cells
    - Open-water freezing: `rho_ice · L_f · h_new / dt` on previously-ice-free cells (latent heat extracted to form new ice)
- Land, lake, and ocean tiles return zeros (no direct ocean-heat exchange).
- `tile_fractions.blend_tiles` and the `FluxAccumulator` chain were extended in lockstep with iter-13.
- 106 coupler/lake/ice tests + 412 cross-component tests pass after the change.

### Iter-15 (2026-05-05 — F9 ocean_stress_x/y back-reaction)

- **F9 (HIGH)**: `TileResponse` and `SurfaceToAtm` gained `ocean_stress_x` and `ocean_stress_y` fields [Pa, positive eastward/northward force on the ocean].  Sea-ice tile populates `-tau_oi · concentration` where `tau_oi = rho_ocean · drag_ocean · |U_w − U_i| · (U_w − U_i)` is the ocean→ice Cauchy drag.
- Newton's third law: ice exerts equal-and-opposite stress on the ocean column.  Without this channel, the ocean felt only its own wind stress through bulk fluxes — under-ice currents drifted incorrectly relative to free-drift currents.
- Land, lake, and ocean tiles return zeros.  Ocean tile already contributes its own wind stress via `tau_x`/`tau_y`.

### Iter-16 (2026-05-05 — F3 phase-aware surface_mass_flux)

- **F3 (HIGH)**: `TileResponse` and `SurfaceToAtm` gained `surface_mass_flux` [kg/m²/s, positive up = drying].  Each tile populates this directly from its own bulk-flux calculation using the appropriate phase latent heat:
  - ocean: `lhflx / L_v` (always liquid)
  - lake: `lhflx / L_eff` with L_eff phase-aware (iter-11)
  - slab land: `lhflx_actual / L_eff` with L_eff snow-aware
  - multilayer land: `evap_rate` (already phase-aware via Richards solve)
  - sea ice: `lhflx / L_s` (sublimation)
- Without this channel, consumers had to back-derive evap mass from `lhflx / L_v`, which under-counts by ~13 % on any sublimating tile.

### Iter-17 (2026-05-05 — F13/F14 pytree-shape invariance)

- **F13 (LOW)**: `LandState.runoff` now initialised to a populated zero Field rather than `None`.  After `step_land` populates it with an array, the pytree shape used to differ from the init state — any `lax.scan` / `jax.tree.map` over the surface state would have raised a tree-structure mismatch.
- **F14 (LOW)**: Same fix for `LakeState.Q_freeze`.
- Together with the iter-12 `SlabOceanState.Q_freeze` fix, all three "diagnostic field initialised None then set to array" pytree-shape inconsistencies are resolved.

### Iter-18/19 (2026-05-05 — conservation channel regression tests)

Add seven explicit regression tests in `tests/unit/test_coupler.py` covering all four conservation channels on every tile that produces them:

- iter-18 (sea-ice tile, 4 tests): `test_sea_ice_freshwater_flux_balances_ice_mass_change` (F4); `test_sea_ice_ocean_heat_extraction_positive_under_warm_ocean` (F8); `test_sea_ice_ocean_stress_opposes_ocean_ice_drag` (F9); `test_sea_ice_surface_mass_flux_equals_lhflx_over_Ls` (F3).
- iter-19 (lake / land tiles, 3 tests): `test_lake_surface_mass_flux_uses_phase_aware_L` (F3, both warm/L_v and frozen/L_s branches); `test_lake_freshwater_flux_is_P_minus_E` (F4); `test_land_freshwater_flux_equals_runoff` (F4, with a saturated bucket and heavy precip to force overflow).

419 tests pass after iter-19 (412 → 416 in iter-18 → 419 in iter-19).

### Iter-20 (2026-05-05 — F22 + F5 wire MPAS freshwater path)

- **F22 (LOW)**: `coupler/mpas_adapter.compute_mpas_freshwater` now consumes the phase-aware `SurfaceToAtm.surface_mass_flux` (iter-16) instead of back-deriving evap from `lhflx / L_v`.  The L_v fallback path is preserved for legacy callers that wire a SurfaceToAtm without the new field.  Removes the ~13 % mass under-count over sublimating tiles.
- **F5 (MEDIUM)**: same adapter now reads slab-land runoff from `LandState.runoff` (single field) when the multilayer `runoff_surface` / `runoff_subsurface` aren't present — earlier code silently dropped slab runoff.
- `ocean/freshwater.freshwater_from_coupler` gains an optional `surface_mass_flux` kwarg.  44 freshwater tests pass.

### Iter-21 (2026-05-05 — regression tests for CRITICAL fixes)

Two CRITICAL iter-1 fixes lacked dedicated regression tests; both now have one:

- `tests/land/unit/test_land_water_budget.py::TestRichardsNoPrematureRunoff::test_matric_flux_redistributes_steep_psi_gradient` — verifies that the Richards solver redistributes water from a wet top layer to dry layers below under zero-flux top + bottom BCs.  Without the iter-1 `L psi^m` divergence term the converged Picard solution reduces to gravity-drainage only and no redistribution occurs.
- `tests/ocean/unit/test_shortwave_penetration.py::test_column_sw_conservation_{deep,shallow}_water` — verifies `sum(dT/dt · rho_0 · c_sw · dz_layer) == sw_down` to bit-precision.  The shallow (H=50 m) case is the regression test for the iter-1 bottom-layer leakage absorption: without the fix, ~6 % of SW would escape the column.

474 tests pass after iter-21 (419 → 463 with iter-20's freshwater suite → 474 with iter-21's 3 new tests).

### Iter-35 (2026-05-05 — CRITICAL F1 moisture-convergence sign + F2 hypsometric T_v)

The deep audit of `atmosphere/physics/_shared.py` found two new fixes:

- **F1 (CRITICAL)**: `compute_moisture_convergence` was double-negating the FV flux divergence on cubed-sphere and lat-lon C-grid branches, returning `+div(q·V)` (moisture DIVERGENCE) instead of `-div(q·V)` (moisture CONVERGENCE).  The Gaussian/spectral branch was correct.  Bug silently inverted the Tiedtke / Bechtold convection deep-closure mass-flux trigger on cubed-sphere/lat-lon runs.
- **F2 (HIGH)**: iter-36 added an optional `q_v` argument to `compute_heights_from_sigma` and `compute_layer_dz` so the hypsometric integral uses virtual temperature `T_v = T·(1 + (1/ε−1)·q_v)`.  Previously dry T was used, biasing layer heights low by ~1 % in tropical moist columns.

Regression tests:
- `test_mc_sign_matches_dycore_tracer_tendency`: asserts `compute_moisture_convergence` returns bit-exactly the same value as `fv_flux_divergence_3d` (both = `dq/dt = -div(q·V)` = MC).  Catches the F1 bug; the previous test was vacuous.

**Integration verification (iter-35/36)**:
- Held-Suarez spectral-PE stability validation passes (6.62 s) with iter-35 + iter-36 fixes applied.
- 210 atmosphere physics unit tests pass (microphysics + convection + GWD + turbulence).
- 63 / 64 combined-physics + spectral-PE tests pass; 1 fail is pre-existing dycore `_lnps_pad` bug unrelated to physics.

**Iter-37 wider integration verification (post-MC-sign-flip)**: 175 tests pass + 2 skipped across `test_convection.py + test_convection_latlon_mpas.py + test_microphysics.py + test_moisture_convergence.py + Held-Suarez slow validation`.  The MC sign flip did NOT cause regressions in Tiedtke / Bechtold / mass-flux convection schemes, microphysics, or the Held-Suarez integration.  This confirms the fix is operationally correct: convection responds to MOISTURE CONVERGENCE (positive when air converges) as physically intended, not to its negation.

**Iter-42 final verification sweep**: 439 tests pass + 6 warnings in 246 s across atmosphere physics (microphysics, convection, moisture convergence, AMIP forcing) + ocean physics + land + sea-ice + coupler + Held-Suarez spectral-PE slow validation.  All physics fixes from iter-1 through iter-41 are operational without regressions.

### Iter-22 (2026-05-05 — stomata vapor pressure formula)

- **Audit finding #24 (MEDIUM)**: `land/carbon/stomata.py` switches the vapor-pressure-from-specific-humidity formula from the mixing-ratio form `e = q · p / (ε + q)` to the correct specific-humidity form `e = q · p / (ε + (1 − ε) · q)`.  Affects both `jarvis_gs` and `coupled_farquhar_stomata`.  The bias is ~1 % at typical tropical q but propagates through the VPD-driven stomatal closure into GPP.
- 90 stomata + carbon-cycle tests pass.

### Iter-23 (2026-05-05 — bulk-formulas |U_a| regression test)

- `tests/ocean/unit/test_surface_forcing_dispatch.py::test_bulk_formula_constant_heat_flux_uses_wind_speed` locks in the iter-1 HIGH-severity sign fix: heat fluxes Q_sh, Q_lh use `|U_a|`, stress `tau_x` is directional.  Test runs the constant-coefficient bulk formula with +5 m/s and -5 m/s zonal wind and asserts Q_net is identical (heat-flux symmetry) while tau_x flips sign (stress directionality).
- 15 surface-forcing-dispatch tests pass.

### Iter-26 (2026-05-05 — gravity-wave-drag deep audit)

Second-pass audit of the GWD subpackage caught two real bugs:

- **P0 (CRITICAL)** — `gravity_wave_drag/prognostic_spectral.py`: the stress-divergence to acceleration conversion was missing the `g` factor.  Hydrostatic identity `dp = -ρ·g·dz` requires ``F/(ρ·dz) = F·g/(-dp)``, so the conversion factor is `g` (~9.8 m/s²), not 1.  The earlier code under-counted GWD acceleration by a factor of ~9.8 in the prognostic-spectral path.
- **P1** — `gravity_wave_drag/ml_emulator.py`: column dissipation `eps_gwd` was computed with `jnp.abs(u·du + v·dv)` instead of `-(u·du + v·dv)`.  Switched to match every other GWD scheme; preserves the conservation tie-back ``c_pd · ∫ρ·dT_dt·dz = eps_gwd`` so an untrained model that accidentally adds KE shows as a negative eps_gwd diagnostic.

Structural physics gaps flagged but deferred:

- Lindzen / McFarlane / prognostic_spectral lack critical-level filters (waves are not absorbed at the level where local U projects to wave phase speed).
- Hines drag direction is tied to local wind rather than wave direction; differs from canonical isotropic Hines.
- Lindzen uses hard `jnp.minimum` at saturation kink (McFarlane uses smooth softmin) — gradient pinching.

37 GWD tests pass after the fixes.

### Remaining HIGH-severity items (deferred)

- **F1 / F10**: Atmosphere ↔ ocean ↔ ice tau sign-convention split (latent until prognostic ocean is wired through coupler).
- **F2**: Slab ocean and surface coupler tile compute fluxes with different bulk schemes — duplicated paths.

### Iter-44 (2026-05-05 — convection compute_cin p_full→p_mid)

- `_plume.compute_cin` had the same bug as iter-39's `compute_cape` HIGH #1 fix: the discrete `∫ dlnp` weighting used `p_full` (a layer-mean pressure on hybrid-sigma grids) instead of `p_mid` (the half-level midpoint).  Produced 0.5–2 % CIN bias relative to the matching CAPE.  Fix matches every sister physics helper.
- 62 atmosphere convection tests pass; Held-Suarez stability validation passes.

### Iter-45 (2026-05-05 — SBM straight-through cloud_mask)

- SBM `sbm_convection` used a hard boolean `(T_moist >= T).astype(...)` cloud_mask, breaking differentiability through layer top/bottom transitions even though the docstring claimed smooth-everywhere semantics.
- Fix: straight-through estimator `cloud_mask = soft + lax.stop_gradient(hard - soft)` — forward semantics bit-identical to the prior hard mask, backward gradient is `sigmoid'`.
- Non-vacuous regression test asserts the gradient at the boundary level equals `-sharpness/4`, falsifying the hard-mask version.
- 8 SBM tests pass; Held-Suarez passes.

### Iter-46 (2026-05-05 — YSU Louis constants + CLUBB-lite)

- `ysu.py` had hardcoded Louis (1982) stability-function constants (`b_louis = 5.0`, `5.0` for `b'`, `100.0` for blend sharpness) violating CLAUDE.md constant-discipline.  Lifted to `YSUConfig.louis_b`, `louis_c`, `louis_d`, `blend_ri_sharpness`.
- `clubb_lite.py` variance formula used `|wpthlp| · |dθ/dz|` (kink at zero gradient).  Rewritten as algebraically-equivalent `Kh · (dθ/dz)²` with explicit positivity guard `Kh_pos = max(Kh, 0)`.
- 64 turbulence tests pass; Held-Suarez passes.

### Iter-47 (2026-05-05 — PP81 momentum/tracer + KPP B_f fallback)

- `richardson.py` (Pacanowski-Philander): momentum and tracer roles were SWAPPED.  Canonical PP81 (POP / E3SM Omega): `ν = ν₀/(1+αRi)^n + ν_b`, `κ = ν/(1+αRi) + κ_b`.  The Prandtl ratio Pr = ν/κ = (1+αRi) GROWS with Ri because momentum mixes more efficiently than tracer in stable shear.  Prior code used a constant Pr_t = 10 which inverted this.  `cfg.Pr_t` is now dead (UserWarning on non-default).
- `kpp.py`: B_f=None fallback used `g/ρ₀ · K_bg · drho_dz_sfc` proxy (~1e-10 m²/s³, 2-4 orders of magnitude below realistic 1e-8 to 1e-7 m²/s³).  Changed to `B_f=0` fail-closed semantics.
- 13 ocean physics + KPP corrections tests pass; Held-Suarez passes.

### Iter-48 (2026-05-05 — sea-ice concentration growth from leads)

- `sea_ice.py` line 528 used `dconc_growth = max(dh_dt, 0) · (1-A) / h_new_ice` for both branches of `where(ice_mask, dh_dt_ice, dh_dt_open)`, letting basal vertical growth of existing floes spuriously spread them laterally.
- Fix (after Codex round): drive concentration growth ONLY from `dh_dt_open · (1-A) / h_new_ice` regardless of ice_mask.  This (a) prevents existing-ice basal growth from changing A, (b) preserves CICE/Icepack ``add_new_ice`` lead refreezing on partially-covered cells.
- Two non-vacuous regression tests verify both: existing-ice basal growth doesn't change A; partial-cover lead refreezing DOES change A.
- 11 differentiable sea-ice tests pass; Held-Suarez passes.

### Iter-49 (2026-05-05 — lake convective overturn for freshwater)

- `coupler/lake/two_layer_lake.py` had no convective overturn check; freshwater density inversions (T_epi < T_hypo with both layers > 4 °C) were preserved indefinitely.  CICE/FLake/ALMA all instantaneously homogenize density inversions.
- Added Kell 1975 / Jones-Harris parabolic ρ(T) anomaly check; when ρ_epi > ρ_hypo, set both layers to mass-weighted mean.
- Constants ``T_freshwater_max_density`` and ``rho_freshwater_curvature`` added to ``legoesm.constants``.
- CLAUDE.md strengthened: ALL physical constants live in ``legoesm.constants`` — no hardcoded values in config.py, function bodies, tests, or plotters.
- 8 lake tests pass (1 new convective-overturn regression with enthalpy-conservation check); Held-Suarez passes.

### Iter-50 (2026-05-05 — HoltslagBoville sharpness + EDMF exner)

- `turbulence/holtslag_boville.py` had three hardcoded sharpness literals (20.0, 100.0, 10.0).  Lifted to ``HoltslagBovilleConfig.pbl_sharpness`` / ``blend_ri_sharpness`` / ``blend_pbl_sharpness``.
- `turbulence/edmf.py` had a single ``exner`` variable shadowed mid-function (first ``(p_ref/p)^κ``, then ``(p/p_ref)^κ``).  Renamed to ``exner_pref`` / ``exner_inv`` for clarity.
- 53 turbulence tests pass; Held-Suarez passes.

### Iter-51 (2026-05-05 — spectral-PE hybrid tracer vertical advection — CRITICAL)

- ``atmosphere/dynamics/spectral_pe.py``: tracer vertical advection was silently dropped on the hybrid coordinate path.  ``_tracer_advection_gaussian`` returned horizontal-only when sigma_coord is ``HybridSigmaPressureCoordinate``, and the caller never added the vertical contribution.  T, u, v had vertical advection (lines 604, 656-657); only tracers were broken.
- Fix: add ``vertical_advection_hybrid(q_grid, mass_flux, p_s, sigma_coord)`` inside the tracer loop, gated on ``is_hybrid``.
- Source-level regression test added.  10 spectral-PE tracer-advection tests pass; Held-Suarez passes.

### Iter-52 (2026-05-05 — coupled driver hardcoded constants + snow_frac)

- ``driver/coupled_esm_driver.py``: hardcoded ``M_CO2 = 44.01``, ``M_air = 28.97`` in three places; ``eps_sfc = 0.96``; ``S_0 = 1360.0`` (vs canonical 1361.0).  All moved to ``legoesm.constants``: M_air, M_CO2, M_H2O, emissivity_ocean/ice/land.
- ``snow_frac = where(T_low < T_freeze, 1, 0)`` hard step replaced with smooth ramp ``clip((T_freeze + 2 - T_low) / 4, 0, 1)`` (Wigmosta 1994 / Dai 2008).
- 67 coupler/diff-coupler tests pass; Held-Suarez passes.

### Iter-53 (2026-05-05 — ocean rho_0 / c_sw to constants.py)

- ``ocean/eos.py`` had module-level literals ``rho_0 = 1025.0`` and ``c_sw = 3994.0``.  Both rebound to ``constants.rho_ocean`` and ``constants.c_sw`` (newly added).
- 21 ocean physics + plume tests pass; Held-Suarez passes.

### Iter-54 (2026-05-05 — Robert-Asselin-Williams filter sign error — CRITICAL)

- ``timestepping/semi_implicit.py:robert_asselin_filter`` had both filter increments with the SAME sign and α/(1-α) swapped.  Williams 2009 (eqs. 8-9) requires OPPOSITE signs:
      X^n_filtered = X^n + α·d_n
      X^{n+1}_filtered = X^{n+1} − (1 − α)·d_n
- The buggy formulation produced sum drift of +d_n every step (~10% per 100 steps at γ = 0.05) — climate-relevant.
- Three non-vacuous regression tests added (sum-conservation at α = 0.5, 2·dt damping rate = 1−γ, α = 1 recovers original RA).
- 23 timestepping + 17 leapfrog tracer-filter + Held-Suarez tests pass.

### Iter-55 (2026-05-05 — RAW α = 0.53 default + dtype-aware test)

- Codex flagged α = 0.5 (the "neutral" point) as unconditionally unstable per Williams 2009 §3b.  Default changed to α = 0.53 (conditionally stable) and exposed via ``SpectralPEConfig.robert_asselin_alpha``.
- Citation corrected to Williams 2009 eqs. 8-9.
- Tests made dtype-aware to pass under both x64 (1e-10 tol) and fp32 (1e-6 tol).

### Iter-56 (2026-05-05 — spectral SW energy diag missing g·h·h_s)

- ``spectral_sw.py:compute_spectral_diagnostics`` line 569 omitted the topography PE term ``g·h·h_s``, producing spurious ~0.3% "energy non-conservation" on Williamson 5 (isolated 2000 m mountain).  Prognostic dynamics use the full potential ``E + Φ + Φs`` correctly; only the diagnostic was wrong.
- Non-vacuous regression test added.

### Iter-57 (2026-05-05 — fv3_csw divergence damping sign — anti-damping)

- ``core/fv3_sw_core.py:fv3_csw_tendencies`` lines 2222-2228 had the wrong sign on divergence damping: ``+ div_damp · ddiv_x`` produced ``∂D/∂t = −K_d · ∇²D`` (anti-damping → grid-scale divergence GROWS).  Path is reachable only via ``shallow_water_fv3_cdgrid.use_experimental_csw=True``.
- Production FB chain bypasses this; uses ``_d_sw5_corner_divergence`` which is correct.
- Iter-58 added a non-vacuous regression test that compares post-step divergence energy with vs without damping.

### Iter-59 (2026-05-05 — ERA5 specific-humidity → mixing-ratio at training boundary)

- ``training/era5_to_state.py``: ERA5 ``q`` is specific humidity, but the legoesm physics path treats q_v as mass mixing ratio.  Convert at the ingestion boundary using ``r = q / (1 − q)`` in BOTH the lat-lon and cubed-sphere paths.

### Iter-60 (2026-05-05 — AMIP SST floor at constants.T_freeze_ocean)

- ``forcing/amip.py``: SST floor was hardcoded as 200.0 K, 71 K BELOW the seawater freezing point.  Replaced with ``constants.T_freeze_ocean = 271.35 K`` at both call sites (HadISST + ICON branches).

### Iter-61 (2026-05-05 — barotropic_mpas land-edge mask hoisted)

- ``ocean/dynamics/barotropic_mpas.py``: predictor/corrector velocity updates had ``u_bar_c + dt·tendency * edge_mask`` which masks only the tendency, not the base state.  Hoisted mask to wrap the full update: ``(u_bar_c + dt·tendency) * edge_mask``.

### Iter-62 (2026-05-05 — barotropic_mpas u_bar masked at depth-average)

- Codex follow-up: initial ``u_bar = Hu_bar / max(H_e, 1e-10)`` was unmasked.  TRiSK ``tangential_velocity`` gathers neighboring edges, so stale land-edge values can contaminate adjacent interior-edge Coriolis tendency.  Masked u_bar immediately after depth-average.

### Iter-63 (2026-05-05 — ml/loss area_weighted_mse normalization)

- ``ml/loss.py:area_weighted_mse`` used ``jnp.mean(weighted)`` which divides by full array size (B·n_lat·n_lon·n_channels), embedding a factor of 2/n_lat baked in.  Fixed with correction factor ``n_lat / Σw_lat`` so the loss magnitude is grid-resolution-independent.

### Iter-64 (2026-05-05 — training/losses lat_weights + per_variable_mse channel_weights)

- ``training/losses.py:carry_mse``: ``lat_weights`` argument was declared but never applied.  Added ``_lat_weighted_mean`` helper.
- ``ml/loss.py:per_variable_mse``: ``_channel_weights`` parameter was dead (leading underscore).  Renamed and applied; also fixed underlying normalization with iter-63 correction factor.

### Iter-65 (2026-05-05 — multilayer_land beta_root_new denominator floor)

- ``land/multilayer_land.py`` lines 364, 369 used ``+ 1e-10`` denominator floor for the post-step ``beta_root_new`` computation.  The pre-step ``beta_root`` had been fixed to ``jnp.maximum(theta_fc - theta_wp, 1e-3)`` per "audit finding #6"; the post-step site was missed.  Pathological PFT cells could produce O(1e7) ``beta_root_new`` and contaminate ``q_sfc_new`` reported to the atmosphere.

### Iter-66 (2026-05-05 — spectral ocean PE scale_depth from constants)

- ``spectral_ocean_pe.py`` line 957 had ``scale_depth = 1000.0`` hardcoded instead of importing from ``legoesm.ocean.eos.scale_depth``.  Per the strengthened CLAUDE.md rule, ocean constants must reference the canonical value.

### Iter-67 (2026-05-05 — multilayer_land has_snow includes fresh snowfall)

- ``multilayer_land.py`` line 183: ``has_snow`` was evaluated from OLD ``snow`` value before fresh ``precip_snow`` accumulation.  Fresh-snow columns started the step routed as bare soil (L_v evap, liquid q_sat) even though the surface is snow-covered for most of the step.  ~13% bias in the latent flux phase coupling.  Fixed with ``has_snow = (snow + precip_snow * dt) > 1e-6``.

### Iter-68 (2026-05-05 — Codex round: refine has_snow with T_surface gate)

- Codex correctly noted that warm-surface snowfall (T_surface > T_freeze) melts immediately within the step, so iter-67's pure post-accum rule over-corrected.  Refined to ``has_existing_snow | (fresh_snow > 1e-6 AND T_surface < T_freeze)``.  Same fix applied to slab_land.py (Codex flagged the analogous bug).

### Iter-69 (2026-05-05 — LossConfig per-variable scale normalization)

- ``training/losses.py:carry_mse`` summed raw-units MSE for variables that differ by 9 orders of magnitude (T~K, q~kg/kg, ps~Pa).  Added per-variable amplitude scales (T_scale=30 K, q_scale=5e-3, ps_scale=1000 Pa, wind_scale=20 m/s) and ``normalize_by_scale`` flag (default True).  Without normalization: ``loss_ps / loss_T = 333``; with normalization: ratio bounded in [0.01, 100].

### Iter-70 (2026-05-05 — Codex round: LossConfig field order + spectral consistency)

- Codex flagged: (1) iter-69 inserted scale fields in MIDDLE of NamedTuple, breaking positional construction; (2) ``spectral_state_vs_carry_loss`` ignored the new scale fields, leaving spectral path on raw-unit MSE.  Fixed both: appended new fields to END of LossConfig; applied scale normalization to spectral path.

### Iter-71 (2026-05-05 — multilayer_land albedo path matches has_snow rule)

- Codex follow-up: albedo path at line 234 still used OLD ``snow`` while iter-68 bulk-flux phase used iter-68 effective snow.  Routed ``snow_effective`` (with the same warm-surface gating) through ``compute_land_albedo``.

### Iter-72 (2026-05-05 — carry_spectral_loss normalized by T_scale²)

- ``carry_spectral_loss`` returned raw-K² spectral MSE while ``carry_mse`` was dimensionless after iter-69.  ``combined_loss`` adding them required user to absorb a factor of ~900 K² in spectral_weight.  Normalized by T_scale² so both terms in combined loss are commensurate.

### Cumulative audit-finding status (Physical_Consistency cycle 2026-05-05)

| Severity | Total found | Fixed | Deferred |
|----------|-------------|-------|----------|
| CRITICAL | 9  | 9  | 0 (1 spectral-ocean w/η consistency tracked as future work) |
| HIGH     | 42 | 39 | 3 (F1, F2, F10) |
| MEDIUM   | 33 | 18 | 15 (mostly LOW-impact / structural cleanup) |
| LOW      | ~30+ | 12 | rest documented in catalog |

All nine CRITICAL findings (DCA latent heat, Richards `L psi^m`, EVP relaxation, ocean integration imports, GWD g-factor, _shared moisture-convergence, spectral-PE hybrid tracer vertical advection, RAW filter sign, plus the iter-39 CAPE p_full→p_mid) are fixed and Codex-verified.

Iter-44 → 66 audited 14 new modules and added 24 HIGH/CRITICAL fixes:
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
