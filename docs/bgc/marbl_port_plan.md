# MARBL port plan for legoESM ocean BGC

Status: design, no code yet (2026-10-15). Branch `ggn/bgc_modelling`.

## Decisions (user-confirmed 2026-10-15)
- Full MARBL base tracer set (32 tracers), developed first in a 1D column, then 3D, then global.
- Carbon is the base currency (MARBL-style): fixed C:N per pool, variable P:C for autotrophs.
- `marbl_lite.py` and `marbl_cesm_simplified.py` are deleted and replaced by a single `marbl` scheme.
  `npzd` stays as the simple scheme. `npzd_v2` / `marbl_cesm` literals are removed from
  `BiogeoConfig`, `run_omip.py`, `__init__.py`, tests (no compat aliases, per CLAUDE.md).
- No model run before all observational inputs are on disk and verified.
- (2026-10-15) MARBL tracers in mmol/m^3 (MARBL convention); convert x1e-3 at the
  carbonate / gas-exchange interface, which stay in mol/m^3.
- (2026-10-15) Column diffusivities from KPP. KPP `_boundary_layer_depth` (`kpp.py:45`) currently
  ignores u* and B_f; fix it to the LMD94 bulk-Richardson form (with w_s, unresolved shear V_t^2)
  before using it. This changes 3D ocean results too. Drive KPP with potential density
  (`wright_eos(T, S, 0)`), not in-situ density.

- (2026-10-15) Source of truth: MARBL checkout `/work/bb1596/b384036/MARBL` @ f00d642. Process code
  ported from that commit (lvariable_NtoC = .false. branch); default parameters from
  `defaults/settings_cesm2.1.yaml` (fixed N:C, 32 tracers, published CESM2 tuning). Hard-coded
  Fortran parameters (Q, Qp_zoo, DOM remin rates, refractory fractions, ...) from
  `src/marbl_settings_mod.F90` / `marbl_pft_mod.F90` at that commit. NOTE: settings_latest.yaml and
  the Fortran fallback defaults differ from cesm2.1 (variable N:C, retuned Fe scavenging etc.).

## Progress log
- 2026-10-15 KPP fix written (untested, awaiting `scripts/slurm/bgc_stage1_tests.sbatch`):
  LMD94 V_t^2 with w_s (`kpp_turbulent_velocity_scale`), scalar flux-profile branches,
  locally referenced N^2 / Ri_b (`eos.locally_referenced_buoyancy_frequency`), linear crossing
  interpolation, `h_bl` in `VerticalMixingOutput`, shared `kpp_surface_fluxes`.
  API: `kpp_vertical_mixing` no longer takes `rho` / `h_bl_prev`, takes `eos_fn`; `KPPConfig.c_b`
  removed; new fields `a_s, zeta_s, phi_stable, phi_unstable, beta_T`. Not done: Ekman / M-O
  limits on h_bl (eq. 24). Still in-situ: Richardson (PP) scheme and enhanced-diffusion convection.
- 2026-10-15 written (untested): `carbonate.py` (full MARBL co2calc, custom_root Newton),
  `gas_exchange.py` (W14 for O2 too), `eos.potential_temperature`, `marbl/` package
  (config, light, autotrophs, zooplankton, dom, iron, nitrogen, particles, surface, interface),
  `ocean/column/model.py` (KPP + MARBL column), tests `test_marbl.py`, `test_ocean_column.py`.
  Old schemes npzd_v2 / marbl_cesm / marbl_lite are KEPT until stage 2 (user decision 2026-10-15:
  chl / ORAS5 experiment scripts depend on them). Only change to them: marbl_cesm's O2 flux call uses
  the MARBL-consistent `air_sea_o2_flux` (W14 k = 0.251, W14 Schmidt) instead of W92 (0.39).
  MARBL not yet dispatched from `BiogeoConfig` (3-D coupling = stage 2).
- Still to do for the first column run: station data builder (WOA23 T/S/nutrients/O2, GLODAP DIC/ALK/
  PI_TCO2 -> ALT_CO2), surface forcing source (ERA5 on Levante, to be confirmed by the survey job),
  dust/Fe deposition values, run script + sbatch.
- Column design consequence: with u=v=0 KPP h_bl under pure wind is only
  Ri_c Cv kappa u* V_tc / N (~0.6 m for u*=1e-2, N=1e-2). The 1D column must carry u, v
  (wind stress + Coriolis + implicit A_v) so resolved shear drives entrainment.

## Reusable pieces found (2026-10-15 survey)
- Implicit diffusion: `ocean/physics/vertical_mixing/implicit_solver.py:49` `implicit_vertical_diffusion_ocean`
  (k=0 surface, zero-flux BCs, batched). Surface fluxes applied separately.
- Grid: `ocean/vertical.py` `create_ocean_z_star`. Shortwave: `ocean/physics/shortwave_penetration.py`.
- Wind stress from U10: `coupler/bulk_flux.py:99` `compute_most_fluxes` (stress sign = on atmosphere; negate).
- WOA reading: `ocean/init_woa.py` (annual only, private helpers -> needs public monthly/station wrapper).
- Existing SLURM template: `scripts/run_levante_gpu_scaling.sh` (GPU). Column jobs: CPU partition, account bb1596.

## Known defects in the current code (fix or delete during the port)
1. 3D model does not advect/diffuse/mix `state.biogeo` (`ocean_model_latlon_cgrid.py`, T/S only).
2. Fe overwritten with a fixed profile each step (`carbon_cycle.py:326-336`).
3. `marbl_cesm` adds biological DIC/ALK twice (v2 block + marbl block in `carbon_cycle.py`).
4. DOC remineralization subtracts DIC (`marbl_lite.py:422`, `marbl_cesm_simplified.py:440`).
5. POC sources in N units, DOC in C units.
6. PO4/DOP are Redfield copies of N terms.
7. Post-hoc softplus positivity + Chl clipping in `step_ocean_biogeochemistry`.
8. `O2_new` read via `'O2_new' in dir()`.
9. `run_omip.py`: JRA55 branches return 4-tuple, caller unpacks 5; BGC never stepped on JRA55 path;
   `NO3_target` lookup before WOA paths are stored.
10. `carbonate.py`: single-pass Follows solver, borate only, no pressure, no Omega.
11. `run_bgc_column.py`: scipy `solve_banded` + host clipping (not jit/grad-safe), stale docstring.

## Tracers (MARBL base set, 32)
Units: mmol/m^3 (Fe, spFe, diatFe, diazFe, Lig also mmol/m^3; Chl in mg/m^3), as in MARBL, so parameter
values can be compared 1:1 with MARBL defaults. Conversion from WOA/GLODAP (umol/kg) uses density
from `ocean.eos`. OPEN: confirm vs. repo-wide mol/m^3 convention used by `npzd`.

| Group | Tracers |
|---|---|
| Inorganic nutrients | PO4, NO3, SiO3, NH4, Fe, Lig |
| Gases / carbonate | O2, DIC, DIC_ALT_CO2, ALK, ALK_ALT_CO2 |
| Dissolved organic | DOC, DON, DOP, DOPr, DONr, DOCr |
| Zooplankton | zooC |
| Small phyto | spChl, spC, spP, spFe, spCaCO3 |
| Diatoms | diatChl, diatC, diatP, diatFe, diatSi |
| Diazotrophs | diazChl, diazC, diazP, diazFe |

Particulate pools (POC, P_CaCO3, P_SiO2, dust, P_iron) are NOT tracers: as in MARBL they are
computed diagnostically each step as a top-down flux through the column (`lax.scan` over levels),
with in-layer remineralization. No sinking CFL limit.

## Module layout (`src/legoesm/ocean/biogeochemistry/marbl/`)
| File | Content |
|---|---|
| `config.py` | `MarblConfig` NamedTuple (per-PFT params as small arrays), `MarblState` (32 tracers), `MarblDiagnostics` |
| `autotrophs.py` | T-function, Liebig nutrient limitation (N w/ NH4 pref, P incl. DOP, Fe, Si), Geider light/Chl, variable Qp, uptake, CaCO3 production, N2 fixation |
| `zooplankton.py` | grazing (MARBL default functional form + preferences), routing of grazed matter |
| `losses.py` | linear mortality, quadratic aggregation, routing to DOM / POC / zoo |
| `dom.py` | labile + refractory DOM remineralization, refractory photodegradation |
| `particles.py` | Armstrong ballast model, O2-dependent POC remin length, CaCO3 dissolution vs Omega, sediment burial / bottom flux |
| `iron.py` | ligand tracer, free-Fe scavenging, sedimentary + dust + hydrothermal sources |
| `nitrogen.py` | nitrification (light-inhibited), water-column + sediment denitrification, suboxic switch |
| `carbonate.py` (shared, upgraded) | Newton/Brent solver with full alkalinity (borate, water, PO4, SiO3), pressure correction, Omega calcite/aragonite; reused by `npzd` |
| `interface.py` | `marbl_source_sink(state, forcing, cfg) -> tendencies, diags`; surface fluxes via existing `gas_exchange.py` |

Constants go in `legoesm/constants.py` (e.g. Redfield-type fixed ratios only if they are physical constants;
tunable stoichiometry/rates stay in `MarblConfig`). Reuse: `gas_exchange.py`, Weiss/Lueker in `carbonate.py`,
`ocean.eos` for density, `diagnostics.column_integrals` pattern for budgets.

## Stage 1: 1D column
- Rewrite `scripts/run_bgc_column.py` in pure JAX: implicit vertical diffusion via tridiagonal solve
  (search for existing tridiagonal solver in `ocean/physics/` first), `lax.scan` time loop, no host clipping.
- Stations: BATS (31.7N, 64.2W), HOT/ALOHA (22.75N, 158W), Papa (50N, 145W).
- Physical forcing per station: T/S (WOA23 monthly), MLD climatology, surface PAR/shortwave, U10, dust/Fe deposition.
- Init: WOA23 NO3/PO4/SiO3/O2, GLODAPv2 DIC/ALK, Fe from GEOTRACES or MARBL-style analytic profile.

Acceptance tests (pytest, `JAX_ENABLE_X64=1`):
- C, N, P, Si, Fe, O2(+ALK) budgets close to round-off with closed boundaries; with fluxes, change == boundary fluxes.
- Non-negativity without clipping for 10 model years at all three stations.
- `jax.grad` of a scalar loss w.r.t. `MarblConfig` params finite and matches finite differences.
- Carbonate solver vs. reference values (Dickson / CO2SYS check values).
- 10-year seasonal cycle vs. station climatologies (chl, NPP, NO3, DIC, export).

## Stage 2: 3D
- Add BGC tracer transport (advection, diffusion, GM/Redi, implicit vertical mixing) reusing T/S code paths.
- Fix JRA55 path in `run_omip.py`; WOA23/GLODAP init for `marbl`; full BGC diagnostics output.
- Check global tracer inventories conserved under transport alone.

## Stage 3: global
- Other grids (cubed sphere, MPAS), MPI single-rank vs multi-rank equivalence, calibration vs WOA23,
  GLODAP, satellite Chl, NPP, SOCAT/Landschuetzer.

## Data status
- On Levante: satellite Chl (`/pool/data/ICDC/ocean/{globcolour_chloro,modis_*_chl}`), T/S (`en4`, `waghc`, `woce_climatology`).
- Download script: `scripts/fetch_bgc_obs.sh` (WOA23 nutrients/O2/T/S, GLODAPv2.2016b mapped + v2.2023 bottle). Not yet run/verified.
- Still missing: GEOTRACES Fe, BATS/HOT/Papa station data, NPP, SOCAT/Landschuetzer, dust/N deposition, rivers, MLD climatology.
