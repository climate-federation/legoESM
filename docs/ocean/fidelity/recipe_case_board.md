# Ocean case board

**GENERATED — do not edit by hand.** Source: `packages/ocean/legoesm/ocean/fidelity/recipe_case_board.py`; regenerate with `scripts/validate/ocean_fidelity/build_recipe_case_board.py`. Completeness + freshness enforced by `tests/ocean/fidelity/test_recipe_case_board.py`.

One unified inventory of every ocean case (idealized + oracle-comparison, no distinction): what it tests, whether a true oracle exists in another model, and — per recipe — whether legoESM reproduces it. One experiment run under many recipes = many result rows (below) / many filled cells (matrix), NOT many cases. Holes (`⬜ todo`) are the roadmap; `⛔ blocked` cites the issue. The **tests** column is a physics summary so duplicate/overlapping cases are visible (use it to retire repeats).

**27 cases, 30 case×recipe results** — ✅ verified: 3, ⛔ blocked: 1, 🟡 partial: 1, ⬜ todo: 25

## Detail (one row per case × recipe)

| case | tests | grids | oracle | recipe | status | note |
|---|---|---|---|---|---|---|
| `baroclinic_adjustment` | stratified baroclinic instability; eddy growth + inverse cascade (unforced) | latlon_cgrid (β-plane) | Oceananigans WENOVectorInvariant(9)+ImplicitFreeSurface | `oceananigans + power_law stack` | ✅ verified | EKE 0.86×, cascade k_e 5.11 vs 5.19, stable 40 d, no backstop — PR #672 |
| `barotropic_gyre` | wind-driven barotropic gyre spin-up (steady, laminar window) | latlon_cgrid | Oceananigans ImplicitFreeSurface gyre | `oceananigans (implicit_cn)` | ✅ verified | surface-u pattern corr 0.93 days 1–20 (kinematic wind-stress unit fix) |
| `internal_tide` | tidal flow over topography; internal-tide generation (Flat-y, free surface) | latlon_cgrid (meridionally_flat) | Oceananigans internal_tide | `oceananigans (meridionally_flat + upwind3)` | ✅ verified | corr ≥0.6 through ~2 days; dispersion cleared by upwind3 — #576 |
| `silvestri_jet_forced` | forced (τ=50 d restoring) eddy-resolving baroclinic jet; Silvestri 2024 §5 | latlon_cgrid (spherical channel) | Oceananigans Silvestri 2024 §5 | `oceananigans + power_law stack` | ⛔ blocked | over-energizes ~7× under sustained restoring (eddy-mean equilibration) — issue #673 |
| `bickley_jet` | single-layer Bickley-jet barotropic instability; vortex roll-up (eddy dissipation) | latlon_cgrid (unit sphere) | Oceananigans WENOVectorInvariant | `oceananigans (implicit_cn)` | 🟡 partial | tracks oracle to t6 (corr 0.97) then under-dissipates; enstrophy ~1.2–1.9× late |
| `acc_channel` | ACC-like re-entrant channel with ridge; eddy saturation + transport | latlon_cgrid, mpas | Veros ACC | `veros_faithful_v1` | ⬜ todo | extensive Veros ACC work exists — re-confirm via scorecard |
| `advection_gyre` | passive-tracer advection in a wind-driven gyre | latlon_cgrid | MITgcm tutorial_advection_in_gyre | `mitgcm` | ⬜ todo | comparison driver exists; not re-assessed |
| `baroclinic_gyre` | wind-driven regional gyre with surface restoring + thermal wind | latlon_regional, mpas_regional | _(idealized — none)_ | `default_wright_v1` | ⬜ todo | matrix-covered |
| `barotropic_wave` | barotropic gravity-wave propagation (Gaussian SSH) | latlon_cgrid, cubed_sphere, mpas, spectral | _(idealized — none)_ | `default_wright_v1` | ⬜ todo | analytic dispersion; matrix-covered |
| `dino` | Double-gyre Idealized North Ocean (Kamm et al. 2025); diabatic basin | latlon_cgrid, mpas_regional | Kamm et al. 2025 / NEMO-DINO | `nemo_dino_v1` | ⬜ todo | DINO reference |
| `eady_instability` | Eady instability from a meridional temperature front | latlon_channel | _(idealized — none)_ | `eady_weno5_v1` | ⬜ todo | POSSIBLE DUPLICATE of eady_uniform — review for retire |
| `eady_uniform` | classical Eady instability (uniform N², linear shear), re-entrant channel | latlon_channel, mpas_channel | analytic Eady growth rate | `eady_weno5_v1` | ⬜ todo | analytic growth-rate benchmark |
| `geostrophic_adjustment` | geostrophic adjustment from a temperature front; free-surface coupling | latlon_cgrid, cubed_sphere, mpas | Oceananigans | `oceananigans` | ⬜ todo | comparison driver exists; not re-assessed post-#501 |
| `global_barotropic_wind` | global 3-belt wind-stress barotropic circulation | latlon_cgrid, mpas | _(idealized — none)_ | `default_wright_v1` | ⬜ todo | matrix-covered |
| `global_omip` | global forced (OMIP) ocean climate; SST/MOC vs reference GCM | latlon_cgrid (tripole), mpas | Veros / NEMO (OMIP) | `veros_faithful_v1` | ⬜ todo | Veros global transfer (1°/4°/flexible) — re-confirm via scorecard |
|  |  |  |  | `nemo_v1` | ⬜ todo | NEMO-faithful dycore — re-confirm via scorecard |
|  |  |  |  | `omip_nemo_match_tripole_v1` | ⬜ todo | tripole eORCA025 NEMO-climate-match (SST RMSE ~1.15) |
|  |  |  |  | `omip_nemo_match_mpas_v1` | ⬜ todo | MPAS ico6 NEMO-climate-match (SST RMSE ~0.84, best grid) |
| `global_overturning` | global baroclinic overturning (Wolfe & Cessi 2010 idealized THC) | latlon_cgrid, mpas | Wolfe & Cessi 2010 (idealized) | `legoesm_linear_v1` | ⬜ todo | idealized THC benchmark |
| `gridmode_decay` | 2Δx grid-mode decay rate (numerical dissipation probe) | latlon_cgrid | Oceananigans | `oceananigans` | ⬜ todo | diagnostic comparison; not re-assessed |
| `held_larichev` | Held–Larichev eddying channel; APE→KE cascade saturation (k⁻³) | latlon_channel, mpas_channel | Held & Larichev (spectral law) | `default_wright_v1` | ⬜ todo | spectral-slope benchmark |
| `inertia_gravity_wave` | Poincaré (inertia-gravity) wave; Bishnu et al. 2024 | latlon_cgrid, cubed_sphere, mpas | analytic / Bishnu 2024 | `default_wright_v1` | ⬜ todo | analytic benchmark; matrix-covered |
| `isomip_plus` | ISOMIP+ ice-shelf cavity benchmark (Asay-Davis 2016) | latlon_regional | ISOMIP+ intercomparison | `legoesm_linear_v1` | ⬜ todo | ice-shelf-cavity benchmark |
| `lock_exchange` | density-driven gravity current; RPE mixing (Petersen 2015) | latlon_cgrid | Petersen et al. 2015 (RPE benchmark) | `default_wright_v1` | ⬜ todo | RPE diagnostic benchmark; matrix-covered |
| `munk_gyre` | Munk gyre; lateral-viscosity western boundary current | latlon_regional, mpas_regional | analytic Munk BL width | `default_wright_v1` | ⬜ todo | WBC benchmark |
| `neverworld2_lite` | idealized global basin + ACC band (NeverWorld2-lite) | latlon_cgrid | NeverWorld2 (idealized) | `legoesm_linear_v1` | ⬜ todo | idealized reference |
| `overflow` | dense water descending a bathymetric slope | latlon_cgrid, cubed_sphere | _(idealized — none)_ | `default_wright_v1` | ⬜ todo | matrix-covered |
| `phillips_two_layer` | Phillips two-layer baroclinic instability | latlon_cgrid, cubed_sphere, mpas | _(idealized — none)_ | `default_wright_v1` | ⬜ todo | matrix-covered |
| `rest_state` | rest-state stability / PGF balance (η≈0); conservation control | latlon_cgrid, cubed_sphere, mpas | _(idealized — none)_ | `default_wright_v1` | ⬜ todo | analytic truth (η≈0); covered by matrix — confirm status |
| `stommel_gyre_tracer` | passive tracer in a wind-driven Stommel gyre (Hecht 2000) | latlon_cgrid, mpas | _(idealized — none)_ | `default_wright_v1` | ⬜ todo | matrix-covered |

## Matrix (case × recipe)

Legend: ✅ verified · 🟩 works · 🟡 partial · ⛔ blocked · ⬜ todo · blank = not run

| case | oceananigans + power_law stack | oceananigans (implicit_cn) | oceananigans (meridionally_flat + upwind3) | oceananigans | veros_faithful_v1 | mitgcm | nemo_v1 | omip_nemo_match_tripole_v1 | omip_nemo_match_mpas_v1 | default_wright_v1 | eady_weno5_v1 | legoesm_linear_v1 | nemo_dino_v1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `baroclinic_adjustment` | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |
| `barotropic_gyre` |  | ✅ |  |  |  |  |  |  |  |  |  |  |  |
| `internal_tide` |  |  | ✅ |  |  |  |  |  |  |  |  |  |  |
| `silvestri_jet_forced` | ⛔ |  |  |  |  |  |  |  |  |  |  |  |  |
| `bickley_jet` |  | 🟡 |  |  |  |  |  |  |  |  |  |  |  |
| `acc_channel` |  |  |  |  | ⬜ |  |  |  |  |  |  |  |  |
| `advection_gyre` |  |  |  |  |  | ⬜ |  |  |  |  |  |  |  |
| `baroclinic_gyre` |  |  |  |  |  |  |  |  |  | ⬜ |  |  |  |
| `barotropic_wave` |  |  |  |  |  |  |  |  |  | ⬜ |  |  |  |
| `dino` |  |  |  |  |  |  |  |  |  |  |  |  | ⬜ |
| `eady_instability` |  |  |  |  |  |  |  |  |  |  | ⬜ |  |  |
| `eady_uniform` |  |  |  |  |  |  |  |  |  |  | ⬜ |  |  |
| `geostrophic_adjustment` |  |  |  | ⬜ |  |  |  |  |  |  |  |  |  |
| `global_barotropic_wind` |  |  |  |  |  |  |  |  |  | ⬜ |  |  |  |
| `global_omip` |  |  |  |  | ⬜ |  | ⬜ | ⬜ | ⬜ |  |  |  |  |
| `global_overturning` |  |  |  |  |  |  |  |  |  |  |  | ⬜ |  |
| `gridmode_decay` |  |  |  | ⬜ |  |  |  |  |  |  |  |  |  |
| `held_larichev` |  |  |  |  |  |  |  |  |  | ⬜ |  |  |  |
| `inertia_gravity_wave` |  |  |  |  |  |  |  |  |  | ⬜ |  |  |  |
| `isomip_plus` |  |  |  |  |  |  |  |  |  |  |  | ⬜ |  |
| `lock_exchange` |  |  |  |  |  |  |  |  |  | ⬜ |  |  |  |
| `munk_gyre` |  |  |  |  |  |  |  |  |  | ⬜ |  |  |  |
| `neverworld2_lite` |  |  |  |  |  |  |  |  |  |  |  | ⬜ |  |
| `overflow` |  |  |  |  |  |  |  |  |  | ⬜ |  |  |  |
| `phillips_two_layer` |  |  |  |  |  |  |  |  |  | ⬜ |  |  |  |
| `rest_state` |  |  |  |  |  |  |  |  |  | ⬜ |  |  |  |
| `stommel_gyre_tracer` |  |  |  |  |  |  |  |  |  | ⬜ |  |  |  |
