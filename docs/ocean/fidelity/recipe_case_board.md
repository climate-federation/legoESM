# Ocean case board

**GENERATED — do not edit by hand.** Source: `packages/ocean/legoesm/ocean/fidelity/recipe_case_board.py`; regenerate with `scripts/validate/ocean_fidelity/build_recipe_case_board.py`. Completeness + freshness enforced by `tests/ocean/fidelity/test_recipe_case_board.py`.

One unified inventory of every ocean case (idealized + oracle-comparison, no distinction): what it tests, whether a true oracle exists in another model, and — per recipe — whether legoESM reproduces it. One experiment run under many recipes = many result rows (below) / many filled cells (matrix), NOT many cases. Holes (`⬜ todo`) are the roadmap; `⛔ blocked` cites the issue. The **tests** column is a physics summary so duplicate/overlapping cases are visible (use it to retire repeats).

**27 cases, 30 case×recipe results** — ✅ verified: 3, ⛔ blocked: 1, 🟡 partial: 1, ⬜ todo: 25

**by reference kind** — 🔬 oracle: 10, 📐 analytic: 7, 📄 published: 8, —: 2

Reference kind = truth strength: 🔬 oracle (runnable model + data) · 📐 analytic (closed-form) · 📄 published (paper/figures, no runnable data) · — none. An oracle is the only kind that supports a tendency/field match.

## Detail (one row per run = case × grid × recipe)

| case | tests | reference | grid | recipe | status | note |
|---|---|---|---|---|---|---|
| `baroclinic_adjustment` | stratified baroclinic instability; eddy growth + inverse cascade (unforced) | 🔬 oracle: Oceananigans WENOVectorInvariant(9)+ImplicitFreeSurface | latlon_cgrid (β-plane) | `oceananigans_v1 (+power_law)` | ✅ verified | EKE 0.86×, cascade k_e 5.11 vs 5.19, stable 40 d, no backstop — PR #672 |
| `barotropic_gyre` | wind-driven barotropic gyre spin-up (steady, laminar window) | 🔬 oracle: Oceananigans ImplicitFreeSurface gyre | latlon_cgrid | `oceananigans_v1` | ✅ verified | surface-u pattern corr 0.93 days 1–20 (kinematic wind-stress unit fix) |
| `internal_tide` | tidal flow over topography; internal-tide generation (Flat-y, free surface) | 🔬 oracle: Oceananigans internal_tide | latlon_cgrid (meridionally_flat) | `oceananigans_v1 (meridionally_flat+oceananigans_up3)` | ✅ verified | corr ≥0.6 through ~2 days; dispersion cleared by the UP3 arm — #576 |
| `silvestri_jet_forced` | forced (τ=50 d restoring) eddy-resolving baroclinic jet; Silvestri 2024 §5 | 🔬 oracle: Oceananigans Silvestri 2024 §5 | latlon_cgrid (spherical channel) | `oceananigans_v1 (+power_law)` | ⛔ blocked | over-energizes ~7× under sustained restoring (eddy-mean equilibration) — issue #673 |
| `bickley_jet` | single-layer Bickley-jet barotropic instability; vortex roll-up (eddy dissipation) | 🔬 oracle: Oceananigans WENOVectorInvariant | latlon_cgrid (unit sphere) | `oceananigans_v1` | 🟡 partial | tracks oracle to t6 (corr 0.97) then under-dissipates; enstrophy ~1.2–1.9× late |
| `acc_channel` | ACC-like re-entrant channel with ridge; eddy saturation + transport | 🔬 oracle: Veros ACC | latlon_cgrid | `veros_faithful_v1` | ⬜ todo | extensive Veros ACC work exists — re-confirm via scorecard (also intended: mpas) |
| `advection_gyre` | passive-tracer advection in a wind-driven gyre | 🔬 oracle: MITgcm tutorial_advection_in_gyre | latlon_cgrid | `mitgcm_v1` | ⬜ todo | comparison driver exists; not re-assessed |
| `baroclinic_gyre` | wind-driven regional gyre with surface restoring + thermal wind | — _(idealized — none)_ | latlon_regional | `default_wright_v1` | ⬜ todo | matrix-covered (also: mpas_regional) |
| `barotropic_wave` | barotropic gravity-wave propagation (Gaussian SSH) | 📐 analytic: gravity-wave dispersion relation | latlon_cgrid | `default_wright_v1` | ⬜ todo | matrix-covered (also: cubed_sphere, mpas, spectral) |
| `dino` | Double-gyre Idealized North Ocean; diabatic basin | 📄 published: Kamm et al. 2025 / NEMO-DINO (no runnable data yet) | latlon_cgrid | `nemo_dino_v1` | ⬜ todo | DINO reference (also: mpas_regional) |
| `eady_instability` | Eady instability from a meridional temperature front | 📐 analytic: Eady growth rate | latlon_channel | `eady_weno5_v1` | ⬜ todo | POSSIBLE DUPLICATE of eady_uniform — review for retire |
| `eady_uniform` | classical Eady instability (uniform N², linear shear), re-entrant channel | 📐 analytic: Eady growth rate | latlon_channel | `eady_weno5_v1` | ⬜ todo | analytic growth-rate benchmark (also: mpas_channel) |
| `geostrophic_adjustment` | geostrophic adjustment from a temperature front; free-surface coupling | 🔬 oracle: Oceananigans | latlon_cgrid | `oceananigans_v1` | ⬜ todo | comparison driver exists; not re-assessed post-#501 (also intended: cubed_sphere, mpas) |
| `global_barotropic_wind` | global 3-belt wind-stress barotropic circulation | — _(idealized — none)_ | latlon_cgrid | `default_wright_v1` | ⬜ todo | matrix-covered (also: mpas) |
| `global_omip` | global forced (OMIP) ocean climate; SST/MOC vs reference GCM | 🔬 oracle: Veros / NEMO (OMIP) | latlon_cgrid | `veros_faithful_v1` | ⬜ todo | Veros global transfer (1°/4°/flexible) — re-confirm via scorecard |
|  |  |  | latlon_cgrid | `legoesm_nemo_like_v1` | ⬜ todo | NEMO-style dycore (barotropic solver is legoESM's own generic arm, not NEMO's) — re-confirm via scorecard |
|  |  |  | latlon_cgrid (tripole) | `omip_nemo_match_tripole_v1` | ⬜ todo | tripole eORCA025 NEMO-climate-match (SST RMSE ~1.15) |
|  |  |  | mpas (ico6) | `omip_nemo_match_mpas_v1` | ⬜ todo | MPAS ico6 NEMO-climate-match (SST RMSE ~0.84, best grid) |
| `global_overturning` | global baroclinic overturning (idealized THC) | 📄 published: Wolfe & Cessi 2010 (idealized THC) | latlon_cgrid | `legoesm_linear_v1` | ⬜ todo | idealized THC benchmark (also: mpas) |
| `gridmode_decay` | 2Δx grid-mode decay rate (numerical dissipation probe) | 🔬 oracle: Oceananigans | latlon_cgrid | `oceananigans_v1` | ⬜ todo | diagnostic comparison; not re-assessed |
| `held_larichev` | Held–Larichev eddying channel; APE→KE cascade saturation (k⁻³) | 📄 published: Held & Larichev 1996 (spectral scaling law) | latlon_channel | `default_wright_v1` | ⬜ todo | spectral-slope benchmark (also: mpas_channel) |
| `inertia_gravity_wave` | Poincaré (inertia-gravity) wave | 📐 analytic: Poincaré dispersion (Bishnu 2024 cross-check) | latlon_cgrid | `default_wright_v1` | ⬜ todo | analytic benchmark; matrix-covered (also: cubed_sphere, mpas) |
| `isomip_plus` | ISOMIP+ ice-shelf cavity benchmark | 📄 published: ISOMIP+ (Asay-Davis 2016) intercomparison | latlon_regional | `legoesm_linear_v1` | ⬜ todo | ice-shelf-cavity benchmark |
| `lock_exchange` | density-driven gravity current; RPE mixing | 📄 published: Petersen et al. 2015 (RPE intercomparison) | latlon_cgrid | `default_wright_v1` | ⬜ todo | RPE diagnostic benchmark; matrix-covered |
| `munk_gyre` | Munk gyre; lateral-viscosity western boundary current | 📐 analytic: Munk boundary-layer width | latlon_regional | `default_wright_v1` | ⬜ todo | WBC benchmark (also: mpas_regional) |
| `neverworld2_lite` | idealized global basin + ACC band (NeverWorld2-lite) | 📄 published: NeverWorld2 intercomparison | latlon_cgrid | `legoesm_linear_v1` | ⬜ todo | idealized reference |
| `overflow` | dense water descending a bathymetric slope | 📄 published: DOME / Petersen 2015 overflow benchmark | latlon_cgrid | `default_wright_v1` | ⬜ todo | matrix-covered (also: cubed_sphere) |
| `phillips_two_layer` | Phillips two-layer baroclinic instability | 📐 analytic: Phillips two-layer instability criterion | latlon_cgrid | `default_wright_v1` | ⬜ todo | matrix-covered (also: cubed_sphere, mpas) |
| `rest_state` | rest-state stability / PGF balance (η≈0); conservation control | 📐 analytic: exact rest state (η≈0) | latlon_cgrid | `default_wright_v1` | ⬜ todo | matrix-covered (also: cubed_sphere, mpas) — confirm status |
| `stommel_gyre_tracer` | passive tracer in a wind-driven Stommel gyre | 📄 published: Hecht et al. 2000 | latlon_cgrid | `default_wright_v1` | ⬜ todo | matrix-covered (also: mpas) |

## Matrix (case × grid × recipe)

Rows are `case @ grid` (grid + recipe = the numerical setup of a run); columns are recipes.
Legend: ✅ verified · 🟩 works · 🟡 partial · ⛔ blocked · ⬜ todo · blank = not run

| case @ grid | oceananigans_v1 (+power_law) | oceananigans_v1 | oceananigans_v1 (meridionally_flat+oceananigans_up3) | veros_faithful_v1 | mitgcm_v1 | legoesm_nemo_like_v1 | omip_nemo_match_tripole_v1 | omip_nemo_match_mpas_v1 | default_wright_v1 | eady_weno5_v1 | legoesm_linear_v1 | nemo_dino_v1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `baroclinic_adjustment` @ latlon_cgrid (β-plane) | ✅ |  |  |  |  |  |  |  |  |  |  |  |
| `barotropic_gyre` @ latlon_cgrid |  | ✅ |  |  |  |  |  |  |  |  |  |  |
| `internal_tide` @ latlon_cgrid (meridionally_flat) |  |  | ✅ |  |  |  |  |  |  |  |  |  |
| `silvestri_jet_forced` @ latlon_cgrid (spherical channel) | ⛔ |  |  |  |  |  |  |  |  |  |  |  |
| `bickley_jet` @ latlon_cgrid (unit sphere) |  | 🟡 |  |  |  |  |  |  |  |  |  |  |
| `acc_channel` @ latlon_cgrid |  |  |  | ⬜ |  |  |  |  |  |  |  |  |
| `advection_gyre` @ latlon_cgrid |  |  |  |  | ⬜ |  |  |  |  |  |  |  |
| `baroclinic_gyre` @ latlon_regional |  |  |  |  |  |  |  |  | ⬜ |  |  |  |
| `barotropic_wave` @ latlon_cgrid |  |  |  |  |  |  |  |  | ⬜ |  |  |  |
| `dino` @ latlon_cgrid |  |  |  |  |  |  |  |  |  |  |  | ⬜ |
| `eady_instability` @ latlon_channel |  |  |  |  |  |  |  |  |  | ⬜ |  |  |
| `eady_uniform` @ latlon_channel |  |  |  |  |  |  |  |  |  | ⬜ |  |  |
| `geostrophic_adjustment` @ latlon_cgrid |  | ⬜ |  |  |  |  |  |  |  |  |  |  |
| `global_barotropic_wind` @ latlon_cgrid |  |  |  |  |  |  |  |  | ⬜ |  |  |  |
| `global_omip` @ latlon_cgrid |  |  |  | ⬜ |  | ⬜ |  |  |  |  |  |  |
| `global_omip` @ latlon_cgrid (tripole) |  |  |  |  |  |  | ⬜ |  |  |  |  |  |
| `global_omip` @ mpas (ico6) |  |  |  |  |  |  |  | ⬜ |  |  |  |  |
| `global_overturning` @ latlon_cgrid |  |  |  |  |  |  |  |  |  |  | ⬜ |  |
| `gridmode_decay` @ latlon_cgrid |  | ⬜ |  |  |  |  |  |  |  |  |  |  |
| `held_larichev` @ latlon_channel |  |  |  |  |  |  |  |  | ⬜ |  |  |  |
| `inertia_gravity_wave` @ latlon_cgrid |  |  |  |  |  |  |  |  | ⬜ |  |  |  |
| `isomip_plus` @ latlon_regional |  |  |  |  |  |  |  |  |  |  | ⬜ |  |
| `lock_exchange` @ latlon_cgrid |  |  |  |  |  |  |  |  | ⬜ |  |  |  |
| `munk_gyre` @ latlon_regional |  |  |  |  |  |  |  |  | ⬜ |  |  |  |
| `neverworld2_lite` @ latlon_cgrid |  |  |  |  |  |  |  |  |  |  | ⬜ |  |
| `overflow` @ latlon_cgrid |  |  |  |  |  |  |  |  | ⬜ |  |  |  |
| `phillips_two_layer` @ latlon_cgrid |  |  |  |  |  |  |  |  | ⬜ |  |  |  |
| `rest_state` @ latlon_cgrid |  |  |  |  |  |  |  |  | ⬜ |  |  |  |
| `stommel_gyre_tracer` @ latlon_cgrid |  |  |  |  |  |  |  |  | ⬜ |  |  |  |
