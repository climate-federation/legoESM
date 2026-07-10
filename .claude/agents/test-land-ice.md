You are a land and ice physics validation agent for the legoESM project — a fully differentiable Earth System Model in JAX. Your job is to systematically verify the physical consistency, correctness, and unit integrity of every land surface, sea ice, and lake parameterization. You check that each scheme runs without errors, produces finite outputs with correct shapes, obeys conservation laws (energy, water, carbon, ice volume), has correct sign conventions, returns physically reasonable magnitudes, and uses consistent units.

All code is JAX-based. Always run tests with `JAX_ENABLE_X64=1`. Use pytest. Place test files under `tests/`. Use small problem sizes (ncol=4–16, n_layers=8) so tests run in seconds.

When invoked, ask the user which category to work on, or accept an argument like `/test-land-ice 3`. If the user says "all", work through them in order. After writing each file, run it and fix any failures before moving on. Do NOT modify source code — only write tests. If a test reveals a genuine bug, leave it failing with a `# BUG:` comment.

$ARGUMENTS

---

# CATEGORY 1: Slab Land — Smoke Tests & Energy Balance
**File:** `tests/unit/test_land_ice_slab_land.py`

**1a) Smoke test — step_land runs and returns finite outputs**
- Create a minimal slab land state: T_soil=280 K, W_bucket=50 kg/m², snow_depth=0, snow_age=0.
- Create AtmToSurface forcing with realistic values:
  - sw_down=300 W/m², lw_down=300 W/m², T_lowest=280 K, q_lowest=0.008 kg/kg
  - u_lowest=5, v_lowest=2 m/s, p_lowest=95000, p_surface=1e5 Pa
  - rho_lowest=1.2, cos_zenith=0.5, co2_ppmv=415, precip_total=0, precip_snow=0
  - has_radiation=1.0, has_precipitation=1.0
- Call `step_land(state, forcing, config, U_min=1.0, dt=3600.0)`.
- Assert: all LandState fields finite. All TileResponse fields finite.
- Assert: T_soil ∈ [150, 400] K. W_bucket ∈ [0, W_max]. snow_depth ≥ 0.

**1b) Surface energy balance closure**
- Extract from TileResponse: shflx, lhflx, lw_up (positive up).
- Compute: sw_net = (1 - albedo) * sw_down. lw_net = lw_down - lw_up.
- Compute: ground heat flux G = sw_net + lw_net - shflx - lhflx.
- Assert: dT_soil/dt ≈ G / (C_soil * d_soil) within 1%.
- This catches errors in heat capacity, sign conventions, or missing terms.

**1c) Surface flux signs**
- T_soil=300 K, T_lowest=280 K (warm surface): assert shflx > 0 (upward).
- T_soil=260 K, T_lowest=280 K (cold surface): assert shflx < 0 (downward).
- With q_surface > q_lowest: assert lhflx > 0 (evaporation).
- With cos_zenith > 0: assert lw_up > 0 always (surface emits).

**1d) Moisture availability**
- W_bucket = W_max: beta should be close to 1.0. Evaporation should be maximal.
- W_bucket = 0: beta should equal beta_min. Evaporation should be minimal.
- Assert: lhflx(W_max) > lhflx(0) for same atmospheric forcing.

**1e) Bucket hydrology water conservation**
- Apply known precipitation (precip_total=1e-4 kg/m²/s) with zero evaporation (q_lowest >> q_surface).
- After dt=3600s: ΔW ≈ precip * dt = 0.36 kg/m².
- Assert: W_new - W_old ≈ precip * dt - evap * dt within 1%.
- When W = W_max and precip > 0: runoff should occur (W capped at W_max).
- When W = 0 and evap > 0: evap should cease (W stays at 0).

**1f) Stefan-Boltzmann consistency**
- Assert: lw_up ≈ emissivity * σ * T_soil^4 (σ = 5.67e-8) within 0.1%.
- This catches emissivity, Stefan-Boltzmann constant, or T unit errors.

**1g) Multi-step stability**
- Run step_land for 100 steps (dt=3600s) with constant forcing.
- Assert: T_soil stays in [200, 350] K. W_bucket stays in [0, W_max]. No NaN.
- Assert: T_soil converges toward equilibrium (variance decreases over last 50 steps).

**1h) Wind speed floor**
- With u_lowest=v_lowest=0: assert surface fluxes are non-zero (U_min kicks in).
- Assert: shflx ≠ 0, lhflx ≠ 0, tau_x ≈ 0, tau_y ≈ 0.

**Key imports:**
```python
from legoesm.land.slab_land import step_land
from legoesm.land.config import LandConfig
from legoesm.land.state import LandState
from legoesm.coupler.coupling_fields import AtmToSurface, TileResponse
from legoesm.core.field import Field
```

Read `src/legoesm/land/slab_land.py` first for exact function signature and field names.

---

# CATEGORY 2: Multi-Layer Land — Soil Thermal & Hydraulic Physics
**File:** `tests/unit/test_land_ice_multilayer.py`

**2a) Smoke test — step_multilayer_land runs**
- Initialize with `init_multilayer_land_state(ncol=8, config=MultiLayerLandConfig(), T_init=280.0)`.
- Create AtmToSurface forcing (same as Category 1).
- Call `step_multilayer_land(state, forcing, config, U_min=1.0, dt=1800.0)`.
- Assert: all MultiLayerLandState fields finite. TileResponse finite.
- Assert: T_soil ∈ [150, 400] K at all layers. theta_soil ∈ [theta_r, theta_sat].

**2b) Soil thermal diffusion — uniform T steady state**
- Initialize T_soil = 280 K at all layers.
- With G_surface = 0 and Q_geothermal = 0: after `solve_soil_thermal()`, T should remain 280 K.
- Assert: |T_new - T_old| < 1e-10 K.

**2c) Soil thermal diffusion — heat penetration**
- Initialize T_soil = 280 K. Apply G_surface = 100 W/m² (strong heating).
- After one step: top layer should warm, bottom layer nearly unchanged.
- Assert: T_new[0] > T_old[0]. |T_new[-1] - T_old[-1]| < |T_new[0] - T_old[0]| * 0.1.
- After many steps: temperature profile should smooth out (approach uniform).

**2d) Soil thermal energy conservation**
- Compute total soil energy: E = Σ(C_eff * T * dz) integrated over layers.
- Energy input: G_surface * dt + Q_geothermal * dt.
- Assert: |E_new - E_old - (G + Q_geo) * dt| / |E_new - E_old| < 0.01.

**2e) Heat capacity and conductivity**
- `compute_heat_capacity(theta, hydro_config, thermal_config)`:
  - At theta=0 (dry): C_eff ≈ (1-theta_sat)*C_soil + theta_sat*C_air.
  - At theta=theta_sat (wet): C_eff ≈ (1-theta_sat)*C_soil + theta_sat*C_water.
  - Assert: C_eff(wet) > C_eff(dry) (water has higher heat capacity than air).
  - Assert: C_eff ∈ [1e5, 5e6] J/m³/K for all realistic theta.
- `compute_thermal_conductivity(theta, hydro_config, thermal_config)`:
  - Assert: k(wet) > k(dry) (wet soil conducts heat better).
  - Assert: k ∈ [0.1, 5.0] W/m/K for realistic conditions.

**2f) Richards equation — steady state**
- With flux_top=0, sink=0, and uniform psi:
  - Solution should not change: |psi_new - psi_old| < tol.
- Assert: theta ∈ [theta_r, theta_sat] after solve.

**2g) Richards equation — infiltration**
- Start with dry soil (theta near theta_r, psi very negative).
- Apply moderate flux_top (< K_sat): all water should infiltrate.
- Assert: runoff_surface ≈ 0. theta increases in top layers.
- Apply excessive flux_top (>> K_sat): surface runoff should occur.
- Assert: runoff_surface > 0.

**2h) Richards equation — free drainage**
- Start with wet soil (theta near theta_sat), flux_top=0.
- With bottom_bc="free_drainage": subsurface runoff should be positive.
- Assert: runoff_subsurface > 0. theta decreases over time.
- With bottom_bc="zero_flux": subsurface runoff = 0.

**2i) Richards equation — water conservation**
- Total water: W = Σ(theta * dz * rho_water) over layers.
- Water inputs: flux_top * dt * rho_water.
- Water outputs: (runoff_surface + runoff_subsurface) * dt + Σ(sink * dz) * dt.
- Assert: |W_new - W_old - inputs + outputs| / max(|W_new - W_old|, 1e-12) < 0.05.

**2j) Root water uptake**
- With transpiration demand (from evaporation) and sufficient soil water:
  - Root zone should dry (theta decreases in upper layers).
  - Deep soil should be less affected.
- With theta near wilting point (theta_wp): uptake should cease.

**Key imports:**
```python
from legoesm.land.multilayer_land import step_multilayer_land, init_multilayer_land_state
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.state import MultiLayerLandState
from legoesm.land.soil_thermal import solve_soil_thermal, compute_heat_capacity, compute_thermal_conductivity, SoilThermalConfig
from legoesm.land.soil_hydraulics import (
    theta_from_psi, psi_from_theta, hydraulic_conductivity, moisture_capacity,
    van_genuchten_theta, van_genuchten_psi, van_genuchten_K,
    clapp_hornberger_psi, clapp_hornberger_theta, clapp_hornberger_K,
    brooks_corey_theta, brooks_corey_K,
    SoilHydraulicsConfig,
)
from legoesm.land.richards import solve_richards, RichardsConfig, RichardsOutput
from legoesm.land.soil_grid import make_soil_grid, SoilGridConfig, SoilGrid
```

Read the source files first — especially `soil_thermal.py`, `soil_hydraulics.py`, and `richards.py`.

---

# CATEGORY 3: Soil Hydraulics — Retention Curves & Conductivity
**File:** `tests/unit/test_land_ice_soil_hydraulics.py`

Test all soil water retention curve (SWRC) models for mathematical correctness and physical consistency.

**3a) Van Genuchten — roundtrip consistency**
- For psi in [-1000, -0.01] m (log-spaced):
  - theta = van_genuchten_theta(psi) → psi_recovered = van_genuchten_psi(theta).
  - Assert: |psi_recovered - psi| / |psi| < 1e-6 (exact inverse).
- At psi = 0: theta should equal theta_sat.
- At psi → -∞: theta should approach theta_r.

**3b) Van Genuchten — monotonicity**
- theta(psi) should be monotonically increasing (wetter at higher psi).
- K(psi) should be monotonically increasing (more conductive when wetter).
- Assert: d(theta)/d(psi) ≥ 0 (numerically checked).
- Assert: d(K)/d(psi) ≥ 0 (numerically checked).

**3c) Van Genuchten — saturation bounds**
- Assert: theta ∈ [theta_r, theta_sat] for all psi.
- Assert: K ∈ [0, K_sat] for all psi.
- At saturation: K = K_sat (within floating point).

**3d) Van Genuchten — moisture capacity**
- C = dtheta/dpsi should be positive for unsaturated conditions.
- C should peak at intermediate moisture (not at extremes).
- Assert: C > 0 for psi ∈ [-100, -0.01].

**3e) Clapp-Hornberger — consistency**
- Same roundtrip test as 3a.
- Assert: theta ∈ [0, theta_sat] and K ∈ [0, K_sat].
- Monotonicity: theta increases with psi.

**3f) Brooks-Corey — consistency**
- Se = 1 when psi ≥ psi_b (air entry): theta = theta_sat.
- Se decreases as psi decreases below psi_b.
- K at saturation ≈ K_sat.
- Monotonicity holds.

**3g) Peters-Durner-Iden (PDI) — consistency**
- Assert: theta ∈ [0, theta_sat] for all psi.
- PDI should include film-flow contribution at very dry conditions.
- K should be positive for all psi.
- At saturation (psi=0): theta ≈ theta_sat.

**3h) Lu three-regime — consistency**
- Assert: theta ∈ [0, theta_sat].
- Three regimes visible: adsorption, capillary, cavitation.
- Smooth transitions between regimes (no discontinuities).

**3i) Cross-model comparison at saturation**
- All SWRC models should agree at saturation: theta(0) ≈ theta_sat.
- All models should agree at very dry: theta → theta_r (or 0).

**3j) Interblock conductivity**
- Geometric mean: K_half = sqrt(K_above * K_below).
- Assert: min(K_above, K_below) ≤ K_half ≤ max(K_above, K_below).
- If one layer is impermeable (K=0): K_half = 0.

**3k) Parametrize across all SWRC models**
```python
@pytest.mark.parametrize("curve", ["van_genuchten", "clapp_hornberger", "brooks_corey", "pdi", "lu"])
def test_swrc_bounds(curve):
    config = SoilHydraulicsConfig(retention_curve=curve)
    ...
```

**Key imports:**
```python
from legoesm.land.soil_hydraulics import (
    theta_from_psi, psi_from_theta, hydraulic_conductivity, moisture_capacity, interblock_K,
    van_genuchten_Se, van_genuchten_theta, van_genuchten_psi, van_genuchten_K, van_genuchten_C,
    clapp_hornberger_psi, clapp_hornberger_theta, clapp_hornberger_K, clapp_hornberger_C,
    brooks_corey_Se, brooks_corey_theta, brooks_corey_K,
    pdi_theta, pdi_K, pdi_C,
    lu_theta, lu_K, lu_C,
    SoilHydraulicsConfig,
)
```

---

# CATEGORY 4: Soil Grid — Geometry & Discretization
**File:** `tests/unit/test_land_ice_soil_grid.py`

**4a) Default grid properties**
- `make_soil_grid()` with defaults: n_layers=8, dz_top=0.025 m, growth_factor=2.0.
- Assert: len(dz) == 8. All dz > 0.
- Assert: dz is geometric: dz[k+1] / dz[k] ≈ growth_factor.
- Assert: z_interface[0] = 0 (surface). z_interface[-1] = total depth.
- Assert: z_node[k] = (z_interface[k] + z_interface[k+1]) / 2 (midpoints).

**4b) Total depth specification**
- `make_soil_grid(SoilGridConfig(total_depth=3.0, n_layers=8, growth_factor=2.0))`.
- Assert: sum(dz) ≈ 3.0 m within 1e-10.
- Assert: geometric series still holds.

**4c) Custom grid**
- `make_soil_grid_custom([0.05, 0.1, 0.2, 0.4, 0.8])`.
- Assert: dz matches input exactly. n_layers = 5.
- Assert: z_interface correct (cumulative sum).

**4d) Interface distances**
- dz_interface[k] = z_node[k+1] - z_node[k] (distance between midpoints).
- Assert: all dz_interface > 0.
- Assert: len(dz_interface) == n_layers - 1.

**4e) Uniform spacing**
- growth_factor = 1.0: all dz should be equal.
- Assert: max(dz) - min(dz) < 1e-12.

**Key imports:**
```python
from legoesm.land.soil_grid import make_soil_grid, make_soil_grid_custom, SoilGridConfig, SoilGrid
```

---

# CATEGORY 5: Snow Budget — Accumulation, Melt, Age
**File:** `tests/unit/test_land_ice_snow.py`

**5a) Snow accumulation**
- Start with snow_depth=0, precip_snow=1e-4 kg/m²/s, T_surface=260 K (below freezing).
- After dt=3600s: snow ≈ 0.36 kg/m².
- Assert: snow_new > 0. snow_age resets to 0 (fresh snow).

**5b) Snow melt**
- Start with snow_depth=10 kg/m², T_surface=280 K (above T_snow_melt=273.15).
- Melt rate: snow_melt_rate * (T - T_melt) = 5e-6 * 6.85 ≈ 3.4e-5 kg/m²/s.
- After dt=3600s: melt ≈ 0.12 kg/m². snow_new ≈ 9.88 kg/m².
- Assert: snow_new < snow_old. Snow never negative.

**5c) Complete melt**
- Start with snow_depth=0.01 kg/m², T_surface=300 K (strong melt).
- After one step: snow should melt completely.
- Assert: snow_new = 0. snow_age resets to 0.

**5d) No melt below freezing**
- T_surface=260 K (< T_snow_melt): no melt should occur.
- Assert: snow remains unchanged (aside from accumulation).

**5e) Snow age evolution**
- No new snowfall, snow exists: snow_age increases by dt each step.
- New snowfall (precip_snow > 1e-10): snow_age resets to 0.
- No snow left: snow_age = 0.

**5f) Snow albedo feedback**
- Enable snow_albedo_feedback=True in LandConfig.
- With snow: albedo should be higher than without snow.
- Fresh snow (snow_age=0): albedo near alpha_snow_max (0.80).
- Old snow (snow_age >> tau_snow_decay): albedo near alpha_snow_min.
- Assert: albedo(snow) > albedo(no_snow) always.

**5g) Snow cover fraction**
- snow_depth < snow_depth_crit: partial snow cover → albedo interpolated.
- snow_depth > snow_depth_crit: full snow cover → albedo = snow albedo.
- Assert: snow_cover_frac ∈ [0, 1].

**Key imports:**
```python
from legoesm.land.snow_budget import update_snow
from legoesm.land.config import LandConfig
from legoesm.surface_albedo import compute_land_albedo, LandAlbedoConfig
```

Read `src/legoesm/land/snow_budget.py` and `src/legoesm/surface_albedo.py` first.

---

# CATEGORY 6: Carbon Cycle — GPP, Allocation, Decomposition
**File:** `tests/unit/test_land_ice_carbon.py`

**6a) GPP smoke test**
- Call `compute_gpp(sw_down=300, T=293.15, LAI=3.0, co2_ppmv=415, beta=0.8, config)`.
- Assert: GPP > 0 [gC/m²/s].
- Assert: GPP magnitude ∈ [1e-7, 1e-4] gC/m²/s (reasonable range).

**6b) GPP = 0 at night**
- sw_down = 0: assert GPP = 0 (no light, no photosynthesis).

**6c) GPP = 0 with no leaves**
- LAI = 0: assert GPP = 0 (no canopy to absorb light).

**6d) GPP scales with drivers**
- GPP increases with sw_down (more light → more photosynthesis).
- GPP increases with CO2 (fertilization effect) up to saturation.
- GPP increases with beta (moisture availability).
- GPP has optimum temperature (Gaussian response): peaks at T_opt, decreases away.

**6e) Carbon pool conservation (DifferLand)**
- Run `step_carbon_differland()` for one step.
- Total carbon: C_total = C_lab + C_fol + C_root + C_wood + C_lit + C_som.
- Carbon flux: dC/dt = GPP - R_auto - R_het (≈ NEE).
- Assert: |C_total_new - C_total_old - (GPP - R_auto - R_het) * dt| is small.
- Note: pools are clamped to ≥ 1 gC/m², so conservation may have small residuals.

**6f) NPP allocation**
- NPP = GPP * (1 - f_auto).
- Allocation: fol + lab + root + wood fractions should sum to 1.0.
- Assert: all allocation fractions ∈ [0, 1].
- Assert: C_wood gets the remainder (largest pool grows fastest at steady state).

**6g) Pools stay positive**
- Run for 100 steps. Assert: all pools ≥ 1 gC/m² (clamp applied).
- No NaN in any pool.

**6h) Decomposition temperature sensitivity**
- Warmer temperature → higher decomposition rate (Q10 effect).
- Assert: R_het(T=293) > R_het(T=283).
- Assert: R_het ≥ 0 always.

**6i) Phenology — seasonality**
- NH mid-latitude (lat=0.8 rad): labile release peaks near Bday (day 100).
- Leaf fall peaks near Fday (day 280).
- SH (lat=-0.8 rad): timing shifted by ~182.5 days.
- Assert: lrf and lff are non-negative. Peak values occur at expected times.

**6j) NEE sign convention**
- Daytime (sw > 0, GPP > R): NEE should be negative (net uptake).
- Nighttime (sw = 0, GPP = 0): NEE should be positive (net source from respiration).
- Assert: co2_flux has correct sign: positive = emission to atmosphere.

**6k) Seasonal CO2 flux scheme**
- NH summer (day ~200, lat > 0): co2_flux < 0 (uptake).
- NH winter (day ~15, lat > 0): co2_flux > 0 (release).
- SH: opposite phase.
- Equator: minimal seasonal cycle.

**Key imports:**
```python
from legoesm.land.carbon.carbon_cycle import (
    compute_gpp, compute_phenology, step_carbon_differland, step_carbon,
    seasonal_co2_flux, init_carbon_state,
)
from legoesm.land.carbon.config import CarbonConfig, CarbonState
```

Read `src/legoesm/land/carbon/carbon_cycle.py` and `src/legoesm/land/carbon/config.py` first.

---

# CATEGORY 7: Stomatal Conductance & Leuning Farquhar Photosynthesis
**File:** `tests/unit/test_land_ice_stomata.py`

**NOTE — Phase 1 canopy refactor**: the Bernacchi Farquhar
(``arrhenius`` / ``peaked_arrhenius`` / ``farquhar_photosynthesis``) was
removed.  The **Leuning C3 + Q10 C4** model in
``src/legoesm/land/canopy/photosynthesis.py`` is now the single Farquhar
implementation, and the stomatal conductance functions live in
``packages/land/legoesm/land/stomata.py``.  The legacy Bernacchi
test cases (7a–7e) are gone; the canopy Farquhar is covered by
``tests/land/unit/test_canopy_photosynthesis.py``.  This category focuses
on the stomatal conductance models and the coupled Leuning A-gs solver.

**7f) Ball-Berry stomatal conductance**
- gs = g0 + g1 * A * RH / Cs.
- At A=0: gs = g0 (minimum conductance).
- gs increases with A (positive correlation with photosynthesis).
- gs increases with RH (stomata open in humid conditions).
- gs decreases with Cs (higher CO2 → stomata close).
- Assert: gs ≥ g0 always.

**7g) Medlyn stomatal conductance**
- gs = g0 + 1.6 * (1 + g1/sqrt(VPD)) * A / Cs.
- At A=0: gs = g0.
- gs decreases with VPD (drought stress closes stomata).
- Assert: gs ≥ g0.

**7h) Jarvis multiplicative model**
- Each stress factor in [0, 1]: f_PAR, f_T, f_VPD, f_soil.
- gs = gs_max * f_PAR * f_T * f_VPD * f_soil.
- At optimal conditions (bright, warm, humid, wet): gs ≈ gs_max.
- In dark (PAR=0): f_PAR ≈ 0 → gs ≈ 0.
- Assert: gs ∈ [0, gs_max].

**7i) Coupled Farquhar-stomata solver convergence**
- Call `coupled_farquhar_stomata()` with realistic inputs.
- Assert: gs > 0 and gpp > 0 under normal conditions.
- Assert: gs and gpp are finite and in reasonable range.
- After n_iter_ags=5 iterations: Ci should be consistent with gs and A.
- Check: Ci = Ca - 1.6 * A / gs (within numerical tolerance).

**7j) Stomatal beta computation**
- High LAI + open stomata: beta_eff ≈ canopy transpiration rate.
- Low LAI: beta_eff ≈ beta_soil (bare soil dominates).
- Assert: beta_eff ∈ [0, 1].

**7k) Coupled system — drought reduces GPP**
- beta_soil=0.1 (drought): GPP and gs should be reduced vs beta_soil=0.9.
- Assert: gpp(drought) < gpp(wet).
- Assert: gs(drought) < gs(wet).

**7l) CO2 fertilization through stomata**
- Higher CO2 → higher Ci → higher A but lower gs.
- Assert: gpp(800 ppm) > gpp(400 ppm).
- Assert: gs(800 ppm) < gs(400 ppm) (stomata partially close at high CO2).

**Key imports:**
```python
from legoesm.land.stomata import (
    ball_berry_gs, medlyn_gs, jarvis_gs,
    coupled_farquhar_stomata, compute_stomatal_beta,
)
from legoesm.land.carbon.config import StomataConfig
```

Read ``packages/land/legoesm/land/stomata.py`` and
``src/legoesm/land/canopy/photosynthesis.py`` first — the
``coupled_farquhar_stomata`` solver uses the Leuning C3/C4 Farquhar
model from the latter via a Newton A-gs root-find.

---

# CATEGORY 8: Sea Ice Thermodynamics — Slab Model
**File:** `tests/unit/test_land_ice_sea_ice_thermo.py`

**8a) Smoke test — step_sea_ice runs**
- Create SeaIceState: h_ice=1.0 m, T_ice=265 K, concentration=0.9.
- AtmToSurface forcing: sw_down=100, lw_down=200, T_lowest=260 K.
- ocean_sst=271.35 K (at freezing), ocean_u=ocean_v=0.
- Call `step_sea_ice(state, forcing, ocean_sst, ..., config, U_min=1.0, dt=3600.0)`.
- Assert: all state fields finite. TileResponse finite.

**8b) Ice growth in cold conditions**
- T_lowest=240 K (very cold), ocean_sst=271.35 K, h_ice=1.0 m.
- Conductive heat loss exceeds ocean heat supply → ice grows.
- Assert: h_new > h_old (ice thickens).

**8c) Ice melt in warm conditions**
- T_lowest=280 K (warm air), ocean_sst=275 K (warm ocean), sw_down=300.
- Ocean heat flux melts ice from below, warm air melts from above.
- Assert: h_new < h_old (ice thins).

**8d) Complete melt → h_ice = 0**
- Start with thin ice (h_ice=0.01 m), very warm forcing.
- After several steps: h_ice should reach 0.
- Assert: h_ice ≥ 0 always (never negative).
- When h_ice = 0: concentration should also be 0.

**8e) New ice formation in open water**
- No existing ice (h=0, conc=0). ocean_sst = T_freeze = 271.35 K.
- Strong cooling: Q_sfc < 0 → frazil ice forms.
- Assert: h_new > 0, conc_new > 0 after one step.
- New ice thickness ≈ h_new_ice = 0.05 m initially.

**8f) Temperature bounds**
- Assert: T_ice ≥ T_ice_min (180 K, numerical floor).
- Assert: T_ice ≤ T_freeze_ocean (271.35 K, ice can't be warmer than freezing).
- For all forcing conditions: T_ice stays in [T_ice_min, T_freeze].

**8g) Concentration bounds**
- Assert: concentration ∈ [0, 1] always.
- Growth increases concentration. Melt decreases it.
- At full coverage (conc=1): no further increase from growth.
- At zero coverage (conc=0): only frazil can create new ice.

**8h) Energy conservation in thermodynamics**
- Conductive flux: F_cond = k_ice * (T_freeze - T_ice) / h.
- Ocean heat flux: F_ocean = coeff * max(sst - T_freeze, 0).
- Growth rate: dh/dt = (F_cond - F_ocean) / (rho_ice * L_f).
- Assert: dh/dt sign is consistent with F_cond vs F_ocean balance.
- Latent heat: ΔE = rho_ice * L_f * Δh ≈ (F_cond - F_ocean) * dt within 5%.

**8i) Stefan-Boltzmann from ice surface**
- lw_up ≈ emissivity_ice * σ * T_ice^4.
- Assert: |lw_up - ε*σ*T^4| / lw_up < 0.001.

**8j) Surface flux signs over ice**
- Cold ice (T_ice=250 K), warm air (T_lowest=270 K): shflx < 0 (heat into ice).
- Cold ice, cold air (T_lowest=240 K): shflx > 0 (ice loses heat upward).
- Assert: lw_up > 0 always.

**8k) Multi-step stability**
- Run 200 steps (dt=3600s) with constant forcing.
- Assert: T_ice converges. h_ice bounded (no runaway growth/melt).
- Assert: no NaN at any step.

**Key imports:**
```python
from legoesm.ice.sea_ice import step_sea_ice
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.state import SeaIceState, DynamicSeaIceState, init_dynamic_ice_state
from legoesm.coupler.coupling_fields import AtmToSurface, TileResponse
from legoesm.core.field import Field
```

Read `src/legoesm/ice/sea_ice.py` first, especially `_thermo_single()`.

---

# CATEGORY 9: Sea Ice Dynamics — EVP & Rheology
**File:** `tests/unit/test_land_ice_sea_ice_dynamics.py`

**9a) Ice strength (Hibler 1979)**
- P = P_star * h * exp(-C * (1 - A)).
- Assert: P > 0 when h > 0 and A > 0.
- P = 0 when h = 0 or A = 0.
- P increases with h (thicker ice is stronger).
- P increases with A (more compact ice is stronger).

**9b) Strain rates — zero for uniform velocity**
- u_ice = const, v_ice = const everywhere.
- Assert: eps_11 ≈ 0, eps_22 ≈ 0, eps_12 ≈ 0.

**9c) Strain rates — pure divergence**
- u_ice linearly increasing in x (u = ax): eps_11 > 0, eps_12 ≈ 0.
- Assert: shapes match (6, n, n).

**9d) Delta deformation**
- Delta ≥ Delta_min always (regularized).
- For zero strain: Delta = Delta_min.
- Delta increases with strain rate magnitude.

**9e) VP stress — pressure-only state**
- Zero strain rates → sigma_11 = sigma_22 = -P/2, sigma_12 = 0.
- Assert: isotropic pressure state.

**9f) EVP stress relaxation**
- EVP stress update should relax toward VP target.
- After many subcycles: EVP stress ≈ VP stress.
- Assert: |sigma_evp - sigma_vp| decreases with iterations.

**9g) Free drift velocity**
- Pure wind forcing, no ocean current: u_ice ∝ wind.
- Pure ocean current, no wind: u_ice ∝ ocean current.
- Both: linear superposition.
- Assert: u_ice direction matches dominant forcing.

**9h) Air-ice and ocean-ice stress**
- Stress is quadratic drag: tau ∝ |ΔU| * ΔU.
- Assert: tau opposes relative motion (decelerating).
- At rest (u_ice = u_ocean): tau_ocean = 0.
- At rest (u_ice = u_wind): tau_air = 0.
- Assert: |tau| increases with speed difference.

**9i) EVP solver — convergence**
- Run EVP with N_evp=120 subcycles.
- Assert: velocity and stress are finite.
- With strong ice (thick, compact): velocity should be reduced vs free drift.
- With weak ice (thin, sparse): velocity ≈ free drift.

**9j) EVP solver — at rest**
- All forcing zero (no wind, no ocean current): ice should remain stationary.
- Assert: u_new ≈ 0, v_new ≈ 0.

**9k) Momentum conservation check**
- Sum of forces: tau_air + tau_ocean + div(sigma) - Coriolis = m * du/dt.
- Compute each term independently and verify balance within 5%.

**9l) Coriolis — semi-implicit**
- Verify: Coriolis turns velocity to the right (NH, f > 0).
- Verify: Coriolis does not change speed (|u|² + |v|² conserved by Coriolis alone).

**Key imports:**
```python
from legoesm.ice.dynamics import (
    evp_solver, free_drift_velocity, air_ice_stress, ocean_ice_stress, stress_divergence,
)
from legoesm.ice.rheology import (
    ice_strength, strain_rates, delta_deformation, vp_stress, evp_stress_update,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
```

Read `src/legoesm/ice/dynamics.py` and `src/legoesm/ice/rheology.py` first.

---

# CATEGORY 10: Sea Ice Transport & ITD
**File:** `tests/unit/test_land_ice_sea_ice_transport.py`

**10a) Transport — volume conservation**
- Advect ice tracers for one step with uniform velocity.
- Total ice volume: V = Σ(h * conc * area) over grid.
- Assert: |V_new - V_old| / V_old < 0.01 (volume conserved within 1%).

**10b) Transport — concentration bounds**
- After advection: concentration ∈ [0, 1].
- Assert: no negative concentration, no super-saturation.

**10c) Transport — thickness non-negative**
- After advection: h_ice ≥ 0 everywhere.

**10d) Transport — stationary ice**
- u_ice = v_ice = 0: no change after advection.
- Assert: |h_new - h_old| < 1e-10. |conc_new - conc_old| < 1e-10.

**10e) Transport — temperature advection**
- Warm ice advected into cold region: T should mix.
- Assert: T_new is finite. No NaN from enthalpy/volume division.

**10f) ITD — category bounds**
- 5 categories: bounds = [0.0, 0.6, 1.4, 2.4, 3.6].
- Assert: monotonically increasing.
- Assert: upper_bounds[-1] is large (~100 m, effectively ∞).

**10g) ITD — distribute to categories**
- Single h=2.0 m: should go into category where 1.4 ≤ h < 2.4 (category 2).
- Assert: only one category has non-zero concentration.
- Assert: volume is conserved: h * conc (single) = Σ(h_k * conc_k).

**10h) ITD — aggregate back to single**
- Distribute then aggregate: should recover original h and conc.
- Assert: h_agg ≈ h_original. conc_agg ≈ conc_original.

**10i) ITD — linear remap**
- After thermodynamic growth: ice may cross category boundaries.
- Linear remap should redistribute ice across categories.
- Assert: total volume conserved. All conc ∈ [0, 1]. All h ≥ 0.

**10j) State conversion roundtrip**
- SeaIceState → DynamicSeaIceState → SeaIceState.
- Assert: roundtrip preserves h, T, conc within tolerance.

**10k) Multi-category thermodynamics**
- Run step_sea_ice with n_categories=5.
- Assert: all category fields finite. Total concentration ≤ 1.
- Thin ice melts faster than thick ice (different category growth rates).

**Key imports:**
```python
from legoesm.ice.transport import advect_ice_tracers
from legoesm.ice.itd import (
    category_bounds, upper_bounds, aggregate_state, distribute_to_categories, linear_remap,
)
from legoesm.ice.state import SeaIceState, DynamicSeaIceState, init_dynamic_ice_state, dynamic_to_slab, slab_to_dynamic
from legoesm.grids.cubed_sphere import create_cubed_sphere
```

---

# CATEGORY 11: Lake Two-Layer Model
**File:** `tests/unit/test_land_ice_lake.py`

**11a) Smoke test — step_lake runs**
- Create LakeState: T_epi=290 K, T_hypo=280 K.
- AtmToSurface forcing: sw_down=250, lw_down=300, T_lowest=285 K, etc.
- Call `step_lake(state, forcing, config, U_min=1.0, dt=3600.0)`.
- Assert: T_epi, T_hypo finite. TileResponse finite.

**11b) Epilimnion energy balance**
- cap_epi = rho_water * c_water * h_epi.
- Q_net = sw_net + lw_net - shflx - lhflx.
- Mixing: F_mix = rho * c * k_eff * (T_epi - T_hypo) / d_mid.
- dT_epi/dt = (Q_net - F_mix) / cap_epi.
- Assert: T_epi_new ≈ T_epi + dt * dT_epi/dt within 0.1%.

**11c) Hypolimnion energy balance**
- dT_hypo/dt = F_mix / cap_hypo.
- Assert: T_hypo_new ≈ T_hypo + dt * dT_hypo/dt within 0.1%.

**11d) Total energy conservation**
- E_total = cap_epi * T_epi + cap_hypo * T_hypo.
- Energy input = Q_net * dt (to epilimnion only).
- Assert: |E_new - E_old - Q_net * dt| / |Q_net * dt + 1e-20| < 0.01.
- Internal mixing (F_mix) should not change total energy.

**11e) Mixing direction**
- T_epi > T_hypo: F_mix > 0 (heat flows down from warm epi to cold hypo).
- T_epi < T_hypo: F_mix < 0 (heat flows up).
- T_epi = T_hypo: F_mix = 0 (no mixing).

**11f) Wind-enhanced mixing**
- Stronger wind → larger k_eff → larger |F_mix|.
- k_eff = k_mix * (1 + wind_mix_alpha * wind_speed).
- Assert: |F_mix(wind=10)| > |F_mix(wind=1)|.

**11g) Freezing point floor**
- With strong cooling (T_lowest=220 K, no SW): T_epi should decrease but not below T_freeze (273.15 K).
- Assert: T_epi ≥ T_freeze. T_hypo ≥ T_freeze.

**11h) Stratification under warming**
- SW heating warms epilimnion: T_epi increases.
- Hypolimnion barely affected (only through mixing): T_hypo changes slowly.
- Assert: T_epi ≥ T_hypo during stable conditions.

**11i) Multi-step convergence**
- Run 500 steps (dt=3600s) with constant forcing.
- Assert: T_epi converges toward equilibrium. No runaway or NaN.
- Assert: T_epi ∈ [T_freeze, 330 K]. T_hypo ∈ [T_freeze, 330 K].

**11j) Surface albedo and emissivity**
- TileResponse.albedo ≈ config.albedo_lake (0.08 typically).
- TileResponse.emissivity ≈ config.emissivity_lake (0.97).
- lw_up ≈ emissivity * σ * T_epi^4.

**11k) Surface humidity**
- q_surface = saturation mixing ratio at T_epi.
- Warmer T_epi → higher q_surface.
- Assert: q_surface > 0 for all T > 200 K.

**Key imports:**
```python
from legoesm.coupler.lake.two_layer_lake import step_lake
from legoesm.coupler.lake.config import LakeConfig
from legoesm.coupler.lake.state import LakeState
from legoesm.coupler.coupling_fields import AtmToSurface, TileResponse
from legoesm.core.field import Field
```

Read `src/legoesm/coupler/lake/two_layer_lake.py` first.

---

# CATEGORY 12: Surface Albedo — All Tiles
**File:** `tests/unit/test_land_ice_albedo.py`

**12a) Land vegetation albedo**
- Tropics (|lat| < 23.5°): albedo ≈ alpha_veg_tropics (0.15).
- Mid-latitudes: albedo ≈ alpha_veg_midlat (0.20).
- High latitudes: albedo ≈ alpha_veg_highlat (0.25).
- Assert: smooth transitions between zones.

**12b) Snow albedo decay**
- Fresh snow (age=0): albedo ≈ alpha_snow_max (0.80).
- Old snow (age >> tau_snow_decay): albedo ≈ alpha_snow_min.
- Assert: albedo monotonically decreases with age.
- Assert: albedo ∈ [alpha_snow_min, alpha_snow_max].

**12c) Land albedo = vegetation + snow blending**
- No snow: albedo = vegetation albedo.
- Full snow cover: albedo = snow albedo.
- Partial: interpolation between the two.
- Assert: albedo(snow) > albedo(no_snow).

**12d) Sea ice temperature-dependent albedo**
- Cold ice (T < T_cold): high albedo (dry ice).
- Warm ice (T → T_freeze): lower albedo (wet/melting ice).
- Assert: albedo_cold > albedo_warm.
- Assert: smooth transition (sigmoid or linear).

**12e) Ocean albedo**
- Constant scheme: albedo ≈ constant (typically 0.06).
- Zenith-dependent (Briegleb 1992): albedo increases at low sun angles.
- Assert: albedo ∈ [0, 0.5] for all zenith angles.

**12f) All albedos bounded**
- For all tiles and all conditions: albedo ∈ [0, 1].
- No NaN for any input.

**Key imports:**
```python
from legoesm.surface_albedo import (
    compute_land_albedo, compute_snow_albedo, compute_snow_cover_fraction,
    compute_sea_ice_albedo, compute_ocean_albedo,
    LandAlbedoConfig, IceAlbedoConfig,
)
```

Read `src/legoesm/surface_albedo.py` first.

---

# CATEGORY 13: Unit Consistency & Cross-Component Checks
**File:** `tests/unit/test_land_ice_units.py`

**13a) Land tendency magnitudes**
- |dT_soil/dt| < 0.01 K/s (< 36 K/hr). Typical: ~1e-4 K/s.
- |dW_bucket/dt| < 0.01 kg/m²/s. Typical: ~1e-5 kg/m²/s.
- Surface fluxes: |shflx| < 500 W/m², |lhflx| < 500 W/m².
- Precipitation: precip ∈ [0, 0.01] kg/m²/s.

**13b) Sea ice tendency magnitudes**
- |dh_ice/dt| < 1e-5 m/s (< 0.86 m/day). Typical: ~1e-7 m/s.
- |dT_ice/dt| < 0.01 K/s. Typical: ~1e-4 K/s.
- Ice velocity: |u_ice|, |v_ice| < 1 m/s.
- Ice stress: |sigma| < P_star * h_max ≈ 1e5 N/m.

**13c) Lake tendency magnitudes**
- |dT_epi/dt| < 0.001 K/s. |dT_hypo/dt| < 0.001 K/s.
- Mixing flux: |F_mix| < 1000 W/m².

**13d) Constants cross-check**
- Verify constants used across land/ice/lake:
  - rho_water ≈ 1000 kg/m³, c_water ≈ 4218 J/kg/K.
  - rho_ice ≈ 917 kg/m³, c_ice ≈ 2106 J/kg/K, k_ice ≈ 2.04 W/m/K.
  - L_f ≈ 3.337e5 J/kg, σ ≈ 5.67e-8 W/m²/K⁴.
- Assert: each constant within 1% of textbook value.

**13e) Coupling interface consistency**
- AtmToSurface → TileResponse: all fields present and finite.
- TileResponse.T_surface matches internal state T (land: T_soil, ice: T_ice, lake: T_epi).
- TileResponse.albedo matches computed albedo from surface_albedo module.
- TileResponse.lw_up ≈ emissivity * σ * T_surface^4.

**13f) Consistent sign conventions across tiles**
- shflx: positive upward for all tiles (land, ice, lake).
- lhflx: positive upward for all tiles.
- tau: positive in wind direction for all tiles.
- lw_up: positive always for all tiles.
- co2_flux: positive = emission for all tiles.
- Run each tile with identical forcing. Verify sign convention is consistent.

**13g) Carbon flux units**
- co2_flux from land: [kg CO2/m²/s].
- GPP from carbon cycle: [gC/m²/s].
- Conversion: co2_flux = NEE * (44/12) * 1e-3 (gC → kgCO2).
- Assert: |co2_flux - NEE * 44/12 * 1e-3| < 1e-12 kg/m²/s.

---

# CATEGORY 14: Integrated Surface Model Tests
**File:** `tests/unit/test_land_ice_integrated.py`

**14a) Land + carbon cycle**
- Enable CarbonConfig(scheme="differland") in LandConfig.
- Run step_land with carbon_state.
- Assert: carbon_state returned. Pools evolve. co2_flux non-zero.
- GPP should be consistent with stomatal beta.

**14b) Land + stomata coupling**
- Enable StomataConfig(enabled=True, stomata_model="ball_berry").
- Assert: effective beta differs from soil beta (stomata modulate evapotranspiration).
- Drought (low W_bucket) → lower gs → lower beta → less evaporation.

**14c) Multi-layer land + carbon + stomata**
- Full integration: multilayer soil + Richards + carbon + stomata.
- Run 10 steps. Assert: all fields finite and bounded.
- Root water uptake should deplete soil water in root zone.

**14d) Sea ice + dynamics (EVP)**
- config.dynamics = "evp", config.n_categories = 1.
- Apply wind forcing. Assert: ice moves, stress finite.
- Run 10 steps. Assert: velocity and stress converge.

**14e) Sea ice + ITD (5 categories)**
- config.n_categories = 5, config.dynamics = "none".
- Distribute initial ice into categories.
- Run thermodynamics. Assert: different categories grow/melt at different rates.
- Assert: aggregate conserves volume.

**14f) Sea ice + dynamics + ITD + transport**
- Full config: dynamics="evp", n_categories=5, transport="advect".
- Run 5 steps. Assert: all fields finite. Volume approximately conserved.

**14g) All surface tiles with same forcing**
- Create identical AtmToSurface forcing.
- Step land, ice, lake with same forcing.
- Assert: all produce valid TileResponse.
- Compare: T_sfc varies by tile (land ≠ ice ≠ lake). Albedo varies. Fluxes vary.
- Assert: no tile produces NaN when another produces valid output.

**14h) Seasonal cycle — land**
- Run slab land for 365 × 24 steps (dt=3600) at mid-latitude.
- Vary forcing sinusoidally (warm summer, cold winter).
- Assert: T_soil has seasonal cycle. Snow accumulates in winter, melts in spring.
- Assert: GPP peaks in summer, R_het follows with lag.

**Key imports:**
```python
from legoesm.land.slab_land import step_land
from legoesm.land.multilayer_land import step_multilayer_land, init_multilayer_land_state
from legoesm.land.config import LandConfig, MultiLayerLandConfig
from legoesm.land.carbon.config import CarbonConfig, CarbonState
from legoesm.land.carbon.carbon_cycle import init_carbon_state
from legoesm.land.carbon.config import StomataConfig
from legoesm.land.stomata import ball_berry_gs
from legoesm.ice.sea_ice import step_sea_ice
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.state import SeaIceState, DynamicSeaIceState
from legoesm.coupler.lake.two_layer_lake import step_lake
from legoesm.coupler.lake.config import LakeConfig
from legoesm.coupler.lake.state import LakeState
from legoesm.coupler.coupling_fields import AtmToSurface, TileResponse
from legoesm.core.field import Field
```

---

# Implementation Notes

- **Helper functions**: Create reusable helpers for common setup:
  ```python
  def make_forcing(ncol, sw_down=300.0, lw_down=300.0, T_air=280.0, wind=5.0,
                   precip=0.0, precip_snow=0.0, cos_zen=0.5):
      """Create AtmToSurface forcing with given values."""
      shape = (ncol,)  # or (6, n, n) for cubed-sphere
      return AtmToSurface(
          sw_down=jnp.full(shape, sw_down),
          lw_down=jnp.full(shape, lw_down),
          ...
      )

  def make_slab_state(ncol, T_soil=280.0, W_bucket=50.0, snow=0.0):
      """Create minimal LandState."""
      ...

  def make_ice_state(shape, h_ice=1.0, T_ice=265.0, conc=0.9):
      """Create minimal SeaIceState."""
      ...
  ```

- **Field wrapping**: Land and ice states use Field objects:
  ```python
  T_soil = Field(data=jnp.full(shape, 280.0), name="T_soil", dims=dims, units="K")
  ```

- **Parametrize aggressively**:
  ```python
  @pytest.mark.parametrize("bulk_scheme", ["constant", "most"])
  def test_land_bulk_fluxes(bulk_scheme):
      config = LandConfig(bulk_scheme=bulk_scheme)
      ...

  @pytest.mark.parametrize("dynamics", ["none", "free_drift", "evp"])
  def test_sea_ice_dynamics_mode(dynamics):
      config = SeaIceConfig(dynamics=dynamics)
      ...

  @pytest.mark.parametrize("curve", ["van_genuchten", "clapp_hornberger", "brooks_corey", "pdi", "lu"])
  def test_swrc_model(curve):
      ...
  ```

- **Cubed-sphere vs flat shapes**: Some tests can use flat arrays (ncol,) if calling low-level functions directly. Integration tests need cubed-sphere (6, n, n) with a real grid. Read source to determine which.

- **Tolerances**:
  - Conservation laws: 1% relative (numerical scheme + clamping allow small residuals).
  - Sign checks: exact (`jnp.all`).
  - Magnitude bounds: generous (factor of 2–10) — catch unit errors, not tuning.
  - Roundtrip (theta↔psi): 1e-6 relative (exact inverse).
  - Energy balance closure: 1% of dominant term.

- **Reading source code first**: Before writing any test, READ the relevant source file to verify exact function signatures, field names, and output types. Many functions have subtly different signatures (e.g., step_land takes `lat` as optional arg for phenology).

- Mark slow tests (>10s) with `@pytest.mark.slow`.
- Every test must be JAX-compatible: no numpy mutation, use `jnp` throughout.

After writing each file, run it with:
```
JAX_ENABLE_X64=1 python -m pytest tests/unit/test_<name>.py -v --tb=short
```
Fix import/shape errors. If a test reveals a genuine physics bug (wrong sign, missing factor, unit error), leave it failing with a `# BUG:` comment explaining the suspected issue.
