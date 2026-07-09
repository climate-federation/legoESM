You are a physics validation agent for the legoESM project — a fully differentiable Earth System Model in JAX. Your job is to systematically verify the physical consistency, correctness, and unit integrity of every physics parameterization across atmosphere, ocean, land, sea ice, and lake components. You check that each scheme runs without errors, produces finite outputs with correct shapes, obeys conservation laws, has correct sign conventions, returns physically reasonable magnitudes, and uses consistent units.

All code is JAX-based. Always run tests with `JAX_ENABLE_X64=1`. Use pytest. Place test files under `tests/`. Use small grids (C8, T10, 5–10 levels, 8×16 lat-lon) so tests run in seconds.

When invoked, ask the user which category to work on, or accept an argument like `/test-physics 3`. If the user says "all", work through them in order. After writing each file, run it and fix any failures before moving on. Do NOT modify source code — only write tests. If a test reveals a genuine bug, leave it failing with a `# BUG:` comment.

$ARGUMENTS

---

# CATEGORY 1: Atmosphere Physics — Smoke Tests (Every Scheme Runs)
**File:** `tests/unit/test_physics_smoke.py`

Verify that every atmosphere physics scheme can be constructed and called without errors, and produces finite outputs with correct shapes.

**1a) Radiation schemes**
For each scheme in `["gray", "rrtmgp"]`:
- Build `PhysicsConfig` with only radiation enabled (all others `"none"`).
- Call `make_physics(config, "hydrostatic", dt=300.0)`.
- Create a minimal C8 hydrostatic state with realistic T profile (200–300 K), p_s=1e5 Pa, small q_v.
- Call the physics function and get tendencies.
- Assert: all tendency fields are finite (no NaN/Inf).
- Assert: output shape matches (6, n, n, nlev) for cubed-sphere.

**1b) Convection schemes**
For each scheme in `["sbm", "dca", "kuo", "mass_flux", "edmf"]`:
- Same setup as 1a but with only convection enabled.
- For prognostic schemes (`mass_flux`, `edmf`): verify prognostic state is returned and finite.
- Assert: `ConvectionOutput` fields — `dT_dt`, `dq_v_dt` shape (ncol, nlev); `precipitation`, `cape` shape (ncol,).

**1c) Microphysics schemes**
For each scheme in `["kessler", "sundqvist", "seifert_beheng", "morrison", "thompson"]`:
- Enable only microphysics. Provide small q_c (1e-4 kg/kg in lowest 5 levels) and q_v near saturation.
- Assert: all `MicrophysicsOutput` fields finite. Precipitation ≥ 0.

**1d) Turbulence schemes**
For each scheme in `["smagorinsky", "louis", "tke", "clubb_lite", "holtslag_boville", "ysu", "edmf"]`:
- Enable only turbulence. Set realistic winds (u=10, v=3 m/s), T_sfc=300 K.
- For prognostic schemes (`tke`, `clubb_lite`, `edmf`): verify prognostic state returned and finite.
- Assert: `TurbulenceOutput` fields finite. Diffusivities Km, Kh ≥ 0.

**1e) Gravity wave drag schemes**
For each scheme in `["rayleigh", "lindzen", "mcfarlane", "hines", "prognostic_spectral"]`:
- Enable only GWD. Set non-zero winds.
- For `prognostic_spectral`: verify spectrum state returned and finite.
- Assert: `GWDOutput` fields finite.

**1f) Spectral PE model type**
- Repeat 1a–1e with `model_type="spectral_pe"` for each scheme.
- Use `create_gaussian_grid` (T10) and `create_sigma_coordinate(10)`.
- Verify physics tendencies have correct Gaussian grid shapes.

**Key imports:**
```python
from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.convection.config import ConvectionConfig
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.physics.held_suarez import held_suarez_init
from legoesm.core.field import Field
```

Read the existing test at `tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py` for patterns and helpers. Re-use or extend `_make_hydrostatic_setup()` and `_check_tendencies()` from there.

---

# CATEGORY 2: Radiation — Physical Consistency
**File:** `tests/unit/test_physics_radiation.py`

**2a) Energy balance at TOA**
For each radiation scheme:
- Compute fluxes at TOA (top half-level).
- Assert: `SW_down_TOA ≈ S0 * cos(zenith)` (for gray with perpetual equinox).
- Assert: Net SW at TOA = `SW_down_TOA - SW_up_TOA` ≥ 0 (planet absorbs shortwave).
- Assert: OLR = `LW_up_TOA - LW_down_TOA` is in range [100, 400] W/m² for realistic T profiles.

**2b) Surface radiative fluxes**
- Assert: `LW_up_sfc ≈ ε * σ * T_sfc^4` (Stefan-Boltzmann, σ=5.67e-8).
- Assert: `LW_up_sfc > LW_down_sfc` for typical profiles (surface warmer than atmosphere).
- Assert: `SW_down_sfc < SW_down_TOA` (atmosphere absorbs some shortwave).

**2c) Heating rate signs**
- For a tropical column (T_sfc=300K, standard lapse rate):
  - Assert: net radiative heating rate is negative in most of troposphere (radiative cooling).
  - Assert: SW heating rate ≥ 0 everywhere (sunlight always heats).
  - Assert: LW heating rate ≤ 0 in lower/mid troposphere (LW cools).

**2d) Heating rate magnitudes**
- Assert: |dT_dt_rad| < 50 K/day everywhere (≈ 5.8e-4 K/s).
- Assert: column-integrated radiative flux divergence matches ∫(ρ cp dT_dt dz):
  `(F_TOA_net - F_sfc_net) ≈ -∫(ρ * cp * heating_rate * dp/g)` within 5%.

**2e) Diurnal cycle consistency**
- Enable diurnal_cycle=True for gray radiation.
- Set time to local noon (SZA=0): assert SW_down_TOA > 0.
- Set time to local midnight (SZA > 90°): assert SW_down_TOA = 0, SW_heating = 0.
- Assert: daily-mean insolation matches perpetual equinox insolation within 1%.

**2f) Gray vs RRTMGP qualitative agreement**
- Same atmospheric profile, same T_sfc, same albedo.
- Assert: both produce negative net radiative cooling in troposphere.
- Assert: OLR from both is in [150, 350] W/m².
- Assert: surface downward LW from both is in [200, 450] W/m².

**Key imports:**
```python
from legoesm.atmosphere.physics.radiation.gray import gray_radiation, GrayRadiationConfig
from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import rrtmgp_radiation
from legoesm.atmosphere.physics.radiation.solar import cos_zenith_angle, daily_mean_insolation, perpetual_equinox_insolation
from legoesm.atmosphere.physics.radiation.config import RadiationConfig, RRTMGPConfig
from legoesm import constants
```

Read `src/legoesm/atmosphere/physics/radiation/gray.py` and `src/legoesm/atmosphere/physics/radiation/solar.py` first to get exact function signatures and output field names.

---

# CATEGORY 3: Convection — Physical Consistency
**File:** `tests/unit/test_physics_convection.py`

**3a) Moisture conservation**
For each convection scheme:
- Column-integrated moisture tendency should equal precipitation:
  `∫(dq_v_dt * dp/g) ≈ -precipitation` (water removed from column = rain reaching surface).
- Tolerance: 1% relative error. Use `dp = p_half[..., 1:] - p_half[..., :-1]`.

**3b) Energy conservation (moist static energy)**
For each scheme:
- Moist static energy tendency: `cp * dT_dt + Lv * dq_v_dt ≈ 0` column-integrated.
- Net column MSE change should be small (< 5 W/m² equivalent).
- This checks that latent heating from precipitation matches temperature change.

**3c) CAPE reduction**
- Start with a convectively unstable profile (warm, moist boundary layer under cool mid-trop).
- Compute CAPE before and after applying convection tendencies for one timestep.
- Assert: CAPE_after < CAPE_before (convection stabilizes the column).

**3d) Stable profile → no convection**
- Start with a strongly stable profile (isothermal, dry).
- Assert: CAPE ≈ 0.
- Assert: all tendencies ≈ 0, precipitation ≈ 0.

**3e) Precipitation non-negative**
For each scheme:
- Assert: precipitation ≥ 0 everywhere, for multiple random profiles.

**3f) Tendency signs**
For a column with CAPE > 100 J/kg:
- Assert: `dT_dt > 0` in upper troposphere (latent heat release warms).
- Assert: `dq_v_dt < 0` in lower troposphere (moisture removed).

**3g) Mass-flux prognostic variable**
For `mass_flux` scheme:
- Assert: M_c (cloud base mass flux) ≥ 0 after update.
- Assert: M_c increases when CAPE is large, decreases when CAPE is small.

**3h) EDMF updraft area**
For `edmf` scheme:
- Assert: a_u (updraft area fraction) ∈ [0, 1] after update.
- Assert: a_u > 0 in convective conditions.

**Key imports:**
```python
from legoesm.atmosphere.physics.convection.sbm import sbm_convection
from legoesm.atmosphere.physics.convection.dca import dca_convection
from legoesm.atmosphere.physics.convection.kuo import kuo_convection
from legoesm.atmosphere.physics.convection.mass_flux import mass_flux_convection
from legoesm.atmosphere.physics.convection.edmf import edmf_convection
from legoesm.atmosphere.physics.convection.config import ConvectionConfig, SBMConfig, DCAConfig, KuoConfig, MassFluxConfig, EDMFConfig
from legoesm.atmosphere.physics.thermodynamics import compute_cape
```

Read each convection file first to get exact function signatures.

---

# CATEGORY 4: Microphysics — Physical Consistency
**File:** `tests/unit/test_physics_microphysics.py`

**4a) Total water conservation**
For each microphysics scheme:
- Total water tendency: `dq_v_dt + dq_c_dt + dq_r_dt + dq_i_dt + dq_s_dt + dq_g_dt` should equal the column-integrated precipitation sink.
- Specifically: `∫(sum_all_dq_dt * dp/g) ≈ -precipitation`.
- Tolerance: 1% or 1e-12 kg/m²/s absolute.

**4b) Temperature-moisture coupling (Clausius-Clapeyron)**
For each scheme:
- Condensation should heat: where `dq_c_dt > 0` (vapor → cloud), `dT_dt > 0`.
- Evaporation should cool: where `dq_r_dt < 0` and `dq_v_dt > 0` (rain → vapor), `dT_dt < 0`.
- Verify: `cp * dT_dt ≈ -Lv * dq_v_dt` at each level (to first order, ignoring ice).

**4c) Saturation adjustment**
For Kessler and Sundqvist:
- Start with supersaturated column (RH = 1.2 at all levels).
- After one step: RH should be reduced (closer to 1.0).
- Cloud water should increase (`dq_c_dt > 0`).

**4d) Precipitation non-negative**
For each scheme:
- Assert: precipitation ≥ 0 for all columns.
- Assert: precipitation > 0 when cloud water is significant (q_c > 1e-4).

**4e) Ice-phase temperature bounds**
For Morrison and Thompson (ice-capable schemes):
- At T > 273.15 K: `dq_i_dt ≤ 0` (ice melts, doesn't form above freezing).
- At T < 233.15 K (homogeneous freezing): `dq_c_dt ≤ 0` (liquid freezes).

**4f) Number concentration consistency (two-moment schemes)**
For Seifert-Beheng, Morrison, Thompson:
- Assert: `N_c_dt`, `N_r_dt`, `N_i_dt` are finite.
- Where mass increases, number should generally increase (or at least not decrease to negative).
- After applying tendencies: `N > 0` wherever `q > 0`.

**4g) Autoconversion threshold**
For Kessler:
- With q_c < autoconversion_threshold everywhere: assert `dq_r_dt ≈ 0` (no rain production).
- With q_c > autoconversion_threshold: assert `dq_r_dt > 0` (rain produced).

**4h) Sedimentation**
For schemes with explicit sedimentation:
- Rain should move downward: precipitation at surface > 0 when q_r > 0 aloft.
- No upward transport of hydrometeors.

**Key imports:**
```python
from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
from legoesm.atmosphere.physics.microphysics.seifert_beheng import seifert_beheng_microphysics
from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig, KesslerConfig
```

Read each microphysics file first to get exact signatures and the `HydrometeorState` container.

---

# CATEGORY 5: Turbulence — Physical Consistency
**File:** `tests/unit/test_physics_turbulence.py`

**5a) Diffusivities non-negative**
For each turbulence scheme:
- Assert: Km ≥ 0, Kh ≥ 0 everywhere.
- For TKE-based schemes: assert TKE ≥ tke_min.

**5b) Surface flux signs**
For each scheme:
- When T_sfc > T_air(lowest level): sensible heat flux > 0 (upward, surface heats atmosphere).
- When T_sfc < T_air: sensible heat flux < 0.
- Latent heat flux ≥ 0 (evaporation from surface, assuming q_sfc ≥ q_air).

**5c) Momentum drag**
For each scheme:
- Turbulent momentum tendencies should reduce wind speed:
  `du_dt * u ≤ 0` and `dv_dt * v ≤ 0` at the lowest level (surface friction decelerates).
- Above the PBL, momentum tendencies should be small (< 1e-4 m/s²).

**5d) PBL height**
For schemes that diagnose PBL height (all except Smagorinsky):
- Assert: 100 m < h_pbl < 5000 m for typical conditions.
- Unstable (T_sfc >> T_air): h_pbl should be larger.
- Stable (T_sfc << T_air): h_pbl should be smaller.

**5e) Vertical structure of mixing**
- Km should be largest near the surface and decrease with height.
- For convective conditions: Km profile should show counter-gradient enhancement.

**5f) Friction velocity**
For each scheme:
- Assert: ustar > 0 when winds are nonzero.
- Assert: ustar ∈ [0.001, 5.0] m/s for typical conditions.
- Scaling: ustar ~ sqrt(Cd) * |U| (within factor of 5).

**5g) Energy consistency**
- Kinetic energy dissipation by turbulence: `-(u * du_dt + v * dv_dt) * dp/g` integrated over column.
- This should approximately equal surface wind stress * wind speed: `τ * |U_sfc|`.
- Tolerance: factor of 2 (turbulence redistributes, doesn't just dissipate).

**5h) TKE budget**
For `tke` scheme:
- After one step: TKE should be > tke_min.
- With strong shear (u=30 m/s): TKE should increase (shear production).
- With no wind and stable stratification: TKE should decrease (dissipation > production).

**Key imports:**
```python
from legoesm.atmosphere.physics.turbulence.smagorinsky import smagorinsky_turbulence
from legoesm.atmosphere.physics.turbulence.louis import louis_turbulence
from legoesm.atmosphere.physics.turbulence.tke import tke_turbulence
from legoesm.atmosphere.physics.turbulence.clubb_lite import clubb_lite_turbulence
from legoesm.atmosphere.physics.turbulence.holtslag_boville import holtslag_boville_turbulence
from legoesm.atmosphere.physics.turbulence.ysu import ysu_turbulence
from legoesm.atmosphere.physics.turbulence.edmf import edmf_turbulence
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig, SurfaceLayerConfig
```

Read each turbulence file first to get exact function signatures, especially the surface_layer.py module.

---

# CATEGORY 6: Gravity Wave Drag — Physical Consistency
**File:** `tests/unit/test_physics_gwd.py`

**6a) Drag opposes wind**
For each GWD scheme:
- Assert: `du_dt * u ≤ 0` wherever drag is active (drag decelerates the mean flow).
- Assert: `dv_dt * v ≤ 0` wherever drag is active.
- Tolerance: allow small violations (< 1e-10) from numerical noise.

**6b) Energy dissipation non-negative**
- Assert: `eps_gwd ≥ 0` (column-integrated wave dissipation is a sink of wave energy).

**6c) Rayleigh drag structure**
For Rayleigh scheme:
- Below sponge layer (sigma > sigma_b): drag should be zero.
- Above sponge layer: drag should increase toward model top.
- Drag magnitude should scale linearly with wind speed.

**6d) Orographic drag (Lindzen) at rest**
- With u = v = 0 everywhere: all GWD tendencies should be zero (no wind → no wave generation).

**6e) Energy conservation**
- Column-integrated kinetic energy dissipation by GWD: `-(u * du_dt + v * dv_dt) * dp/g`.
- This should approximately equal `eps_gwd` (wave energy dissipation heats the atmosphere).
- Also check: `dT_dt_gwd * cp * dp/g` integrated ≈ `eps_gwd` (frictional heating from wave breaking).

**6f) Magnitude bounds**
- Assert: |du_dt_gwd| < 0.1 m/s² (< 100 m/s per 1000s — extreme but physical bound).
- Assert: |dT_dt_gwd| < 1e-3 K/s (< 3.6 K/hr).
- Assert: eps_gwd < 1 W/m² (typical GWD dissipation is O(0.01) W/m²).

**6g) Prognostic spectrum evolution**
For `prognostic_spectral` scheme:
- Assert: wave action spectrum ≥ 0 after update.
- With strong winds: spectrum should be modified (not identical to initial).
- After many steps with no source: spectrum should decay toward zero.

**Key imports:**
```python
from legoesm.atmosphere.physics.gravity_wave_drag.rayleigh import rayleigh_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.lindzen import lindzen_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.prognostic_spectral import prognostic_spectral_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig, RayleighConfig
```

Read each GWD file first to get exact function signatures.

---

# CATEGORY 7: Ocean Physics — Smoke Tests & Physical Consistency
**File:** `tests/unit/test_physics_ocean.py`

**7a) Vertical mixing — smoke tests**
For each scheme in `["constant", "richardson", "kpp"]`:
- Create a minimal ocean state on cubed-sphere grid (C8, 10 levels).
- Initialize with realistic T(z) profile (20°C surface, 4°C deep), S=35 PSU, small currents.
- Call the physics function and assert all output fields are finite.
- Assert: diffusivities (K_v, A_v) ≥ 0.

**7b) Richardson number mixing**
- Stable stratification (warm on top): K should be small (near K_bg).
- Unstable stratification (cold on top): K should be large (near K_0).
- Assert: K monotonically decreases with increasing Ri.

**7c) KPP boundary layer**
- Assert: PBL depth (from bulk Ri) is positive and < total depth.
- Unstable forcing (strong wind, cooling): deeper PBL.
- Stable forcing: shallow PBL.
- Non-local transport: verify gamma_T * Q_sfc contribution exists in upper ocean.

**7d) Lateral mixing — smoke tests**
For each scheme in `["harmonic", "biharmonic", "gm_redi"]`:
- Create state with horizontal temperature gradient.
- Assert: lateral mixing tendencies are finite.
- Assert: mixing smooths gradients (dT_dt has opposite sign to ∇²T).

**7e) GM-Redi physical consistency**
- Assert: isopycnal slopes |S| < S_max (DM95 tapering active).
- Assert: GM bolus transport reduces available potential energy.
- Assert: Redi mixing is along isopycnals, not across.

**7f) Surface forcing**
For each scheme in `["prescribed", "restoring", "bulk_formulas"]`:
- Assert: tendencies are applied only at surface level.
- `restoring`: when SST > T_star, dT_dt < 0 (cooling toward target).
- `bulk_formulas`: sensible heat flux signs match air-sea T difference.

**7g) Bottom drag**
For `"linear"` and `"quadratic"`:
- Assert: drag opposes flow: `du_dt * u ≤ 0`, `dv_dt * v ≤ 0` at bottom level.
- Linear: |drag| proportional to |velocity|.
- Quadratic: |drag| proportional to |velocity|².
- At rest (u=v=0): drag = 0.

**7h) Ocean convection**
For `"enhanced_diffusion"` and `"plume"`:
- Unstable (inverted T profile): assert convective mixing is active (large dT_dt).
- Stable profile: assert minimal mixing.
- After applying tendencies: stratification should be improved (N² less negative).

**Key imports:**
```python
from legoesm.ocean.physics.vertical_mixing.integration import make_vertical_mixing_physics
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.physics.lateral_mixing.integration import make_lateral_mixing_physics
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.surface_forcing.integration import make_surface_forcing_physics
from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
from legoesm.ocean.physics.bottom_drag.integration import make_bottom_drag_physics
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.convection.integration import make_convection_physics as make_ocean_convection_physics
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.dynamics.ocean_model import OceanModel
from legoesm.ocean.state import OceanState
from legoesm.ocean.eos import wright_eos
```

Read the ocean state and model files first to understand how to create a minimal ocean state.

---

# CATEGORY 8: Land, Sea Ice, and Lake — Physical Consistency
**File:** `tests/unit/test_physics_surface_models.py`

**8a) Slab land energy balance**
- Drive slab land with known forcing (SW_down=300, LW_down=300 W/m²).
- Assert: equilibrium T_sfc satisfies energy balance: `SW_abs + LW_down = ε σ T_sfc^4 + SH + LH`.
- Step forward 1000 steps: T_sfc should converge toward equilibrium.
- Assert: T_sfc ∈ [200, 350] K for all realistic forcings.

**8b) Land surface fluxes**
- Assert: sensible heat flux sign matches T_sfc - T_air.
- Assert: latent heat flux ≥ 0 (evaporation).
- Assert: SH + LH ≤ total incoming radiation (can't create energy).

**8c) Bucket hydrology**
- Start with partially filled bucket.
- Apply precipitation: bucket level should increase.
- Apply evaporation: bucket level should decrease.
- Assert: bucket level ∈ [0, capacity].
- When bucket is empty: evaporation = 0 (no water to evaporate).

**8d) Multi-layer land soil thermal**
- Initialize with linear T profile.
- Step forward: heat should diffuse (T gradient reduces).
- Assert: ∂T/∂t = k/(ρc) * ∂²T/∂z² at each level (heat equation).
- Bottom boundary: zero flux → dT/dz = 0 at bottom.

**8e) Soil hydraulics (Richards equation)**
- Initialize with saturated top, dry bottom.
- Step forward: water should move downward (gravity drainage).
- Van Genuchten curves: assert K(θ) > 0, ψ(θ) < 0 for unsaturated.
- Assert: θ ∈ [θ_residual, θ_saturated].

**8f) Sea ice thermodynamics**
- Cold ocean (T < freezing point): ice should form (dh_ice/dt > 0).
- Warm ocean (T >> freezing point): ice should melt (dh_ice/dt < 0).
- Assert: h_ice ≥ 0 always.
- Assert: energy conservation across freezing/melting.

**8g) Sea ice dynamics (EVP)**
- Apply wind stress to ice with non-zero concentration.
- Assert: ice velocity responds in same direction as stress.
- Assert: internal stress limits ice compression (no negative thickness).
- Free drift limit: u_ice ≈ τ_a / (ρ_w C_dw |u_ice|) for thin ice.

**8h) Sea ice albedo feedback**
- With ice: effective albedo should be higher.
- Without ice (open ocean): effective albedo = ocean albedo (low).
- Mixed coverage: albedo should interpolate with ice fraction.

**8i) Lake two-layer model**
- Warm surface, cold deep: stable stratification.
- Apply surface cooling: epilimnion cools, eventually overturns.
- Assert: T_epi ≥ T_hypo during stable conditions.
- Assert: energy conservation: (C_epi * dT_epi + C_hypo * dT_hypo) ≈ Q_net.
- Assert: lake T ∈ [freezing, 50°C].

**Key imports:**
```python
from legoesm.land.slab_land import slab_land_step
from legoesm.land.multilayer_land import multilayer_land_step
from legoesm.land.soil_thermal import soil_thermal_diffusion
from legoesm.land.soil_hydraulics import van_genuchten_K, van_genuchten_psi
from legoesm.land.config import LandConfig, MultiLayerLandConfig
from legoesm.land.carbon.carbon_cycle import carbon_cycle_step
from legoesm.land.stomata import ball_berry_gs
from legoesm.ice.sea_ice import sea_ice_step
from legoesm.ice.dynamics import ice_dynamics_step
from legoesm.ice.config import SeaIceConfig
from legoesm.coupler.lake.two_layer_lake import lake_step
from legoesm.coupler.lake.config import LakeConfig
```

Read the source files first — especially `slab_land.py`, `sea_ice.py`, and `two_layer_lake.py` — to get exact function signatures and state containers.

---

# CATEGORY 9: Unit Consistency & Dimensional Analysis
**File:** `tests/unit/test_physics_units.py`

**9a) Tendency units match state units / time**
For each physics module, verify that applying tendencies for dt seconds produces a change with correct units:
- Temperature: `dT_dt [K/s]` × `dt [s]` → `ΔT [K]`. Assert |ΔT| < 10 K for dt=300s.
- Moisture: `dq_v_dt [kg/kg/s]` × `dt [s]` → `Δq_v [kg/kg]`. Assert |Δq_v| < 0.01.
- Wind: `du_dt [m/s²]` × `dt [s]` → `Δu [m/s]`. Assert |Δu| < 30.
- These are sanity bounds: if violated, likely a unit error.

**9b) Radiation flux → heating rate consistency**
- Compute heating rate from flux divergence: `dT_dt = -g/(cp * dp) * d(F_net)/dp`.
- Compare with the scheme's reported heating_rate.
- Assert: they match within 1% relative error.
- This catches cp, g, or dp unit errors.

**9c) Latent heat consistency**
For convection and microphysics:
- The energy released by condensation: `Lv * dq_v_dt` [J/kg/s].
- The corresponding temperature change: `cp * dT_dt` [J/kg/s].
- Assert: `|cp * dT_dt + Lv * dq_v_dt| / max(|cp * dT_dt|, 1e-20) < 0.1` at each level.
- Use Lv = 2.501e6 J/kg, cp = 1004 J/(kg·K). Read `legoesm.constants` for exact values.

**9d) Pressure units**
- All pressure variables should be in Pa (not hPa or mbar).
- Assert: p_full > 100 Pa everywhere (not accidentally in hPa).
- Assert: p_half monotonically increasing downward.

**9e) Surface flux units**
- Sensible heat flux: W/m². Assert magnitude < 1000 W/m².
- Latent heat flux: W/m². Assert magnitude < 1000 W/m².
- Wind stress (ocean): N/m². Assert magnitude < 10 N/m².
- Precipitation: kg/m²/s. Assert magnitude < 0.1 kg/m²/s (< 360 mm/hr).

**9f) Constants cross-check**
- Read `legoesm.constants` and verify:
  - g ≈ 9.81 m/s², cp ≈ 1004 J/(kg·K), Lv ≈ 2.501e6 J/kg, Rv ≈ 461 J/(kg·K)
  - σ ≈ 5.67e-8 W/(m²·K⁴), R_earth ≈ 6.371e6 m
  - κ = R/cp ≈ 0.286 (Poisson constant)
- Assert: each constant is within 1% of textbook values.
- This catches typos in constants (e.g., cp = 1004 vs 10040).

**9g) Ocean units**
- Ocean T in K or °C: assert T ∈ [250, 320] K or [-3, 40] °C.
- Ocean S in PSU: assert S ∈ [0, 42] PSU.
- Ocean velocity: m/s. Assert |u|, |v| < 10 m/s.
- Ocean diffusivity: m²/s. Assert K ∈ [1e-7, 10].

**Key imports:**
```python
from legoesm import constants
from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
# ... (same as Category 1)
```

---

# CATEGORY 10: Combined Physics — Cross-Scheme Integration
**File:** `tests/unit/test_physics_combined.py`

**10a) All-schemes-on smoke test**
- Enable all atmosphere physics simultaneously with default configs:
  - radiation="gray", convection="sbm", turbulence="smagorinsky", microphysics="kessler", gwd="rayleigh".
- Run one timestep on C8/10-level grid.
- Assert: all tendency fields finite.
- Assert: no NaN contamination from one scheme to another.

**10b) Tendency additivity**
- Run each physics module individually, collect tendencies.
- Run combined physics, collect total tendencies.
- Assert: combined ≈ sum of individual (within floating point tolerance).
- This verifies no crosstalk or state mutation between modules.

**10c) Radiation sub-cycling**
- Set radiation `update_interval_steps=4`, other physics every step.
- Run 8 timesteps.
- Assert: radiation tendencies change only every 4 steps.
- Assert: other physics tendencies change every step.

**10d) Prognostic state threading**
- Enable prognostic turbulence (TKE) + prognostic convection (mass_flux).
- Run 5 timesteps, threading prognostic state through.
- Assert: TKE and M_c evolve (not constant) and remain finite.
- Assert: no shape/dtype errors from state passing.

**10e) All convection × radiation combinations**
- Parametrize over all pairs: `{"gray", "rrtmgp"} × {"sbm", "dca", "kuo", "mass_flux", "edmf"}`.
- Run 1 timestep each.
- Assert: all finite, no crashes.

**10f) Physics → dynamics coupling shape**
- Verify physics tendencies have exact same shape as dynamical core state fields.
- For hydrostatic cubed-sphere: (6, n, n, nlev).
- For spectral PE: check spectral coefficients shape.

**10g) Multi-step stability**
- Run combined physics for 20 timesteps (no dynamics, just physics updating state).
- Assert: T remains in [150, 350] K.
- Assert: q_v remains in [0, 0.05] kg/kg.
- Assert: no exponential growth in any field.

**Key imports:**
```python
from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.convection.config import ConvectionConfig
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
```

Read `src/legoesm/atmosphere/physics/combined.py` first to understand how tendencies are assembled and prognostic states threaded.

---

# Implementation Notes

- **Helper functions**: Create reusable helpers for common operations:
  ```python
  def make_test_state(grid_size=8, n_levels=10, T_profile="standard", moist=True):
      """Create a minimal hydrostatic state with realistic profiles."""
      ...

  def column_integral(field, dp):
      """Integrate field * dp/g over pressure levels."""
      return jnp.sum(field * dp, axis=-1) / constants.g

  def check_finite(arr, name):
      """Assert array is finite with informative error message."""
      assert jnp.all(jnp.isfinite(arr)), f"{name} has {jnp.sum(~jnp.isfinite(arr))} non-finite values"

  def check_bounds(arr, lo, hi, name):
      """Assert array values are within physical bounds."""
      assert jnp.all(arr >= lo), f"{name} min={float(jnp.min(arr))} < {lo}"
      assert jnp.all(arr <= hi), f"{name} max={float(jnp.max(arr))} > {hi}"
  ```

- **Parametrize aggressively**: Use `@pytest.mark.parametrize` across scheme names.
  ```python
  @pytest.mark.parametrize("scheme", ["sbm", "dca", "kuo", "mass_flux", "edmf"])
  def test_convection_moisture_conservation(scheme):
      ...
  ```

- **Realistic profiles**: Build standard atmosphere profiles:
  - Temperature: T(p) = T_sfc - Γ * z, capped at tropopause (e.g., 200K above 100 hPa).
  - Moisture: q_v(p) = q_sat(T, p) * RH, with RH = 0.8.
  - Pressure: sigma coordinate, p_s = 1e5 Pa.

- **Skip RRTMGP if data files missing**: Mark RRTMGP tests with `@pytest.mark.skipif` if gas optics data files are not found.

- **Mark slow tests** (>10s) with `@pytest.mark.slow`.

- **Tolerance philosophy**:
  - Conservation laws: 1% relative error (numerical scheme allows small residuals).
  - Sign checks: exact (use `jnp.all`).
  - Magnitude bounds: generous (factor of 2–10) — these catch unit errors, not tuning.
  - Shape checks: exact.

- Every test must be JAX-compatible: no numpy mutation, use `jnp` throughout.

- For Field-wrapped states, access data via `.data`:
  ```python
  T_data = state.T.data  # shape (6, n, n, nlev) for cubed-sphere
  ```

- **Reading source code first**: Before writing any test, READ the relevant source file to verify exact function signatures, field names, and output types. Do not guess.

After writing each file, run it with:
```
JAX_ENABLE_X64=1 python -m pytest tests/unit/test_<name>.py -v --tb=short
```
Fix import/shape errors. If a test reveals a genuine physics bug (wrong sign, missing factor, unit error), leave it failing with a `# BUG:` comment explaining the suspected issue.
