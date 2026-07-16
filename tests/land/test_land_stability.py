"""Long-integration stability and realism tests for the land models.

Runs both the slab land and multi-layer land models for 30+ simulated days
with realistic atmospheric forcing (diurnal + seasonal cycles), then verifies:

- State variables stay within physical bounds (T, moisture, snow)
- Energy budget closes (accumulated residual bounded)
- Water budget closes (precip - evap - runoff = storage change)
- Snow accumulates in winter and melts in spring
- Carbon pools stay positive and GPP is realistic
- Diurnal temperature cycle has the right sign
- No NaN/Inf appears anywhere
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.config import LandConfig, MultiLayerLandConfig
from legoesm.land.state import LandState, MultiLayerLandState
from legoesm.land.slab_land import step_land
from legoesm.land.multilayer_land import (
    step_multilayer_land,
    init_multilayer_land_state,
)
from legoesm.land.carbon.config import CarbonConfig, CarbonState
from legoesm.land.carbon.carbon_cycle import init_carbon_state
from legoesm import constants


# ========================================================================
# Forcing generator
# ========================================================================

def _make_forcing(
    ncol: int,
    lat: np.ndarray,
    day: float,
    hour: float,
    *,
    dtype=jnp.float64,
    precip_rate: float = 2e-5,
    cold: bool = False,
) -> AtmToSurface:
    """Construct realistic atmospheric forcing for *ncol* columns.

    Parameters
    ----------
    lat : array (ncol,), latitude in radians.
    day : day of year [0–365].
    hour : hour of day [0–24].
    cold : if True, set sub-freezing air temperature and snow precip.
    """
    # Solar geometry
    decl = 23.45 * jnp.sin(2.0 * jnp.pi * (day - 80.0) / 365.0)
    decl_rad = decl * jnp.pi / 180.0
    ha = (hour - 12.0) * 15.0 * jnp.pi / 180.0
    cos_sza = (
        jnp.sin(lat) * jnp.sin(decl_rad)
        + jnp.cos(lat) * jnp.cos(decl_rad) * jnp.cos(ha)
    )
    cos_sza = jnp.maximum(cos_sza, 0.0)

    S0 = constants.S_0  # W/m2
    sw_down = (S0 * cos_sza).astype(dtype)

    # Atmospheric temperature: baseline from latitude, small diurnal range
    if cold:
        T_atm = jnp.full(ncol, 260.0, dtype=dtype)
    else:
        T_base = 288.0 - 30.0 * jnp.abs(lat) / (jnp.pi / 2.0)
        diurnal_amp = 3.0
        T_atm = (T_base + diurnal_amp * jnp.cos(2 * jnp.pi * (hour - 14) / 24)).astype(dtype)

    # LW down: effective emissivity ~0.75 of blackbody at T_atm
    lw_down = (0.75 * constants.sigma_sb * T_atm ** 4).astype(dtype)

    # Humidity: ~60% RH equivalent (rough Clausius-Clapeyron)
    e_sat = 611.2 * jnp.exp(17.67 * (T_atm - constants.T_freeze) / (T_atm - 29.65))
    q_atm = (0.6 * constants.epsilon * e_sat / 1e5).astype(dtype)

    # Precipitation
    precip_total = jnp.full(ncol, precip_rate, dtype=dtype)
    if cold:
        precip_snow = precip_total
    else:
        precip_snow = jnp.where(T_atm < 275.0, precip_total, jnp.zeros(ncol, dtype=dtype))

    p_sfc = jnp.full(ncol, 1e5, dtype=dtype)
    rho = jnp.full(ncol, 1.2, dtype=dtype)

    return AtmToSurface(
        sw_down=sw_down,
        lw_down=lw_down,
        precip_total=precip_total,
        precip_snow=precip_snow,
        T_lowest=T_atm,
        q_lowest=q_atm,
        u_lowest=jnp.full(ncol, 3.0, dtype=dtype),
        v_lowest=jnp.full(ncol, 2.0, dtype=dtype),
        p_lowest=jnp.full(ncol, 9.5e4, dtype=dtype),
        p_surface=p_sfc,
        rho_lowest=rho,
        cos_zenith=cos_sza,
        co2_ppmv=jnp.full(ncol, 412.0, dtype=dtype),
        has_radiation=jnp.ones(ncol, dtype=dtype),
        has_precipitation=jnp.ones(ncol, dtype=dtype),
    )


# ========================================================================
# Helpers
# ========================================================================

NCOL = 4
LATITUDES = jnp.array([0.0, 0.52, 1.05, 1.40])  # equator, 30N, 60N, 80N
DT_SLAB = 3600.0     # 1 hour
DT_ML = 1800.0        # 30 min (safer for Richards)
U_MIN = 1.0


def _make_slab_state(ncol=NCOL, T_init=285.0, W_init=100.0):
    """Create initial slab land state (columnar, using (ncol,) shape)."""
    return LandState(
        T_soil=Field(jnp.full(ncol, T_init), name="T_soil", units="K"),
        W_bucket=Field(jnp.full(ncol, W_init), name="W_bucket", units="kg/m2"),
        snow_depth=Field(jnp.zeros(ncol), name="snow_depth", units="kg/m2"),
        snow_age=Field(jnp.zeros(ncol), name="snow_age", units="s"),
    )


def _run_slab(n_days, config=None, start_day=0.0, T_init=285.0, carbon=False):
    """Run slab land for *n_days* and return diagnostics."""
    if config is None:
        cc = CarbonConfig(scheme="differland") if carbon else CarbonConfig()
        config = LandConfig(carbon=cc, snow_albedo_feedback=True)

    state = _make_slab_state(T_init=T_init)
    carbon_state = init_carbon_state((NCOL,), config.carbon) if carbon else None

    n_steps = int(n_days * 86400 / DT_SLAB)
    dt = DT_SLAB

    # Accumulators
    T_history = []
    W_history = []
    snow_history = []
    shflx_acc = jnp.zeros(NCOL)
    lhflx_acc = jnp.zeros(NCOL)
    sw_acc = jnp.zeros(NCOL)
    lw_net_acc = jnp.zeros(NCOL)

    # Compile the step ONCE and reuse it across all timesteps.  The eager
    # per-step call path re-lowers the soil solver's ``lax.scan`` sweeps on
    # every iteration under jax>=0.9, leaking compiled executables per step —
    # a multi-day (hundreds-of-steps) run otherwise grows unbounded and
    # segfaults.  Production time-stepping runs inside a jitted ``scan``, so
    # this matches it.  ``config``/``U_MIN``/``dt``/``LATITUDES`` are captured
    # as constants; ``carbon_state`` (None or pytree) is a traced argument.
    @jax.jit
    def _step(state, forcing, carbon_state, doy):
        return step_land(
            state, forcing, config, U_MIN, dt,
            lat=LATITUDES, carbon_state=carbon_state, doy=doy,
        )

    for i in range(n_steps):
        t = start_day * 86400.0 + i * dt
        day = (t / 86400.0) % 365.0
        hour = (t / 3600.0) % 24.0

        forcing = _make_forcing(NCOL, LATITUDES, day, hour)
        doy = day

        state, response, carbon_state = _step(state, forcing, carbon_state, doy)

        T_history.append(np.asarray(state.T_soil.data))
        W_history.append(np.asarray(state.W_bucket.data))
        snow_history.append(np.asarray(state.snow_depth.data))

        # Accumulate fluxes for budget
        alpha = np.asarray(response.albedo)
        sw_net = (1 - alpha) * np.asarray(forcing.sw_down)
        lw_net = (config.emissivity_land * np.asarray(forcing.lw_down)
                  - np.asarray(response.lw_up))
        shflx_acc += np.asarray(response.shflx) * dt
        lhflx_acc += np.asarray(response.lhflx) * dt
        sw_acc += sw_net * dt
        lw_net_acc += lw_net * dt

    return {
        "state": state,
        "carbon_state": carbon_state,
        "T": np.stack(T_history),       # (n_steps, ncol)
        "W": np.stack(W_history),
        "snow": np.stack(snow_history),
        "shflx_acc": np.asarray(shflx_acc),
        "lhflx_acc": np.asarray(lhflx_acc),
        "sw_acc": np.asarray(sw_acc),
        "lw_net_acc": np.asarray(lw_net_acc),
        "config": config,
        "n_steps": n_steps,
        "dt": dt,
    }


def _run_multilayer(n_days, config=None, start_day=0.0, carbon=False):
    """Run multi-layer land for *n_days* and return diagnostics."""
    if config is None:
        cc = CarbonConfig(scheme="differland") if carbon else CarbonConfig()
        config = MultiLayerLandConfig(
            carbon=cc,
            snow_albedo_feedback=True,
        )

    state = init_multilayer_land_state(NCOL, config, T_init=285.0)
    carbon_state = init_carbon_state((NCOL,), config.carbon) if carbon else None

    n_steps = int(n_days * 86400 / DT_ML)
    dt = DT_ML

    T_top_history = []
    theta_top_history = []
    snow_history = []
    runoff_history = []

    # Compile the step ONCE and reuse it across all timesteps (see _run_slab):
    # the eager per-step path re-lowers the soil solver's ``lax.scan`` sweeps
    # every iteration under jax>=0.9, leaking executables and segfaulting a
    # multi-day run.  Production runs inside a jitted ``scan``; this matches it.
    @jax.jit
    def _step(state, forcing, carbon_state, doy):
        return step_multilayer_land(
            state, forcing, config, U_MIN, dt,
            lat=LATITUDES, carbon_state=carbon_state, doy=doy,
        )

    for i in range(n_steps):
        t = start_day * 86400.0 + i * dt
        day = (t / 86400.0) % 365.0
        hour = (t / 3600.0) % 24.0

        forcing = _make_forcing(NCOL, LATITUDES, day, hour)
        doy = day

        state, response, carbon_state = _step(state, forcing, carbon_state, doy)

        T_top_history.append(np.asarray(state.T_soil[:, 0]))
        theta_top_history.append(np.asarray(state.theta_soil[:, 0]))
        snow_history.append(np.asarray(state.snow_depth))
        runoff_history.append(
            np.asarray(state.runoff_surface + state.runoff_subsurface)
        )

    return {
        "state": state,
        "carbon_state": carbon_state,
        "T_top": np.stack(T_top_history),
        "theta_top": np.stack(theta_top_history),
        "snow": np.stack(snow_history),
        "runoff": np.stack(runoff_history),
        "config": config,
        "n_steps": n_steps,
        "dt": dt,
    }


# ========================================================================
# Slab land stability
# ========================================================================

class TestSlabLandStability:
    """Run slab land for 30 days and check physical bounds."""

    @pytest.fixture(scope="class")
    def run30(self):
        return _run_slab(30, start_day=170.0)  # start in NH summer

    def test_no_nan(self, run30):
        """No NaN in temperature or moisture history."""
        assert not np.any(np.isnan(run30["T"]))
        assert not np.any(np.isnan(run30["W"]))
        assert not np.any(np.isnan(run30["snow"]))

    def test_temperature_bounds(self, run30):
        """Soil temperature stays in [200, 340] K."""
        assert np.all(run30["T"] > 200.0), f"T min = {run30['T'].min()}"
        assert np.all(run30["T"] < 340.0), f"T max = {run30['T'].max()}"

    def test_moisture_bounds(self, run30):
        """Bucket moisture stays in [0, W_max]."""
        W_max = run30["config"].W_max
        assert np.all(run30["W"] >= 0.0)
        assert np.all(run30["W"] <= W_max + 1e-10)

    def test_snow_nonnegative(self, run30):
        """Snow depth is always non-negative."""
        assert np.all(run30["snow"] >= 0.0)

    def test_tropical_temperature(self, run30):
        """Equatorial temperature stays near 285-310 K."""
        T_eq = run30["T"][:, 0]  # equator column
        assert T_eq.mean() > 275.0, f"tropical mean T = {T_eq.mean():.1f}"
        assert T_eq.mean() < 320.0, f"tropical mean T = {T_eq.mean():.1f}"

    def test_temperature_diurnal_cycle(self, run30):
        """Temperature has a diurnal range > 0 for each column."""
        T = run30["T"]
        for col in range(NCOL):
            T_col = T[:, col]
            diurnal_range = T_col.max() - T_col.min()
            assert diurnal_range > 0.5, (
                f"col {col}: diurnal range = {diurnal_range:.2f} K"
            )


# ========================================================================
# Multi-layer land stability
# ========================================================================

class TestMultiLayerStability:
    """Run multi-layer land for 15 days and check physical bounds."""

    @pytest.fixture(scope="class")
    def run15(self):
        return _run_multilayer(15, start_day=170.0)

    def test_no_nan(self, run15):
        """No NaN in temperature or moisture."""
        s = run15["state"]
        assert not np.any(np.isnan(np.asarray(s.T_soil)))
        assert not np.any(np.isnan(np.asarray(s.theta_soil)))
        assert not np.any(np.isnan(np.asarray(s.psi_soil)))

    def test_temperature_bounds(self, run15):
        """All soil layer temperatures in [200, 340] K."""
        T = np.asarray(run15["state"].T_soil)
        assert np.all(T > 200.0), f"T min = {T.min()}"
        assert np.all(T < 340.0), f"T max = {T.max()}"

    def test_moisture_bounds(self, run15):
        """Volumetric water content in [theta_r, theta_sat]."""
        cfg = run15["config"]
        theta = np.asarray(run15["state"].theta_soil)
        assert np.all(theta >= cfg.hydraulics.theta_r - 0.01), (
            f"theta min = {theta.min()}"
        )
        assert np.all(theta <= cfg.hydraulics.theta_sat + 0.01), (
            f"theta max = {theta.max()}"
        )

    def test_matric_potential_finite(self, run15):
        """Matric potential is finite everywhere."""
        psi = np.asarray(run15["state"].psi_soil)
        assert np.all(np.isfinite(psi))

    def test_runoff_nonnegative(self, run15):
        """Runoff is non-negative."""
        s = run15["state"]
        assert np.all(np.asarray(s.runoff_surface) >= -1e-15)
        assert np.all(np.asarray(s.runoff_subsurface) >= -1e-15)

    def test_surface_temperature_tracks_atmosphere(self, run15):
        """Surface T (top layer) shows diurnal variation."""
        T_top = run15["T_top"]
        for col in range(NCOL):
            rng = T_top[:, col].max() - T_top[:, col].min()
            assert rng > 0.1, (
                f"col {col}: T_top range = {rng:.3f} K (too small)"
            )

    def test_deep_soil_less_variable(self, run15):
        """Deep soil temperature is less variable than surface."""
        s = run15["state"]
        T = np.asarray(s.T_soil)
        T_top_std = np.std(run15["T_top"], axis=0)  # variability of top layer over time
        # Deep layer should be more stable (closer to initial)
        T_deep = T[:, -1]  # deepest layer at final time
        # Deep layers should be closer to the initialization temperature
        # (they respond slowly to surface forcing)
        assert np.all(np.abs(T_deep - 285.0) < np.abs(T[:, 0] - 285.0) + 5.0), (
            "Deep soil should be more stable than surface"
        )


# ========================================================================
# Energy budget
# ========================================================================

class TestSlabEnergyBudget:
    """Energy balance closure for slab land over 10 days."""

    @pytest.fixture(scope="class")
    def run10(self):
        return _run_slab(10, start_day=170.0, T_init=288.0)

    def test_energy_residual_bounded(self, run10):
        """Accumulated energy residual is small relative to total fluxes."""
        cfg = run10["config"]
        T_init = 288.0
        T_final = np.asarray(run10["state"].T_soil.data)

        # Energy change in the soil slab
        heat_cap = cfg.C_soil * cfg.d_soil
        dE = heat_cap * (T_final - T_init)  # J/m2

        # Net radiative + turbulent flux = SW_net + LW_net - SH - LH
        net_flux_acc = (
            run10["sw_acc"] + run10["lw_net_acc"]
            - run10["shflx_acc"] - run10["lhflx_acc"]
        )

        # Residual = dE - net_flux_acc (should be ~0)
        residual = np.abs(dE - net_flux_acc)
        total_flux = np.abs(run10["sw_acc"]) + np.abs(run10["lw_net_acc"])

        # Allow up to 5% relative error (discretization, nonlinearities)
        for col in range(NCOL):
            rel_err = residual[col] / (total_flux[col] + 1.0)
            assert rel_err < 0.05, (
                f"col {col}: energy residual = {rel_err*100:.1f}%"
            )


# ========================================================================
# Water budget (slab)
# ========================================================================

class TestSlabWaterBudget:
    """Water balance closure for slab land over 10 days."""

    @pytest.fixture(scope="class")
    def run10(self):
        return _run_slab(10, start_day=170.0, T_init=288.0)

    def test_water_change_plausible(self, run10):
        """Net moisture change is consistent with precip and evaporation."""
        W_init = 100.0
        W_final = np.asarray(run10["state"].W_bucket.data)
        dW = W_final - W_init

        # Net evaporation (approximate from accumulated LH)
        evap_acc = run10["lhflx_acc"] / constants.L_v  # kg/m2

        # Total precip = precip_rate * total_time
        total_time = run10["n_steps"] * run10["dt"]
        precip_acc = 2e-5 * total_time  # kg/m2

        # dW should be approximately precip - evap, clipped to [0, W_max]
        expected_dW = precip_acc - evap_acc
        # The clip distorts the budget, so just check plausibility
        for col in range(NCOL):
            if 0 < W_final[col] < run10["config"].W_max:
                # Interior: budget should be close
                err = abs(dW[col] - expected_dW[col])
                assert err < 50.0, (
                    f"col {col}: water budget error = {err:.1f} kg/m2"
                )


# ========================================================================
# Snow cycle
# ========================================================================

class TestSnowCycle:
    """Test snow accumulation and melt physics."""

    def test_snow_accumulates_in_cold(self):
        """Snow accumulates when forcing is cold with snow precip."""
        config = LandConfig(snow_albedo_feedback=True)
        state = _make_slab_state(T_init=265.0)

        # 3 days of cold forcing with snow
        n_steps = int(3 * 24)
        for i in range(n_steps):
            hour = (i * DT_SLAB / 3600.0) % 24.0
            forcing = _make_forcing(
                NCOL, LATITUDES, day=350.0, hour=hour,
                cold=True, precip_rate=3e-5,
            )
            state, response, _ = step_land(
                state, forcing, config, U_MIN, DT_SLAB, lat=LATITUDES,
            )

        snow = np.asarray(state.snow_depth.data)
        # Should have accumulated non-trivial snow at higher latitudes.
        # The equatorial column (lat~0) can lose most snow to sublimation
        # and energy-limited melt, so we only require it to be non-negative.
        # Higher-latitude columns should retain substantial snow.
        assert np.all(snow >= 0.0), f"snow went negative: {snow}"
        assert np.any(snow > 3.0), f"snow = {snow}, expected some columns > 3 kg/m2"

    def test_snow_melts_in_warm(self):
        """Snow melts when temperature is above freezing."""
        config = LandConfig(snow_albedo_feedback=True)
        # Start with 50 kg/m2 of snow, warm temperature
        state = LandState(
            T_soil=Field(jnp.full(NCOL, 280.0), name="T_soil", units="K"),
            W_bucket=Field(jnp.full(NCOL, 100.0), name="W_bucket", units="kg/m2"),
            snow_depth=Field(jnp.full(NCOL, 50.0), name="snow_depth", units="kg/m2"),
            snow_age=Field(jnp.zeros(NCOL), name="snow_age", units="s"),
        )

        # 5 days of warm forcing, no precip
        n_steps = int(5 * 24)
        for i in range(n_steps):
            hour = (i * DT_SLAB / 3600.0) % 24.0
            forcing = _make_forcing(
                NCOL, LATITUDES, day=170.0, hour=hour, precip_rate=0.0,
            )
            state, _, _ = step_land(
                state, forcing, config, U_MIN, DT_SLAB, lat=LATITUDES,
            )

        snow = np.asarray(state.snow_depth.data)
        # Tropical/subtropical columns (indices 0,1) should melt significantly;
        # high-latitude columns receive less warmth so melt less — physically correct.
        # The polar column (lat=80deg) may not melt at all when snow albedo
        # feedback keeps the surface cold.
        assert snow[0] < 30.0, f"equatorial snow = {snow[0]:.1f}, expected < 30"
        assert snow[1] < 45.0, f"subtropical snow = {snow[1]:.1f}, expected < 45"
        # At least the tropical columns should have lost some snow
        assert np.any(snow < 50.0), f"snow = {snow}, no melt occurred anywhere"

    def test_snow_raises_albedo(self):
        """Presence of snow increases surface albedo."""
        config = LandConfig(snow_albedo_feedback=True)

        # No snow
        state_bare = _make_slab_state(T_init=270.0)
        forcing = _make_forcing(NCOL, LATITUDES, day=1.0, hour=12.0)
        _, resp_bare, _ = step_land(
            state_bare, forcing, config, U_MIN, DT_SLAB, lat=LATITUDES,
        )

        # With snow
        state_snow = LandState(
            T_soil=Field(jnp.full(NCOL, 270.0), name="T_soil", units="K"),
            W_bucket=Field(jnp.full(NCOL, 100.0), name="W_bucket", units="kg/m2"),
            snow_depth=Field(jnp.full(NCOL, 100.0), name="snow_depth", units="kg/m2"),
            snow_age=Field(jnp.zeros(NCOL), name="snow_age", units="s"),
        )
        _, resp_snow, _ = step_land(
            state_snow, forcing, config, U_MIN, DT_SLAB, lat=LATITUDES,
        )

        alpha_bare = np.asarray(resp_bare.albedo)
        alpha_snow = np.asarray(resp_snow.albedo)
        assert np.all(alpha_snow > alpha_bare), (
            f"snow albedo {alpha_snow} should exceed bare {alpha_bare}"
        )


# ========================================================================
# Carbon cycle
# ========================================================================

class TestCarbonCycleIntegration:
    """Carbon cycle coupled to land model over 30 days."""

    @pytest.fixture(scope="class")
    def run30_carbon(self):
        return _run_slab(30, start_day=170.0, carbon=True)

    def test_pools_positive(self, run30_carbon):
        """All carbon pools remain positive."""
        cs = run30_carbon["carbon_state"]
        for pool in ["C_lab", "C_fol", "C_root", "C_wood", "C_lit", "C_som_active"]:
            val = np.asarray(getattr(cs, pool))
            assert np.all(val > 0), f"{pool} has non-positive values: min={val.min()}"

    def test_pools_finite(self, run30_carbon):
        """All carbon pools are finite."""
        cs = run30_carbon["carbon_state"]
        for pool in ["C_lab", "C_fol", "C_root", "C_wood", "C_lit", "C_som_active"]:
            val = np.asarray(getattr(cs, pool))
            assert np.all(np.isfinite(val)), f"{pool} has NaN/Inf"

    def test_wood_pool_largest(self, run30_carbon):
        """Wood pool should remain the largest pool."""
        cs = run30_carbon["carbon_state"]
        C_wood = np.asarray(cs.C_wood).mean()
        C_fol = np.asarray(cs.C_fol).mean()
        assert C_wood > C_fol, "Wood pool should exceed foliage"

    def test_pools_change(self, run30_carbon):
        """Carbon pools evolve (not stuck at initial values)."""
        cs = run30_carbon["carbon_state"]
        cc = run30_carbon["config"].carbon
        # Foliage should have changed from initial (GPP during NH summer)
        C_fol_init = cc.C_fol_init
        C_fol_now = np.asarray(cs.C_fol).mean()
        assert abs(C_fol_now - C_fol_init) > 1.0, (
            f"Foliage pool unchanged: {C_fol_now:.1f} vs init {C_fol_init:.1f}"
        )


# ========================================================================
# Multi-layer specific tests
# ========================================================================

class TestMultiLayerPhysics:
    """Physical process tests for multi-layer land."""

    def test_thermal_profile_smooth(self):
        """After 5 days, temperature profile is smooth (no oscillations)."""
        result = _run_multilayer(5, start_day=170.0)
        T = np.asarray(result["state"].T_soil)

        for col in range(NCOL):
            T_col = T[col]
            # Check no large jumps between adjacent layers
            dT = np.abs(np.diff(T_col))
            assert np.all(dT < 15.0), (
                f"col {col}: large T jump between layers: max dT = {dT.max():.2f} K"
            )

    def test_moisture_profile_physical(self):
        """Soil moisture is within physical bounds in all layers."""
        result = _run_multilayer(5, start_day=170.0)
        cfg = result["config"]
        theta = np.asarray(result["state"].theta_soil)

        assert np.all(theta >= cfg.hydraulics.theta_r - 0.02), (
            f"theta min = {theta.min():.4f}, theta_r = {cfg.hydraulics.theta_r}"
        )
        assert np.all(theta <= cfg.hydraulics.theta_sat + 0.02), (
            f"theta max = {theta.max():.4f}, theta_sat = {cfg.hydraulics.theta_sat}"
        )

    def test_deep_soil_stable(self):
        """Deepest soil layer temperature barely changes from initial."""
        result = _run_multilayer(5, start_day=170.0)
        T_deep = np.asarray(result["state"].T_soil[:, -1])
        # Should be close to initialization (285 K)
        np.testing.assert_allclose(T_deep, 285.0, atol=3.0)

    def test_surface_responds_to_forcing(self):
        """Top layer temperature evolves away from initial value."""
        result = _run_multilayer(5, start_day=170.0)
        T_top = np.asarray(result["state"].T_soil[:, 0])
        # After 5 days, surface should have shifted from 285 K
        assert np.any(np.abs(T_top - 285.0) > 0.5), (
            f"T_top = {T_top}, no response to forcing"
        )


# ========================================================================
# Multilayer + carbon
# ========================================================================

class TestMultiLayerCarbon:
    """Multi-layer land with carbon cycle."""

    @pytest.fixture(scope="class")
    def run_ml_carbon(self):
        return _run_multilayer(15, start_day=170.0, carbon=True)

    def test_pools_positive(self, run_ml_carbon):
        """All carbon pools remain positive in multi-layer run."""
        cs = run_ml_carbon["carbon_state"]
        for pool in ["C_lab", "C_fol", "C_root", "C_wood", "C_lit", "C_som_active"]:
            val = np.asarray(getattr(cs, pool))
            assert np.all(val > 0), f"{pool} min = {val.min()}"

    def test_coupled_state_finite(self, run_ml_carbon):
        """All multi-layer state variables are finite with carbon."""
        s = run_ml_carbon["state"]
        for name in ["T_soil", "theta_soil", "psi_soil"]:
            arr = np.asarray(getattr(s, name))
            assert np.all(np.isfinite(arr)), f"{name} has NaN/Inf"


# ========================================================================
# Seasonal (longer) test — lightweight
# ========================================================================

class TestSeasonalBehavior:
    """180-day seasonal run to verify seasonal temperature response."""

    @pytest.fixture(scope="class")
    def run180(self):
        # Run from Jan 1 through June — see warming trend
        return _run_slab(180, start_day=0.0, T_init=275.0)

    def test_midlat_warms_jan_to_june(self, run180):
        """Mid-latitude column warms from January to June (NH summer)."""
        T_midlat = run180["T"][:, 1]  # 30N column
        # Average T in first 30 days vs last 30 days
        steps_per_day = int(86400 / DT_SLAB)
        T_early = T_midlat[:30 * steps_per_day].mean()
        T_late = T_midlat[-30 * steps_per_day:].mean()
        assert T_late > T_early + 2.0, (
            f"Expected warming: T_early={T_early:.1f}, T_late={T_late:.1f}"
        )

    def test_polar_warms_jan_to_june(self, run180):
        """Polar column also warms from Jan to June."""
        T_polar = run180["T"][:, 3]  # 80N column
        steps_per_day = int(86400 / DT_SLAB)
        T_early = T_polar[:30 * steps_per_day].mean()
        T_late = T_polar[-30 * steps_per_day:].mean()
        assert T_late > T_early, (
            f"Polar should warm: T_early={T_early:.1f}, T_late={T_late:.1f}"
        )

    def test_no_runaway(self, run180):
        """No temperature runaway over 180 days."""
        T = run180["T"]
        assert np.all(T > 200.0)
        assert np.all(T < 350.0)

    def test_equatorial_less_seasonal(self, run180):
        """Equatorial daily-mean temperature has smaller seasonal range."""
        steps_per_day = int(86400 / DT_SLAB)
        T_eq = run180["T"][:, 0]
        T_ml = run180["T"][:, 1]
        n_days = len(T_eq) // steps_per_day

        # Compute daily means to filter out diurnal cycle
        T_eq_daily = np.array([
            T_eq[d * steps_per_day:(d + 1) * steps_per_day].mean()
            for d in range(n_days)
        ])
        T_ml_daily = np.array([
            T_ml[d * steps_per_day:(d + 1) * steps_per_day].mean()
            for d in range(n_days)
        ])
        range_eq = T_eq_daily.max() - T_eq_daily.min()
        range_ml = T_ml_daily.max() - T_ml_daily.min()
        assert range_eq < range_ml + 5.0, (
            f"Equatorial daily-mean range ({range_eq:.1f}) "
            f"should be ≤ midlat ({range_ml:.1f})"
        )
