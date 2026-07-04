"""Category 2: Radiation -- Physical Consistency.

Verifies energy balance at TOA, surface fluxes (Stefan-Boltzmann),
heating rate signs and magnitudes, diurnal cycle, and flux-divergence
consistency for gray radiation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.radiation.gray import gray_radiation
from legoesm.atmosphere.physics.radiation.config import (
    GrayRadiationConfig,
    RadiationConfig,
)
from legoesm.atmosphere.physics.radiation.solar import (
    cos_zenith_angle,
    daily_mean_insolation,
    perpetual_equinox_insolation,
)


# ---------------------------------------------------------------------------
# Helper: build a realistic column
# ---------------------------------------------------------------------------

def _make_tropical_column(nlev=20):
    """Build a single tropical column with standard lapse rate.

    Returns T, p_full, p_half, T_sfc, lat, q_v as (1, nlev) etc.
    """
    p_s = 1.0e5  # Pa
    # Sigma levels uniformly spaced
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    p_half = sigma_half * p_s   # (nlev+1,)
    p_full = sigma_full * p_s   # (nlev,)

    # Standard lapse rate: T = T_sfc - gamma * z, approximated via
    # T(sigma) = T_sfc * sigma^(R*gamma/g) with gamma ~ 6.5 K/km
    T_sfc_val = 300.0
    T = T_sfc_val * jnp.clip(sigma_full, 0.01, None) ** 0.19
    T = jnp.maximum(T, 200.0)

    # Reshape to (1, nlev) column format
    T = T[None, :]
    p_full = p_full[None, :]
    p_half = p_half[None, :]
    T_sfc = jnp.array([T_sfc_val])
    lat = jnp.array([0.0])  # equator

    # Moist profile: 80% RH
    es = 611.2 * jnp.exp(17.67 * (T - constants.T_freeze) / (T - 29.65))
    q_sat = constants.epsilon * es / jnp.maximum(p_full - es, 1.0)
    q_v = 0.8 * q_sat

    return T, p_full, p_half, T_sfc, lat, q_v


# ============================================================================
# 2a  Energy balance at TOA
# ============================================================================

class TestTOAEnergyBalance:
    """Test TOA flux consistency for gray radiation."""

    def test_sw_down_toa_matches_insolation(self):
        """SW_down at TOA should approximately equal insolation."""
        T, p_full, p_half, T_sfc, lat, q_v = _make_tropical_column()
        config = GrayRadiationConfig()
        insol = perpetual_equinox_insolation(lat, config.S_0)

        out = gray_radiation(T, p_full, p_half, T_sfc, lat, q_v, insol, config)
        sw_down_toa = out.sw_flux_down[0, 0]

        # At TOA (sigma=0), tau_sw should be ~0, so SW_down_TOA ~ insolation
        assert float(jnp.abs(sw_down_toa - insol[0])) < 1.0, (
            f"SW_down_TOA={float(sw_down_toa):.1f} != insol={float(insol[0]):.1f}"
        )

    def test_net_sw_toa_non_negative(self):
        """Planet absorbs shortwave: net SW at TOA >= 0."""
        T, p_full, p_half, T_sfc, lat, q_v = _make_tropical_column()
        config = GrayRadiationConfig()
        insol = perpetual_equinox_insolation(lat, config.S_0)

        out = gray_radiation(T, p_full, p_half, T_sfc, lat, q_v, insol, config)
        net_sw_toa = out.sw_flux_down[0, 0] - out.sw_flux_up[0, 0]
        assert float(net_sw_toa) >= 0.0, f"Net SW TOA = {float(net_sw_toa):.1f} < 0"

    def test_olr_in_physical_range(self):
        """OLR should be in [100, 400] W/m^2 for realistic profiles."""
        T, p_full, p_half, T_sfc, lat, q_v = _make_tropical_column()
        config = GrayRadiationConfig()
        insol = perpetual_equinox_insolation(lat, config.S_0)

        out = gray_radiation(T, p_full, p_half, T_sfc, lat, q_v, insol, config)
        olr = float(out.lw_flux_up[0, 0] - out.lw_flux_down[0, 0])
        assert 100.0 <= olr <= 400.0, f"OLR = {olr:.1f} W/m^2 out of [100, 400]"


# ============================================================================
# 2b  Surface radiative fluxes
# ============================================================================

class TestSurfaceFluxes:
    """Test surface flux physical consistency."""

    def test_lw_up_sfc_stefan_boltzmann(self):
        """LW_up at surface should be approximately eps * sigma * T_sfc^4."""
        T, p_full, p_half, T_sfc, lat, q_v = _make_tropical_column()
        config = GrayRadiationConfig()  # eps = 1.0
        insol = perpetual_equinox_insolation(lat, config.S_0)

        out = gray_radiation(T, p_full, p_half, T_sfc, lat, q_v, insol, config)
        lw_up_sfc = float(out.lw_flux_up[0, -1])
        expected = float(config.sfc_emissivity * constants.sigma_sb * T_sfc[0] ** 4)
        # Allow 5% for reflected LW component
        rel_err = abs(lw_up_sfc - expected) / expected
        assert rel_err < 0.05, (
            f"LW_up_sfc={lw_up_sfc:.1f} vs Stefan-Boltzmann={expected:.1f}, "
            f"rel_err={rel_err:.3f}"
        )

    def test_lw_up_sfc_exceeds_lw_down_sfc(self):
        """Surface is warmer than atmosphere: LW_up_sfc > LW_down_sfc."""
        T, p_full, p_half, T_sfc, lat, q_v = _make_tropical_column()
        config = GrayRadiationConfig()
        insol = perpetual_equinox_insolation(lat, config.S_0)

        out = gray_radiation(T, p_full, p_half, T_sfc, lat, q_v, insol, config)
        lw_up_sfc = float(out.lw_flux_up[0, -1])
        lw_down_sfc = float(out.lw_flux_down[0, -1])
        assert lw_up_sfc > lw_down_sfc, (
            f"LW_up_sfc={lw_up_sfc:.1f} <= LW_down_sfc={lw_down_sfc:.1f}"
        )

    def test_sw_down_sfc_less_than_toa(self):
        """Atmosphere absorbs some SW: SW_down_sfc < SW_down_TOA."""
        T, p_full, p_half, T_sfc, lat, q_v = _make_tropical_column()
        config = GrayRadiationConfig()
        insol = perpetual_equinox_insolation(lat, config.S_0)

        out = gray_radiation(T, p_full, p_half, T_sfc, lat, q_v, insol, config)
        sw_down_toa = float(out.sw_flux_down[0, 0])
        sw_down_sfc = float(out.sw_flux_down[0, -1])
        assert sw_down_sfc < sw_down_toa, (
            f"SW_down_sfc={sw_down_sfc:.1f} >= SW_down_TOA={sw_down_toa:.1f}"
        )


# ============================================================================
# 2c  Heating rate signs
# ============================================================================

class TestHeatingRateSigns:
    """Test heating rate sign conventions."""

    def test_sw_heating_non_negative(self):
        """SW always heats: SW heating rate >= 0 everywhere."""
        T, p_full, p_half, T_sfc, lat, q_v = _make_tropical_column()
        config = GrayRadiationConfig()
        insol = perpetual_equinox_insolation(lat, config.S_0)

        out = gray_radiation(T, p_full, p_half, T_sfc, lat, q_v, insol, config)
        sw_hr = out.sw_heating_rate[0]  # (nlev,)
        min_sw_hr = float(jnp.min(sw_hr))
        assert min_sw_hr >= -1e-10, (
            f"SW heating rate has negative values: min = {min_sw_hr:.2e} K/s"
        )

    def test_lw_cooling_in_troposphere(self):
        """LW should cool in lower/mid troposphere for tropical column."""
        T, p_full, p_half, T_sfc, lat, q_v = _make_tropical_column()
        config = GrayRadiationConfig()
        insol = perpetual_equinox_insolation(lat, config.S_0)

        out = gray_radiation(T, p_full, p_half, T_sfc, lat, q_v, insol, config)
        lw_hr = out.lw_heating_rate[0]  # (nlev,)
        # Mid-troposphere: roughly sigma 0.3 to 0.8 -> levels 6 to 16 in 20-level
        nlev = lw_hr.shape[0]
        mid_start = nlev // 4
        mid_end = 3 * nlev // 4
        mid_lw = lw_hr[mid_start:mid_end]
        # Most mid-tropospheric levels should have negative LW heating (cooling)
        n_cooling = int(jnp.sum(mid_lw < 0))
        n_total = mid_end - mid_start
        frac_cooling = n_cooling / n_total
        assert frac_cooling > 0.5, (
            f"Only {frac_cooling:.0%} of mid-trop levels have LW cooling "
            f"(expected > 50%)"
        )


# ============================================================================
# 2d  Heating rate magnitudes
# ============================================================================

class TestHeatingRateMagnitudes:
    """Test heating rate magnitude bounds."""

    def test_heating_rate_bounded(self):
        """|dT/dt| < 50 K/day = 5.8e-4 K/s."""
        T, p_full, p_half, T_sfc, lat, q_v = _make_tropical_column()
        config = GrayRadiationConfig()
        insol = perpetual_equinox_insolation(lat, config.S_0)

        out = gray_radiation(T, p_full, p_half, T_sfc, lat, q_v, insol, config)
        max_hr = float(jnp.max(jnp.abs(out.heating_rate)))
        max_K_per_day = max_hr * 86400.0
        assert max_K_per_day < 50.0, (
            f"|dT/dt| = {max_K_per_day:.1f} K/day exceeds 50 K/day"
        )

    def test_flux_divergence_matches_heating(self):
        """Column-integrated flux divergence matches integral of rho*cp*dT/dt*dz.

        (F_TOA_net - F_sfc_net) ~ -integral(rho * cp * heating_rate * dp/g)
        """
        T, p_full, p_half, T_sfc, lat, q_v = _make_tropical_column()
        config = GrayRadiationConfig()
        insol = perpetual_equinox_insolation(lat, config.S_0)

        out = gray_radiation(T, p_full, p_half, T_sfc, lat, q_v, insol, config)

        # Net upward flux at TOA and surface
        F_net_up_toa = (out.lw_flux_up[0, 0] + out.sw_flux_up[0, 0]
                        - out.lw_flux_down[0, 0] - out.sw_flux_down[0, 0])
        F_net_up_sfc = (out.lw_flux_up[0, -1] + out.sw_flux_up[0, -1]
                        - out.lw_flux_down[0, -1] - out.sw_flux_down[0, -1])
        flux_divergence = float(F_net_up_toa - F_net_up_sfc)

        # Integrated heating: -integral(cp * dT/dt * dp/g)
        dp = p_half[0, 1:] - p_half[0, :-1]
        integrated_heating = -float(
            jnp.sum(constants.c_pd * out.heating_rate[0] * dp / constants.g)
        )

        # These should match within 5%
        if abs(flux_divergence) > 1.0:
            rel_err = abs(flux_divergence - integrated_heating) / abs(flux_divergence)
            assert rel_err < 0.05, (
                f"Flux div = {flux_divergence:.2f} W/m^2, "
                f"Integrated heating = {integrated_heating:.2f} W/m^2, "
                f"rel_err = {rel_err:.3f}"
            )


# ============================================================================
# 2e  Diurnal cycle consistency
# ============================================================================

class TestDiurnalCycle:
    """Test diurnal cycle solar geometry."""

    def test_noon_positive_insolation(self):
        """At local noon (hour=12, lon=0), SW should be positive."""
        lat = jnp.array([0.3])   # ~17 degrees N
        lon = jnp.array([0.0])   # prime meridian
        # At equinox (day 80), noon UTC, lon=0 => local noon
        cos_sza = cos_zenith_angle(lat, lon, day_of_year=80.0, hour=12.0)
        assert float(cos_sza[0]) > 0.0, f"cos(SZA) = {float(cos_sza[0]):.3f} <= 0 at noon"

    def test_midnight_zero_insolation(self):
        """At local midnight, SW_down_TOA should be zero."""
        lat = jnp.array([0.3])
        lon = jnp.array([0.0])
        # Midnight UTC at lon=0 is hour=0
        cos_sza = cos_zenith_angle(lat, lon, day_of_year=80.0, hour=0.0)
        # At equinox, midnight should have negative cos_sza (sun below horizon)
        assert float(cos_sza[0]) <= 0.0, (
            f"cos(SZA) = {float(cos_sza[0]):.3f} > 0 at midnight"
        )

    def test_daily_mean_matches_perpetual_equinox(self):
        """Daily-mean insolation at equinox matches perpetual-equinox formula."""
        lats = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 50)
        S_0 = 1360.0

        Q_daily = daily_mean_insolation(lats, day_of_year=80.0, S_0=S_0)
        Q_perp = perpetual_equinox_insolation(lats, S_0=S_0)

        # At equinox these should match within 1%
        # Avoid division by zero near poles
        mask = Q_perp > 10.0
        rel_err = jnp.abs(Q_daily - Q_perp) / Q_perp
        max_rel_err = float(jnp.max(rel_err * mask))
        assert max_rel_err < 0.01, (
            f"Daily-mean vs perpetual equinox: max rel err = {max_rel_err:.4f}"
        )

    def test_daily_mean_insolation_polar_ad_safe(self):
        """Iter-92 regression: daily-mean insolation must produce finite
        gradients at polar day / polar night.

        Previous code clipped ``cos(h_s)`` exactly to ``[-1, 1]`` before
        calling ``arccos``, but ``d/dx arccos(x) = -1/sqrt(1 - x²)``
        blows up at the boundary.  Combined with the zero subgradient
        through ``clip`` itself, JAX evaluates ``0 · ∞`` on the backward
        pass and emits ``NaN`` for every column inside the polar-day or
        polar-night cap.  Sweep lat × doy and confirm both the forward
        value and ``d/dlat``, ``d/dDOY`` are finite.
        """
        from legoesm.atmosphere.physics.radiation.solar import (
            daily_mean_insolation, daylight_fraction,
        )
        for lat_deg in (88.0, 80.0, -80.0, -88.0):
            lat = jnp.deg2rad(lat_deg)
            for doy in (1.0, 80.0, 172.0, 265.0):
                Q = daily_mean_insolation(lat, doy)
                g_lat = jax.grad(daily_mean_insolation, argnums=0)(lat, doy)
                g_doy = jax.grad(daily_mean_insolation, argnums=1)(lat, doy)
                f = daylight_fraction(lat, doy)
                g_f = jax.grad(daylight_fraction, argnums=0)(lat, doy)
                assert jnp.isfinite(Q), f"NaN Q at lat={lat_deg}, doy={doy}"
                assert jnp.isfinite(g_lat), (
                    f"NaN dQ/dlat at lat={lat_deg}, doy={doy}"
                )
                assert jnp.isfinite(g_doy), (
                    f"NaN dQ/dDOY at lat={lat_deg}, doy={doy}"
                )
                assert jnp.isfinite(f), f"NaN daylight at lat={lat_deg}, doy={doy}"
                assert jnp.isfinite(g_f), (
                    f"NaN d(daylight)/dlat at lat={lat_deg}, doy={doy}"
                )


# ============================================================================
# 2f  Gray radiation qualitative checks
# ============================================================================

class TestGrayQualitative:
    """Qualitative consistency checks for gray radiation."""

    def test_net_radiative_cooling_troposphere(self):
        """Tropical troposphere should have net radiative cooling."""
        T, p_full, p_half, T_sfc, lat, q_v = _make_tropical_column()
        config = GrayRadiationConfig()
        insol = perpetual_equinox_insolation(lat, config.S_0)

        out = gray_radiation(T, p_full, p_half, T_sfc, lat, q_v, insol, config)
        total_hr = out.heating_rate[0]  # (nlev,)
        nlev = total_hr.shape[0]
        # Mid-troposphere
        mid_start = nlev // 4
        mid_end = 3 * nlev // 4
        mid_mean = float(jnp.mean(total_hr[mid_start:mid_end]))
        # Net radiative effect should be cooling (negative)
        assert mid_mean < 0.0, (
            f"Mid-trop mean heating = {mid_mean:.2e} K/s (expected negative)"
        )

    def test_olr_increases_with_temperature(self):
        """Warmer surface -> higher OLR."""
        config = GrayRadiationConfig()

        olrs = []
        for T_sfc_val in [280.0, 300.0, 320.0]:
            T, p_full, p_half, T_sfc, lat, q_v = _make_tropical_column()
            T_sfc = jnp.array([T_sfc_val])
            insol = perpetual_equinox_insolation(lat, config.S_0)
            out = gray_radiation(T, p_full, p_half, T_sfc, lat, None, insol, config)
            olr = float(out.lw_flux_up[0, 0])
            olrs.append(olr)

        assert olrs[0] < olrs[1] < olrs[2], (
            f"OLR not increasing with T_sfc: {olrs}"
        )


# ============================================================================
# 2f  Per-column surface albedo (blended-albedo threading, 2026-06-10)
# ============================================================================

class TestPerColumnAlbedo:
    """Gray SW must consume a per-column surface albedo when given one,
    fall back byte-identically to ``config.sfc_albedo`` otherwise, and
    stay differentiable w.r.t. the albedo."""

    def _run(self, sfc_albedo=None, config=None):
        T, p_full, p_half, T_sfc, lat, q_v = _make_tropical_column()
        cfg = config or GrayRadiationConfig()
        insol = jnp.array([400.0])
        return gray_radiation(
            T=T, p_full=p_full, p_half=p_half, sfc_temperature=T_sfc,
            lat=lat, q_v=q_v, insolation=insol, config=cfg,
            sfc_albedo=sfc_albedo,
        )

    def test_none_matches_config_scalar(self):
        cfg = GrayRadiationConfig(sfc_albedo=0.21)
        out_none = self._run(config=cfg)
        out_explicit = self._run(sfc_albedo=jnp.array([0.21]), config=cfg)
        assert jnp.array_equal(out_none.sw_flux_up, out_explicit.sw_flux_up)
        assert jnp.array_equal(out_none.heating_rate, out_explicit.heating_rate)

    def test_higher_albedo_reflects_more(self):
        dark = self._run(sfc_albedo=jnp.array([0.06]))
        bright = self._run(sfc_albedo=jnp.array([0.65]))
        # Same downwelling beam, more reflected upward SW everywhere.
        assert jnp.array_equal(dark.sw_flux_down, bright.sw_flux_down)
        assert float(bright.sw_flux_up[0, 0]) > float(dark.sw_flux_up[0, 0])
        # Reflection ratio at the surface equals the albedo ratio.
        r = float(bright.sw_flux_up[0, -1] / dark.sw_flux_up[0, -1])
        assert abs(r - 0.65 / 0.06) < 1e-6

    def test_albedo_gradient_flows(self):
        def loss(alpha):
            out = self._run(sfc_albedo=alpha)
            return jnp.sum(out.sw_flux_up[:, 0])  # reflected SW at TOA

        g = jax.grad(loss)(jnp.array([0.3]))
        assert bool(jnp.all(jnp.isfinite(g)))
        assert float(jnp.abs(g[0])) > 0.0

    def test_pipeline_blend_reaches_gray(self):
        """Integration: sea-ice fraction must change gray SW through the
        pipeline's blended albedo (previously albedo_col was dropped)."""
        import numpy as np
        from legoesm.driver.config import (
            DycoreConfig, ExperimentConfig, GridConfig,
        )
        from legoesm.driver.physics_pipeline import build_physics_pipeline
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import make_hybrid_levels

        nlev = 8
        grid = create_cubed_sphere(4)
        sigma = make_hybrid_levels(nlev)
        s2 = grid.grid_lat.shape
        s3 = (*s2, nlev)
        config = ExperimentConfig(
            grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=nlev),
            dycore=DycoreConfig(
                model_type="hydrostatic", discretization="cdgrid"),
            convection="none", radiation="gray", microphysics="none",
        )
        config.validate_strict()
        pipe = build_physics_pipeline(grid, sigma, config)

        T = jnp.asarray(np.broadcast_to(
            300.0 - 60.0 * np.linspace(0, 1, nlev)[::-1], s3).copy())
        p_s = jnp.full(s2, 1.0e5)
        q_v = jnp.full(s3, 5e-3)
        sst = jnp.full(s2, 290.0)
        lat = jnp.asarray(grid.grid_lat)
        lon = jnp.asarray(grid.grid_lon)

        def rad(sic):
            return pipe.compute_radiation_core(
                T, p_s, q_v, sst, sic, lat, lon,
                jnp.asarray(80.0), jnp.asarray(43200.0),
                None, jnp.asarray(1361.0), None, None,
            )

        out_ocean = rad(jnp.zeros(s2))
        out_ice = rad(jnp.ones(s2))
        # Ice (albedo 0.65) vs ocean (0.06): less SW absorbed at the
        # surface and a different heating profile.
        assert float(jnp.max(out_ocean[1] - out_ice[1])) > 1.0  # sw_net_sfc
        assert float(jnp.max(jnp.abs(out_ocean[0] - out_ice[0]))) > 0.0


class TestSfcEmissivityOverrideThreading:
    """Coupler-threaded dynamic surface emissivity reaches the radiation solver.

    The land tile exports a LAI-dependent ``eps_eff`` (consistent with its
    conservative ``LW_out`` / emission-weighted ``T_surface``); the coupler
    tile-blends it and threads it as ``sfc_emissivity_override`` so the
    atmospheric LW boundary uses the SAME emissivity.  Gray radiation ignores
    surface emissivity, so we assert at the solver boundary: the emissivity
    array passed to ``radiation_fn`` (position 10, consumed by RRTMGP as
    ``sfc_emissivity``) equals the override, and the static config blend when
    no override is supplied.
    """

    def _build(self):
        import numpy as np
        from legoesm.driver.config import (
            DycoreConfig, ExperimentConfig, GridConfig,
        )
        from legoesm.driver.physics_pipeline import build_physics_pipeline
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import make_hybrid_levels

        nlev = 8
        grid = create_cubed_sphere(4)
        sigma = make_hybrid_levels(nlev)
        s2 = grid.grid_lat.shape
        s3 = (*s2, nlev)
        config = ExperimentConfig(
            grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=nlev),
            dycore=DycoreConfig(
                model_type="hydrostatic", discretization="cdgrid"),
            convection="none", radiation="gray", microphysics="none",
        )
        config.validate_strict()
        pipe = build_physics_pipeline(grid, sigma, config)
        T = jnp.asarray(np.broadcast_to(
            300.0 - 60.0 * np.linspace(0, 1, nlev)[::-1], s3).copy())
        p_s = jnp.full(s2, 1.0e5)
        q_v = jnp.full(s3, 5e-3)
        sst = jnp.full(s2, 290.0)
        lat = jnp.asarray(grid.grid_lat)
        lon = jnp.asarray(grid.grid_lon)
        ncol = int(np.prod(s2))
        return pipe, (T, p_s, q_v, sst, jnp.zeros(s2), lat, lon), s2, (ncol, nlev)

    def _capture_solver_emissivity(self, pipe, args, emis_ovr, ncol_nlev):
        from types import SimpleNamespace
        ncol, nlev = ncol_nlev
        captured = {}

        def stub(*a, **k):
            captured["emis"] = a[10]  # emis_col is the 11th positional arg
            z2 = jnp.zeros((ncol, nlev))
            z3 = jnp.zeros((ncol, nlev + 1))
            return SimpleNamespace(
                heating_rate=z2, sw_flux_down=z3, sw_flux_up=z3,
                lw_flux_down=z3, lw_flux_up=z3,
            )

        T, p_s, q_v, sst, sic, lat, lon = args
        orig = pipe.radiation_fn
        pipe.radiation_fn = stub
        try:
            pipe.compute_radiation_core(
                T, p_s, q_v, sst, sic, lat, lon,
                jnp.asarray(80.0), jnp.asarray(43200.0),
                None, jnp.asarray(1361.0), None, None,
                sfc_emissivity_override=emis_ovr,
            )
        finally:
            pipe.radiation_fn = orig
        return captured["emis"]

    def test_override_controls_solver_emissivity(self):
        pipe, args, s2, ncol_nlev = self._build()
        emis = self._capture_solver_emissivity(
            pipe, args, jnp.full(s2, 0.80), ncol_nlev)
        assert jnp.allclose(emis, 0.80)

    def test_none_uses_static_config_blend(self):
        pipe, args, s2, ncol_nlev = self._build()
        emis = self._capture_solver_emissivity(pipe, args, None, ncol_nlev)
        # Static all-ocean blend = config ocean emissivity, NOT the override,
        # and physical.  (Byte-identical to the pre-feedback behaviour.)
        assert not jnp.allclose(emis, 0.80)
        assert bool(jnp.all(emis > 0.0)) and bool(jnp.all(emis <= 1.0))
