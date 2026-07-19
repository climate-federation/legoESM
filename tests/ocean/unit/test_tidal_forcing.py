"""Unit tests for the astronomical (equilibrium) tidal forcing.

Exercises the leaf module ``ocean.physics.tidal_forcing`` directly:

* each constituent's angular frequency matches its known period
  (M2=12.4206h, S2=12h, K1=23.9345h, O1=25.819h) with an independent
  omega/period cross-check;
* the degree-2 species geometry (semidiurnal ~ cos^2 peaks at the equator,
  diurnal ~ sin2phi peaks at +/-45 deg, long-period ~ (3sin^2-1)/2);
* the global area-weighted mean of eta_eq ~ 0 (no degree-0 part);
* the momentum forcing a = +g grad(eta_eq) points TOWARD the equilibrium high
  (sub-lunar bulge) — the load-bearing sign choice;
* the scalar Love / SAL / amplitude corrections;
* differentiability wrt time and a config amplitude, and JIT safety;
* the enabled=False application hook is a byte-identical no-op (regression guard);
* dispatch hardening on the species code.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)  # sci test: float64 frequency identities

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.grids.latlon import create_latlon_grid  # noqa: E402
from legoesm.grids.operators_latlon_cgrid import (  # noqa: E402
    gradient_x_cgrid,
    gradient_y_cgrid,
)
from legoesm.ocean.physics import tidal_forcing as tf  # noqa: E402
from legoesm.ocean.physics.tidal_forcing import (  # noqa: E402
    TidalForcingConfig,
    _CONSTITUENTS,
    _SPECIES_DIURNAL,
    _SPECIES_LONG_PERIOD,
    _SPECIES_SEMIDIURNAL,
    apply_tidal_forcing,
    equilibrium_tide_elevation,
    tidal_acceleration,
)

TWO_PI = 2.0 * np.pi

# Independent (textbook) constituent periods [h] — the "known period" source of
# truth the stored omega is checked against (NOT derived from the table).
_KNOWN_PERIOD_H = {"M2": 12.4206, "S2": 12.000, "K1": 23.9345, "O1": 25.819}


def _species_only_config(species: int, **kw) -> TidalForcingConfig:
    """Config enabling ONLY the given species (all constituents of that species)."""
    return TidalForcingConfig(
        enabled=True,
        include_semidiurnal=(species == _SPECIES_SEMIDIURNAL),
        include_diurnal=(species == _SPECIES_DIURNAL),
        include_long_period=(species == _SPECIES_LONG_PERIOD),
        **kw,
    )


def _species_amplitude_sum(species: int) -> float:
    return float(sum(c.amplitude_m for c in _CONSTITUENTS if c.species == species))


# ---------------------------------------------------------------------------
# 1. Frequencies vs known periods
# ---------------------------------------------------------------------------
def test_table_membership_and_species_counts():
    names = [c.name for c in _CONSTITUENTS]
    assert names == ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "Mf", "Mm"]
    counts = {s: sum(c.species == s for c in _CONSTITUENTS) for s in (0, 1, 2)}
    assert counts == {_SPECIES_SEMIDIURNAL: 4, _SPECIES_DIURNAL: 4, _SPECIES_LONG_PERIOD: 2}


def test_omega_matches_stored_period_crosscheck():
    """2*pi/omega == period (independent literals) — catches a transcription typo."""
    for c in _CONSTITUENTS:
        period_from_omega_s = TWO_PI / c.omega_rad_s
        stored_period_s = c.period_hours * 3600.0
        assert abs(period_from_omega_s - stored_period_s) / stored_period_s < 1e-6, c.name


def test_omega_matches_known_textbook_periods():
    by = {c.name: c for c in _CONSTITUENTS}
    for name, period_h in _KNOWN_PERIOD_H.items():
        implied_h = (TWO_PI / by[name].omega_rad_s) / 3600.0
        assert abs(implied_h - period_h) < 1e-3, (name, implied_h, period_h)
    # M2 magnitude explicitly (task spec: omega ~ 1.405e-4 rad/s)
    assert abs(by["M2"].omega_rad_s - 1.405e-4) < 1e-7
    # K1 is the lunisolar diurnal ~ sidereal day => omega ~ Earth's rotation rate.
    assert abs(by["K1"].omega_rad_s - constants.Omega) < 1e-6


# ---------------------------------------------------------------------------
# 2. Species geometry
# ---------------------------------------------------------------------------
def test_semidiurnal_geometry_peaks_at_equator():
    lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 91)
    lon = jnp.zeros_like(lat)  # lambda=0 (+ t=0, chi=0) => cos(arg)=1 for all
    cfg = _species_only_config(_SPECIES_SEMIDIURNAL)
    eta = equilibrium_tide_elevation(lat, lon, 0.0, cfg)
    scale = cfg.beta_sal * cfg.love_factor * cfg.amplitude_scale
    expected = scale * _species_amplitude_sum(_SPECIES_SEMIDIURNAL) * jnp.cos(lat) ** 2
    np.testing.assert_allclose(np.asarray(eta), np.asarray(expected), atol=1e-12)
    # peaks at the equator, vanishes at the poles
    assert int(np.argmax(np.asarray(eta))) == 45  # equator index (91 pts -> mid)
    assert abs(float(eta[0])) < 1e-9 and abs(float(eta[-1])) < 1e-9


def test_diurnal_geometry_peaks_at_45deg():
    lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 361)
    lon = jnp.zeros_like(lat)
    cfg = _species_only_config(_SPECIES_DIURNAL)
    eta = equilibrium_tide_elevation(lat, lon, 0.0, cfg)
    scale = cfg.beta_sal * cfg.love_factor * cfg.amplitude_scale
    expected = scale * _species_amplitude_sum(_SPECIES_DIURNAL) * jnp.sin(2.0 * lat)
    np.testing.assert_allclose(np.asarray(eta), np.asarray(expected), atol=1e-12)
    # zero at the equator and poles, +max near +45 deg, antisymmetric
    i_eq = 180
    assert abs(float(eta[i_eq])) < 1e-9  # equator
    assert abs(float(eta[0])) < 1e-9 and abs(float(eta[-1])) < 1e-9  # poles
    lat_at_max = float(lat[int(np.argmax(np.asarray(eta)))])
    assert abs(lat_at_max - np.pi / 4) < 2.0 * (np.pi / 360)  # within one cell of +45
    np.testing.assert_allclose(np.asarray(eta), -np.asarray(eta)[::-1], atol=1e-12)


def test_long_period_geometry_structure():
    lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 361)
    lon = jnp.zeros_like(lat)
    cfg = _species_only_config(_SPECIES_LONG_PERIOD)
    eta = equilibrium_tide_elevation(lat, lon, 0.0, cfg)
    scale = cfg.beta_sal * cfg.love_factor * cfg.amplitude_scale
    amp = _species_amplitude_sum(_SPECIES_LONG_PERIOD)
    expected = scale * amp * (3.0 * jnp.sin(lat) ** 2 - 1.0) / 2.0
    np.testing.assert_allclose(np.asarray(eta), np.asarray(expected), atol=1e-12)
    # negative at the equator (-amp/2 * scale), positive at the poles (+amp * scale)
    i_eq = 180
    assert float(eta[i_eq]) < 0.0
    assert float(eta[0]) > 0.0 and float(eta[-1]) > 0.0
    # zero crossing where sin^2(phi) = 1/3  -> |phi| = arcsin(1/sqrt(3)) ~ 35.26 deg
    phi0 = np.arcsin(1.0 / np.sqrt(3.0))
    eta_at_cross = float(equilibrium_tide_elevation(jnp.array([phi0]), jnp.array([0.0]), 0.0, cfg)[0])
    assert abs(eta_at_cross) < 1e-9


# ---------------------------------------------------------------------------
# 3. Global area-weighted mean ~ 0 (no degree-0 part)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("t", [0.0, 1.0e4, 3.73e4, 1.0e5, 5.0e5])
def test_global_area_weighted_mean_near_zero(t):
    grid = create_latlon_grid(90, 180)
    cfg = TidalForcingConfig(enabled=True)  # all species
    eta = equilibrium_tide_elevation(grid.lat2d, grid.lon2d, t, cfg)
    weight = jnp.broadcast_to(grid.cos_lat[:, None], eta.shape)  # cell area ~ cos(lat)
    area_mean = float(jnp.sum(eta * weight) / jnp.sum(weight))
    # Longitude-dependent species vanish EXACTLY in the lon sum; the residual is
    # the latitude-quadrature error of the long-period (3sin^2-1)/2 term.
    assert abs(area_mean) < 1e-4, (t, area_mean)


# ---------------------------------------------------------------------------
# 4. Momentum forcing sign / direction: a = +g grad(eta_eq) toward the bulge
# ---------------------------------------------------------------------------
def test_acceleration_points_toward_equilibrium_high():
    grid = create_latlon_grid(60, 120)
    cfg = _species_only_config(_SPECIES_SEMIDIURNAL)  # ~ cos^2(phi) cos(2 lambda)
    eta = equilibrium_tide_elevation(grid.lat2d, grid.lon2d, 0.0, cfg)
    a_x, a_y = tidal_acceleration(grid, 0.0, cfg)
    assert a_x.shape == (60, 121) and a_y.shape == (61, 120)

    # Comprehensive: on every INTERIOR u-face, a_x has the SAME sign as
    # (eta_east - eta_west) => the tractive force points UP the elevation
    # gradient, TOWARD the higher eta (the bulge). A -g grad choice would give
    # the OPPOSITE sign everywhere and fail this.
    eta_east = np.asarray(eta[:, 1:])
    eta_west = np.asarray(eta[:, :-1])
    a_x_interior = np.asarray(a_x[:, 1:-1])
    assert np.max(np.abs(a_x_interior)) > 0.0  # non-vacuous
    assert np.min(a_x_interior * (eta_east - eta_west)) >= -1e-30

    # Concrete: the M2-family bulge sits at lambda=0 (cell 0) at t=0; just EAST of
    # it the elevation falls, so the eastward acceleration on the first interior
    # u-face is NEGATIVE (points back west, toward the bulge).
    i_eq = int(np.argmin(np.abs(np.asarray(grid.lat2d[:, 0]))))
    assert float(np.asarray(eta)[i_eq, 0]) == float(np.max(np.asarray(eta)[i_eq, :]))
    assert float(a_x[i_eq, 1]) < 0.0


def test_acceleration_is_plus_g_grad_eta():
    """Discrete-consistency (both components) + the model PGF equivalence.

    a = +g*grad(eta_eq) exactly, and the tidal contribution is the difference
    between the standard PGF applied to (eta - eta_eq) vs eta:
    -g*grad(eta - eta_eq) == -g*grad(eta) + a_tide.
    """
    grid = create_latlon_grid(48, 96)
    cfg = TidalForcingConfig(enabled=True)
    eta = equilibrium_tide_elevation(grid.lat2d, grid.lon2d, 1.2e4, cfg)
    a_x, a_y = tidal_acceleration(grid, 1.2e4, cfg)
    np.testing.assert_allclose(
        np.asarray(a_x), np.asarray(constants.g * gradient_x_cgrid(eta, grid)), rtol=0, atol=0)
    np.testing.assert_allclose(
        np.asarray(a_y), np.asarray(constants.g * gradient_y_cgrid(eta, grid)), rtol=0, atol=0)

    # Model PGF form: replacing eta by (eta_model - eta_eq) in -g*grad(eta) adds
    # exactly the tidal acceleration. Use a synthetic model SSH eta_model.
    eta_model = 0.3 * jnp.sin(grid.lat2d) * jnp.cos(grid.lon2d)
    pgf_with_tide = -constants.g * gradient_x_cgrid(eta_model - eta, grid)
    pgf_plain = -constants.g * gradient_x_cgrid(eta_model, grid)
    # Mathematically exact (the gradient operator is linear); FP re-association of
    # the compact difference + 1/dx division gives ~1e-11 absolute slack.
    np.testing.assert_allclose(
        np.asarray(pgf_with_tide), np.asarray(pgf_plain + a_x), rtol=1e-5, atol=1e-9)


def test_jit_safe_with_static_config():
    """config MUST be a static (compile-time) arg — its bools gate control flow."""
    grid = create_latlon_grid(16, 32)
    # config passed as a STATIC argument (static_argnums); t_seconds is traced.
    a_x, a_y = jax.jit(
        lambda t, cfg: tidal_acceleration(grid, t, cfg),
        static_argnums=(1,),
    )(4.2e4, TidalForcingConfig(enabled=True))
    assert np.all(np.isfinite(np.asarray(a_x))) and np.all(np.isfinite(np.asarray(a_y)))


# ---------------------------------------------------------------------------
# Scalar corrections (Love / SAL / amplitude)
# ---------------------------------------------------------------------------
def test_scalar_corrections_scale_the_elevation():
    lat = jnp.array([[0.2]])
    lon = jnp.array([[0.5]])
    raw = equilibrium_tide_elevation(
        lat, lon, 3.0e4, TidalForcingConfig(enabled=True, love_factor=1.0, beta_sal=1.0, amplitude_scale=1.0)
    )
    default = equilibrium_tide_elevation(lat, lon, 3.0e4, TidalForcingConfig(enabled=True))
    doubled = equilibrium_tide_elevation(lat, lon, 3.0e4, TidalForcingConfig(enabled=True, amplitude_scale=2.0))
    np.testing.assert_allclose(np.asarray(default), 0.69 * 0.94 * np.asarray(raw), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(doubled), 2.0 * np.asarray(default), rtol=1e-12)


def test_species_selector_is_additive():
    lat = jnp.array([[0.3]])
    lon = jnp.array([[0.7]])
    t = 2.0e4
    full = equilibrium_tide_elevation(lat, lon, t, TidalForcingConfig(enabled=True))
    parts = sum(
        equilibrium_tide_elevation(lat, lon, t, _species_only_config(s))
        for s in (_SPECIES_SEMIDIURNAL, _SPECIES_DIURNAL, _SPECIES_LONG_PERIOD)
    )
    np.testing.assert_allclose(np.asarray(full), np.asarray(parts), rtol=1e-12)


# ---------------------------------------------------------------------------
# 5. Differentiability + JIT
# ---------------------------------------------------------------------------
def test_grad_wrt_time_and_config_amplitude_finite():
    grid = create_latlon_grid(24, 48)

    def scalar_of_accel(t, love, scale):
        cfg = TidalForcingConfig(enabled=True, love_factor=love, amplitude_scale=scale)
        a_x, a_y = tidal_acceleration(grid, t, cfg)
        return jnp.sum(a_x ** 2) + jnp.sum(a_y ** 2)

    g_t, g_love, g_scale = jax.grad(scalar_of_accel, argnums=(0, 1, 2))(1.0e4, 0.69, 1.0)
    for g in (g_t, g_love, g_scale):
        assert np.isfinite(float(g))
    assert float(g_t) != 0.0 and float(g_love) != 0.0 and float(g_scale) != 0.0


def test_grad_of_elevation_wrt_time_finite():
    lat = jnp.array([[0.4]])
    lon = jnp.array([[1.0]])
    cfg = TidalForcingConfig(enabled=True)
    g = jax.grad(lambda t: equilibrium_tide_elevation(lat, lon, t, cfg)[0, 0])(5.0e3)
    assert np.isfinite(float(g)) and float(g) != 0.0


def test_jit_safe():
    grid = create_latlon_grid(16, 32)
    cfg = TidalForcingConfig(enabled=True)
    f = jax.jit(lambda lat, lon, t: equilibrium_tide_elevation(lat, lon, t, cfg))
    eta = f(grid.lat2d, grid.lon2d, 4.2e4)
    assert np.all(np.isfinite(np.asarray(eta)))


# ---------------------------------------------------------------------------
# 6. Application hook: disabled = byte-identical no-op (regression guard)
# ---------------------------------------------------------------------------
def test_apply_disabled_is_byte_identical_noop():
    grid = create_latlon_grid(16, 32)
    du = jnp.ones((16, 33))  # u-tendency (n_lat, n_lon+1)
    dv = jnp.full((17, 32), -2.0)  # v-tendency (n_lat+1, n_lon)
    du_off, dv_off = apply_tidal_forcing(du, dv, grid, 1234.0, TidalForcingConfig(enabled=False))
    # returns the SAME objects untouched -> bit-identical to no-tides
    assert du_off is du and dv_off is dv


def test_apply_enabled_adds_the_acceleration():
    grid = create_latlon_grid(16, 32)
    du = jnp.ones((16, 33))
    dv = jnp.full((17, 32), -2.0)
    cfg = TidalForcingConfig(enabled=True)
    du_on, dv_on = apply_tidal_forcing(du, dv, grid, 1234.0, cfg)
    a_x, a_y = tidal_acceleration(grid, 1234.0, cfg)
    # The wrapper casts the tide to the tendency dtype BEFORE adding, so it must
    # be compared against the same cast. This is not a test nicety: production
    # passes float32 F_slow_u/v accumulators, and the cast is what stops a
    # float64 tide from promoting them and breaking the barotropic fori_loop
    # carry-type invariant. Comparing against the un-cast `du + a_x` would pin
    # the pre-cast behaviour this change deliberately removed.
    np.testing.assert_allclose(
        np.asarray(du_on), np.asarray(du + a_x.astype(du.dtype)), rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(dv_on), np.asarray(dv + a_y.astype(dv.dtype)), rtol=1e-12)
    assert du_on.dtype == du.dtype and dv_on.dtype == dv.dtype
    assert not np.array_equal(np.asarray(du_on), np.asarray(du))


def test_apply_preserves_tendency_dtype_carry_invariant():
    """The load-bearing reason the wrapper exists: it must NOT promote the
    tendency dtype. tidal_acceleration can return float64 under x64, and the
    barotropic substep carries float32 F_slow_u/v through a fori_loop whose
    carry type must not change. Feeding a float32 accumulator must return
    float32 -- the exact guarantee the inline call sites hand-rolled and that
    anyone using the wrapper per its docstring now inherits.
    """
    grid = create_latlon_grid(16, 32)
    f_slow_u = jnp.ones((16, 33), dtype=jnp.float32)
    f_slow_v = jnp.full((17, 32), -2.0, dtype=jnp.float32)
    cfg = TidalForcingConfig(enabled=True)
    out_u, out_v = apply_tidal_forcing(f_slow_u, f_slow_v, grid, 1234.0, cfg)
    assert out_u.dtype == jnp.float32 and out_v.dtype == jnp.float32
    assert not np.array_equal(np.asarray(out_u), np.asarray(f_slow_u))


def test_apply_no_time_is_byte_identical_noop():
    """t_seconds=None cannot evaluate an equilibrium tide; the wrapper returns
    the inputs unchanged rather than silently substituting a frozen t=0 tide.
    This matches the inline call sites' `and t_seconds is not None` guard.
    """
    grid = create_latlon_grid(16, 32)
    du = jnp.ones((16, 33))
    dv = jnp.full((17, 32), -2.0)
    du_off, dv_off = apply_tidal_forcing(
        du, dv, grid, None, TidalForcingConfig(enabled=True))
    assert du_off is du and dv_off is dv


def test_apply_none_config_is_byte_identical_noop():
    """config=None (no tidal_forcing on the model config) is a no-op -- the
    call sites pass getattr(config, 'tidal_forcing', None), which is None on
    any ocean config without the field."""
    grid = create_latlon_grid(16, 32)
    du = jnp.ones((16, 33))
    dv = jnp.full((17, 32), -2.0)
    du_off, dv_off = apply_tidal_forcing(du, dv, grid, 1234.0, None)
    assert du_off is du and dv_off is dv


# ---------------------------------------------------------------------------
# Dispatch hardening
# ---------------------------------------------------------------------------
def test_unknown_species_code_raises():
    with pytest.raises(ValueError):
        tf._species_geometry(7, jnp.array(0.0), jnp.array(0.0))
    with pytest.raises(ValueError):
        tf._species_included(7, TidalForcingConfig())


# ---------------------------------------------------------------------------
# End-to-end WIRE: the opt-in body force actually reaches the barotropic step
# (LatLonCGridOceanModel.integrate_scan -> _step_impl -> barotropic substeps).
# ---------------------------------------------------------------------------
_WIRE_DT = 1800.0
_WIRE_NSTEPS = 4


def _tide_basin(tide_cfg, n_lat=8, n_lon=16, nlev=4):
    """Flat-bottom all-ocean basin at REST (u=v=eta=0, uniform T/S) so the ONLY
    forcing is the tide — the tide-on vs tide-off difference is purely the tidal
    body force. Default split-explicit barotropic solver (explicit_substep)."""
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    grid = create_latlon_grid(n_lat, n_lon)
    z = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    land = jnp.ones((n_lat, n_lon))
    H = jnp.full((n_lat, n_lon), 4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, land_mask_override=land, H_bathy_override=H)
    cfg = LatLonCGridOceanConfig.from_flat(
        enable_runtime_checks=False)._replace(tidal_forcing=tide_cfg)
    return state, LatLonCGridOceanModel(grid, z, cfg)


def test_wire_disabled_is_byte_identical():
    """Tide OFF (default) is invariant to t0 and matches the default-instance
    config — the config-field addition + gate is a bit-exact no-op."""
    st_a, m_a = _tide_basin(TidalForcingConfig(enabled=False))
    fa, _ = m_a.integrate_scan(st_a, n_steps=_WIRE_NSTEPS, dt=_WIRE_DT)
    # the off path ignores t0_seconds entirely (xs=None branch)
    fa2, _ = m_a.integrate_scan(st_a, n_steps=_WIRE_NSTEPS, dt=_WIRE_DT, t0_seconds=9.9e5)
    np.testing.assert_array_equal(np.asarray(fa.u.data), np.asarray(fa2.u.data))
    np.testing.assert_array_equal(np.asarray(fa.eta.data), np.asarray(fa2.eta.data))
    # default TidalForcingConfig() (enabled=False) gives the identical trajectory
    st_b, m_b = _tide_basin(TidalForcingConfig())
    fb, _ = m_b.integrate_scan(st_b, n_steps=_WIRE_NSTEPS, dt=_WIRE_DT)
    np.testing.assert_array_equal(np.asarray(fa.u.data), np.asarray(fb.u.data))
    np.testing.assert_array_equal(np.asarray(fa.v.data), np.asarray(fb.v.data))
    np.testing.assert_array_equal(np.asarray(fa.eta.data), np.asarray(fb.eta.data))


def test_wire_enabled_drives_flow_and_is_time_dependent():
    """Tide ON drives a current from rest, and the forcing is NOT frozen in time
    (a t0 phase shift changes the trajectory) — the traced-t wire works."""
    st_off, m_off = _tide_basin(TidalForcingConfig(enabled=False))
    st_on, m_on = _tide_basin(TidalForcingConfig(
        enabled=True, include_diurnal=False, include_long_period=False))
    f_off, _ = m_off.integrate_scan(st_off, n_steps=_WIRE_NSTEPS, dt=_WIRE_DT)
    f_on, _ = m_on.integrate_scan(st_on, n_steps=_WIRE_NSTEPS, dt=_WIRE_DT)
    assert np.isfinite(np.asarray(f_on.u.data)).all()
    du = np.max(np.abs(np.asarray(f_on.u.data) - np.asarray(f_off.u.data)))
    dv = np.max(np.abs(np.asarray(f_on.v.data) - np.asarray(f_off.v.data)))
    assert du > 1e-6 or dv > 1e-6      # the tide drove a current from rest
    # time-dependence: shifting the phase (t0) changes the outcome
    f_shift, _ = m_on.integrate_scan(
        st_on, n_steps=_WIRE_NSTEPS, dt=_WIRE_DT, t0_seconds=6.0e4)
    assert np.max(np.abs(np.asarray(f_on.u.data) - np.asarray(f_shift.u.data))) > 1e-8


@pytest.mark.parametrize("solver", ["rigid_lid", "implicit_cn", "implicit_unsplit"])
def test_wire_rejects_unsupported_barotropic_solver(solver):
    """tidal_forcing + any non-split-explicit barotropic solver raises loudly at
    construction (dispatch hardening — no silent drop of an enabled forcing).
    Pins ALL three solvers that route around barotropic_substeps_latlon_cgrid."""
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.vertical import create_ocean_z_star
    grid = create_latlon_grid(8, 16)
    z = create_ocean_z_star(n_levels=4, H_max=4000.0)
    cfg = LatLonCGridOceanConfig.from_flat(
        barotropic_solver=solver)._replace(
        tidal_forcing=TidalForcingConfig(enabled=True))
    with pytest.raises(ValueError, match="tidal_forcing"):
        LatLonCGridOceanModel(grid, z, cfg)


def test_wire_differentiable_end_to_end():
    """grad of a scalar of the stepped state wrt the tidal reference time is
    finite — reverse-mode AD flows through the barotropic tidal wire."""
    st, m = _tide_basin(TidalForcingConfig(
        enabled=True, include_diurnal=False, include_long_period=False))

    def loss(t0):
        f, _ = m.integrate_scan(st, n_steps=2, dt=_WIRE_DT, t0_seconds=t0)
        # u (u-faces) and v (v-faces) are on DIFFERENT C-grid staggers -> sum
        # each separately (they do not broadcast together).
        return jnp.sum(f.u.data ** 2) + jnp.sum(f.v.data ** 2)

    g = jax.grad(loss)(0.0)
    assert np.isfinite(float(g))


def test_yaml_nested_config_builds_tidal_forcing_config():
    """`ocean.tidal_forcing: {...}` YAML builds a real TidalForcingConfig (not a
    raw dict) with defaults preserved, and rejects a member typo."""
    from legoesm.ocean.config import OceanExperimentConfig
    doc = {
        "grid": {"type": "latlon_cgrid", "n_lat": 16, "n_lon": 32},
        "ocean": {"tidal_forcing": {"enabled": True, "love_factor": 0.7,
                                    "include_diurnal": False}},
    }
    tf_cfg = OceanExperimentConfig(doc).to_ocean_config().tidal_forcing
    assert isinstance(tf_cfg, TidalForcingConfig)
    assert tf_cfg.enabled is True and tf_cfg.love_factor == 0.7
    assert tf_cfg.include_diurnal is False       # explicit override
    assert tf_cfg.beta_sal == 0.94               # untouched default preserved
    with pytest.raises(ValueError, match="TidalForcingConfig field"):
        OceanExperimentConfig({
            "grid": {"type": "latlon_cgrid", "n_lat": 8, "n_lon": 16},
            "ocean": {"tidal_forcing": {"enabld": True}},  # typo
        }).to_ocean_config()


def test_wire_enabled_step_without_time_raises():
    """Dispatch hardening: tidal_forcing.enabled + a bare step(state, dt)
    (no t_seconds — the production-driver calling pattern) must raise
    loudly instead of running a silently tide-free simulation."""
    state, model = _tide_basin(TidalForcingConfig(enabled=True))
    with pytest.raises(ValueError, match="t_seconds"):
        model.step(state, _WIRE_DT)
    # With the time supplied the same call steps fine.
    out = model.step(state, _WIRE_DT, t_seconds=jnp.asarray(0.0))
    assert jnp.all(jnp.isfinite(out.eta.data))
