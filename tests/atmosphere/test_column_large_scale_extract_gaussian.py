"""Gaussian/spectral column-forcing extractor (iter 83) — the 4th grid family.

A worst column flagged on the spectral (Gaussian) dycore grid can now have its
large-scale forcing extracted for the LES spin-off (lat-lon + cubed-sphere +
MPAS already worked).  The Gaussian path computes the divergence + advection
PSEUDOSPECTRALLY (reusing the dycore's ``vordiv_from_uv_3d`` SH operators), with
the advective tendency in flux form ``−V·∇φ = −∇·(φV) + φ·∇·V``; the grid-agnostic
``omega_from_divergence`` continuity is reused unchanged.  Geostrophic forcing is
supplied (iter 90) from the spectral Φ gradient (geographic east/north) except
within the equatorial cutoff (``u_geo/v_geo = None`` → f×V there).

Float64 required (spectral transforms).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from legoesm.atmosphere.forcing.idealized.column_forcing import ColumnLargeScaleState  # noqa: E402
from legoesm.atmosphere.forcing.column_large_scale_extract import (  # noqa: E402
    _gradient_gaussian_3d,
    extract_column_forcing,
    extract_column_forcing_gaussian,
)
from legoesm.atmosphere.physics._shared import exner_function  # noqa: E402
from legoesm.grids.gaussian import create_gaussian_grid  # noqa: E402
from legoesm.grids.vertical import create_sigma_coordinate  # noqa: E402

_NLEV = 6
_COL = (11, 20)         # an interior (i_lat, i_lon) of the n_max=21 grid (22×44)


def _grid_sigma():
    return create_gaussian_grid(n_max=21, dealiasing="linear"), create_sigma_coordinate(_NLEV)


def _state(grid, *, theta_const=True, wind="solid_body"):
    n_lat, n_lon = grid.n_lat, grid.n_lon
    p_s = jnp.full((n_lat, n_lon), 1.0e5)
    p_full = p_s[..., None] * jnp.asarray(create_sigma_coordinate(_NLEV).sigma_full)
    if theta_const:
        T = 290.0 * exner_function(p_full)               # θ uniform across all cells
    else:
        T = (290.0 + 5.0 * jnp.asarray(grid.cos_lat)[:, None, None] ** 2) \
            * exner_function(p_full)
    cos_lat = jnp.asarray(grid.cos_lat)[:, None, None]
    lon = jnp.asarray(grid.lon)[None, :, None]
    # q_v VARIES with LONGITUDE (so a zonal wind advects it ⇒ non-vacuous moisture
    # advection; a lat-only q with v=0 would give zero u·∇q).
    q_v = 0.01 + 4.0e-3 * jnp.cos(lon) * cos_lat * jnp.ones((n_lat, n_lon, _NLEV))
    if wind == "solid_body":          # u = U cos φ, v = 0 → NON-divergent (∇·v = 0)
        u = 20.0 * cos_lat * jnp.ones((n_lat, n_lon, _NLEV))
        v = jnp.zeros((n_lat, n_lon, _NLEV))
    else:                             # divergent: u varies with longitude
        u = 15.0 * jnp.cos(lon) * jnp.ones((n_lat, n_lon, _NLEV))
        v = jnp.zeros((n_lat, n_lon, _NLEV))
    # A real spectral state is float64 (the SH transforms require it); jnp defaults
    # to float32 even under x64, so cast explicitly.
    return tuple(jnp.asarray(x, dtype=jnp.float64) for x in (T, q_v, p_s, u, v))


def test_extract_gaussian_shapes_and_finite():
    grid, sigma = _grid_sigma()
    T, q_v, p_s, u, v = _state(grid, theta_const=False, wind="divergent")
    out = extract_column_forcing_gaussian(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=float(grid.lat[_COL[0]]), col_index=_COL)
    assert isinstance(out, ColumnLargeScaleState)
    for name in ("T", "p_full", "q_v", "omega", "theta_adv", "qv_adv"):
        arr = np.asarray(getattr(out, name))
        assert arr.shape == (_NLEV,) and np.all(np.isfinite(arr)), name
    # _COL is at lat[11] ≈ 4° — WITHIN the equatorial cutoff, so no geostrophic wind.
    assert out.u_geo is None and out.v_geo is None


def test_extract_gaussian_is_surface_flux_free():
    """The gaussian/spectral extractor produces NO surface BC (prescribe='none', no
    T_s/w_th_s/w_qv_s) — the cross-dispatch invariant matching iter-148's
    build_column_les_setup guard (a surface flux wired into ANY extractor without the
    LES surface-flux bottom BC would make that guard reject the column). This completes
    the surface-flux-free lock across ALL 4 grid families (cubed/latlon/voronoi/gaussian)."""
    grid, sigma = _grid_sigma()
    T, q_v, p_s, u, v = _state(grid, theta_const=False, wind="divergent")
    out = extract_column_forcing_gaussian(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=float(grid.lat[_COL[0]]), col_index=_COL)
    assert out.prescribe == "none"
    assert out.T_s is None and out.w_th_s is None and out.w_qv_s is None


def test_extract_gaussian_constant_theta_zero_advection_nonvacuous_moisture():
    """Uniform θ ⇒ the flux-form advection cancels (−∇·(θV)+θ·∇·V = 0) to spectral
    precision; a lon-VARYING q_v + zonal wind ⇒ NON-zero moisture advection."""
    grid, sigma = _grid_sigma()
    T, q_v, p_s, u, v = _state(grid, theta_const=True, wind="divergent")
    out = extract_column_forcing_gaussian(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=float(grid.lat[_COL[0]]), col_index=_COL)
    # Codex: tolerance (two independent transform chains leave a spectral-precision
    # residual ~1.6e-10), not bitwise exact — but ≪ a real advection tendency.
    np.testing.assert_allclose(np.asarray(out.theta_adv), 0.0, atol=1e-9)
    assert float(np.max(np.abs(np.asarray(out.qv_adv)))) > 1e-12


def test_extract_gaussian_solid_body_rotation_zero_omega():
    """ANALYTIC ANCHOR: a solid-body-rotation wind (u=U cos φ, v=0) is NON-divergent
    (∇·v = 0) ⇒ ω ≈ 0 to spectral precision."""
    grid, sigma = _grid_sigma()
    T, q_v, p_s, u, v = _state(grid, theta_const=True, wind="solid_body")
    out = extract_column_forcing_gaussian(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=float(grid.lat[_COL[0]]), col_index=_COL)
    np.testing.assert_allclose(np.asarray(out.omega), 0.0, atol=1e-6)


def test_extract_gaussian_divergent_flow_nonzero_omega():
    """A DIVERGENT wind (u varies with longitude) ⇒ non-zero ω (the continuity
    chain is genuinely exercised, not vacuously zero)."""
    grid, sigma = _grid_sigma()
    T, q_v, p_s, u, v = _state(grid, theta_const=True, wind="divergent")
    out = extract_column_forcing_gaussian(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=float(grid.lat[_COL[0]]), col_index=_COL)
    assert np.all(np.isfinite(np.asarray(out.omega)))
    assert float(np.max(np.abs(np.asarray(out.omega)))) > 1e-4


def test_dispatch_routes_gaussian_matches_direct():
    grid, sigma = _grid_sigma()
    T, q_v, p_s, u, v = _state(grid, theta_const=False, wind="divergent")
    kw = dict(T=T, q_v=q_v, p_s=p_s, sigma_coord=sigma,
              lat_rad=float(grid.lat[_COL[0]]), col_index=_COL)
    via = extract_column_forcing(u=u, v=v, grid=grid, **kw)
    direct = extract_column_forcing_gaussian(u=u, v=v, grid=grid, **kw)
    np.testing.assert_array_equal(np.asarray(via.omega), np.asarray(direct.omega))
    np.testing.assert_array_equal(np.asarray(via.theta_adv), np.asarray(direct.theta_adv))


def test_dispatch_gaussian_wrong_arity_raises():
    grid, sigma = _grid_sigma()
    T, q_v, p_s, u, v = _state(grid)
    with pytest.raises(ValueError, match="Gaussian col_index"):
        extract_column_forcing(
            T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
            lat_rad=0.1, col_index=(11, 20, 0))       # arity 3, not 2


def test_extract_gaussian_float32_raises():
    """The spectral path requires float64 — a float32 input fails LOUDLY (Codex)."""
    grid, sigma = _grid_sigma()
    T, q_v, p_s, u, v = _state(grid)
    with pytest.raises(TypeError, match="requires float64"):
        extract_column_forcing_gaussian(
            T=T.astype(jnp.float32), q_v=q_v, u=u, v=v, p_s=p_s, grid=grid,
            sigma_coord=sigma, lat_rad=float(grid.lat[_COL[0]]), col_index=_COL)


def test_dispatch_gaussian_mismatched_field_shape_raises():
    """A field whose shape disagrees with T fails with a targeted message (Codex)."""
    grid, sigma = _grid_sigma()
    T, q_v, p_s, u, v = _state(grid)
    with pytest.raises(ValueError, match="Gaussian `u` must match"):
        extract_column_forcing(
            T=T, q_v=q_v, u=u[:, :, :-1], v=v, p_s=p_s, grid=grid,
            sigma_coord=sigma, lat_rad=0.1, col_index=_COL)


# --- Geostrophic forcing (iter 90) ------------------------------------------
_MIDLAT_COL = (16, 20)   # lat[16] ≈ 44°, well outside the 5° equatorial cutoff


def _uniform_geo_state(grid):
    """A HORIZONTALLY-uniform state (uniform θ + constant q_v + constant p_s) ⇒ a
    horizontally-uniform Φ ⇒ ZERO geostrophic wind (the analytic anchor)."""
    n_lat, n_lon = grid.n_lat, grid.n_lon
    p_s = jnp.full((n_lat, n_lon), 1.0e5, dtype=jnp.float64)
    p_full = p_s[..., None] * jnp.asarray(create_sigma_coordinate(_NLEV).sigma_full)
    T = (290.0 * exner_function(p_full)).astype(jnp.float64)        # uniform θ
    q_v = jnp.full((n_lat, n_lon, _NLEV), 5e-3, dtype=jnp.float64)  # constant
    return T, q_v, p_s


def test_spectral_gradient_3d_direct_analytic():
    """The PROMOTED shared operator ``legoesm.grids.gaussian.spectral_gradient_3d``
    (iter 90) on the SH spectrum of Φ = C·sin(lat): ∂Φ/∂x = 0, ∂Φ/∂y = (C/a)cos(lat)."""
    from legoesm.grids.gaussian import sh_analysis_3d, spectral_gradient_3d
    grid, _ = _grid_sigma()
    lat = np.asarray(grid.grid_lat)
    a = float(grid.radius)
    C = 500.0
    phi = jnp.asarray((C * np.sin(lat))[:, :, None] * np.ones((1, 1, 2)), dtype=jnp.float64)
    dfdx, dfdy = spectral_gradient_3d(grid, sh_analysis_3d(grid, phi))
    interior = np.abs(lat) < np.deg2rad(80.0)
    assert float(np.max(np.abs(np.asarray(dfdx)[:, :, 0][interior]))) < 1e-12
    np.testing.assert_allclose(
        np.asarray(dfdy)[:, :, 0][interior], ((C / a) * np.cos(lat))[interior], rtol=1e-8)


def test_gradient_gaussian_analytic():
    """``_gradient_gaussian_3d`` matches the analytic gradient of Φ = C·sin(lat):
    ∂Φ/∂x = 0 (no longitude dependence), ∂Φ/∂y = (C/a)·cos(lat)."""
    grid, _ = _grid_sigma()
    lat = np.asarray(grid.grid_lat)
    a = float(grid.radius)
    C = 1000.0
    phi = (C * np.sin(lat))[:, :, None] * np.ones((1, 1, 2))
    dfdx, dfdy = _gradient_gaussian_3d(jnp.asarray(phi, dtype=jnp.float64), grid)
    dfdx, dfdy = np.asarray(dfdx), np.asarray(dfdy)
    interior = np.abs(lat) < np.deg2rad(80.0)            # away from the cos-φ floor
    assert float(np.max(np.abs(dfdx[:, :, 0][interior]))) < 1e-12   # no zonal grad
    dfdy_analytic = (C / a) * np.cos(lat)
    np.testing.assert_allclose(
        dfdy[:, :, 0][interior], dfdy_analytic[interior], rtol=1e-8)


def test_extract_gaussian_geostrophic_uniform_state_zero_wind():
    """ANALYTIC ANCHOR: a horizontally-uniform state ⇒ ∇Φ = 0 ⇒ the geostrophic
    wind is EXACTLY zero (to spectral precision) at a mid-latitude column."""
    grid, sigma = _grid_sigma()
    T, q_v, p_s = _uniform_geo_state(grid)
    out = extract_column_forcing_gaussian(
        T=T, q_v=q_v, u=jnp.zeros_like(T), v=jnp.zeros_like(T), p_s=p_s, grid=grid,
        sigma_coord=sigma, lat_rad=float(grid.lat[_MIDLAT_COL[0]]),
        col_index=_MIDLAT_COL)
    assert out.u_geo is not None and out.v_geo is not None
    np.testing.assert_allclose(np.asarray(out.u_geo), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.asarray(out.v_geo), 0.0, atol=1e-9)


def test_extract_gaussian_geostrophic_meridional_gradient_zonal_jet():
    """A meridional temperature (⇒ geopotential) gradient at a mid-latitude column
    gives a FINITE, predominantly-ZONAL geostrophic wind (|u_geo| > |v_geo|) — the
    thermal-wind signature, same anchor as the Voronoi geostrophic (iter 82)."""
    grid, sigma = _grid_sigma()
    n_lat, n_lon = grid.n_lat, grid.n_lon
    p_s = jnp.full((n_lat, n_lon), 1.0e5, dtype=jnp.float64)
    p_full = p_s[..., None] * jnp.asarray(sigma.sigma_full)
    cos2 = jnp.asarray(grid.cos_lat)[:, None, None] ** 2     # latitude-only structure
    T = ((290.0 + 30.0 * cos2) * exner_function(p_full)).astype(jnp.float64)
    q_v = jnp.full((n_lat, n_lon, _NLEV), 5e-3, dtype=jnp.float64)
    out = extract_column_forcing_gaussian(
        T=T, q_v=q_v, u=jnp.zeros_like(T), v=jnp.zeros_like(T), p_s=p_s, grid=grid,
        sigma_coord=sigma, lat_rad=float(grid.lat[_MIDLAT_COL[0]]),
        col_index=_MIDLAT_COL)
    ug, vg = np.asarray(out.u_geo), np.asarray(out.v_geo)
    assert np.all(np.isfinite(ug)) and np.all(np.isfinite(vg))
    assert float(np.max(np.abs(ug))) > 1.0                  # a real jet (m/s)
    assert float(np.max(np.abs(ug))) > float(np.max(np.abs(vg)))   # predominantly zonal


def test_extract_gaussian_geostrophic_orographic_phis_runs():
    """The optional orographic ``phis`` threads through the Gaussian spectral
    gradient (a single-LEVEL surface field through SH analysis): terrain gives a
    finite, NON-zero, σ-independent geostrophic wind where the flat state gives 0."""
    from legoesm import constants
    grid, sigma = _grid_sigma()
    T, q_v, p_s = _uniform_geo_state(grid)
    kw = dict(T=T, q_v=q_v, u=jnp.zeros_like(T), v=jnp.zeros_like(T), p_s=p_s,
              grid=grid, sigma_coord=sigma,
              lat_rad=float(grid.lat[_MIDLAT_COL[0]]), col_index=_MIDLAT_COL)
    flat = extract_column_forcing_gaussian(**kw)
    phis = constants.g * (800.0 * jnp.sin(2.0 * jnp.asarray(grid.grid_lat)))
    oro = extract_column_forcing_gaussian(**kw, phis=phis)
    assert bool(jnp.all(jnp.isfinite(oro.u_geo))) and bool(jnp.all(jnp.isfinite(oro.v_geo)))
    du = np.asarray(oro.u_geo) - np.asarray(flat.u_geo)
    dv = np.asarray(oro.v_geo) - np.asarray(flat.v_geo)
    assert float(np.max(np.abs(du)) + np.max(np.abs(dv))) > 1e-3
    np.testing.assert_allclose(du, du[0], atol=1e-9)               # σ-independent
    np.testing.assert_allclose(dv, dv[0], atol=1e-9)


def test_extract_gaussian_geostrophic_equatorial_is_none():
    """Within the equatorial cutoff (lat[11] ≈ 4°) no geostrophic reference wind is
    supplied (f → 0 is ill-posed) — the LES plane Coriolis falls back to f×V."""
    grid, sigma = _grid_sigma()
    T, q_v, p_s = _uniform_geo_state(grid)
    out = extract_column_forcing_gaussian(
        T=T, q_v=q_v, u=jnp.zeros_like(T), v=jnp.zeros_like(T), p_s=p_s, grid=grid,
        sigma_coord=sigma, lat_rad=float(grid.lat[11]), col_index=(11, 20))
    assert out.u_geo is None and out.v_geo is None
