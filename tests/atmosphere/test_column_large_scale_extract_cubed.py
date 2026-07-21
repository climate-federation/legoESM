"""Cubed-sphere column large-scale forcing extractor + grid dispatcher.

Mirror of ``test_column_large_scale_extract.py`` for the model's native
cubed-sphere grid (stage-4 grid-side extractor, iter 27): a quiescent uniform
state gives zero forcing (analytic anchor), a non-uniform state gives finite
forcing whose column value matches an independent recomputation (gather/wiring
cross-check), and the dispatcher routes by grid type + rejects bad arity /
unsupported grids.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.forcing.idealized.column_forcing import ColumnLargeScaleState
from legoesm.atmosphere.forcing.column_large_scale_extract import (
    _gradient_cubed_geographic_3d,
    advective_tendency,
    extract_column_forcing,
    extract_column_forcing_cubed_sphere,
    omega_from_divergence,
)
from legoesm.atmosphere.physics._shared import exner_function
from legoesm.core.operators_3d import divergence_3d, gradient_x_3d, gradient_y_3d
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.factory import create_grid
from legoesm.grids.vertical import create_sigma_coordinate

jax.config.update("jax_enable_x64", True)

_RES, _NLEV = 8, 4
_COL = (2, 3, 5)  # (face, i, j)


def _grid_and_sigma():
    return create_grid("cubed_sphere", resolution=_RES), create_sigma_coordinate(_NLEV)


def _uniform_state(grid):
    shp = (6, _RES, _RES, _NLEV)
    T = jnp.full(shp, 280.0)
    q_v = jnp.full(shp, 0.01)
    u = jnp.zeros(shp)
    v = jnp.zeros(shp)
    p_s = jnp.full((6, _RES, _RES), 1.0e5)
    return T, q_v, u, v, p_s


def _nonuniform_state(grid):
    lat = jnp.asarray(grid.grid_lat)[..., None]   # (6,n,n,1)
    lon = jnp.asarray(grid.grid_lon)[..., None]
    lev = jnp.arange(_NLEV, dtype=jnp.float64)
    # Smooth global fields with non-zero horizontal gradient.
    T = 280.0 + 12.0 * jnp.sin(lon) * jnp.cos(lat) + 0.5 * lev
    q_v = 0.012 + 0.004 * jnp.cos(lon) * jnp.cos(lat) + 1e-4 * lev  # nlev levels
    u = jnp.full((6, _RES, _RES, _NLEV), 6.0)
    v = jnp.full((6, _RES, _RES, _NLEV), -3.0)
    p_s = jnp.full((6, _RES, _RES), 1.0e5) + 50.0 * jnp.sin(jnp.asarray(grid.grid_lon))
    return T, q_v, u, v, p_s


def test_extract_cubed_uniform_state_zero_forcing():
    """Quiescent horizontally-uniform state ⇒ zero advection AND zero subsidence."""
    grid, sigma = _grid_and_sigma()
    T, q_v, u, v, p_s = _uniform_state(grid)
    ls = extract_column_forcing_cubed_sphere(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=0.3, col_index=_COL,
    )
    assert isinstance(ls, ColumnLargeScaleState)
    assert ls.T.shape == (_NLEV,)
    np.testing.assert_allclose(np.asarray(ls.theta_adv), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(ls.qv_adv), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(ls.omega), 0.0, atol=1e-8)


def _recompute_column(T, q_v, u, v, p_s, grid, sigma, col):
    """Independent full-grid recompute of (theta_adv, qv_adv, omega) at one column
    via the same 4D-native operators, gathered at ``col`` — pins the gather."""
    p_full = p_s[..., None] * jnp.asarray(sigma.sigma_full, dtype=T.dtype)
    theta = T / exner_function(p_full)
    th_adv = advective_tendency(
        u, v, gradient_x_3d(theta, grid), gradient_y_3d(theta, grid))
    q_adv = advective_tendency(
        u, v, gradient_x_3d(q_v, grid), gradient_y_3d(q_v, grid))
    omega = omega_from_divergence(divergence_3d(u, v, grid), p_s, sigma)
    f, i, j = col
    return th_adv[f, i, j, :], q_adv[f, i, j, :], omega[f, i, j, :]


@pytest.mark.parametrize("col", [_COL, (0, 0, 0), (4, 0, 7)])
def test_extract_cubed_nonuniform_finite_and_wiring(col):
    """Non-uniform state ⇒ finite, non-zero forcing; theta_adv / qv_adv / omega
    at the gathered column each match an independent recompute (validates the
    (face,i,j) gather + assembly), including face-corner / edge columns."""
    grid, sigma = _grid_and_sigma()
    T, q_v, u, v, p_s = _nonuniform_state(grid)
    ls = extract_column_forcing_cubed_sphere(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=0.3, col_index=col,
    )
    assert bool(jnp.all(jnp.isfinite(ls.theta_adv)))
    assert bool(jnp.all(jnp.isfinite(ls.qv_adv)))
    assert bool(jnp.all(jnp.isfinite(ls.omega)))
    assert float(jnp.max(jnp.abs(ls.theta_adv))) > 0.0
    assert float(jnp.max(jnp.abs(ls.omega))) > 0.0

    th_exp, q_exp, om_exp = _recompute_column(
        T, q_v, u, v, p_s, grid, sigma, col)
    # rtol is float32-level: the cubed-sphere grid metric (dx/dy/h*_ext) is
    # float32 and XLA may reassociate the halo adds between the fused extractor
    # and this recompute.  A wrong-column gather would differ by O(1), so each
    # field's match still pins the (face,i,j) wiring separately.
    np.testing.assert_allclose(np.asarray(ls.theta_adv), np.asarray(th_exp), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(ls.qv_adv), np.asarray(q_exp), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(ls.omega), np.asarray(om_exp), rtol=1e-5)


def test_extract_cubed_constant_scalar_nonzero_wind_zero_advection():
    """Constant θ/q_v with NON-zero advecting wind ⇒ advective tendency still ≈0
    (gradient of a constant is exactly 0 on the cube) — non-vacuous gradient-path
    check that the trivial u=v=0 anchor cannot give."""
    grid, sigma = _grid_and_sigma()
    shp = (6, _RES, _RES, _NLEV)
    ls = extract_column_forcing_cubed_sphere(
        T=jnp.full(shp, 290.0), q_v=jnp.full(shp, 0.008),
        u=jnp.full(shp, 7.0), v=jnp.full(shp, -4.0),
        p_s=jnp.full((6, _RES, _RES), 1.0e5),
        grid=grid, sigma_coord=sigma, lat_rad=0.2, col_index=_COL,
    )
    np.testing.assert_allclose(np.asarray(ls.theta_adv), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.asarray(ls.qv_adv), 0.0, atol=1e-12)


def test_extract_cubed_is_surface_flux_free():
    """The extractor produces NO surface boundary condition (prescribe='none', no
    T_s / w_th_s / w_qv_s) — the surface-flux-free design the forced LES requires.

    This is the EXTRACTOR-side guard rail matching iter-148's build_column_les_setup
    check: that guard FAILS LOUD if a forcing requests a surface BC the LES can't yet
    apply. So if a future developer wires the GCM column's SST into the extractor as a
    surface flux (prescribe='fluxes') to improve convective columns, build_column_les_
    setup would suddenly reject EVERY such column — this test catches that here, at the
    extractor, pointing the implementer to also add the LES surface-flux bottom BC and
    relax the guard before shipping it."""
    grid, sigma = _grid_and_sigma()
    T, q_v, u, v, p_s = _nonuniform_state(grid)
    ls = extract_column_forcing_cubed_sphere(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=0.3, col_index=_COL,
    )
    assert ls.prescribe == "none"
    assert ls.T_s is None and ls.w_th_s is None and ls.w_qv_s is None


def test_extract_cubed_jit():
    grid, sigma = _grid_and_sigma()
    T, q_v, u, v, p_s = _nonuniform_state(grid)
    fn = jax.jit(
        lambda T_, q_, u_, v_, ps_: extract_column_forcing_cubed_sphere(
            T=T_, q_v=q_, u=u_, v=v_, p_s=ps_, grid=grid, sigma_coord=sigma,
            lat_rad=0.3, col_index=_COL,
        ).omega
    )
    assert bool(jnp.all(jnp.isfinite(fn(T, q_v, u, v, p_s))))


def test_dispatch_routes_cubed_sphere():
    grid, sigma = _grid_and_sigma()
    T, q_v, u, v, p_s = _nonuniform_state(grid)
    via_dispatch = extract_column_forcing(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=0.3, col_index=_COL,
    )
    direct = extract_column_forcing_cubed_sphere(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=0.3, col_index=_COL,
    )
    np.testing.assert_allclose(
        np.asarray(via_dispatch.omega), np.asarray(direct.omega), rtol=1e-12)


def test_dispatch_routes_latlon():
    grid = create_grid("latlon", resolution=8)
    sigma = create_sigma_coordinate(_NLEV)
    shp = (8, 16, _NLEV)
    T = jnp.full(shp, 285.0)
    ls = extract_column_forcing(
        T=T, q_v=jnp.full(shp, 0.01), u=jnp.zeros(shp), v=jnp.zeros(shp),
        p_s=jnp.full((8, 16), 1.0e5), grid=grid, sigma_coord=sigma,
        lat_rad=0.1, col_index=(3, 4),
    )
    assert ls.T.shape == (_NLEV,)


def test_dispatch_wrong_arity_raises():
    grid, sigma = _grid_and_sigma()
    T, q_v, u, v, p_s = _uniform_state(grid)
    with pytest.raises(ValueError, match="cubed-sphere col_index must be"):
        extract_column_forcing(
            T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
            lat_rad=0.3, col_index=(3, 5),  # missing face → wrong arity
        )


def test_dispatch_unknown_grid_raises():
    sigma = create_sigma_coordinate(_NLEV)

    class _BogusGrid:
        pass

    with pytest.raises(ValueError, match="unsupported grid type"):
        extract_column_forcing(
            T=jnp.zeros((1, 1, _NLEV)), q_v=jnp.zeros((1, 1, _NLEV)),
            u=jnp.zeros((1, 1, _NLEV)), v=jnp.zeros((1, 1, _NLEV)),
            p_s=jnp.zeros((1, 1)), grid=_BogusGrid(), sigma_coord=sigma,
            lat_rad=0.0, col_index=(0, 0),
        )


def _cubed_geo_north_relerr(n):
    """Relative error of the metric-correct geographic north-gradient of Φ = C·sin(lat)
    (analytic ∂Φ/∂north = (C/a)cos(lat)) on an n×n×6 cube: returns (full-grid max,
    panel-boundary-band max, east-leakage) over all cells with |lat| < 88° (only the
    polar singular rows excluded — INCLUDING the panel edges/corners where the iter-
    91/92 naive rotation had its non-convergent ~10–17% error)."""
    g = create_cubed_sphere(n)
    lat = np.asarray(g.lat)
    a, C = float(g.radius), 1000.0
    phi = jnp.asarray((C * np.sin(lat))[..., None], dtype=jnp.float64)
    ge, gn = _gradient_cubed_geographic_3d(phi, g)
    ge, gn = np.asarray(ge)[..., 0], np.asarray(gn)[..., 0]
    gn_an = (C / a) * np.cos(lat)
    fin = np.abs(lat) < np.deg2rad(88.0)
    err = np.abs(gn - gn_an) / np.max(np.abs(gn_an[fin]))
    band = np.zeros_like(lat, bool)             # the panel boundary ring (edges+corners)
    band[:, [0, -1], :] = True
    band[:, :, [0, -1]] = True
    full = float(np.max(err[fin]))
    edge = float(np.max(err[fin & band]))
    east_leak = float(np.max(np.abs(ge[fin])) / np.max(np.abs(gn[fin])))
    return full, edge, east_leak


def test_gradient_cubed_geographic_converges_including_edges():
    """The METRIC-CORRECT geographic gradient (iter 93) CONVERGES EVERYWHERE — the
    analytic Φ=C·sin(lat) north-gradient relative error is small AND shrinks with
    resolution on the FULL grid INCLUDING the panel boundary band (the exact region
    where the iter-91 naive rotation / iter-92 scalar-c inverse had a NON-convergent
    ~10–17% error — Codex iter-93: the test must not mask the edges)."""
    f24, e24, leak24 = _cubed_geo_north_relerr(24)
    f48, e48, _ = _cubed_geo_north_relerr(48)
    assert f24 < 0.01 and e24 < 0.01        # full + the EDGE BAND both < 1% at n=24
    assert f48 < f24 and e48 < e24          # both CONVERGE with resolution
    assert leak24 < 0.01                    # east leakage tiny (rotation not mixed)


def _interior_midlat_column(grid, lo=15.0, hi=55.0):
    lat = np.rad2deg(np.asarray(grid.grid_lat))
    for f in range(6):
        for i in range(2, _RES - 2):
            for j in range(2, _RES - 2):
                if lo <= abs(lat[f, i, j]) <= hi:
                    return (f, i, j), float(np.deg2rad(lat[f, i, j]))
    raise AssertionError("no interior mid-lat cubed column found")


def test_extract_cubed_geostrophic_uniform_state_zero_wind():
    """ANALYTIC ANCHOR: a horizontally-uniform state ⇒ ∇Φ = 0 EXACTLY ⇒ zero
    geostrophic wind (the gradient of a constant is 0 regardless of the metric)."""
    grid, sigma = _grid_and_sigma()
    T, q_v, u, v, p_s = _uniform_state(grid)
    col, lat_rad = _interior_midlat_column(grid)
    ls = extract_column_forcing_cubed_sphere(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=lat_rad, col_index=col)
    assert ls.u_geo is not None and ls.v_geo is not None
    np.testing.assert_allclose(np.asarray(ls.u_geo), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.asarray(ls.v_geo), 0.0, atol=1e-9)


def test_extract_cubed_geostrophic_meridional_gradient_zonal_jet():
    """A meridional temperature (⇒ geopotential) gradient at an INTERIOR mid-latitude
    column gives a finite, predominantly-ZONAL geostrophic wind (|u_geo| > |v_geo|) —
    the thermal-wind signature (same anchor as Voronoi iter 82 / Gaussian iter 90)."""
    grid, sigma = _grid_and_sigma()
    p_s = jnp.full((6, _RES, _RES), 1.0e5)
    p_full = p_s[..., None] * jnp.asarray(sigma.sigma_full)
    cos2 = jnp.cos(jnp.asarray(grid.grid_lat))[..., None] ** 2
    T = (290.0 + 30.0 * cos2) * exner_function(p_full)
    q_v = jnp.full((6, _RES, _RES, _NLEV), 5e-3)
    z = jnp.zeros((6, _RES, _RES, _NLEV))
    col, lat_rad = _interior_midlat_column(grid)
    ls = extract_column_forcing_cubed_sphere(
        T=T, q_v=q_v, u=z, v=z, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=lat_rad, col_index=col)
    ug, vg = np.asarray(ls.u_geo), np.asarray(ls.v_geo)
    assert np.all(np.isfinite(ug)) and np.all(np.isfinite(vg))
    assert float(np.max(np.abs(ug))) > 1.0                        # a real jet (m/s)
    assert float(np.max(np.abs(ug))) > float(np.max(np.abs(vg)))  # predominantly zonal


def test_extract_cubed_geostrophic_orographic_phis_runs():
    """The optional orographic ``phis`` threads through the cubed-sphere 4D-native
    geographic gradient (a single-LEVEL surface field): terrain gives a finite,
    NON-zero, σ-independent geostrophic wind where the flat uniform state gives 0."""
    from legoesm import constants
    grid, sigma = _grid_and_sigma()
    T, q_v, u, v, p_s = _uniform_state(grid)
    col, lat_rad = _interior_midlat_column(grid)
    kw = dict(T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
              lat_rad=lat_rad, col_index=col)
    flat = extract_column_forcing_cubed_sphere(**kw)
    phis = constants.g * (800.0 * jnp.sin(2.0 * jnp.asarray(grid.grid_lat)))
    oro = extract_column_forcing_cubed_sphere(**kw, phis=phis)
    assert bool(jnp.all(jnp.isfinite(oro.u_geo))) and bool(jnp.all(jnp.isfinite(oro.v_geo)))
    du = np.asarray(oro.u_geo) - np.asarray(flat.u_geo)
    dv = np.asarray(oro.v_geo) - np.asarray(flat.v_geo)
    assert float(np.max(np.abs(du)) + np.max(np.abs(dv))) > 1e-3   # terrain moved the wind
    np.testing.assert_allclose(du, du[0], atol=1e-10)              # σ-independent
    np.testing.assert_allclose(dv, dv[0], atol=1e-10)


def test_extract_cubed_geostrophic_equatorial_is_none():
    """Within the equatorial cutoff no geostrophic reference wind is supplied (f→0
    ill-posed) — the LES plane Coriolis falls back to f×V; ω/advection still set."""
    grid, sigma = _grid_and_sigma()
    T, q_v, u, v, p_s = _nonuniform_state(grid)
    ls = extract_column_forcing_cubed_sphere(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=float(jnp.deg2rad(2.0)), col_index=_COL)   # within the 5° cutoff
    assert ls.u_geo is None and ls.v_geo is None
    assert ls.omega.shape == (_NLEV,) and bool(jnp.all(jnp.isfinite(ls.omega)))
