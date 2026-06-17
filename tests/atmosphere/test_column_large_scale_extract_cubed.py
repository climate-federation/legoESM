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

from legoesm.atmosphere.column_forcing import ColumnLargeScaleState
from legoesm.atmosphere.dynamics.column_large_scale_extract import (
    advective_tendency,
    extract_column_forcing,
    extract_column_forcing_cubed_sphere,
    omega_from_divergence,
)
from legoesm.atmosphere.physics._shared import exner_function
from legoesm.core.operators_3d import divergence_3d, gradient_x_3d, gradient_y_3d
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
    q_v = 0.012 + 0.004 * jnp.cos(lon) * jnp.cos(lat)
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
