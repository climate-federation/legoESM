"""T+S pair advection fast path == per-tracer calls, bitwise.

The pair fast path stacks both tracers along the LEVEL axis for the
level-separable horizontal reconstructions
(``_LEVEL_SEPARABLE_H_SCHEMES``) and computes the vertical flux
divergence per tracer — outputs must be BIT-IDENTICAL to two
single-tracer ``_compute_advection_flux_div`` calls (and the RK3 pair
step to two ``_ssp_rk3_tracer_step`` calls).  Non-separable schemes
must fall back, also bit-identical.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    _LEVEL_SEPARABLE_H_SCHEMES,
    _compute_advection_flux_div,
    compute_advection_flux_div_pair,
    _ssp_rk3_tracer_pair_step,
    _ssp_rk3_tracer_step,
)

N_LAT, N_LON, NLEV = 8, 16, 5
DT = 600.0


def _problem(seed=0):
    rng = np.random.default_rng(seed)
    grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
    T = jnp.asarray(280.0 + rng.standard_normal((N_LAT, N_LON, NLEV)))
    S = jnp.asarray(35.0 + 0.1 * rng.standard_normal((N_LAT, N_LON, NLEV)))
    mfu = jnp.asarray(
        0.1 * rng.standard_normal((N_LAT, N_LON + 1, NLEV)))
    mfv = jnp.asarray(
        0.1 * rng.standard_normal((N_LAT + 1, N_LON, NLEV)))
    w = jnp.asarray(
        1e-4 * rng.standard_normal((N_LAT, N_LON, NLEV + 1)))
    h = jnp.asarray(
        10.0 + np.abs(rng.standard_normal((N_LAT, N_LON, NLEV))))
    h_u = jnp.asarray(
        10.0 + np.abs(rng.standard_normal((N_LAT, N_LON + 1, NLEV))))
    h_v = jnp.asarray(
        10.0 + np.abs(rng.standard_normal((N_LAT + 1, N_LON, NLEV))))
    return grid, T, S, mfu, mfv, w, h, h_u, h_v


@pytest.mark.parametrize(
    "scheme",
    sorted(_LEVEL_SEPARABLE_H_SCHEMES) + ["weno5"],  # weno5 = fallback lane
)
def test_pair_flux_div_matches_singles_bitwise(scheme):
    grid, T, S, mfu, mfv, w, h, h_u, h_v = _problem(1)
    (dh_a, dv_a), (dh_b, dv_b) = compute_advection_flux_div_pair(
        T, S, scheme, mfu, mfv, w, h, h_u, h_v, grid, DT,
    )
    dh_T, dv_T = _compute_advection_flux_div(
        T, scheme, mfu, mfv, w, h, h_u, h_v, grid, DT,
    )
    dh_S, dv_S = _compute_advection_flux_div(
        S, scheme, mfu, mfv, w, h, h_u, h_v, grid, DT,
    )
    np.testing.assert_array_equal(np.asarray(dh_a), np.asarray(dh_T))
    np.testing.assert_array_equal(np.asarray(dv_a), np.asarray(dv_T))
    np.testing.assert_array_equal(np.asarray(dh_b), np.asarray(dh_S))
    np.testing.assert_array_equal(np.asarray(dv_b), np.asarray(dv_S))


@pytest.mark.parametrize("scheme", ["tvd", "upwind", "centered"])
def test_pair_rk3_matches_singles_bitwise(scheme):
    grid, T, S, mfu, mfv, w, h, h_u, h_v = _problem(2)
    h_new = h * 1.001
    active = jnp.ones_like(h)
    a_new, b_new = _ssp_rk3_tracer_pair_step(
        T, S, scheme, mfu, mfv, w, h, h_new, h_u, h_v, grid, DT, active,
    )
    T_ref = _ssp_rk3_tracer_step(
        T, scheme, mfu, mfv, w, h, h_new, h_u, h_v, grid, DT, active,
    )
    S_ref = _ssp_rk3_tracer_step(
        S, scheme, mfu, mfv, w, h, h_new, h_u, h_v, grid, DT, active,
    )
    np.testing.assert_array_equal(np.asarray(a_new), np.asarray(T_ref))
    np.testing.assert_array_equal(np.asarray(b_new), np.asarray(S_ref))


def test_pair_respects_land_mask():
    grid, T, S, mfu, mfv, w, h, h_u, h_v = _problem(3)
    active = jnp.asarray(
        (np.random.default_rng(4).random((N_LAT, N_LON, NLEV)) > 0.3)
        .astype(np.float64))
    a_new, b_new = _ssp_rk3_tracer_pair_step(
        T, S, "tvd", mfu, mfv, w, h, h * 1.001, h_u, h_v, grid, DT, active,
    )
    np.testing.assert_array_equal(
        np.asarray(jnp.where(active > 0.5, 0.0, a_new)),
        np.asarray(jnp.where(active > 0.5, 0.0, T)),
    )
    np.testing.assert_array_equal(
        np.asarray(jnp.where(active > 0.5, 0.0, b_new)),
        np.asarray(jnp.where(active > 0.5, 0.0, S)),
    )
