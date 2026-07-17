"""Spectral/Gaussian compare-side adapter (iter 84).

A spectral run's prognostic state is a SpectralHydrostaticState (vor_hat/div_hat
— SH coefficients), with no grid u/v for the worst-column ranking / wind RMSE or
the iter-83 Gaussian LES extractor.  ``grid_winds_from_spectral`` synthesizes it
to a grid HydrostaticState via the dycore's own ``spectral_pe_to_grid``; ANY
non-spectral state passes through unchanged.  This is the spectral analog of the
MPAS edge→cell reconstruction and the UPSTREAM of the Gaussian extractor.

Float64 required (spectral transforms).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from types import SimpleNamespace  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (  # noqa: E402
    SpectralHydrostaticState,
    spectral_pe_to_grid,
)
from legoesm.atmosphere.physics._shared import exner_function  # noqa: E402
from legoesm.core.field import Field  # noqa: E402
from legoesm.core.state import HydrostaticState  # noqa: E402
from legoesm.grids.gaussian import (  # noqa: E402
    create_gaussian_grid,
    sh_analysis,
    sh_analysis_3d,
    vordiv_from_uv_3d,
)
from legoesm.grids.vertical import create_sigma_coordinate  # noqa: E402
from legoesm.training.compare_reanalysis import grid_winds_from_spectral  # noqa: E402
from legoesm.training.run_to_column_mean import amip_column_state  # noqa: E402

_NLEV = 5


def _spectral_state(grid, sigma):
    """A SpectralHydrostaticState built from a KNOWN solid-body-rotation wind
    (u=U cos φ, v=0) + uniform T/p_s, so the synthesized grid wind is recoverable."""
    n_lat, n_lon = grid.n_lat, grid.n_lon
    p_s = jnp.full((n_lat, n_lon), 1.0e5, dtype=jnp.float64)
    p_full = p_s[..., None] * jnp.asarray(sigma.sigma_full)
    t_grid = (290.0 * exner_function(p_full)).astype(jnp.float64)
    u0 = (20.0 * jnp.asarray(grid.cos_lat)[:, None, None]
          * jnp.ones((n_lat, n_lon, _NLEV))).astype(jnp.float64)
    v0 = jnp.zeros((n_lat, n_lon, _NLEV), dtype=jnp.float64)
    vor_hat, div_hat = vordiv_from_uv_3d(grid, u0, v0)
    state = SpectralHydrostaticState(
        vor_hat=Field(vor_hat), div_hat=Field(div_hat),
        T_hat=Field(sh_analysis_3d(grid, t_grid)),
        lnps_hat=Field(sh_analysis(grid, jnp.log(p_s))),
        phis_hat=Field(sh_analysis(grid, jnp.zeros((n_lat, n_lon)))),
        tracers={"q_v": Field(jnp.full((n_lat, n_lon, _NLEV), 5e-3))})
    return state, u0, v0


def test_grid_winds_from_spectral_synthesizes_grid_state():
    grid = create_gaussian_grid(n_max=21, dealiasing="linear")
    sigma = create_sigma_coordinate(_NLEV)
    state, u0, v0 = _spectral_state(grid, sigma)

    out = grid_winds_from_spectral(state, grid, sigma)
    # A grid HydrostaticState (has u/v Fields), NOT the spectral state.
    assert isinstance(out, HydrostaticState)
    assert not isinstance(out, SpectralHydrostaticState)
    u = np.asarray(out.u.data)
    v = np.asarray(out.v.data)
    assert u.shape == (grid.n_lat, grid.n_lon, _NLEV)
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(v))
    # Round-trip: the synthesized grid wind recovers the known solid-body wind
    # (uv -> vor/div -> uv is the inverse, per vordiv_from_uv_3d).
    np.testing.assert_allclose(u, np.asarray(u0), atol=1e-6)
    np.testing.assert_allclose(v, np.asarray(v0), atol=1e-6)
    # Consistent with the dycore's own diagnostic synthesis.
    g = spectral_pe_to_grid(state, grid, sigma)
    np.testing.assert_array_equal(u, np.asarray(g["u"]))


def test_grid_winds_from_spectral_passthrough_non_spectral():
    """A grid HydrostaticState (lat-lon/cubed) is returned UNCHANGED (grid/sigma
    unused) — the caller can apply the adapter unconditionally."""
    grid_state = HydrostaticState(
        u=Field(jnp.ones((2, 3, _NLEV))), v=Field(jnp.zeros((2, 3, _NLEV))),
        T=Field(jnp.full((2, 3, _NLEV), 280.0)), p_s=Field(jnp.full((2, 3), 1.0e5)),
        phis=Field(jnp.zeros((2, 3))))
    out = grid_winds_from_spectral(grid_state, None, None)
    assert out is grid_state          # same object, unchanged


def test_amip_column_state_spectral_end_to_end():
    """The AMIP compare path works for a SPECTRAL run: amip_column_state converts
    the spectral driver state to grid winds → a ColumnState with grid u/v (the
    worst-column ranking + the Gaussian extractor are now reachable)."""
    grid = create_gaussian_grid(n_max=21, dealiasing="linear")
    sigma = create_sigma_coordinate(_NLEV)
    state, u0, _ = _spectral_state(grid, sigma)
    driver = SimpleNamespace(
        state=state, q_v=jnp.full((grid.n_lat, grid.n_lon, _NLEV), 5e-3),
        grid=grid, sigma=sigma,
        get_sst_sic=lambda day: (jnp.full((grid.n_lat, grid.n_lon), 295.0), None))

    cs = amip_column_state(driver)
    assert cs.u.shape == (grid.n_lat, grid.n_lon, _NLEV)   # grid winds, not spectral
    assert np.all(np.isfinite(np.asarray(cs.u)))
    np.testing.assert_allclose(np.asarray(cs.u), np.asarray(u0), atol=1e-6)
    np.testing.assert_allclose(np.asarray(cs.sst_K), 295.0)
