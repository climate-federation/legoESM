"""Spectral hybrid pressure-gradient balance over terrain (spectral twin of
lat-lon #1029, tests/atmosphere/dycore/regression/test_latlon_pgf_rest_balance_1029.py).

An isothermal atmosphere at rest over the DCMIP 2012 §2-0-0 mountain must get
no push from the dynamics.  Before the fix the spectral hybrid lane carried
``R_d T_ref (1 - B p_s/p) grad ln p_s`` (~3000 m/s/day on the B=0 levels of the
CAM6 L32 table); it now differences the SB81 full-level ln p the geopotential
integrates.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm import constants
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    SpectralPEConfig, spectral_pe_tendencies, spectral_pe_to_grid)
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.vertical import make_cam6_l32_levels
from tests.test_cases.dcmip2012.rest_state_topography import (
    rest_state_topography_init_spectral)

_DAY = 86400.0
_REST_LIMIT_MS_PER_DAY = 1.0   # G1 (user-approved): every level


@pytest.fixture(autouse=True)
def _fp64_policy():
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


@pytest.fixture(scope="module")
def setup():
    grid = create_gaussian_grid(31, radius=constants.R_earth)
    coord = make_cam6_l32_levels()
    state = rest_state_topography_init_spectral(grid, coord)
    return grid, coord, state


def _wind_tendency(state, grid, coord, cfg):
    t = spectral_pe_tendencies(state, grid, coord, cfg, None)
    f = spectral_pe_to_grid(t, grid, coord)
    return np.asarray(f["u"]) * _DAY, np.asarray(f["v"]) * _DAY


def test_hybrid_rest_over_terrain_every_level(setup):
    grid, coord, state = setup
    du, dv = _wind_tendency(state, grid, coord, SpectralPEConfig())
    per_level = np.maximum(np.abs(du).max(axis=(0, 1)), np.abs(dv).max(axis=(0, 1)))
    assert np.all(np.isfinite(per_level))
    assert per_level.max() < _REST_LIMIT_MS_PER_DAY, (
        f"rest-state push per level [m/s/day]: {np.round(per_level, 3).tolist()}")


def test_hybrid_explicit_tendency_independent_of_t_ref(setup):
    grid, coord, state = setup
    a = _wind_tendency(state, grid, coord, SpectralPEConfig(si_T_ref=300.0))
    b = _wind_tendency(state, grid, coord, SpectralPEConfig(si_T_ref=250.0))
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, rtol=1e-10, atol=1e-10 * np.abs(x).max())


def test_hybrid_pgf_check_grads_order2(setup):
    """AD through the changed hybrid PGF and heating terms, with a smooth wind
    so the kappa*T*v.grad(ln p) heating is exercised (zero at rest)."""
    grid, coord, state = setup
    cfg = SpectralPEConfig()
    from legoesm.grids.gaussian import sh_analysis, sh_analysis_3d
    rng = np.random.default_rng(0)
    lat = np.asarray(grid.lat2d)
    lon = np.asarray(grid.lon2d)
    nlev = state.vor_hat.data.shape[-1]
    vor = 3e-5 * (np.sin(2 * lat) * np.cos(lon))[..., None] * np.ones(nlev)
    div = 4e-6 * (np.cos(lat) * np.sin(2 * lon))[..., None] * np.ones(nlev)
    state = state._replace(
        vor_hat=state.vor_hat.replace(data=sh_analysis_3d(grid, jnp.asarray(vor)).astype(
            state.vor_hat.data.dtype)),
        div_hat=state.div_hat.replace(data=sh_analysis_3d(grid, jnp.asarray(div)).astype(
            state.div_hat.data.dtype)))
    bump = jnp.asarray(0.01 * np.cos(lat) * rng.uniform(0.5, 1.0, lat.shape))
    d_hat = sh_analysis(grid, bump)

    def tend(eps):
        s = state._replace(lnps_hat=state.lnps_hat.replace(
            data=state.lnps_hat.data + eps * d_hat))
        t = spectral_pe_tendencies(s, grid, coord, cfg, None)
        return jnp.real(t.div_hat.data).ravel(), jnp.real(t.T_hat.data).ravel()

    d0, t0 = tend(jnp.asarray(0.3))
    sd, st = jnp.max(jnp.abs(d0)), jnp.max(jnp.abs(t0))

    def f(eps):
        d, t = tend(eps)
        return jnp.concatenate([d / sd, t / st])

    check_grads(f, (jnp.asarray(0.3),), order=2, modes=("fwd", "rev"))
    _, (_, dt_deps) = jax.jvp(tend, (jnp.asarray(0.3),), (jnp.asarray(1.0),))
    assert float(jnp.max(jnp.abs(dt_deps))) > 0.0


def test_semi_implicit_refused_on_hybrid_levels(setup):
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        SpectralPrimitiveEquationModel)
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.neural_gcm_spectral import _make_spectral_integrator
    grid, coord, _ = setup
    si = SpectralPEConfig(semi_implicit=True)
    with pytest.raises(NotImplementedError, match="hybrid"):
        SpectralPrimitiveEquationModel(grid, coord, si)
    with pytest.raises(NotImplementedError, match="hybrid"):
        _make_spectral_integrator(si, grid, coord, 600.0, "ssp_rk3")
    for name in ("leapfrog", "leapfrog_si", "Leapfrog_SI"):
        with pytest.raises(NotImplementedError, match="hybrid"):
            SpectralPrimitiveEquationModel(
                grid, coord, SpectralPEConfig(time_integrator=name))
    # sigma levels and the explicit path on hybrid levels still build
    SpectralPrimitiveEquationModel(grid, create_sigma_coordinate(8), si)
    SpectralPrimitiveEquationModel(grid, coord, SpectralPEConfig())
