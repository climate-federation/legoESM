"""Spectral PE: T damping corrected to pressure surfaces (opt-in).

DCMIP 2012 2-0-0 (constant-lapse-rate atmosphere at rest over a 2 km ridged
mountain) on the spectral dycore: the post-step spectral filter (and the
hyperdiffusion) act on T along the terrain-following eta surfaces, where a
stratified T varies, so they heat / cool the column over terrain with no wind
present (T21 matrix config: 5.3 K/day; T42: 101 K/day).

``SpectralPEConfig.T_damping_pressure_surfaces`` adds to each linear T damping
L a quadratic-in-p correction toward L on the pressure surfaces,
L_p T ~ L T - T_p B L(pi) - 1/2 T_pp B^2 [L(pi^2) - 2 pi L(pi)], with T_p, T_pp
from the AREA-MEAN T profile (first term: CAM3, Collins et al. 2004, eqs.
3.79-3.81 / 3.84).  Where the local lapse rate differs from the mean a
first-order error remains.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy


@pytest.fixture(autouse=True)
def _fp64_policy():
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


_NMAX, _NLEV, _DT, _DAY = 21, 20, 600.0, 86400.0


def _setup(on: bool, *, filt: float = 0.01, hyperdiff: float = 2.338e15,
           h_0: float = 2000.0, isothermal: bool = False):
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import standard_hybrid_levels
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        SpectralPEConfig, SpectralPrimitiveEquationModel)
    from tests.test_cases.dcmip2012 import rest_state_topography as rst
    grid = create_gaussian_grid(_NMAX)
    coord = standard_hybrid_levels(_NLEV)
    cfg = SpectralPEConfig(hyperdiff_coeff=hyperdiff, spectral_filter_order=8,
                           spectral_filter_strength=filt,
                           T_damping_pressure_surfaces=on)
    init = (rst.rest_state_topography_init_spectral if isothermal
            else rst.lapse_rate_state_topography_init_spectral)
    state = init(grid, coord, h_0=h_0)
    return grid, coord, cfg, SpectralPrimitiveEquationModel(grid, coord, cfg), state


def _filter_heating(on: bool, **kw):
    """Per-level max |dT/dt| [K/day] of the post-step filter (+ the 2/3 state
    truncation that follows it in the step) at rest, and its global
    mass-weighted heating [W/m^2]."""
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.grids.gaussian import sh_synthesis, sh_synthesis_3d
    from legoesm.grids.vertical import dp_from_hybrid
    grid, coord, _cfg, model, s = _setup(on, **kw)
    trunc = model._apply_state_truncation
    dT = sh_synthesis_3d(grid, trunc(model._apply_spectral_filter(s)).T_hat.data
                         - trunc(s).T_hat.data) / _DT
    dp = dp_from_hybrid(coord, jnp.exp(sh_synthesis(grid, s.lnps_hat.data)))
    area = grid.grid_area
    wm2 = float(jnp.sum(constants.c_pd * jnp.sum(dT * dp, -1) / constants.g * area)
                / jnp.sum(area))
    return np.asarray(jnp.abs(dT).max(axis=(0, 1))) * _DAY, wm2


def test_local_quadratic_dT_dp_exact_for_quadratic_profiles():
    """Nonuniform levels, all levels including the one-sided ends."""
    import jax.numpy as jnp
    from legoesm.grids.vertical import local_quadratic_dT_dp
    p = jnp.array([[1.0e3, 5.0e3, 1.6e4, 3.0e4, 5.5e4, 8.0e4, 9.6e4]])
    a, b, c = 210.0, 1.1e-3, -4.0e-9
    Tp, Tpp = local_quadratic_dT_dp(a + b * p + c * p * p, p)
    np.testing.assert_allclose(np.asarray(Tp), np.asarray(b + 2 * c * p), rtol=1e-9)
    np.testing.assert_allclose(np.asarray(Tpp), 2 * c, rtol=1e-6)
    with pytest.raises(ValueError, match="nlev >= 3"):
        local_quadratic_dT_dp(p[:, :2], p[:, :2])


def test_filter_heating_at_rest_over_terrain_is_corrected():
    """Measured (fp64, T21 L20 hybrid): interior peak 5.3 -> 0.2 K/day; the
    global heating stays small (uncorrected -0.03, corrected +0.01 W/m^2;
    CAM3's first-order ln(pi) form alone gives +0.35 at T21, +69 at T42)."""
    off, wm2_off = _filter_heating(False)
    on, wm2_on = _filter_heating(True)
    assert off[1:-1].max() > 2.0, off            # the defect is there
    assert on[1:-1].max() < 0.1 * off[1:-1].max(), (on, off)
    assert on[-1] < 0.1 * off[-1] and on[0] <= off[0] + 1e-9, (on, off)
    assert abs(wm2_on) < 0.1, wm2_on


def test_hyperdiffusion_heating_at_rest_over_terrain_is_corrected():
    """Explicit del^4 on T (filter off, coefficient x100 so it dominates): the
    tendency at rest IS the damping; corrected to pressure surfaces it nearly
    vanishes."""
    import jax.numpy as jnp
    from legoesm.grids.gaussian import sh_synthesis_3d
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import spectral_pe_tendencies
    out = {}
    for on in (False, True):
        grid, coord, cfg, _m, s = _setup(on, filt=0.0, hyperdiff=2.338e17)
        dT = sh_synthesis_3d(grid, spectral_pe_tendencies(s, grid, coord, cfg).T_hat.data)
        out[on] = np.asarray(jnp.abs(dT).max(axis=(0, 1))) * _DAY
    assert out[False].max() > 0.1, out[False]
    assert out[True].max() < 0.1 * out[False].max(), out


def test_default_off_is_unchanged():
    """Default: the filter and implicit-hyperdiffusion paths return exactly
    the plain multiplicative damping (bitwise check against the base commit is
    out-of-tree, see the commit message)."""
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        SpectralPEConfig, apply_spectral_filter_to_state)
    assert SpectralPEConfig().T_damping_pressure_surfaces is False
    _g, _c, _cfg, model, s = _setup(False)
    a = model._apply_spectral_filter(s)
    b = apply_spectral_filter_to_state(s, model._spectral_filter)
    for k in ("vor_hat", "div_hat", "T_hat", "lnps_hat"):
        assert np.array_equal(np.asarray(getattr(a, k).data),
                              np.asarray(getattr(b, k).data)), k
    # implicit hyperdiffusion: exactly the multiplicative factor
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        SpectralPrimitiveEquationModel)
    grid, coord, cfg, _m, s = _setup(False, filt=0.0, hyperdiff=2.338e17)
    model = SpectralPrimitiveEquationModel(
        grid, coord, cfg._replace(implicit_hyperdiff=True))
    model._ensure_hyperdiff_filter(_DT)
    got = np.asarray(model._apply_implicit_hyperdiff(s).T_hat.data)
    want = np.asarray(s.T_hat.data * model._hyperdiff_filter[:, None])
    assert np.array_equal(got, want)


def test_option_refuses_fewer_than_three_levels():
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        SpectralPEConfig, SpectralPrimitiveEquationModel)
    with pytest.raises(ValueError, match="nlev >= 3"):
        SpectralPrimitiveEquationModel(
            create_gaussian_grid(_NMAX), create_sigma_coordinate(2),
            SpectralPEConfig(T_damping_pressure_surfaces=True))


def test_correction_touches_T_only_and_vanishes_when_flat_or_isothermal():
    """Mass (lnps) and winds untouched; zero for a flat surface (L(pi) = 0)
    and, to round-off, for an isothermal column (T_p = T_pp = 0)."""
    _g, _c, _cfg, m_on, s = _setup(True)
    _g, _c, _cfg, m_off, _s = _setup(False)
    a, b = m_on._apply_spectral_filter(s), m_off._apply_spectral_filter(s)
    for k in ("vor_hat", "div_hat", "lnps_hat"):
        assert np.array_equal(np.asarray(getattr(a, k).data),
                              np.asarray(getattr(b, k).data)), k
    assert np.abs(np.asarray(a.T_hat.data - b.T_hat.data)).max() > 1e-3
    for kw in ({"h_0": 0.0}, {"isothermal": True}):
        _g, _c, _cfg, m_on, s = _setup(True, **kw)
        _g, _c, _cfg, m_off, _s = _setup(False, **kw)
        d = np.abs(np.asarray(m_on._apply_spectral_filter(s).T_hat.data
                              - m_off._apply_spectral_filter(s).T_hat.data)).max()
        assert d < 1e-9, (kw, d)


def test_implicit_hyperdiffusion_heating_at_rest_over_terrain_is_corrected():
    """Post-step implicit hyperdiffusion factor exp(-nu eig dt) (filter off,
    coefficient x100): the same correction with L = factor - 1."""
    import jax.numpy as jnp
    from legoesm.grids.gaussian import sh_synthesis_3d
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        SpectralPrimitiveEquationModel)
    out = {}
    for on in (False, True):
        grid, coord, cfg, _m, s = _setup(on, filt=0.0, hyperdiff=2.338e17)
        model = SpectralPrimitiveEquationModel(
            grid, coord, cfg._replace(implicit_hyperdiff=True))
        model._ensure_hyperdiff_filter(_DT)
        s = model._apply_state_truncation(s)      # the step's order
        post = model._apply_implicit_hyperdiff(s).T_hat.data
        dT = sh_synthesis_3d(grid, post - s.T_hat.data) / _DT
        out[on] = np.asarray(jnp.abs(dT).max(axis=(0, 1))) * _DAY
        # runs after the 2/3 truncation: must not repopulate the cut band
        cut = ~np.asarray(model._dealias_state, dtype=bool)
        assert cut.any() and np.abs(np.asarray(post)[cut]).max() == 0.0
    assert out[False].max() > 0.1, out[False]
    assert out[True].max() < 0.1 * out[False].max(), out
