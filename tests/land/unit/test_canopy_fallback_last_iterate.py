"""Unconverged two-leaf canopy columns take their last iterate's fluxes (CLM5).

``CanopyConfig.fallback_from_last_iterate`` (default True since 2026-11-20).  CLM5 stops its
canopy iteration after forty passes and keeps the last iterate, adding the
energy error to vegetation sensible heat (Technical Description 2.5.3.2,
step 12; eqs. 2.5.135, 2.5.138-2.5.139).  legoESM evaluated an unconverged
column's fluxes at the cold state instead (canopy at air temperature); over
hot dry soil that gave 1,000-1,400 W/m2 of sensible heat, the guards rejected
it and the column was held (enso_attribution docs/85 15.4: all 38 held
land column-steps of a 1 day smoke).

Tests: with the flag on, an unconverged column is evaluated at its last
iterate when admissible (not at the cold state), is accepted, and closes
Rn_ext = SH + LE + G exactly; with the flag off nothing changes.  The held
columns of the run (vegetated, stopped at the cap near a root) are not
reproduced by a synthetic column here: the synthetic bare columns converge in
two iterations.  float64.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land import multilayer_land as ml
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.soil_grid import SoilGridConfig
from legoesm.land.surface_scheme import TwoLeafCanopyConfig

pytestmark = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="float64 energy gates; run with JAX_ENABLE_X64=1")

_N = 4


def _forcing(T=300.0):
    o = jnp.ones(_N)
    ps = 1e5 * o
    return AtmToSurface(
        sw_down=900.0 * o, lw_down=400.0 * o, precip_total=0 * o, precip_snow=0 * o,
        T_lowest=T * o, q_lowest=0.008 * o, u_lowest=jnp.array([0.7, 1.4, 2.8, 3.6]),
        v_lowest=0 * o, p_lowest=0.99 * ps, p_surface=ps,
        rho_lowest=ps / (constants.R_d * T), cos_zenith=0.85 * o, co2_ppmv=340 * o,
        has_radiation=o, has_precipitation=o)


def _run(flag, max_iters, T_init):
    sch = TwoLeafCanopyConfig(max_iters=max_iters, fallback_from_last_iterate=flag,
                              implicit_ground_flux=False)  # one variable
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=10, dz_top=0.025, growth_factor=2.0,
                                 total_depth=3.0),
        surface_scheme=sch)
    st = ml.init_multilayer_land_state(_N, cfg, T_init=T_init, theta_init=0.12)
    f = _forcing()
    return jax.jit(lambda s: ml.step_multilayer_land_with_diagnostics(
        s, f, cfg, 1.0, 1800.0, lat=jnp.full(_N, -0.55)))(st)


def test_unconverged_column_uses_its_last_iterate_and_closes():
    new, r, _c, so = _run(True, 1, 307.0)
    unsolved = ~np.asarray(so.converged).astype(bool)
    assert unsolved.any()                                   # non-vacuity
    acc = unsolved & ~np.asarray(so.held).astype(bool)
    assert acc.any()
    # not the cold state: the cold state has leaves at the air temperature
    assert np.all(np.abs(np.asarray(so.Tf_Sun)[acc] - 300.0) > 1e-6)
    # the land boundary closes exactly for accepted unsolved columns
    E = np.asarray(so.Rn_ext - r.shflx - r.lhflx - so.G_soil)
    np.testing.assert_allclose(E[acc], 0.0, atol=1e-9)
    # flag off: the same columns sit at the cold state
    _new0, _r0, _c0, so0 = _run(False, 1, 307.0)
    np.testing.assert_allclose(np.asarray(so0.Tf_Sun)[unsolved], 300.0, atol=1e-9)


def test_last_iterate_gradient_matches_finite_differences_near_root(monkeypatch):
    """A column stopped at the cap near its root gets the root's gradient.

    The convergence gate is disabled (rtol = atol = 0) so every column stops
    at the cap three iterations in, close to its root (residual ~1e-14 of the
    seed's), as the held columns of a real run do.  d(SH+LE)/d(T_air) by
    reverse mode matches central differences; with the old zero cotangent for
    unconverged columns it was ~0.11 of them
    (scripts/validate/canopy_fallback_gradient_fd.py).
    """
    from legoesm.land.canopy import solver as canopy_solver
    monkeypatch.setattr(canopy_solver, "_LM_RTOL", 0.0)
    monkeypatch.setattr(canopy_solver, "_LM_ATOL", 0.0)
    sch = TwoLeafCanopyConfig(max_iters=3, implicit_ground_flux=False)
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=10, dz_top=0.025, growth_factor=2.0,
                                 total_depth=3.0),
        surface_scheme=sch)
    st = ml.init_multilayer_land_state(_N, cfg, T_init=307.0, theta_init=0.12)

    def run(dT):
        f = _forcing(300.0 + dT)
        return ml.step_multilayer_land_with_diagnostics(
            st, f, cfg, 1.0, 1800.0, lat=jnp.full(_N, -0.55))

    _, _, _, so = jax.jit(run)(0.0)
    acc = ~np.asarray(so.converged).astype(bool) & ~np.asarray(so.held).astype(bool)
    assert acc.all()                                        # non-vacuity
    m = jnp.asarray(acc, dtype=jnp.float64)

    def loss(dT):
        _, r, _, _ = run(dT)
        return jnp.sum(m * (r.shflx + r.lhflx))

    ad = float(jax.jit(jax.grad(loss))(0.0))
    lf = jax.jit(loss)
    h = 1e-4
    fd = float((lf(h) - lf(-h)) / (2 * h))
    np.testing.assert_allclose(ad, fd, rtol=1e-3)


def test_flag_on_by_default():
    assert TwoLeafCanopyConfig().fallback_from_last_iterate is True
    with pytest.raises(ValueError, match="fallback_from_last_iterate"):
        TwoLeafCanopyConfig(fallback_from_last_iterate=1).validate()
