"""The adding-method recurrence stores the PHYSICAL TOA fluxes at index -1 of
the (ncol, 1, nlev+2) flux arrays; the top model layer is heated by their
divergence against index -2.  A clipped-quadratic overwrite of that face
(``_replace_top_flux``, removed) zeroed the top layer's radiation on the
production AMIP state (LW 0.00 / SW +0.01 K/day with the overwrite versus
SW +2.31 / LW -0.94 K/day without it) and made the published OLR an
extrapolation (2.6 W/m2 high).  These tests pin the physical face.
"""
from __future__ import annotations

import inspect

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.radiation.rrtmgp import constants as rte_const
from legoesm.atmosphere.physics.radiation.rrtmgp.rte.two_stream import (
    compute_heating_rate,
)

NCOL, NLEV = 2, 6
F_TOA, DFLUX, DP = 250.0, 10.0, 3300.0

# Surface-first: index 0 = bottom halo, 1..NLEV = model layers, -1 = top halo;
# index k holds the net flux at the BOTTOM face of cell k.
_faces = np.arange(NLEV + 2)
_flux = F_TOA - DFLUX * (NLEV + 1 - _faces)          # F_TOA at index -1
FLUX_NET = jnp.asarray(np.broadcast_to(_flux, (NCOL, 1, NLEV + 2)))
DP_ARR = jnp.full((NCOL, 1, NLEV + 2), DP)
PRESSURE = jnp.asarray(np.broadcast_to(1.0e5 - DP * _faces, (NCOL, 1, NLEV + 2)))


def test_top_layer_heating_uses_physical_toa_face():
    heating = compute_heating_rate(FLUX_NET, PRESSURE, dp=DP_ARR)
    top = np.asarray(heating[:, 0, -2])
    expected = -rte_const.G * (
        np.asarray(FLUX_NET[:, 0, -1]) - np.asarray(FLUX_NET[:, 0, -2])
    ) / DP / rte_const.CP_D
    assert top == pytest.approx(expected)
    assert np.all(np.abs(top) > 1e-8)


def test_column_heating_closes_against_toa_face():
    heating = np.asarray(compute_heating_rate(FLUX_NET, PRESSURE, dp=DP_ARR))
    layers = heating[:, 0, 1:NLEV + 1]
    dp_layers = np.asarray(DP_ARR)[:, 0, 1:NLEV + 1]
    integral = np.sum(layers * dp_layers * rte_const.CP_D / rte_const.G, axis=1)
    expected = -(np.asarray(FLUX_NET[:, 0, -1]) - np.asarray(FLUX_NET[:, 0, 1]))
    np.testing.assert_allclose(integral, expected, rtol=1e-12)


def test_no_top_flux_overwrite_reintroduced():
    import legoesm.atmosphere.physics.radiation.rrtmgp.rte.two_stream as two_stream

    assert not hasattr(two_stream, "_replace_top_flux")
    assert "_replace_top_flux" not in inspect.getsource(two_stream.solve_lw)
    assert "_replace_top_flux" not in inspect.getsource(two_stream.solve_sw)


# ---------------------------------------------------------------------------
# Solver-level pin: the physical boundary face survives solve_lw / solve_sw.
# A single clear-sky column with a 10 hPa lid (the production sigma lid), one
# day and one night copy.  With the removed overwrite in place, sw_flux_down
# at the top face was the clipped extrapolation (~15 % below the insolation)
# and the top layer's SW/LW heating was ~0.
# ---------------------------------------------------------------------------
def _lid_columns(nlev=30, p_top=1000.0, p_sfc=101300.0, sfc_T=295.0):
    sig = np.linspace(0.0, 1.0, nlev + 1)
    p_half = p_top + (p_sfc - p_top) * sig                   # uniform sigma
    p_full = 0.5 * (p_half[:-1] + p_half[1:])
    z = -7000.0 * np.log(p_full / p_sfc)
    t_iso = sfc_T - 6.5e-3 * (-7000.0 * np.log(1.0e4 / p_sfc))
    temp = np.clip(np.where(p_full > 1.0e4, sfc_T - 6.5e-3 * z, t_iso),
                   200.0, 305.0)
    q = np.clip(0.015 * (p_full / p_sfc) ** 3, 1e-6, None)
    rep = lambda a: jnp.asarray(np.broadcast_to(a, (2,) + a.shape))  # noqa: E731
    return dict(
        T=rep(temp), p_full=rep(p_full), p_half=rep(p_half),
        sfc_temperature=jnp.asarray([sfc_T, sfc_T]),
        q_v=rep(q), cos_zenith=jnp.asarray([0.5, 0.0]),     # day, night
    )


@pytest.fixture(scope="module")
def lid_solution():
    from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
    cfg = RRTMGPConfig()
    out = RRTMGP.from_legoesm_config(cfg).solve_columns(**_lid_columns())
    return cfg, {k: np.asarray(getattr(out, k)) for k in (
        "sw_flux_down", "sw_flux_up", "lw_flux_down", "lw_flux_up",
        "sw_heating_rate", "lw_heating_rate")}


def test_solver_top_face_is_the_insolation_and_heats_the_top_layer(lid_solution):
    cfg, o = lid_solution
    day = 0
    # TOA-first: index 0 = the top face / the top model layer.
    assert o["sw_flux_down"][day, 0] == pytest.approx(cfg.S_0 * 0.5, rel=1e-6)
    assert o["lw_flux_down"][day, 0] == 0.0
    assert o["sw_heating_rate"][day, 0] > 1e-6          # K/s: ozone SW heating
    assert o["lw_heating_rate"][day, 0] < -1e-7         # K/s: cooling to space
    assert o["sw_flux_up"][day, 0] > 0.0


def test_solver_night_column_is_finite_and_dark(lid_solution):
    _cfg, o = lid_solution
    night = 1
    for k, v in o.items():
        assert np.all(np.isfinite(v[night])), k
    assert np.all(o["sw_flux_down"][night] == 0.0)
    assert np.all(o["sw_heating_rate"][night] == 0.0)
    assert o["lw_heating_rate"][night, 0] < -1e-7
