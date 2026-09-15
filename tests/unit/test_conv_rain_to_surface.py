"""Opt-in routing of the survivor convective rain to surface precipitation.

The MPAS bridge hands the in-updraught rain that survives Bechtold's own
sub-cloud evaporation to the microphysics ``q_r`` tracer at its formation
levels, where the microphysics evaporates it a second time at grid-mean
humidity (dd_ctl day 110: 4.1 kg/m2/day below sigma 0.7, 93 % of the surface
rain rate).  ``ConvectionConfig.rain_to_surface`` routes that rain out of the
column as the ``precip`` field the combined-physics accumulator sums (IFS
cuflxn convention).

Every assertion here fails with the integration.py change reverted: the q_r
key reappears, ``precip`` is None, and the conservation identity breaks.
The bridge is CALLED (spy pattern from test_bechtold_mixing_gradients); the
kernel's returned ``dq_r_conv_dt`` is overridden with a fixed positive
profile so the rain source is known exactly.
"""
from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics.convection import integration as ci  # noqa: E402
from legoesm.atmosphere.physics.convection.config import ConvectionConfig  # noqa: E402
from legoesm.core.field import Field  # noqa: E402
from legoesm.core.state import HydrostaticState  # noqa: E402
from legoesm.grids.vertical import create_sigma_coordinate  # noqa: E402

NCOL, NLEV = 6, 16
DT = 600.0
_DQ_R = np.maximum(
    np.random.default_rng(7).normal(2.0e-7, 1.0e-7, size=(NCOL, NLEV)), 1.0e-9)


def _run_bridge(flag: bool, with_qr: bool = True):
    sigma = create_sigma_coordinate(NLEV)
    p_s = jnp.full((NCOL,), 1.0e5)
    T = jnp.broadcast_to(jnp.linspace(298.0, 215.0, NLEV)[None, :], (NCOL, NLEV))
    q = jnp.broadcast_to(jnp.linspace(0.017, 1e-5, NLEV)[None, :], (NCOL, NLEV))
    tracers = {"q_v": Field(q), "q_c": Field(jnp.zeros((NCOL, NLEV)))}
    if with_qr:
        tracers["q_r"] = Field(jnp.zeros((NCOL, NLEV)))
    state = HydrostaticState(
        u=Field(jnp.zeros((NCOL, NLEV))), T=Field(T), p_s=Field(p_s),
        phis=Field(jnp.zeros((NCOL,))), v=Field(jnp.zeros((NCOL, NLEV))),
        tracers=tracers)
    grid = types.SimpleNamespace(grid_shape_2d=(NCOL,), land_frac=np.zeros(NCOL))
    real = ci.bechtold_convection

    def spy(*a, **k):
        out, m_u, stoch = real(*a, **k)
        return out._replace(dq_r_conv_dt=jnp.asarray(_DQ_R)), m_u, stoch

    ci.bechtold_convection = spy
    try:
        fn = ci.make_convection_physics(
            ConvectionConfig(scheme="bechtold", rain_to_surface=flag),
            model_type="mpas", dt=DT)
        tend, prog_out = fn(state, grid, sigma)
    finally:
        ci.bechtold_convection = real
    return tend, prog_out


def _dp():
    # The bridge integrates with the coordinate's OWN (float32) half pressures;
    # use the same values so the identities below hold to round-off.
    sigma = create_sigma_coordinate(NLEV)
    p_half = np.asarray(sigma.pressure_at_half(jnp.full((NCOL,), 1.0e5)), dtype=np.float64)
    return p_half[:, 1:] - p_half[:, :-1]


def _column(t, key):
    if key not in t.tracer_tendencies:
        return np.zeros(NCOL)
    return (np.asarray(t.tracer_tendencies[key].data).reshape(NCOL, NLEV) * _dp()).sum(-1) / constants.g


def test_flag_off_keeps_the_legacy_q_r_hand_off():
    t, _ = _run_bridge(False)
    assert "q_r" in t.tracer_tendencies
    assert getattr(t, "precip", None) is None


def test_flag_on_emits_the_rain_as_surface_precip_and_conserves_water():
    t_off, _ = _run_bridge(False)
    t_on, _ = _run_bridge(True)
    assert "q_r" not in t_on.tracer_tendencies
    np.testing.assert_array_equal(
        np.asarray(t_on.tracer_tendencies["q_c"].data),
        np.asarray(t_off.tracer_tendencies["q_c"].data))
    assert t_on.precip is not None and t_on.precip.units == "kg/m^2/s"
    precip = np.asarray(t_on.precip.data).reshape(NCOL)
    assert precip.dtype == np.asarray(t_on.dp_s_dt.data).dtype
    expect = (np.maximum(_DQ_R, 0.0) * _dp()).sum(-1) / constants.g
    np.testing.assert_allclose(precip, expect, rtol=1e-12)
    assert np.all(precip > 0.0)
    total_off = _column(t_off, "q_v") + _column(t_off, "q_c") + _column(t_off, "q_r")
    total_on = _column(t_on, "q_v") + _column(t_on, "q_c") + precip
    np.testing.assert_allclose(total_on, total_off, rtol=1e-12, atol=1e-18)


def test_flag_on_without_a_q_r_tracer_does_not_fold_the_rain_into_cloud():
    t_off, _ = _run_bridge(False, with_qr=False)
    t_on, _ = _run_bridge(True, with_qr=False)
    fold = (np.asarray(t_off.tracer_tendencies["q_c"].data)
            - np.asarray(t_on.tracer_tendencies["q_c"].data))
    np.testing.assert_allclose(fold, _DQ_R, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(t_on.precip.data).reshape(NCOL),
        (np.maximum(_DQ_R, 0.0) * _dp()).sum(-1) / constants.g, rtol=1e-12)


def test_conv_precip_carry_is_unchanged_by_the_switch():
    _, prog_off = _run_bridge(False)
    _, prog_on = _run_bridge(True)
    np.testing.assert_array_equal(np.asarray(prog_on["conv_precip"]),
                                  np.asarray(prog_off["conv_precip"]))
    assert np.all(np.asarray(prog_on["conv_precip"]) > 0.0)
