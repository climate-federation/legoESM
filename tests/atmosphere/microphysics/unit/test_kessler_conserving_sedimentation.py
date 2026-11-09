"""Kessler rain sedimentation over layer thickness conserves column water.

``KesslerConfig.conserving_sedimentation`` (default True, user decision
2026-11-09) divides the sedimentation flux divergence by the layer thickness,
so on every oracle fixture column the column water change plus the surface
rain is zero to roundoff.  False is the oracle's level-centre form, which
reports more rain than leaves the column; its per-column excess is pinned
below (measured on main before the change, oracle form, hard saturation limit).
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/atmosphere/oracle_fixtures/kessler.npz"
HARD = dict(saturation_sharpness=1.0e30)
CONSERVE_RTOL = 1.0e-10        # of the column's surface rain (voted 3-0)
# (column water change + surface rain) / surface rain, oracle form, hard limit;
# 0.0 is the fixture's rain-free column.
ORACLE_EXCESS = {
    1200.0: [0.1445202599609262, 0.05111189071162986, 0.13810816314525043,
             0.32301479242621284, 0.042249489910942926, 0.0,
             0.08495940983786825],
    300.0: [0.14859212143325382, 0.05079410078113758, 0.1475823013285174,
            0.33175260247592064, 0.03749621727760283, 0.0,
            0.014387619259817153],
}


def _excess(dt, **cfg):
    from legoesm.atmosphere.physics.microphysics.config import KesslerConfig
    from legoesm.atmosphere.physics.microphysics.kessler import kessler_step_states
    f = np.load(FIXTURE)
    T, qv, qc, qr, pf, rho, dz = (jnp.asarray(f[k]) for k in (
        "col_T", "col_q_v", "col_q_c", "col_q_r", "col_p_full", "col_rho",
        "col_dz"))
    out = kessler_step_states(T, qv, qc, qr, pf, rho, dz, dt,
                              KesslerConfig(**cfg))
    w0 = jnp.sum((qv + qc + qr) * rho * dz, -1)
    w1 = jnp.sum((out[1] + out[2] + out[3]) * rho * dz, -1)
    rain = out[4] * dt
    return np.asarray(w1 - w0 + rain), np.asarray(rain)


def test_default_is_conserving():
    from legoesm.atmosphere.physics.microphysics.config import KesslerConfig
    assert KesslerConfig().conserving_sedimentation is True


@pytest.mark.parametrize("dt", [1200.0, 300.0])
def test_conserving_form_closes_column_water(dt):
    res, rain = _excess(dt, **HARD)
    assert np.all(np.isfinite(res))
    assert rain.max() > 0.0
    assert np.all(np.abs(res) <= CONSERVE_RTOL * rain), res / np.maximum(rain, 1e-300)


@pytest.mark.parametrize("dt", [1200.0, 300.0])
def test_oracle_form_keeps_the_recorded_per_column_excess(dt):
    res, rain = _excess(dt, conserving_sedimentation=False, **HARD)
    rel = np.where(rain > 0, res / np.where(rain > 0, rain, 1.0), 0.0)
    np.testing.assert_allclose(rel, ORACLE_EXCESS[dt], rtol=1e-12, atol=1e-15)


def test_traced_switch_raises():
    from legoesm.atmosphere.physics.microphysics.config import KesslerConfig
    from legoesm.atmosphere.physics.microphysics.kessler import kessler_step_states
    f = np.load(FIXTURE)
    args = [jnp.asarray(f[k]) for k in (
        "col_T", "col_q_v", "col_q_c", "col_q_r", "col_p_full", "col_rho",
        "col_dz")]
    with pytest.raises(ValueError, match="conserving_sedimentation"):
        jax.jit(lambda flag: kessler_step_states(
            *args, 300.0, KesslerConfig(conserving_sedimentation=flag)))(True)


def test_top_layer_courant_is_limited():
    """Heavy rain in a thin top layer: the conserving CFL covers the top level
    too, so the outflow never overdraws the layer and no water is clipped."""
    from legoesm.atmosphere.physics.microphysics.config import KesslerConfig
    from legoesm.atmosphere.physics.microphysics.kessler import kessler_step_states
    f = np.load(FIXTURE)
    T, qv, qc, qr, pf, rho, dz = (np.array(f[k]) for k in (
        "col_T", "col_q_v", "col_q_c", "col_q_r", "col_p_full", "col_rho",
        "col_dz"))
    qr[:, 0] = 5.0e-3
    dz[:, 0] = 20.0
    args = [jnp.asarray(a) for a in (T, qv, qc, qr, pf, rho, dz)]
    out = kessler_step_states(*args, 300.0, KesslerConfig(**HARD))
    w0 = jnp.sum((args[1] + args[2] + args[3]) * args[5] * args[6], -1)
    w1 = jnp.sum((out[1] + out[2] + out[3]) * args[5] * args[6], -1)
    rain = out[4] * 300.0
    res = np.asarray(w1 - w0 + rain)
    assert np.all(np.abs(res) <= CONSERVE_RTOL * np.asarray(rain)), res
