"""#2241 RULE 5: the E3SM-CAM scheme's ``U > 2 m/s`` orographic launch gate.

``E3SMOrographicConfig.oro_wind_ramp_width`` (default 0.5 m/s, the
LindzenConfig form) multiplies the launch stress by
``smoothstep_quintic((U - 2)/w)``; ``0.0`` is E3SM's hard gate.  Column 12 of
the E3SM v3.0.1 orographic oracle fixture (``lindzen_e3sm.npz``, see
``test_lindzen_e3sm_oracle.py``) launches at a source wind U0 = 2.6563 m/s;
scaling its wind by s puts the source wind at s*U0.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    E3SMCAMConfig, E3SMOrographicConfig)
from legoesm.atmosphere.physics.gravity_wave_drag.e3sm_cam import e3sm_cam_gwd

FIXTURE = Path(__file__).resolve().parents[2] / "oracle_fixtures" / "lindzen_e3sm.npz"
_KEYS = ("u", "v", "T", "p_full", "p_half", "z_full", "z_half", "rho", "lat", "sgh")
_COL, _U0 = 12, 2.656279920412854


@pytest.fixture(scope="module")
def col():
    if not jax.config.jax_enable_x64:
        pytest.skip("RULE 5 gaps are recorded in fp64 (JAX_ENABLE_X64=1)")
    f = np.load(FIXTURE)
    return {k: jnp.asarray(f[k][_COL:_COL + 1]) for k in _KEYS} | {"dt": float(f["dt"])}


def _cfg(width):
    return E3SMCAMConfig(source="orographic",
                         orographic=E3SMOrographicConfig(oro_wind_ramp_width=width))


def _du(col, s, width):
    out = e3sm_cam_gwd(col["u"] * s, col["v"] * s, col["T"], col["p_full"], col["p_half"],
                       col["z_full"], col["z_half"], col["rho"], col["lat"], col["dt"],
                       _cfg(width), h_topo_col=col["sgh"])
    return out.du_dt


def test_default_is_the_lindzen_width():
    assert E3SMOrographicConfig().oro_wind_ramp_width == 0.5


def test_hard_limit_is_e3sms_gate(col):
    """Width 0: no drag at and below U = 2 m/s, the full E3SM stress above it
    (a jump); outside the ramp the default is the hard result bit for bit."""
    below, above = 2.0 / _U0 * (1 - 1e-9), 2.0 / _U0 * (1 + 1e-9)
    assert float(jnp.abs(_du(col, below, 0.0)).max()) == 0.0
    assert float(jnp.abs(_du(col, above, 0.0)).max()) > 0.0
    for s in (1.0, 2.6 / _U0):
        np.testing.assert_array_equal(np.asarray(_du(col, s, 0.5)), np.asarray(_du(col, s, 0.0)))


# Gap of the default vs the hard gate per case: max|du(0.5) - du(0)| /
# max|du(0)| at source wind U = s*U0 [m/s].  Measured 2026-11-08 (#2241).
# 1.0 up to 2.1 m/s: the weaker smooth launch stress deposits at other levels,
# so the default's du/dt is zero where the hard one peaks.  0 from ~2.2 m/s:
# the column's max du/dt is saturation-limited, not launch-limited.
_GAP = {2.02: 1.0, 2.05: 1.0, 2.1: 1.0, 2.15: 0.6354419476179015, 2.25: 0.0}


@pytest.mark.parametrize("U", sorted(_GAP))
def test_default_gap_per_case(col, U):
    s = U / _U0
    hard, soft = np.asarray(_du(col, s, 0.0)), np.asarray(_du(col, s, 0.5))
    gap = np.abs(soft - hard).max() / np.abs(hard).max()
    np.testing.assert_allclose(gap, _GAP[U], rtol=1e-6, atol=1e-12)


def test_check_grads_through_the_gate_at_default(col):
    """On the gate (s = 2/U0) the hard stress jumps (check_grads fails at
    width 0); the default is C2 there.  At 2.15 m/s (in the ramp, before the
    column saturates) the derivative is non-zero and differs from the hard
    one: the ramp, not saturation, sets it."""
    def f(s, width=0.5):
        return jnp.sum(_du(col, s, width))
    fd = jax.jit(f)
    check_grads(fd, (jnp.asarray(2.0 / _U0),), order=2, modes=("fwd", "rev"), eps=1e-5)
    s_mid = jnp.asarray(2.15 / _U0)
    check_grads(fd, (s_mid,), order=2, modes=("fwd", "rev"), eps=1e-5)
    g_soft = float(jax.grad(fd)(s_mid))
    g_hard = float(jax.grad(lambda s: f(s, 0.0))(s_mid))
    assert g_soft != 0.0 and g_soft != g_hard
