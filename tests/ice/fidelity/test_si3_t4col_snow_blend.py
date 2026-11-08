"""Column BL99 snow/no-snow switch: hard forward, straight-through gradient (user ruling 3a).

* Forward: with the surrogate width ``snow_ramp_width_m`` > 0 every output is
  bitwise the hard single solve, on every tier-4 fixture (NEMO 5.0.1 SI3
  ORCA1, kt=17281).
* Gradient: d/dh_s of the output equals the central FD of the soft column
  (vote Z22 = c: both systems solved, enthalpies and fluxes blended).
* Closure of the soft column: its enthalpy content change and conductive
  fluxes are the s-blend of the two systems' (content is linear in e at fixed
  h); at h_s = 0 it is the no-snow system exactly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ice import bitz_lipscomb as bl
from legoesm.ice.config import SI3SwitchConfig
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as K

from si3_t4col_fixture import dt, load, zdf_args

jax.config.update("jax_enable_x64", True)
FIXTURES = ("c2_ti_clamp", "c3_qcn_top_k", "c4_fullmelt", "c5_flood_sst", "c6_melt_order",
            "c7_remap_eps", "c8_sal_himin", "c9_exchange")
W = 2.0e-4      # surrogate width = 0.01 x snow_cover_depth_nemo (#2407 vote Y1, Z34)
ON = SI3SwitchConfig(snow_ramp_width_m=W)
OFF = SI3SwitchConfig()


def _zdf(fx, sw, h_s=None):
    e_i, e_s, sz, h_i, hs, t_su, fz = zdf_args(fx)
    hs = hs if h_s is None else jnp.full_like(hs, h_s)
    return bl._si3_zdf_bl99_step(e_i, e_s, sz, h_i, hs, t_su, fz, dt(fx), K,
                                 nemo_version="5.0.1", switches=sw)


def _soft(monkeypatch, fx, h_s, pin=None):
    """The soft column (straight_through returns its soft argument); ``pin`` fixes s."""
    monkeypatch.setattr(bl, "straight_through", lambda hard, soft: soft)
    if pin is not None:
        monkeypatch.setattr(bl, "smoothstep_cubic", lambda t: jnp.full_like(t, pin))
    try:
        return _zdf(fx, ON, h_s)
    finally:
        monkeypatch.undo()


def test_forward_is_bitwise_the_hard_solve_on_every_fixture():
    n_snow = n_bare = 0
    for name in FIXTURES:
        fx = load(name)
        a, b = _zdf(fx, ON), _zdf(fx, OFF)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        h_s = np.asarray(zdf_args(fx)[4])
        n_snow += int((h_s > 0).sum()); n_bare += int((h_s == 0).sum())
    assert n_snow and n_bare, (n_snow, n_bare)          # both branches exercised


def test_gradient_is_the_soft_column_derivative(monkeypatch):
    fx = load("c3_qcn_top_k")
    for h in (0.25 * W, 0.5 * W, 0.9 * W):
        g = jax.grad(lambda x: jnp.sum(_zdf(fx, ON, x).qcn_ice_top))(jnp.asarray(h))
        eps = 1e-6 * W      # vote Z35
        fd = (float(jnp.sum(_soft(monkeypatch, fx, h + eps).qcn_ice_top))
              - float(jnp.sum(_soft(monkeypatch, fx, h - eps).qcn_ice_top))) / (2 * eps)
        assert bool(jnp.isfinite(g)) and float(g) != 0.0
        g_hard = jax.grad(lambda x: jnp.sum(_zdf(fx, OFF, x).qcn_ice_top))(jnp.asarray(h))
        assert abs(float(g) - float(g_hard)) > 1e-3 * abs(float(g_hard))    # the blend term is present
        np.testing.assert_allclose(float(g), fd, rtol=1e-4)
    g0 = jax.grad(lambda x: jnp.sum(_zdf(fx, ON, x).qcn_ice_top))(jnp.asarray(0.0))
    assert bool(jnp.isfinite(g0))


def test_soft_column_is_the_blend_of_the_two_systems(monkeypatch):
    fx = load("c3_qcn_top_k")
    e_i0, e_s0, _, h_i, _, _, _ = zdf_args(fx)
    h = 0.5 * W
    s = float(bl.smoothstep_cubic(jnp.asarray(0.5)))
    mid, snow, bare = (_soft(monkeypatch, fx, h, pin) for pin in (None, 1.0, 0.0))

    def content(r):
        return (np.sum(np.asarray(r.e_ice), -1) * np.asarray(h_i)
                + np.sum(np.asarray(r.e_snow), -1) * h) / 3.0
    assert np.any(content(snow) != content(bare))       # the two systems differ (non-vacuous)
    blend = s * content(snow) + (1 - s) * content(bare)
    np.testing.assert_allclose(content(mid), blend, rtol=1e-13)
    for k in ("qcn_ice_top", "qcn_ice_bottom", "qns_ice"):
        np.testing.assert_allclose(np.asarray(getattr(mid, k)),
                                   s * np.asarray(getattr(snow, k)) + (1 - s) * np.asarray(getattr(bare, k)),
                                   rtol=1e-13, atol=1e-12)
    # Snow layers are inert in the no-snow system (vote Z31).
    np.testing.assert_array_equal(np.asarray(bare.e_snow), np.asarray(e_s0))


def test_soft_column_at_zero_snow_is_the_no_snow_system(monkeypatch):
    """At h_s = 0 the soft column equals the independent hard no-snow solve (Z38)."""
    fx = load("c3_qcn_top_k")
    at0, hard = _soft(monkeypatch, fx, 0.0), _zdf(fx, OFF, 0.0)
    for k in ("e_ice", "qcn_ice_top", "qcn_ice_bottom", "qns_ice", "T_surface", "T_ice"):
        assert np.all(np.isfinite(np.asarray(getattr(at0, k)))), k     # no NaN from the snow system at h_s = 0
        np.testing.assert_array_equal(np.asarray(getattr(at0, k)), np.asarray(getattr(hard, k)))
    # Soft snow temperature is continuous at h_s -> 0+ (Z37).
    tiny = _soft(monkeypatch, fx, 1e-30)
    np.testing.assert_allclose(np.asarray(tiny.T_snow), np.asarray(at0.T_snow), rtol=1e-13)   # Z35 closure tolerance
