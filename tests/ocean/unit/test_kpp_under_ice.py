"""Under-ice attenuation of the KPP velocity scales (KPPConfig.eice; nn_eice).

2026-07-19: the halocline-erosion fix pair closed the Arctic on the TKE grid
(TKEConfig.eice) but F2 is TKE-only — the KPP grids (latlon default, MPAS)
kept the full over-deep MLD + Siberian salt.  KPPConfig.eice mirrors it:
compact ice scales w_m/w_s by (1-eff), shrinking BOTH the bulk-Ri boundary
layer depth (via V_t^2) and the mixing coefficients.  Pinned:

* the attenuation factor helper (mode 0/1/3 mapping + None + raise);
* eice=0 (default) is bit-identical to no ice_frac;
* full ice reduces the diffusivity AND shoals the boundary layer;
* dispatch: unknown eice raises; MPAS KPP bridge rejects eice != 0.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
from legoesm.ocean.physics.vertical_mixing.kpp import (
    _kpp_ice_attenuation,
    _kpp_velocity_scales,
)

jax.config.update("jax_enable_x64", True)


class TestAttenuationHelper:
    def test_off_and_none(self):
        assert _kpp_ice_attenuation(None, 0) == 1.0
        assert _kpp_ice_attenuation(jnp.ones((3,)), 0) == 1.0
        assert _kpp_ice_attenuation(None, 3) == 1.0

    def test_mode1_linear(self):
        fi = jnp.asarray([0.0, 0.5, 1.0])
        att = np.asarray(_kpp_ice_attenuation(fi, 1))[..., 0]
        np.testing.assert_allclose(att, [1.0, 0.5, 0.0], atol=1e-12)

    def test_mode3_steep(self):
        fi = jnp.asarray([0.0, 0.1, 0.25, 0.5])
        att = np.asarray(_kpp_ice_attenuation(fi, 3))[..., 0]
        np.testing.assert_allclose(att, [1.0, 0.6, 0.0, 0.0], atol=1e-12)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="eice"):
            _kpp_ice_attenuation(jnp.ones((2,)), 2)


class TestVelocityScales:
    @staticmethod
    def _inputs(n=4, nlev=10):
        u_star = jnp.full((n,), 0.01)
        B_f = jnp.full((n,), 1.0e-7)          # destabilising (convective)
        d = jnp.broadcast_to(jnp.linspace(1.0, 90.0, nlev), (n, nlev))
        h_bl = jnp.full((n, 1), 100.0)
        return u_star, B_f, d, h_bl

    def test_eice0_bit_identical(self):
        u_star, B_f, d, h_bl = self._inputs()
        cfg0 = KPPConfig()
        cfg3 = KPPConfig(eice=3)
        wm_a, ws_a = _kpp_velocity_scales(u_star, B_f, d, h_bl, cfg0, 1e-10)
        # eice=3 but ice_frac=None -> still no attenuation -> identical
        wm_b, ws_b = _kpp_velocity_scales(u_star, B_f, d, h_bl, cfg3, 1e-10,
                                          ice_frac=None)
        np.testing.assert_array_equal(np.asarray(wm_a), np.asarray(wm_b))
        np.testing.assert_array_equal(np.asarray(ws_a), np.asarray(ws_b))

    def test_full_ice_reduces_scales(self):
        u_star, B_f, d, h_bl = self._inputs()
        cfg = KPPConfig(eice=1)
        wm_o, ws_o = _kpp_velocity_scales(u_star, B_f, d, h_bl, cfg, 1e-10,
                                          ice_frac=jnp.zeros((4,)))
        wm_i, ws_i = _kpp_velocity_scales(u_star, B_f, d, h_bl, cfg, 1e-10,
                                          ice_frac=jnp.ones((4,)))
        # open water == eice-off; full ice floors both scales at 1e-10
        assert float(jnp.max(ws_o)) > 1e-6
        np.testing.assert_allclose(np.asarray(ws_i), 1e-10, atol=1e-12)
        np.testing.assert_allclose(np.asarray(wm_i), 1e-10, atol=1e-12)

    def test_partial_ice_scales_linearly_mode1(self):
        u_star, B_f, d, h_bl = self._inputs()
        cfg = KPPConfig(eice=1)
        _, ws0 = _kpp_velocity_scales(u_star, B_f, d, h_bl, cfg, 1e-10,
                                      ice_frac=jnp.zeros((4,)))
        _, ws_half = _kpp_velocity_scales(u_star, B_f, d, h_bl, cfg, 1e-10,
                                          ice_frac=jnp.full((4,), 0.5))
        # 50% ice -> half the scale (well above the 1e-10 floor here)
        np.testing.assert_allclose(np.asarray(ws_half), 0.5 * np.asarray(ws0),
                                   rtol=1e-9)


def test_config_default_off():
    assert KPPConfig().eice == 0


def test_mpas_kpp_bridge_rejects_eice():
    import inspect
    from legoesm.ocean.physics.vertical_mixing import mpas_integration
    for fn in (mpas_integration.make_kpp_physics_mpas,
               mpas_integration.make_kpp_profiles_mpas):
        src = inspect.getsource(fn)
        # the static eice guard + its NotImplementedError must both be present
        # (the message wraps across string-continuation lines, so match the
        # contiguous fragments, not the wrapped phrase).
        assert 'getattr(config.kpp, "eice", 0)' in src
        assert "NotImplementedError" in src
        assert "MPAS KPP bridge" in src
