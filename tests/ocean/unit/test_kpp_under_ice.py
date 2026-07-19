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
        assert _kpp_ice_attenuation(jnp.ones((6,)), 0) == 1.0
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


class TestFullKPPUnderIce:
    """End-to-end through kpp_vertical_mixing (not just the velocity helper) —
    a strongly-stable haline column (the Arctic winter-halocline regime): full
    ice must REDUCE K_v AND SHOAL the boundary layer vs open water."""

    @staticmethod
    def _column(n=6, nlev=20):
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.eos import wright_eos
        z = create_ocean_z_star(n_levels=nlev, H_max=400.0)
        shape = (n, nlev)
        # WEAKLY stratified surface (34.6->35.0): the KPP boundary layer is
        # V_t-CONTROLLED here (deep, unresolved-shear-limited), the regime
        # where the under-ice w_s reduction actually shoals it.  A strong
        # halocline makes the BL buoyancy-limited (~top cell) so the V_t
        # change is invisible — that column would test only the K reduction.
        S = jnp.broadcast_to(jnp.linspace(34.6, 35.0, nlev), shape).astype(jnp.float64)
        T = jnp.full(shape, 0.0, dtype=jnp.float64)      # near-freezing Arctic
        p = jnp.zeros_like(T)
        rho = wright_eos(T, S, p)
        # wind-driven shear so the bulk-Ri depth is resolved-shear anchored
        # (the zero-shear column is the degenerate Ri_b -> inf edge).
        u = jnp.broadcast_to(jnp.linspace(0.2, 0.0, nlev), shape).astype(jnp.float64)
        v = jnp.zeros(shape, dtype=jnp.float64)
        eta = jnp.zeros((n,), dtype=jnp.float64)
        J = jnp.ones((n,), dtype=jnp.float64)
        B_f = jnp.full((n,), 5.0e-7, dtype=jnp.float64)   # destabilising (brine)
        tau_x = jnp.full((n,), 0.15, dtype=jnp.float64)
        tau_y = jnp.zeros((n,), dtype=jnp.float64)
        return dict(u=u, v=v, T=T, S=S, rho=rho, eta=eta, z_coord=z,
                    J=J, B_f=B_f, tau_x=tau_x, tau_y=tau_y)

    def _run(self, cfg, ice_frac, c):
        from legoesm.ocean.physics.vertical_mixing.kpp import (
            kpp_vertical_mixing, _boundary_layer_depth,
        )
        out = kpp_vertical_mixing(
            c["u"], c["v"], c["T"], c["S"], c["rho"], c["eta"],
            c["z_coord"], c["J"], cfg, tau_x=c["tau_x"], tau_y=c["tau_y"],
            B_f=c["B_f"], apply_diffusion=False, ice_frac=ice_frac)
        u_star = jnp.sqrt(jnp.sqrt(c["tau_x"] ** 2 + c["tau_y"] ** 2) / 1025.0)
        h_bl = _boundary_layer_depth(
            c["rho"], c["T"], c["S"], c["u"], c["v"], c["z_coord"], c["J"],
            u_star, c["B_f"], cfg, ice_frac=ice_frac)
        return out.K_v, h_bl

    def test_full_ice_reduces_K_and_shoals_bl(self):
        c = self._column()
        cfg = KPPConfig(eice=3)
        K_open, h_open = self._run(cfg, jnp.zeros((6,)), c)
        K_ice, h_ice = self._run(cfg, jnp.ones((6,)), c)
        assert float(jnp.sum(K_ice)) < float(jnp.sum(K_open))   # weaker mixing
        assert float(jnp.mean(h_ice)) < float(jnp.mean(h_open)) # shallower BL

    def test_eice0_ice_present_is_noop(self):
        c = self._column()
        cfg0 = KPPConfig(eice=0)
        K_a, h_a = self._run(cfg0, None, c)
        # eice=0: even a full ice_frac must not change anything (gated off)
        K_b, h_b = self._run(cfg0, jnp.ones((6,)), c)
        np.testing.assert_array_equal(np.asarray(K_a), np.asarray(K_b))
        np.testing.assert_array_equal(np.asarray(h_a), np.asarray(h_b))


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
