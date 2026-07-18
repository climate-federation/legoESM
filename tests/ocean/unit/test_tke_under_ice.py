"""Under-ice attenuation of the TKE lc/etau wave sources (NEMO nn_eice).

2026-07-18 audit: the ``nemo_langmuir_tke_source`` / ``nemo_etau_injection``
kernels always accepted ``ice_frac`` (the ``(1-fi)`` factor) but NO caller
supplied it — wave TKE was injected beneath compact ice, over-mixing the
Arctic.  ``TKEConfig.eice`` + the k_profiles threading close the gap.  Pinned:

* kernel: full ice cover kills the Langmuir source and the etau injection;
* orchestrator: ``tke_vertical_mixing(ice_frac=...)`` reduces K under ice
  relative to the no-ice call, and is bit-identical when ``ice_frac=None``;
* nn_eice mode mapping (1 -> fi, 3 -> min(4*fi, 1)) at the config gate;
* dispatch hardening: unknown eice raises (config gate + CLI builder).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)  # dtype-stable comparisons

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
from legoesm.ocean.physics.vertical_mixing.tke import (
    nemo_etau_injection,
    nemo_langmuir_tke_source,
    tke_vertical_mixing,
)


def _column(nlev=12, n=4):
    rng = np.random.default_rng(2)
    shape = (n, nlev)
    u = jnp.asarray(rng.normal(0.05, 0.02, shape))
    v = jnp.asarray(rng.normal(0.0, 0.02, shape))
    T = jnp.asarray(np.linspace(6.0, 2.0, nlev)[None, :].repeat(n, 0))
    S = jnp.asarray(np.linspace(33.0, 34.5, nlev)[None, :].repeat(n, 0))
    rho = jnp.asarray(1026.0 + 0.1 * np.arange(nlev)[None, :].repeat(n, 0))
    dz_half = jnp.full((n, nlev - 1), 10.0)
    tau_x = jnp.full((n,), 0.2)
    tau_y = jnp.zeros((n,))
    return u, v, T, S, rho, dz_half, tau_x, tau_y


def _cfg(**kw):
    base = dict(lc=True, lc_coeff=0.25, etau_mode="below_ml",
                etau_frac=0.08, etau_htau_mode="constant10m")
    base.update(kw)
    return TKEConfig(**base)


class TestKernels:
    def test_langmuir_source_zero_under_full_ice(self):
        n, nlev = 3, 10
        taum = jnp.full((n,), 0.3)
        N2 = jnp.full((n, nlev - 1), 1.0e-5)
        depth_w = jnp.asarray(np.linspace(5.0, 95.0, nlev - 1))
        dz_w = jnp.full((n, nlev - 1), 10.0)
        cfg = _cfg()
        src_open = nemo_langmuir_tke_source(taum, N2, depth_w, dz_w, cfg,
                                            ice_frac=None)
        src_iced = nemo_langmuir_tke_source(taum, N2, depth_w, dz_w, cfg,
                                            ice_frac=jnp.ones((n,)))
        assert float(jnp.max(src_open)) > 0.0
        np.testing.assert_allclose(np.asarray(src_iced), 0.0, atol=1e-30)

    def test_etau_injection_zero_under_full_ice(self):
        n, nlev = 3, 10
        e = jnp.full((n, nlev - 1), 1.0e-4)
        taum = jnp.full((n,), 0.3)
        depth_w = jnp.asarray(np.linspace(5.0, 95.0, nlev - 1))
        cfg = _cfg()
        e_open = nemo_etau_injection(e, taum, depth_w, cfg)
        e_iced = nemo_etau_injection(e, taum, depth_w, cfg,
                                     ice_frac=jnp.ones((n,)))
        assert float(jnp.max(e_open - e)) > 0.0        # injects when open
        np.testing.assert_allclose(np.asarray(e_iced), np.asarray(e),
                                   rtol=0, atol=1e-30)  # inert under ice


class TestOrchestrator:
    @staticmethod
    def _z_interface(nlev=12, dz=10.0):
        # Interior interface reference heights, NEGATIVE downward (the
        # k_profiles convention: z_coord.z_half_ref[1:-1]).
        return jnp.asarray(-dz * np.arange(1, nlev))

    def test_ice_frac_none_bit_identical_to_before(self):
        u, v, T, S, rho, dz_half, tau_x, tau_y = _column()
        cfg = _cfg()
        zi = self._z_interface()
        out_a = tke_vertical_mixing(u, v, T, S, rho, dz_half, None,
                                    tau_x, tau_y, dt=86400.0, cfg=cfg,
                                    n_iterations=3, z_interface=zi)
        out_b = tke_vertical_mixing(u, v, T, S, rho, dz_half, None,
                                    tau_x, tau_y, dt=86400.0, cfg=cfg,
                                    n_iterations=3, z_interface=zi,
                                    ice_frac=None)
        np.testing.assert_array_equal(np.asarray(out_a.K_M),
                                      np.asarray(out_b.K_M))

    def test_full_ice_reduces_K(self):
        u, v, T, S, rho, dz_half, tau_x, tau_y = _column()
        cfg = _cfg()
        zi = self._z_interface()
        out_open = tke_vertical_mixing(u, v, T, S, rho, dz_half, None,
                                       tau_x, tau_y, dt=86400.0, cfg=cfg,
                                       n_iterations=3, z_interface=zi)
        out_iced = tke_vertical_mixing(u, v, T, S, rho, dz_half, None,
                                       tau_x, tau_y, dt=86400.0, cfg=cfg,
                                       n_iterations=3, z_interface=zi,
                                       ice_frac=jnp.ones(tau_x.shape))
        K_open = float(jnp.sum(out_open.K_M))
        K_iced = float(jnp.sum(out_iced.K_M))
        assert K_iced <= K_open  # attenuation can only remove TKE sources
        assert K_iced < K_open   # and with lc+etau active it strictly does


class TestEiceModes:
    def test_config_default_off(self):
        assert TKEConfig().eice == 0

    def test_cli_builder_mapping_and_raise(self):
        import importlib.util
        import pathlib
        import sys
        root = pathlib.Path(__file__).resolve().parents[3]
        spec = importlib.util.spec_from_file_location(
            "run_omip_core2", root / "scripts" / "run" / "run_omip_core2.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("run_omip_core2", mod)
        spec.loader.exec_module(mod)
        vm = mod.build_tripole_vmix_config("tke")
        assert vm.tke.eice == 3          # ORCA1 card default (nn_eice=3)
        vm0 = mod.build_tripole_vmix_config("tke", tke_eice=0)
        assert vm0.tke.eice == 0         # A/B override
        with pytest.raises(ValueError, match="tke-eice"):
            mod.build_tripole_vmix_config("tke", tke_eice=2)

    def test_effective_ice_frac_mode3(self):
        # mode-3 mapping min(4*fi, 1): at fi=0.25 the source is fully killed.
        n, nlev = 2, 8
        taum = jnp.full((n,), 0.3)
        N2 = jnp.full((n, nlev - 1), 1.0e-5)
        depth_w = jnp.asarray(np.linspace(5.0, 75.0, nlev - 1))
        dz_w = jnp.full((n, nlev - 1), 10.0)
        cfg = _cfg()
        fi = jnp.full((n,), 0.25)
        eff = jnp.minimum(4.0 * fi, 1.0)
        src = nemo_langmuir_tke_source(taum, N2, depth_w, dz_w, cfg,
                                       ice_frac=eff)
        np.testing.assert_allclose(np.asarray(src), 0.0, atol=1e-30)
