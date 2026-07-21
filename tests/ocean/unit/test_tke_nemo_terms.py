"""Direct unit tests for the NEMO zdftke surface terms (Langmuir ln_lc +
sub-ML TKE penetration nn_etau) and the NEMO nn_evdm=1 momentum EVD path.

References: NEMO 5.0.1 src/OCE/ZDF/zdftke.F90 (lines 305-370 Langmuir,
492-496 etau, 265 surface TKE) and zdfevd.F90 (nn_evdm); Axell (2002, JGR).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
from legoesm.ocean.physics.vertical_mixing.tke import (
    _NEMO_TKE_EBB,
    _NEMO_TKE_EMIN0,
    _NEMO_TKE_LC_CSD,
    nemo_etau_injection,
    nemo_langmuir_tke_source,
    tke_vertical_mixing,
)

jax.config.update("jax_enable_x64", True)

_RHO0 = 1026.0


# ---------------------------------------------------------------------------
# Langmuir source (Axell 2002 / NEMO ln_lc)
# ---------------------------------------------------------------------------


def _col(nlev=6):
    """Tiny single-column geometry: interfaces at 10,20,...,50 m."""
    depth_w = jnp.asarray([10.0, 20.0, 30.0, 40.0, 50.0])
    dz_w = jnp.full((5,), 10.0)
    return depth_w, dz_w


class TestLangmuirSource:
    def test_zero_wind_zero_source(self):
        depth_w, dz_w = _col()
        N2 = jnp.full((1, 5), 1e-5)
        src = nemo_langmuir_tke_source(jnp.zeros((1,)), N2, depth_w, dz_w,
                                       TKEConfig(lc=True))
        assert np.allclose(np.asarray(src), 0.0)

    def test_positive_above_hlc_zero_below(self):
        """Strong stratification confines the LC depth; the source is >0
        above h_lc (sin>0) and exactly 0 at/below it."""
        depth_w, dz_w = _col()
        cfg = TKEConfig(lc=True)
        taum = jnp.asarray([0.1])                     # |τ| = 0.1 N/m²
        # PE(k) = Σ N²·z·dz; choose N² so PE crosses ½W_lc² between k=1 and 2.
        half_wlc2 = float(_NEMO_TKE_LC_CSD * 0.1)
        # PE(k) with N²=const and dz=10: N²·10·Σz = N²·10·(10+20+...+z_k).
        # Pick N² so PE stays BELOW ½W_lc² through k=1 (Σz=30 ⇒ PE=300·N²)
        # and crosses at k=2 (Σz=60 ⇒ PE=600·N²): h_lc = 30 m.
        N2c = half_wlc2 / (10.0 * (10.0 + 20.0 + 30.0))
        N2 = jnp.full((1, 5), N2c * 1.001)
        src = np.asarray(nemo_langmuir_tke_source(taum, N2, depth_w, dz_w, cfg))[0]
        # h_lc = 30 m (first interface where cumulative PE exceeds ½W_lc²)
        assert src[0] > 0.0 and src[1] > 0.0          # 10, 20 m < h_lc
        assert np.allclose(src[2:], 0.0)              # 30 m+ = at/below h_lc
        # exact value at 10 m: us³·(rn_lc·sin(π·10/30))³/30
        us3 = (2.0 * half_wlc2) ** 1.5
        want = us3 * (cfg.lc_coeff * np.sin(np.pi * 10.0 / 30.0)) ** 3 / 30.0
        np.testing.assert_allclose(src[0], want, rtol=1e-12)

    def test_unstratified_column_hlc_is_bottom(self):
        """N²=0 everywhere ⇒ PE never exceeds ½W_lc² ⇒ h_lc = deepest
        interface (NEMO initialises imlc to the bottom)."""
        depth_w, dz_w = _col()
        src = np.asarray(nemo_langmuir_tke_source(
            jnp.asarray([0.1]), jnp.zeros((1, 5)), depth_w, dz_w,
            TKEConfig(lc=True)))[0]
        # h_lc = 50 m: source positive at all interfaces above 50 m
        assert (src[:4] > 0.0).all()
        assert src[4] == 0.0                          # z == h_lc excluded

    def test_source_is_nonnegative_and_finite(self):
        depth_w, dz_w = _col()
        rng = np.random.default_rng(0)
        N2 = jnp.asarray(rng.standard_normal((4, 5)) * 1e-5)  # signed N²
        taum = jnp.asarray(rng.random(4) * 0.3)
        src = np.asarray(nemo_langmuir_tke_source(
            taum, N2, depth_w, dz_w, TKEConfig(lc=True)))
        assert np.isfinite(src).all() and (src >= 0.0).all()

    def test_grad_finite_at_zero_wind(self):
        """AD-safety: d(source)/d(taum) finite at τ=0 (the x^{3/2} corner)."""
        depth_w, dz_w = _col()
        N2 = jnp.full((5,), 1e-5)

        def total(t):
            return jnp.sum(nemo_langmuir_tke_source(
                t, N2, depth_w, dz_w, TKEConfig(lc=True)))

        g = jax.grad(total)(jnp.asarray(0.0))
        assert np.isfinite(float(g))


# ---------------------------------------------------------------------------
# etau injection (NEMO nn_etau=1)
# ---------------------------------------------------------------------------


class TestEtauInjection:
    def test_none_mode_identity(self):
        depth_w, _ = _col()
        e = jnp.ones((2, 5)) * 1e-3
        out = nemo_etau_injection(e, jnp.asarray([0.1, 0.2]), depth_w,
                                  TKEConfig(etau_mode="none"), rho_0=_RHO0)
        assert out is e

    def test_constant10m_exact_injection(self):
        depth_w, _ = _col()
        cfg = TKEConfig(etau_mode="below_ml")   # htau default constant10m
        e = jnp.full((1, 5), 1e-3)
        taum = jnp.asarray([0.1])
        out = np.asarray(nemo_etau_injection(e, taum, depth_w, cfg,
                                             rho_0=_RHO0))[0]
        e_sfc = max(_NEMO_TKE_EMIN0, _NEMO_TKE_EBB / _RHO0 * 0.1)
        want = 1e-3 + cfg.etau_frac * e_sfc * np.exp(-np.asarray(depth_w) / 10.0)
        np.testing.assert_allclose(out, want, rtol=1e-12)

    def test_zero_wind_still_injects_emin0(self):
        """NEMO's surface TKE has the rn_emin0 floor, so etau injects even
        unforced — faithful behaviour, pinned here."""
        depth_w, _ = _col()
        cfg = TKEConfig(etau_mode="below_ml")
        e = jnp.zeros((1, 5))
        out = np.asarray(nemo_etau_injection(e, jnp.zeros((1,)), depth_w, cfg,
                                             rho_0=_RHO0))[0]
        want = cfg.etau_frac * _NEMO_TKE_EMIN0 * np.exp(-np.asarray(depth_w) / 10.0)
        np.testing.assert_allclose(out, want, rtol=1e-12)

    def test_latitude_htau_formula(self):
        """h_tau = max(0.5, min(30, 45·|sin φ|)): 0.5 m floor at the equator,
        30 m cap poleward of ~41.8°."""
        depth_w, _ = _col()
        cfg = TKEConfig(etau_mode="below_ml", etau_htau_mode="latitude")
        e = jnp.zeros((2, 5))
        lat = jnp.asarray([0.0, 60.0])
        out = np.asarray(nemo_etau_injection(
            e, jnp.zeros((2,)), depth_w, cfg, rho_0=_RHO0, lat_deg=lat))
        base = cfg.etau_frac * _NEMO_TKE_EMIN0
        np.testing.assert_allclose(out[0], base * np.exp(-np.asarray(depth_w) / 0.5),
                                   rtol=1e-12)
        np.testing.assert_allclose(out[1], base * np.exp(-np.asarray(depth_w) / 30.0),
                                   rtol=1e-12)

    def test_latitude_mode_without_lat_raises(self):
        depth_w, _ = _col()
        cfg = TKEConfig(etau_mode="below_ml", etau_htau_mode="latitude")
        with pytest.raises(ValueError, match="lat_deg"):
            nemo_etau_injection(jnp.zeros((1, 5)), jnp.zeros((1,)), depth_w,
                                cfg, rho_0=_RHO0)

    def test_unknown_modes_raise(self):
        depth_w, _ = _col()
        with pytest.raises(ValueError, match="etau_mode"):
            nemo_etau_injection(jnp.zeros((1, 5)), jnp.zeros((1,)), depth_w,
                                TKEConfig(etau_mode="bogus"), rho_0=_RHO0)
        with pytest.raises(ValueError, match="etau_htau_mode"):
            nemo_etau_injection(
                jnp.zeros((1, 5)), jnp.zeros((1,)), depth_w,
                TKEConfig(etau_mode="below_ml", etau_htau_mode="bogus"),
                rho_0=_RHO0)


# ---------------------------------------------------------------------------
# Orchestrator wiring
# ---------------------------------------------------------------------------


def _orchestrator_inputs(nlev=8):
    shape = (3, 4, nlev)
    rng = np.random.default_rng(1)
    T = jnp.asarray(20.0 - 2.0 * np.arange(nlev))[None, None, :] * jnp.ones(shape)
    S = jnp.full(shape, 35.0)
    rho = jnp.asarray(1026.0 + 0.2 * np.arange(nlev))[None, None, :] * jnp.ones(shape)
    u = jnp.asarray(rng.standard_normal(shape) * 0.05)
    v = jnp.asarray(rng.standard_normal(shape) * 0.05)
    dz_half = jnp.full(shape[:-1] + (nlev - 1,), 25.0)
    z_interface = -25.0 * jnp.arange(1, nlev)   # negative heights
    tau_x = jnp.full(shape[:-1], 0.1)
    tau_y = jnp.zeros(shape[:-1])
    return u, v, T, S, rho, dz_half, z_interface, tau_x, tau_y


class TestOrchestratorWiring:
    def test_lc_and_etau_change_output(self):
        u, v, T, S, rho, dz_half, z_int, tx, ty = _orchestrator_inputs()
        base = tke_vertical_mixing(u, v, T, S, rho, dz_half, None, tx, ty,
                                   dt=3600.0, cfg=TKEConfig(), rho_0=_RHO0,
                                   n_iterations=3, z_interface=z_int)
        nemo = tke_vertical_mixing(
            u, v, T, S, rho, dz_half, None, tx, ty, dt=3600.0,
            cfg=TKEConfig(lc=True, etau_mode="below_ml"), rho_0=_RHO0,
            n_iterations=3, z_interface=z_int)
        assert not np.allclose(np.asarray(base.tke_new), np.asarray(nemo.tke_new))
        # both sources ADD energy ⇒ TKE must not decrease anywhere
        assert (np.asarray(nemo.tke_new) >= np.asarray(base.tke_new) - 1e-15).all()

    def test_terms_need_z_interface(self):
        u, v, T, S, rho, dz_half, _, tx, ty = _orchestrator_inputs()
        with pytest.raises(ValueError, match="z_interface"):
            tke_vertical_mixing(u, v, T, S, rho, dz_half, None, tx, ty,
                                dt=3600.0, cfg=TKEConfig(lc=True),
                                rho_0=_RHO0)

    def test_post_mixing_guard_rejects_etau_but_allows_lc(self):
        """lc is now APPLIED on the post-mixing path (the source is computed in
        tke_set_diffusivities and added pre-solve in tke_integrate_post_mixing)
        so the guard no longer rejects it; etau remains blocked (still a
        silent no-op there)."""
        from legoesm.ocean.physics.vertical_mixing.tke import (
            _validate_post_mixing_cfg,
        )
        base = dict(buoyancy_timing="post_mixing_veros", prognostic=True,
                    veros_dz_slots=True, n2_mode="adiabatic",
                    positivity="veros_surface_correction")
        _validate_post_mixing_cfg(TKEConfig(lc=True, **base))  # no raise
        with pytest.raises(ValueError, match="etau"):
            _validate_post_mixing_cfg(TKEConfig(etau_mode="below_ml", **base))

    def test_post_mixing_langmuir_source_is_applied(self):
        """Non-vacuity: under wind, lc=True raises the post-mixing TKE vs
        lc=False (the source ADDS energy), exercising the ctx.langmuir_source
        channel end-to-end."""
        import jax.numpy as jnp

        from legoesm.ocean.fidelity.nemo_recipe import _nemo_tke_config
        from legoesm.ocean.physics.vertical_mixing.tke import (
            tke_integrate_post_mixing,
            tke_set_diffusivities,
        )
        from legoesm.ocean.vertical import create_ocean_z_star

        # 10 m interfaces (GYRE-like near-surface spacing) + weak upper
        # stratification so h_lc sits BELOW several interfaces (the source is
        # correctly zero when the first interface's cumulative PE already
        # exceeds W_lc^2 — a coarse column hides the term).
        nlev = 6
        z = create_ocean_z_star(n_levels=nlev, H_max=60.0)
        shape = (2, 2, nlev)
        T = jnp.asarray(20.0 - 0.001 * np.arange(nlev))[None, None, :] * jnp.ones(shape)
        S = jnp.full(shape, 35.0)
        rho = 1026.0 * (1.0 - 2e-4 * (T - 10.0))
        dz_half = jnp.broadcast_to(z.dz_half_ref, shape[:-1] + (nlev - 1,))
        tke_old = jnp.full(shape[:-1] + (nlev - 1,), 1.0e-4)
        tau = jnp.full(shape[:-1], 0.1)
        tau0 = jnp.zeros(shape[:-1])
        J = jnp.ones(shape[:-1])
        p = jnp.broadcast_to(1026.0 * 9.81 * jnp.abs(z.z_full_ref), shape)

        def eos_fn(T_, S_, p_):
            return 1026.0 * (1.0 - 2e-4 * (T_ - 10.0))

        outs = {}
        for lc in (False, True):
            cfg = _nemo_tke_config()._replace(lc=lc)
            _, _, ctx = tke_set_diffusivities(
                T * 0, T * 0, T, S, rho, dz_half, tke_old, tau, tau0,
                cfg, 1026.0, 9.81, p_cell=p, dz_ref=z.dz_ref, jacobian=J,
                eos_fn=eos_fn, z_interface=z.z_half_ref[1:-1],
                dz_surface=0.5 * z.dz_half_ref[0] * J)
            assert (ctx.langmuir_source is not None) == lc
            n2 = jnp.full(shape[:-1] + (nlev - 1,), 1e-6)
            outs[lc] = tke_integrate_post_mixing(
                ctx, n2, jnp.zeros_like(n2), jnp.zeros(shape[:-1]),
                dt=3600.0, cfg=cfg)
        assert float(jnp.max(outs[True] - outs[False])) > 0.0
        assert (np.asarray(outs[True]) >= np.asarray(outs[False]) - 1e-15).all()


# ---------------------------------------------------------------------------
# nn_evdm=1: momentum EVD through the k_profiles implicit path
# ---------------------------------------------------------------------------


def _kprof_setup(unstable: bool):
    from legoesm.ocean.vertical import create_ocean_z_star
    nlev = 6
    z = create_ocean_z_star(n_levels=nlev, H_max=600.0)
    shape = (2, 2, nlev)
    # unstable: T INCREASES downward (denser water above lighter) everywhere.
    prof = (10.0 + 2.0 * np.arange(nlev)) if unstable else (20.0 - 2.0 * np.arange(nlev))
    T = jnp.asarray(prof)[None, None, :] * jnp.ones(shape)
    f = lambda a: SimpleNamespace(data=a)
    state = SimpleNamespace(
        T=f(T), S=f(jnp.full(shape, 35.0)),
        u=f(jnp.zeros(shape)), v=f(jnp.zeros(shape)),
        eta=f(jnp.zeros(shape[:-1])), H_bathy=f(jnp.full(shape[:-1], 600.0)),
    )
    return state, z


def _physics_cfg(vmix_scheme: str, nu_conv: float):
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.convection.config import (
        EnhancedDiffusionConfig, OceanConvectionConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.config import (
        VerticalMixingConfig,
    )
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme=vmix_scheme),
        convection=OceanConvectionConfig(
            scheme="enhanced_diffusion",
            enhanced_diffusion=EnhancedDiffusionConfig(
                K_conv=100.0, nu_conv=nu_conv),
        ),
    )


class TestMomentumEVD:
    def test_nu_conv_boosts_A_v_under_constant_vmix(self):
        from legoesm.ocean.physics.vertical_mixing.k_profiles import (
            compute_vertical_K_profiles,
        )
        state, z = _kprof_setup(unstable=True)
        _, A_off = compute_vertical_K_profiles(
            state, z, None, _physics_cfg("constant", 0.0))
        _, A_on = compute_vertical_K_profiles(
            state, z, None, _physics_cfg("constant", 100.0))
        # Unstable column ⇒ the momentum EVD must raise A_v substantially.
        assert float(jnp.max(A_on - A_off)) > 50.0

    def test_stable_column_no_boost(self):
        from legoesm.ocean.physics.vertical_mixing.k_profiles import (
            compute_vertical_K_profiles,
        )
        state, z = _kprof_setup(unstable=False)
        _, A_off = compute_vertical_K_profiles(
            state, z, None, _physics_cfg("constant", 0.0))
        _, A_on = compute_vertical_K_profiles(
            state, z, None, _physics_cfg("constant", 100.0))
        #

        # smooth_transition sigmoid leaves a tiny tail; must stay ≪ K_conv.
        assert float(jnp.max(jnp.abs(A_on - A_off))) < 1.0

    def test_kpp_plus_nu_conv_still_raises(self):
        from legoesm.ocean.physics.vertical_mixing.k_profiles import (
            compute_vertical_K_profiles,
        )
        state, z = _kprof_setup(unstable=True)
        with pytest.raises(ValueError, match="nu_conv"):
            compute_vertical_K_profiles(
                state, z, None, _physics_cfg("kpp", 100.0))


# ---------------------------------------------------------------------------
# DINO faithful wiring (nn_evdm=1 + lc/etau in the DINO TKE recipe)
# ---------------------------------------------------------------------------


class TestDinoFaithfulWiring:
    def test_dino_tke_recipe_enables_nemo_terms(self):
        import dataclasses
        from legoesm.ocean.experiments import dino as dmod
        cfg = dataclasses.replace(dmod.DINOConfig(), vmix_scheme="tke")
        vm = dmod._dino_vertical_mixing_config(cfg)
        assert vm.tke.lc is True
        assert vm.tke.etau_mode == "below_ml"
        assert vm.tke.etau_htau_mode == "constant10m"   # DINO nn_htau default

    def test_dino_evd_momentum_gating(self):
        import dataclasses
        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid, dino_lat_lon_model_config,
        )
        base = DINOConfig()
        assert base.evd_on_momentum is True             # NEMO nn_evdm=1
        g = dino_lat_lon_grid(base, n_lon=50)
        # kpp (default): nu_conv must be 0 (guard pairs EVD-momentum w/ TKE)
        _, pc = dino_lat_lon_model_config(g, base, physics=True)
        assert pc.convection.enhanced_diffusion.nu_conv == 0.0
        # tke: nu_conv = K_conv (rn_evd, momentum too)
        cfg_tke = dataclasses.replace(base, vmix_scheme="tke")
        _, pc2 = dino_lat_lon_model_config(g, cfg_tke, physics=True)
        assert pc2.convection.enhanced_diffusion.nu_conv == pytest.approx(100.0)
        # opt-out
        cfg_off = dataclasses.replace(base, vmix_scheme="tke",
                                      evd_on_momentum=False)
        _, pc3 = dino_lat_lon_model_config(g, cfg_off, physics=True)
        assert pc3.convection.enhanced_diffusion.nu_conv == 0.0


# ---------------------------------------------------------------------------
# NEMO nn_bc_surf=1 Dirichlet surface TKE (TKEConfig.surface_bc)
# ---------------------------------------------------------------------------


class TestNemoDirichletSurfaceBC:
    def test_dirichlet_holds_en1_exactly(self):
        """surface_bc='nemo_dirichlet' HOLDS the top interface at
        en(1)=max(rn_emin0, rn_ebb*|tau|/rho0) — the identity row makes it
        exact, and it is ~60x the default Veros flux-BC response."""
        u, v, T, S, rho, dz_half, z_int, tx, ty = _orchestrator_inputs()
        taum = float(np.hypot(np.asarray(tx)[0, 0], np.asarray(ty)[0, 0]))
        e_sfc = max(_NEMO_TKE_EMIN0, _NEMO_TKE_EBB / _RHO0 * taum)

        nemo = tke_vertical_mixing(
            u, v, T, S, rho, dz_half, None, tx, ty, dt=3600.0,
            cfg=TKEConfig(surface_bc="nemo_dirichlet"), rho_0=_RHO0,
            n_iterations=3, z_interface=z_int)
        np.testing.assert_allclose(
            np.asarray(nemo.tke_new)[..., 0], e_sfc, rtol=0, atol=1e-12)

        base = tke_vertical_mixing(
            u, v, T, S, rho, dz_half, None, tx, ty, dt=3600.0,
            cfg=TKEConfig(), rho_0=_RHO0, n_iterations=3, z_interface=z_int)
        # the Dirichlet surface value dominates the flux-BC response
        assert float(np.max(np.asarray(base.tke_new)[..., 0])) < 0.5 * e_sfc

    def test_dirichlet_zero_wind_floor(self):
        """Windless: en(1) sits at the rn_emin0 floor (not the flux-BC zero)."""
        u, v, T, S, rho, dz_half, z_int, _, _ = _orchestrator_inputs()
        zeros = jnp.zeros(u.shape[:-1])
        out = tke_vertical_mixing(
            u, v, T, S, rho, dz_half, None, zeros, zeros, dt=3600.0,
            cfg=TKEConfig(surface_bc="nemo_dirichlet"), rho_0=_RHO0,
            n_iterations=1, z_interface=z_int)
        np.testing.assert_allclose(
            np.asarray(out.tke_new)[..., 0], _NEMO_TKE_EMIN0, rtol=0, atol=1e-12)

    def test_surface_bc_typo_raises(self):
        u, v, T, S, rho, dz_half, z_int, tx, ty = _orchestrator_inputs()
        with pytest.raises(ValueError, match="surface_bc"):
            tke_vertical_mixing(
                u, v, T, S, rho, dz_half, None, tx, ty, dt=3600.0,
                cfg=TKEConfig(surface_bc="bogus"), rho_0=_RHO0,
                n_iterations=1, z_interface=z_int)
