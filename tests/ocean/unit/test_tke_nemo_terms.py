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
    _solve_tke_backward_euler,
    _surface_tke_dirichlet,
    TKEEntryN2Bundle,
    nemo_etau_injection,
    nemo_langmuir_tke_source,
    nemo_literal_langmuir_tke_update,
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
    def test_explicit_vectorized_selector_is_exact_legacy_default(self):
        """The selector must not perturb cards that retain the old path."""
        depth_w, dz_w = _col()
        taum = jnp.asarray([0.07, 0.13])
        N2 = jnp.asarray([[1.0e-5, -2.0e-6, 3.0e-5, 0.0, 1.0e-6],
                          [0.0, 2.0e-5, 0.0, 3.0e-6, -1.0e-6]])
        default = nemo_langmuir_tke_source(
            taum, N2, depth_w, dz_w, TKEConfig(lc=True))
        explicit = nemo_langmuir_tke_source(
            taum, N2, depth_w, dz_w,
            TKEConfig(lc=True, tke_langmuir_evaluation="vectorized"))
        np.testing.assert_array_equal(default, explicit)

    def test_literal_strict_crossing_and_source_order_hand_case(self):
        """A PE tie does not cross; the next level sets h_lc (NEMO ``>``)."""
        depth_w, dz_w = _col()
        # half_wlc2=1 exactly. The first PE contribution is exactly one, so
        # a >= transcription would choose h=10 and return zero everywhere;
        # NEMO's strict > chooses h=20 after the second contribution.
        taum = jnp.asarray([1.0 / float(_NEMO_TKE_LC_CSD)])
        N2 = jnp.asarray([[0.01, 0.001, 0.0, 0.0, 0.0]])
        cfg = TKEConfig(lc=True, tke_langmuir_evaluation="nemo_literal")
        src = np.asarray(nemo_langmuir_tke_source(
            taum, N2, depth_w, dz_w, cfg,
            bottom_level=jnp.asarray([4], dtype=jnp.int32),
            w_active=jnp.ones_like(N2, dtype=bool)))[0]
        us = np.sqrt(2.0)
        want0 = ((us * us * us)
                 * (cfg.lc_coeff * np.sin(np.pi * 10.0 / 20.0)) ** 3
                 / 20.0)
        np.testing.assert_allclose(src[0], want0, rtol=2e-15, atol=0.0)
        np.testing.assert_array_equal(src[1:], 0.0)

    def test_literal_unequal_depth_no_crossing_is_per_column(self):
        """No-crossing fallback is each column's mbkt+1, never global."""
        depth_w, dz_w = _col()
        N2 = jnp.zeros((2, 5))
        taum = jnp.asarray([0.1, 0.1])
        bottom = jnp.asarray([2, 4], dtype=jnp.int32)
        wet = jnp.asarray([[True, True, False, False, False],
                           [True, True, True, True, False]])
        cfg = TKEConfig(lc=True, tke_langmuir_evaluation="nemo_literal")
        actual = np.asarray(nemo_langmuir_tke_source(
            taum, N2, depth_w, dz_w, cfg,
            bottom_level=bottom, w_active=wet))

        half = float(_NEMO_TKE_LC_CSD) * 0.1
        us = np.sqrt(2.0 * half)
        expected = np.zeros((2, 5))
        for column, h_lc in enumerate((30.0, 50.0)):
            for k, depth in enumerate(np.asarray(depth_w)):
                if wet[column, k] and depth < h_lc:
                    zwlc = cfg.lc_coeff * np.sin(np.pi * depth / h_lc)
                    expected[column, k] = us * us * us * zwlc * zwlc * zwlc / h_lc
        np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0.0)

        # Red control: the historical global-deepest fallback gives the
        # shallow column h=50 and must demonstrably disagree.
        historical = np.asarray(nemo_langmuir_tke_source(
            taum, N2, depth_w, dz_w,
            TKEConfig(lc=True, tke_langmuir_evaluation="vectorized")))
        assert not np.array_equal(historical[0], expected[0])

    def test_literal_eager_jit_and_grad(self):
        depth_w, dz_w = _col()
        N2 = jnp.zeros((2, 5))
        bottom = jnp.asarray([2, 4], dtype=jnp.int32)
        wet = jnp.asarray([[True, True, False, False, False],
                           [True, True, True, True, False]])
        cfg = TKEConfig(lc=True, tke_langmuir_evaluation="nemo_literal")

        def total(taum):
            return jnp.sum(nemo_langmuir_tke_source(
                taum, N2, depth_w, dz_w, cfg,
                bottom_level=bottom, w_active=wet))

        taum = jnp.asarray([0.1, 0.2])
        eager = jax.value_and_grad(total)(taum)
        compiled = jax.jit(jax.value_and_grad(total))(taum)
        np.testing.assert_allclose(compiled[0], eager[0], rtol=2e-15, atol=0.0)
        np.testing.assert_allclose(compiled[1], eager[1], rtol=2e-15, atol=0.0)
        assert np.isfinite(np.asarray(compiled[1])).all()

    def test_literal_line463_update_order_is_red_against_rate_first(self):
        depth_w, dz_w = _col()
        N2 = jnp.zeros((2, 5))
        taum = jnp.asarray([0.1, 0.2])
        bottom = jnp.asarray([2, 4], dtype=jnp.int32)
        wet = jnp.asarray([[True, True, False, False, False],
                           [True, True, True, True, False]])
        e_old = jnp.asarray([[1e-4, 2e-4, 3e-4, 4e-4, 5e-4],
                             [5e-4, 4e-4, 3e-4, 2e-4, 1e-4]])
        cfg = TKEConfig(lc=True, tke_langmuir_evaluation="nemo_literal")
        actual = nemo_literal_langmuir_tke_update(
            e_old, 2700.0, taum, N2, depth_w, dz_w, cfg,
            bottom_level=bottom, w_active=wet)

        source = nemo_langmuir_tke_source(
            taum, N2, depth_w, dz_w, cfg,
            bottom_level=bottom, w_active=wet)
        rate_first = e_old + 2700.0 * source
        # This exact element is the red control for moving /h before dt.
        assert np.asarray(actual)[0, 0] != np.asarray(rate_first)[0, 0]

        half = float(_NEMO_TKE_LC_CSD) * 0.1
        zus = np.sqrt(2.0 * half)
        zwlc = cfg.lc_coeff * np.sin(np.pi * 10.0 / 30.0)
        want = (float(e_old[0, 0])
                + ((2700.0 * zus * zus * zus)
                   * (zwlc * zwlc * zwlc)) / 30.0)
        np.testing.assert_array_equal(np.asarray(actual)[0, 0], want)

    def test_literal_requires_column_bottom_and_wmask(self):
        depth_w, dz_w = _col()
        cfg = TKEConfig(lc=True, tke_langmuir_evaluation="nemo_literal")
        with pytest.raises(ValueError, match="bottom_level and w_active"):
            nemo_langmuir_tke_source(
                jnp.asarray([0.1]), jnp.zeros((1, 5)), depth_w, dz_w, cfg)

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
    def test_literal_langmuir_rejects_mixed_generic_source(self):
        u, v, T, S, rho, dz_half, z_int, tx, ty = _orchestrator_inputs(4)
        shape_w = dz_half.shape
        shape_t = T.shape
        cfg = TKEConfig(
            lc=True, tke_langmuir_evaluation="nemo_literal",
            tke_n2_evaluation_stage="step_entry", bottom_tke_bc=True)
        bundle = TKEEntryN2Bundle(
            rn2=jnp.zeros(shape_w), rn2b=jnp.zeros(shape_w),
            gdepw_Kmm=jnp.broadcast_to(-z_int, shape_w),
            e3w_Kmm=dz_half, e3t_Kmm=jnp.ones(shape_t))
        with pytest.raises(ValueError, match="cannot merge"):
            tke_vertical_mixing(
                u, v, T, S, rho, dz_half, None, tx, ty,
                dt=3600.0, cfg=cfg, rho_0=_RHO0, n_iterations=1,
                z_interface=z_int, external_source=jnp.ones(shape_w),
                bottom_dirichlet=jnp.zeros(shape_t[:-1]),
                bottom_level=jnp.full(shape_t[:-1], 2, dtype=jnp.int32),
                w_active=jnp.ones(shape_w, dtype=bool),
                precomputed_n2_bundle=bundle)

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
        assert vm.tke.etau_htau_mode == "latitude"   # nn_htau=1 (namelist_ref default, c1a661da2)

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


# ---------------------------------------------------------------------------
# T3 (Phase-2 #1317): surface BC PLACEMENT — nemo_z0 vs interior_pinned.
#
# NEMO holds en(1) at the z=0 w-point and SOLVES the tridiagonal from jk=2
# (zdftke.F90:264,403-410). ``tke_surface_bc_level="interior_pinned"`` (the
# prior legoESM default) instead pins the CARRIED interior interface 0
# itself — one w-level too deep. This is the #1 ranked suspect from the
# Phase-1 audit (probe: en(20m) 8.6x, avt(30m) ~50x NEMO with placement held
# as the ONLY variable). ``tke_surface_bc_level="nemo_z0"`` fixes it by
# prepending a virtual z=0 Dirichlet row so interior interface 0 becomes a
# genuinely SOLVED row, matching NEMO's jk=2 exactly.
# ---------------------------------------------------------------------------


def _nemo_z0_hand_solve(e_old, K_M, N2, l_eps, dz, e_sfc, dt, c_eps, alpha_tke,
                        avm1=None, shear_sq=None, dz_face_sfc=None):
    """Independent NumPy Thomas solve of NEMO's (N+1)-row surface-Dirichlet
    tridiagonal (zdftke.F90:264-269,403-410), uniform dz, backward_euler
    dissipation, buoyancy sink split sign-aware (N2>=0 here so the whole
    sink is the implicit stable branch) — used to CROSS-CHECK
    ``tke_surface_bc_level='nemo_z0'`` independently of the JAX solver under
    test (same formulas, re-derived from the NEMO lines, not copy-pasted).

    ``e_old``/``K_M``/``N2``/``l_eps`` are length-N (interior interfaces,
    legoESM's carried array); the hand-solve prepends ONE virtual surface
    row internally and returns only the N interior rows.

    ``avm1`` (T3-exact): NEMO's TRUE surface-w-level viscosity avm(jk=1).
    None (default) falls back to ``K_M[0]`` (the documented approximation
    avm(1)~=avm(2), matching ``K_M_surface=None`` in the function under
    test).

    ``shear_sq`` : length-N or None (default 0, the historical omission —
    harmless when the test column's shear-production is negligible vs the
    dissipation/buoyancy terms, but must be threaded for a genuinely
    discriminating exact-avm comparison). ``P_s = K_M*shear_sq`` added to
    the RHS, matching the function-under-test's ``rhs = e_old + dt*P_s``.

    ``dz_face_sfc`` (#1690): distance between the virtual z=0 W-point and
    interior W-point 0 -- NEMO's ``e3t(1)`` in the ``zzd_lw(jk=2)``
    denominator ``e3t(1)*e3w(2)``. ``None`` (default) => ``dz`` (uniform
    column, where the two coincide). Only row 1's subdiagonal uses it.
    """
    N = len(e_old)
    Np1 = N + 1
    _shear_sq = np.zeros(N) if shear_sq is None else np.asarray(shear_sq)
    # Row 0 = virtual surface (Dirichlet); rows 1..N = interior interfaces
    # (row k+1 <-> e_old[k]).
    _avm1 = K_M[0] if avm1 is None else avm1
    Kw = np.concatenate([[_avm1], K_M])          # K at each of the N+1 rows
    a = np.zeros(Np1); b = np.ones(Np1); c = np.zeros(Np1); r = np.zeros(Np1)
    r[0] = e_sfc
    for row in range(1, Np1):
        k = row - 1   # interior index
        up_avm = alpha_tke * 0.5 * (Kw[row] + (Kw[row + 1] if row < N else Kw[row]))
        lo_avm = alpha_tke * 0.5 * (Kw[row] + Kw[row - 1])
        up = dt * up_avm / (dz * dz) if row < N else 0.0
        _lo_face = dz if (row > 1 or dz_face_sfc is None) else dz_face_sfc
        lo = dt * lo_avm / (_lo_face * dz)
        diss_rate = c_eps * np.sqrt(max(e_old[k], 0.0)) / l_eps[k]
        buoy_rate = K_M[k] * max(N2[k], 0.0) / max(e_old[k], 1e-12)
        P_s = K_M[k] * _shear_sq[k]
        a[row] = -lo
        if row < N:
            c[row] = -up
        b[row] = 1.0 + lo + up + dt * (diss_rate + buoy_rate)
        r[row] = e_old[k] + dt * P_s
    for row in range(1, Np1):
        w = a[row] / b[row - 1]
        b[row] -= w * c[row - 1]
        r[row] -= w * r[row - 1]
    x = np.empty(Np1)
    x[-1] = r[-1] / b[-1]
    for row in range(Np1 - 2, -1, -1):
        x[row] = (r[row] - c[row] * x[row + 1]) / b[row]
    return x[1:]   # drop the virtual surface row


class TestNemoZ0SurfaceBCPlacement:
    """T3: ``tke_surface_bc_level='nemo_z0'`` (mandatory Phase-2 column test)."""

    def _column(self, nlev=6):
        """Uniform dz=10 m column, wind-forced, stably stratified with shear
        (a DINO-like near-surface profile)."""
        shape = (1, 1, nlev)
        T = jnp.asarray(np.linspace(18.0, 12.0, nlev))[None, None, :] * jnp.ones(shape)
        S = jnp.full(shape, 35.0)
        rho = jnp.asarray(1025.0 + 0.05 * np.arange(nlev))[None, None, :] * jnp.ones(shape)
        u = jnp.asarray(0.02 * np.arange(nlev))[None, None, :] * jnp.ones(shape)
        v = jnp.zeros(shape)
        dz_half = jnp.full((1, 1, nlev - 1), 10.0)
        z_interface = -10.0 * jnp.arange(1, nlev)
        tau_x = jnp.full((1, 1), 0.1)
        tau_y = jnp.zeros((1, 1))
        dz_ref = jnp.full((nlev,), 10.0)
        jacobian = jnp.ones((1, 1))
        dz_surface = jnp.full((1, 1), 10.0)   # z=0 to first cell centre (5 m
        # in a true midpoint grid; using the full dz here is the SAME value
        # nemo_z0's face/volume slot uses in this synthetic uniform column —
        # only the RELATIVE placement (nemo_z0 vs interior_pinned) is under
        # test, not the exact metric constant.
        return (u, v, T, S, rho, dz_half, z_interface, tau_x, tau_y,
                dz_ref, jacobian, dz_surface)

    @staticmethod
    def _floor_only_solve(*, surface_bc_level, surface_dirichlet):
        """Two uncoupled interior rows isolating the post-solve floor.

        K_M=K_H=P_s=N2=0, so the virtual surface row cannot influence either
        interior row.  Dissipation makes the raw result slightly less than
        e_old=1e-6, hence NEMO's ordinary interior ``rn_emin`` floor gives the
        hand-computed exact result [1e-6, 1e-6].  A non-NEMO surface path is
        expected to retain legoESM's historical first-interface 1e-4 floor.
        """
        shape = (1, 1, 2)
        cfg = TKEConfig(tke_background=1.0e-6, tke_surface_min=1.0e-4)
        zeros = jnp.zeros(shape)
        return np.asarray(_solve_tke_backward_euler(
            e_old=jnp.full(shape, 1.0e-6),
            K_M_old=zeros, K_H_old=zeros, P_s=zeros, N2=zeros,
            l_eps=jnp.ones(shape), dz_half=jnp.ones(shape),
            surface_flux=jnp.zeros((1, 1)), dt=1.0, cfg=cfg,
            dz_surface=jnp.ones((1, 1)),
            dz_face_surface=(jnp.ones((1, 1))
                             if surface_bc_level == "nemo_z0" else None),
            surface_dirichlet=(None if surface_dirichlet is None else
                               jnp.full((1, 1), surface_dirichlet)),
            surface_bc_level=surface_bc_level,
            K_M_surface=(jnp.zeros((1, 1))
                         if surface_bc_level == "nemo_z0" else None),
        ))[0, 0]

    def test_nemo_z0_first_interior_uses_rn_emin_not_rn_emin0(self):
        """Regression: rn_emin0 belongs only to virtual surface jk=1.

        With zero coupling, the hand-computed solved jk=2/jk=3 values are
        both the ordinary interior floor rn_emin=1e-6.  Before the fix,
        legoESM incorrectly repinned jk=2 (carried interface 0) to 1e-4.
        NEMO MY_SRC/zdftke.F90:361 vs :564-565.
        """
        actual = self._floor_only_solve(
            surface_bc_level="nemo_z0", surface_dirichlet=1.0e-4)
        np.testing.assert_array_equal(actual, np.asarray([1.0e-6, 1.0e-6]))

    def test_nemo_z0_surface_min_knob_is_live_at_virtual_dirichlet_row(self):
        """Changing tke_surface_min changes the coupled nemo_z0 solution.

        The field is NEMO's rn_emin0 and therefore belongs to the virtual
        surface Dirichlet row, not the first solved interior interface.  This
        catches the dead-knob regression where the row read a module constant
        and silently ignored the configuration field.
        """
        shape = (1, 1, 2)
        zeros = jnp.zeros(shape)
        taum = jnp.zeros((1, 1))

        def solve(surface_min):
            cfg = TKEConfig(
                surface_bc="nemo_dirichlet",
                tke_surface_bc_level="nemo_z0",
                tke_surface_min=surface_min,
                c_eps=0.0,
            )
            surface = _surface_tke_dirichlet(cfg, taum, _RHO0)
            return np.asarray(_solve_tke_backward_euler(
                e_old=jnp.full(shape, cfg.tke_background),
                K_M_old=jnp.full(shape, 1.0e-3), K_H_old=zeros,
                P_s=zeros, N2=zeros, l_eps=jnp.ones(shape),
                dz_half=jnp.ones(shape), surface_flux=jnp.zeros((1, 1)),
                dt=1.0, cfg=cfg, dz_surface=jnp.ones((1, 1)),
                dz_face_surface=jnp.ones((1, 1)),
                surface_dirichlet=surface, surface_bc_level="nemo_z0",
                K_M_surface=jnp.full((1, 1), 1.0e-3),
            ))

        low = solve(1.0e-4)
        high = solve(4.0e-4)
        assert high[0, 0, 0] > low[0, 0, 0]
        assert not np.array_equal(low, high)

    @pytest.mark.parametrize("surface_dirichlet", [None, 5.0e-5])
    def test_non_nemo_z0_paths_retain_first_interface_surface_min_byte_pin(
            self, surface_dirichlet):
        """Byte pin: flux/default and interior-pinned paths do not change."""
        actual = self._floor_only_solve(
            surface_bc_level="interior_pinned",
            surface_dirichlet=surface_dirichlet)
        np.testing.assert_array_equal(actual, np.asarray([1.0e-4, 1.0e-6]))

    def test_nemo_z0_matches_hand_derived_solve(self):
        """One prognostic step: nemo_z0's solved row-0 EXACTLY matches an
        independently hand-derived NEMO (N+1)-row Thomas solve (formulas
        re-derived from zdftke.F90, not copy-pasted from the implementation
        under test)."""
        (u, v, T, S, rho, dz_half, z_int, tx, ty, dz_ref, jacobian,
         dz_surface) = self._column()
        cfg = TKEConfig(
            surface_bc="nemo_dirichlet", tke_surface_bc_level="nemo_z0")
        tke_old = jnp.full((1, 1, 5), cfg.tke_background)
        out = tke_vertical_mixing(
            u, v, T, S, rho, dz_half, tke_old, tx, ty, dt=1800.0, cfg=cfg,
            rho_0=_RHO0, n_iterations=1, dz_ref=dz_ref, jacobian=jacobian,
            dz_surface=dz_surface, z_interface=z_int)

        # Independent hand-solve on the SAME inputs the orchestrator computed
        # internally: reconstruct N2/shear/K_M/l_eps/e_sfc from the same
        # column (Bougeault-Lacarrere defaults, tke_mxl_choice=2 — the
        # TKEConfig default — so l_k IS the mxl length fed to K_M).
        from legoesm.ocean.physics.vertical_mixing.tke import (
            _NEMO_TKE_EBB, _NEMO_TKE_EMIN0, _safe_stress_modulus,
            compute_K_from_tke, compute_mixing_lengths,
        )
        from legoesm.ocean.physics.vertical_mixing._shared import (
            compute_N2 as _compute_N2, vertical_shear_squared as _vsq,
        )
        from legoesm import constants as _constants
        dz_half_np = np.asarray(dz_half)
        N2_arr = np.asarray(_compute_N2(rho, dz_half, _RHO0, _constants.g))
        taum = float(np.asarray(_safe_stress_modulus(tx, ty))[0, 0])
        e_sfc = max(_NEMO_TKE_EMIN0, _NEMO_TKE_EBB / _RHO0 * taum)
        e_old_np = np.asarray(tke_old)[0, 0]
        l_k, l_eps = compute_mixing_lengths(
            tke_old, jnp.asarray(N2_arr), dz_half, cfg, signed_n2=False)
        K_M, _ = compute_K_from_tke(tke_old, l_k, cfg)
        shear_sq_np = np.asarray(_vsq(u, v, dz_half))[0, 0]
        expected = _nemo_z0_hand_solve(
            e_old_np, np.asarray(K_M)[0, 0], N2_arr[0, 0],
            np.asarray(l_eps)[0, 0], 10.0, e_sfc, 1800.0, cfg.c_eps,
            cfg.alpha_tke, shear_sq=shear_sq_np)
        # NEMO's solved jk=2..jpkm1 rows receive only rn_emin; rn_emin0 is
        # owned by the separate virtual surface row (zdftke.F90:361,564-565).
        expected = np.maximum(expected, cfg.tke_background)
        np.testing.assert_allclose(
            np.asarray(out.tke_new)[0, 0], expected, rtol=1e-10)

    def test_nemo_z0_surface_face_is_full_e3t1_not_the_cell_midpoint(self):
        """#1690: the virtual-surface face distance is NEMO's e3t(1)
        (the LIVE top-cell thickness dz_ref[0]*J), NOT the top cell's midpoint
        depth -z_full_ref[0]*J (= half e3t(1) on a midpoint grid, which is
        what the production orchestrator used to hand the solver).

        Non-vacuity: this column feeds the honest MIDPOINT value (5 m on a
        10 m top cell) in the ``dz_surface`` slot exactly as ``k_profiles``
        does, so the pre-fix wiring solves a DIFFERENT system.  The two
        candidate metrics are asserted to disagree before the model is
        compared to either, so the test cannot pass by both being equal.
        """
        (u, v, T, S, rho, dz_half, z_int, tx, ty, dz_ref, jacobian,
         _dz_surface_full) = self._column()
        # What k_profiles actually computes for the Veros injection volume:
        # -z_full_ref[0]*J = half the 10 m top cell.
        dz_surface_midpoint = jnp.full((1, 1), 5.0)
        cfg = TKEConfig(
            surface_bc="nemo_dirichlet", tke_surface_bc_level="nemo_z0")
        tke_old = jnp.full((1, 1, 5), cfg.tke_background)
        out = tke_vertical_mixing(
            u, v, T, S, rho, dz_half, tke_old, tx, ty, dt=1800.0, cfg=cfg,
            rho_0=_RHO0, n_iterations=1, dz_ref=dz_ref, jacobian=jacobian,
            dz_surface=dz_surface_midpoint, z_interface=z_int)

        from legoesm.ocean.physics.vertical_mixing.tke import (
            _NEMO_TKE_EBB, _NEMO_TKE_EMIN0, _safe_stress_modulus,
            compute_K_from_tke, compute_mixing_lengths,
        )
        from legoesm.ocean.physics.vertical_mixing._shared import (
            compute_N2 as _compute_N2, vertical_shear_squared as _vsq,
        )
        from legoesm import constants as _constants
        N2_arr = np.asarray(_compute_N2(rho, dz_half, _RHO0, _constants.g))
        taum = float(np.asarray(_safe_stress_modulus(tx, ty))[0, 0])
        e_sfc = max(_NEMO_TKE_EMIN0, _NEMO_TKE_EBB / _RHO0 * taum)
        e_old_np = np.asarray(tke_old)[0, 0]
        l_k, l_eps = compute_mixing_lengths(
            tke_old, jnp.asarray(N2_arr), dz_half, cfg, signed_n2=False)
        K_M, _ = compute_K_from_tke(tke_old, l_k, cfg)
        shear_sq_np = np.asarray(_vsq(u, v, dz_half))[0, 0]

        def _hand(face):
            return np.maximum(_nemo_z0_hand_solve(
                e_old_np, np.asarray(K_M)[0, 0], N2_arr[0, 0],
                np.asarray(l_eps)[0, 0], 10.0, e_sfc, 1800.0, cfg.c_eps,
                cfg.alpha_tke, shear_sq=shear_sq_np, dz_face_sfc=face),
                cfg.tke_background)

        expected_e3t1 = _hand(10.0)          # NEMO e3t(1)
        pre_fix_midpoint = _hand(5.0)        # -z_full_ref[0]*J, the defect
        # The discriminator is live: doubling the coupling moves row 0.
        assert not np.allclose(
            expected_e3t1[0], pre_fix_midpoint[0], rtol=1e-6), (
            "metric candidates are degenerate on this column - the "
            "assertion below would be vacuous")
        np.testing.assert_allclose(
            np.asarray(out.tke_new)[0, 0], expected_e3t1, rtol=1e-10)

    def test_nemo_z0_exact_surface_avm_matches_hand_derived_solve(self):
        """T3-EXACT: with ``tke_mxl_choice=3`` (which computes the ln_mxl0
        surface anchor the exact face needs), the nemo_z0 face uses NEMO's
        TRUE avm(jk=1) = MAX(rn_ediff*zmxlm(1)*sqrt(en(1)), avmb) — not the
        avm(1)~=avm(2) approximation — and matches an independent hand-solve
        using that exact value."""
        (u, v, T, S, rho, dz_half, z_int, tx, ty, dz_ref, jacobian,
         dz_surface) = self._column()
        cfg = TKEConfig(
            surface_bc="nemo_dirichlet", tke_surface_bc_level="nemo_z0",
            tke_mxl_choice=3, kappa_convention="veros_sqrte")
        tke_old = jnp.full((1, 1, 5), cfg.tke_background)
        out = tke_vertical_mixing(
            u, v, T, S, rho, dz_half, tke_old, tx, ty, dt=1800.0, cfg=cfg,
            rho_0=_RHO0, n_iterations=1, dz_ref=dz_ref, jacobian=jacobian,
            dz_surface=dz_surface, z_interface=z_int)

        from legoesm.ocean.physics.vertical_mixing.tke import (
            _NEMO_TKE_EBB, _NEMO_TKE_EMIN0, _mxl0_surface_anchor,
            _safe_stress_modulus, compute_K_from_tke, compute_mixing_lengths,
            nemo_surface_avm,
        )
        from legoesm.ocean.physics.vertical_mixing._shared import (
            compute_N2 as _compute_N2, vertical_shear_squared as _vsq,
        )
        from legoesm import constants as _constants
        g = _constants.g   # the orchestrator's own default — a 9.80665
        # literal here would inject a 5e-5 relative N2/anchor mismatch vs
        # the function under test (physics-validator review CONCERN 2).
        shear_sq_np = np.asarray(_vsq(u, v, dz_half))[0, 0]
        N2_arr = np.asarray(_compute_N2(rho, dz_half, _RHO0, g))
        taum_batch = _safe_stress_modulus(tx, ty)          # shape (1, 1)
        taum = float(np.asarray(taum_batch)[0, 0])
        e_sfc = max(_NEMO_TKE_EMIN0, _NEMO_TKE_EBB / _RHO0 * taum)
        e_old_np = np.asarray(tke_old)[0, 0]
        l_anchor = _mxl0_surface_anchor(cfg, taum_batch, _RHO0, g)  # (1, 1)
        dz_cell = dz_ref * jacobian[..., None]
        l_k, l_eps = compute_mixing_lengths(
            tke_old, jnp.asarray(N2_arr), dz_half, cfg, signed_n2=False,
            dz_cell=dz_cell, l_surface_anchor=l_anchor)
        K_M, _ = compute_K_from_tke(tke_old, l_k, cfg)
        e_sfc_batch = jnp.maximum(
            jnp.asarray(_NEMO_TKE_EMIN0), _NEMO_TKE_EBB / _RHO0 * taum_batch)
        avm1 = float(np.asarray(
            nemo_surface_avm(cfg, e_sfc_batch, l_anchor))[0, 0])
        expected = _nemo_z0_hand_solve(
            e_old_np, np.asarray(K_M)[0, 0], N2_arr[0, 0],
            np.asarray(l_eps)[0, 0], 10.0, e_sfc, 1800.0, cfg.c_eps,
            cfg.alpha_tke, avm1=avm1, shear_sq=shear_sq_np)
        expected = np.maximum(expected, cfg.tke_background)
        np.testing.assert_allclose(
            np.asarray(out.tke_new)[0, 0], expected, rtol=1e-9)
        # And it must NOT equal the approximation (avm1 != K_M[0] generically
        # — otherwise this test would not discriminate the fix).
        assert not np.isclose(avm1, float(np.asarray(K_M)[0, 0, 0]), rtol=1e-6)

    def test_nemo_surface_avm_formula(self):
        """Direct unit test of :func:`nemo_surface_avm` — re-derived from
        zdftke.F90:713-715 (avm = MAX(rn_ediff*zmxlm*sqrt(en), avmb))."""
        from legoesm.ocean.physics.vertical_mixing.tke import nemo_surface_avm
        cfg = TKEConfig(kappaM_min=2.0e-4)
        e_sfc = jnp.asarray(4.0)     # en(1)
        l_sfc = jnp.asarray(2.0)     # zmxlm(1)
        out = float(nemo_surface_avm(cfg, e_sfc, l_sfc))
        expected = max(cfg.c_k * 2.0 * np.sqrt(4.0), cfg.kappaM_min)
        assert np.isclose(out, expected, rtol=1e-12)
        # Floor binds when the raw value is tiny.
        out_floor = float(nemo_surface_avm(cfg, jnp.asarray(1e-12),
                                           jnp.asarray(1e-12)))
        assert out_floor == pytest.approx(cfg.kappaM_min)

    def test_nemo_surface_avm_requires_l_sfc(self):
        from legoesm.ocean.physics.vertical_mixing.tke import nemo_surface_avm
        with pytest.raises(ValueError, match="l_sfc"):
            nemo_surface_avm(TKEConfig(), jnp.asarray(1.0), None)

    def test_nemo_z0_face_falls_back_to_approximation_without_choice3(self):
        """Without tke_mxl_choice=3 (no ln_mxl0 anchor available), the face
        keeps the documented approximation avm(1)~=avm(2) — BIT-IDENTICAL to
        the pre-T3-exact behaviour (choice=2, the TKEConfig default)."""
        (u, v, T, S, rho, dz_half, z_int, tx, ty, dz_ref, jacobian,
         dz_surface) = self._column()
        cfg = TKEConfig(
            surface_bc="nemo_dirichlet", tke_surface_bc_level="nemo_z0")
        assert cfg.tke_mxl_choice == 2   # default, no ln_mxl0 anchor
        tke_old = jnp.full((1, 1, 5), cfg.tke_background)
        out = tke_vertical_mixing(
            u, v, T, S, rho, dz_half, tke_old, tx, ty, dt=1800.0, cfg=cfg,
            rho_0=_RHO0, n_iterations=1, dz_ref=dz_ref, jacobian=jacobian,
            dz_surface=dz_surface, z_interface=z_int)
        assert bool(np.all(np.isfinite(np.asarray(out.tke_new))))

    def test_nemo_z0_displaces_profile_vs_interior_pinned(self):
        """Holding the SAME Dirichlet value, 'nemo_z0' SOLVES interface 0 as
        a genuine interior row coupled (via the surface face) to the held
        value one level up, so it sits SYSTEMATICALLY BELOW 'interior_pinned'
        (which instead CLAMPS interface 0 AT the Dirichlet value exactly,
        with no diffusive loss) — the whole profile is displaced.

        NOTE (T3-exact gap-closure, 2026-07-24): the original Phase-1 probe
        measured 'nemo_z0' ABOVE 'interior_pinned' (en(20m) 8.6x) under a
        surface-row assembly bug that DOUBLE-COUNTED the virtual-row
        coupling source in the RHS (fixed the same session this test was
        extended) — that bug artificially inflated 'nemo_z0' by ~2x. With
        the fix, 'nemo_z0' correctly sits BELOW 'interior_pinned' (energy is
        lost crossing the extra diffusive face to reach the solved value);
        the direction here is the physically-correct one, not the probe's
        original (bug-confounded) figure."""
        (u, v, T, S, rho, dz_half, z_int, tx, ty, dz_ref, jacobian,
         dz_surface) = self._column(nlev=10)
        tke_old = jnp.full((1, 1, 9), 1e-6)
        common = dict(
            u_cell=u, v_cell=v, T_cell=T, S_cell=S, rho_cell=rho,
            dz_half=dz_half, tke_old=tke_old, tau_x_surface=tx,
            tau_y_surface=ty, dt=1800.0, rho_0=_RHO0, n_iterations=20,
            dz_ref=dz_ref, jacobian=jacobian, dz_surface=dz_surface,
            z_interface=z_int,
        )
        pinned = tke_vertical_mixing(
            cfg=TKEConfig(surface_bc="nemo_dirichlet",
                         tke_surface_bc_level="interior_pinned"),
            **common)
        z0 = tke_vertical_mixing(
            cfg=TKEConfig(surface_bc="nemo_dirichlet",
                         tke_surface_bc_level="nemo_z0"),
            **common)
        en_pinned = np.asarray(pinned.tke_new)[0, 0]
        en_z0 = np.asarray(z0.tke_new)[0, 0]
        # interior_pinned CLAMPS interface 0 at e_sfc exactly (the Dirichlet
        # row); nemo_z0 SOLVES interface 0 as a genuine interior row coupled
        # to the (larger, near-surface) held value one level up — so its
        # equilibrium value at interface 0 sits BELOW the pinned value
        # (energy is lost crossing the extra diffusive face) yet every
        # deeper interface is displaced systematically differently. Assert
        # the direction the Phase-1 audit found — a substantial (>=50%)
        # nemo_z0 strictly below interior_pinned at EVERY interface (energy
        # lost crossing the extra diffusive face to the virtual surface
        # row) — a substantial (>=10%), monotone-in-depth-growing placement
        # effect (the exact multiplier is column-geometry-dependent).
        assert not np.allclose(en_pinned, en_z0, rtol=1e-3)
        assert bool(np.all(en_z0 < en_pinned))
        ratio = en_z0 / np.maximum(en_pinned, 1e-30)
        assert ratio[1] < 0.9, (
            f"nemo_z0 vs interior_pinned at interface 1 should differ "
            f"substantially (placement is a first-order effect); got "
            f"ratio={ratio[1]}")
        # The displacement GROWS with depth (the surface energy-loss effect
        # compounds through the column) — a monotonicity check that is
        # robust to the exact per-level multiplier.
        assert bool(np.all(np.diff(ratio) <= 1e-9))

    def test_nemo_z0_requires_surface_dirichlet(self):
        """'nemo_z0' has no meaning without a held surface value — raises
        rather than silently keeping the flux BC (dispatch hardening)."""
        u, v, T, S, rho, dz_half, z_int, tx, ty = _orchestrator_inputs()
        with pytest.raises(ValueError, match="nemo_z0"):
            tke_vertical_mixing(
                u, v, T, S, rho, dz_half, None, tx, ty, dt=3600.0,
                cfg=TKEConfig(tke_surface_bc_level="nemo_z0"), rho_0=_RHO0,
                n_iterations=1, z_interface=z_int)

    def test_nemo_z0_requires_the_cell_thickness_metric(self):
        """#1690: nemo_z0 derives NEMO's e3t(1) from dz_ref/jacobian, so
        those are what it demands -- it must NOT silently fall back to the
        midpoint ``dz_surface`` slot."""
        u, v, T, S, rho, dz_half, z_int, tx, ty = _orchestrator_inputs()
        with pytest.raises(ValueError, match="dz_ref"):
            tke_vertical_mixing(
                u, v, T, S, rho, dz_half, None, tx, ty, dt=3600.0,
                cfg=TKEConfig(surface_bc="nemo_dirichlet",
                             tke_surface_bc_level="nemo_z0"),
                rho_0=_RHO0, n_iterations=1, z_interface=z_int)

    def test_default_is_byte_identical(self):
        """interior_pinned (default) is bit-identical to the prior (pre-T3)
        behaviour — i.e. dropping tke_surface_bc_level entirely."""
        u, v, T, S, rho, dz_half, z_int, tx, ty = _orchestrator_inputs()
        base = tke_vertical_mixing(
            u, v, T, S, rho, dz_half, None, tx, ty, dt=3600.0,
            cfg=TKEConfig(surface_bc="nemo_dirichlet"), rho_0=_RHO0,
            n_iterations=3, z_interface=z_int)
        explicit = tke_vertical_mixing(
            u, v, T, S, rho, dz_half, None, tx, ty, dt=3600.0,
            cfg=TKEConfig(surface_bc="nemo_dirichlet",
                         tke_surface_bc_level="interior_pinned"),
            rho_0=_RHO0, n_iterations=3, z_interface=z_int)
        np.testing.assert_array_equal(
            np.asarray(base.tke_new), np.asarray(explicit.tke_new))

    def test_unknown_surface_bc_level_raises(self):
        u, v, T, S, rho, dz_half, z_int, tx, ty = _orchestrator_inputs()
        with pytest.raises(ValueError, match="tke_surface_bc_level"):
            tke_vertical_mixing(
                u, v, T, S, rho, dz_half, None, tx, ty, dt=3600.0,
                cfg=TKEConfig(surface_bc="nemo_dirichlet",
                             tke_surface_bc_level="bogus"),
                rho_0=_RHO0, n_iterations=1, z_interface=z_int)


# ---------------------------------------------------------------------------
# T15 (Phase-2 #1317): bottom TKE BC (zdftke.F90:279-288).
# ---------------------------------------------------------------------------


class TestNemoBottomTkeDirichlet:
    def test_formula_matches_hand_derivation(self):
        """en(mbkt+1) = max(0.001875*CdU_bot*|u_bot|, rn_emin) — re-derived
        by hand from the F90 line, not copy-pasted."""
        from legoesm.ocean.physics.vertical_mixing.tke import (
            nemo_bottom_tke_dirichlet,
        )
        cfg = TKEConfig()
        r = jnp.asarray([2.0e-3])
        u_bot = jnp.asarray([0.3])
        v_bot = jnp.asarray([0.4])
        out = float(np.asarray(
            nemo_bottom_tke_dirichlet(r, u_bot, v_bot, cfg))[0])
        speed = np.hypot(0.3, 0.4)   # 0.5
        expected = max(0.001875 * 2.0e-3 * speed, cfg.tke_background)
        assert np.isclose(out, expected, rtol=1e-12)

    def test_floors_at_tke_background_when_drag_negligible(self):
        from legoesm.ocean.physics.vertical_mixing.tke import (
            nemo_bottom_tke_dirichlet,
        )
        cfg = TKEConfig()
        out = float(np.asarray(nemo_bottom_tke_dirichlet(
            jnp.asarray([0.0]), jnp.asarray([0.0]), jnp.asarray([0.0]), cfg,
        ))[0])
        assert out == pytest.approx(cfg.tke_background)

    def test_bottom_dirichlet_holds_deepest_interface_exactly(self):
        """bottom_tke_bc=True holds e_new[..., -1] at the supplied value
        EXACTLY (a plain Dirichlet identity row — no coupling adjustment
        needed at the bottom, unlike the surface T3 fix)."""
        u, v, T, S, rho, dz_half, z_int, tx, ty = _orchestrator_inputs()
        bottom_val = jnp.full(u.shape[:-1], 5e-4)
        out = tke_vertical_mixing(
            u, v, T, S, rho, dz_half, None, tx, ty, dt=3600.0,
            cfg=TKEConfig(bottom_tke_bc=True), rho_0=_RHO0,
            n_iterations=3, z_interface=z_int, bottom_dirichlet=bottom_val)
        np.testing.assert_allclose(
            np.asarray(out.tke_new)[..., -1], 5e-4, rtol=0, atol=1e-12)

    def test_bottom_tke_bc_requires_bottom_dirichlet(self):
        u, v, T, S, rho, dz_half, z_int, tx, ty = _orchestrator_inputs()
        with pytest.raises(ValueError, match="bottom_tke_bc"):
            tke_vertical_mixing(
                u, v, T, S, rho, dz_half, None, tx, ty, dt=3600.0,
                cfg=TKEConfig(bottom_tke_bc=True), rho_0=_RHO0,
                n_iterations=1, z_interface=z_int)

    def test_bottom_dirichlet_without_gate_raises(self):
        """Silent-no-op guard: passing bottom_dirichlet without the gate
        must fail loudly, not silently ignore the value."""
        u, v, T, S, rho, dz_half, z_int, tx, ty = _orchestrator_inputs()
        with pytest.raises(ValueError, match="bottom_tke_bc=False"):
            tke_vertical_mixing(
                u, v, T, S, rho, dz_half, None, tx, ty, dt=3600.0,
                cfg=TKEConfig(), rho_0=_RHO0, n_iterations=1,
                z_interface=z_int,
                bottom_dirichlet=jnp.full(u.shape[:-1], 5e-4))

    def test_default_is_byte_identical(self):
        u, v, T, S, rho, dz_half, z_int, tx, ty = _orchestrator_inputs()
        base = tke_vertical_mixing(
            u, v, T, S, rho, dz_half, None, tx, ty, dt=3600.0,
            cfg=TKEConfig(), rho_0=_RHO0, n_iterations=3, z_interface=z_int)
        explicit = tke_vertical_mixing(
            u, v, T, S, rho, dz_half, None, tx, ty, dt=3600.0,
            cfg=TKEConfig(bottom_tke_bc=False), rho_0=_RHO0,
            n_iterations=3, z_interface=z_int)
        np.testing.assert_array_equal(
            np.asarray(base.tke_new), np.asarray(explicit.tke_new))

    def test_combines_with_nemo_z0_surface_bc(self):
        """Regression: the kamm card enables BOTH nemo_z0 (T3, prepends a
        virtual row at the FRONT) and bottom_tke_bc (T15, pins the LAST row
        of the pre-extension system) simultaneously — the two must not
        interact (the bottom row's Dirichlet pin must survive the surface
        extension+slice unchanged)."""
        u, v, T, S, rho, dz_half, z_int, tx, ty = _orchestrator_inputs()
        dz_surface = jnp.full(u.shape[:-1], 25.0)
        bottom_val = jnp.full(u.shape[:-1], 3e-4)
        cfg = TKEConfig(surface_bc="nemo_dirichlet",
                        tke_surface_bc_level="nemo_z0", bottom_tke_bc=True)
        out = tke_vertical_mixing(
            u, v, T, S, rho, dz_half, None, tx, ty, dt=3600.0, cfg=cfg,
            rho_0=_RHO0, n_iterations=3, z_interface=z_int,
            dz_ref=jnp.full((u.shape[-1],), 25.0),
            jacobian=jnp.ones(u.shape[:-1]),
            dz_surface=dz_surface, bottom_dirichlet=bottom_val)
        np.testing.assert_allclose(
            np.asarray(out.tke_new)[..., -1], 3e-4, rtol=0, atol=1e-12)
        assert bool(np.all(np.isfinite(np.asarray(out.tke_new))))

    def test_bottom_level_scatters_to_per_column_seafloor(self):
        """T15-EXACT: on a STEPPED-bathymetry column (bottom_level varies per
        column), the Dirichlet pin lands at interior interface bottom_level,
        NOT the array's last row — the fix over the flat-bottom-only
        approximation."""
        u, v, T, S, rho, dz_half, z_int, tx, ty = _orchestrator_inputs()
        N = dz_half.shape[-1]   # 7 interior interfaces (nlev=8)
        # Per-column bottom_level: shallow (2), mid (4), full-depth (N-1=6).
        bl = jnp.asarray([2, 4, N - 1])[:, None] * jnp.ones(
            u.shape[:-1], dtype=jnp.int32)
        bottom_val = jnp.full(u.shape[:-1], 7e-4)
        out = tke_vertical_mixing(
            u, v, T, S, rho, dz_half, None, tx, ty, dt=3600.0,
            cfg=TKEConfig(bottom_tke_bc=True), rho_0=_RHO0,
            n_iterations=3, z_interface=z_int, bottom_dirichlet=bottom_val,
            bottom_level=bl)
        en = np.asarray(out.tke_new)
        bl_np = np.asarray(bl)
        for i in range(bl_np.shape[0]):
            for j in range(bl_np.shape[1]):
                k = bl_np[i, j]
                assert en[i, j, k] == pytest.approx(7e-4, abs=1e-12), (
                    f"column ({i},{j}): pin did not land at bottom_level={k}")
        # The last row of the SHALLOW columns must NOT be pinned (it's above
        # the true seafloor and remains a genuine solved/diffusing row) —
        # distinguishes this from the old unconditional-last-row behaviour.
        assert en[0, 0, -1] != pytest.approx(7e-4, abs=1e-12)

    def test_bottom_level_none_is_byte_identical(self):
        """bottom_level=None keeps the unconditional last-row pin (the prior
        behaviour) — BIT-IDENTICAL."""
        u, v, T, S, rho, dz_half, z_int, tx, ty = _orchestrator_inputs()
        bottom_val = jnp.full(u.shape[:-1], 5e-4)
        common = dict(
            u_cell=u, v_cell=v, T_cell=T, S_cell=S, rho_cell=rho,
            dz_half=dz_half, tke_old=None, tau_x_surface=tx,
            tau_y_surface=ty, dt=3600.0, rho_0=_RHO0, n_iterations=3,
            z_interface=z_int, bottom_dirichlet=bottom_val,
            cfg=TKEConfig(bottom_tke_bc=True),
        )
        base = tke_vertical_mixing(**common)
        explicit_none = tke_vertical_mixing(bottom_level=None, **common)
        np.testing.assert_array_equal(
            np.asarray(base.tke_new), np.asarray(explicit_none.tke_new))

    def test_bottom_level_without_dirichlet_raises(self):
        """Silent-no-op guard: bottom_level selects WHERE the pin lands, not
        whether one exists."""
        u, v, T, S, rho, dz_half, z_int, tx, ty = _orchestrator_inputs()
        bl = jnp.zeros(u.shape[:-1], dtype=jnp.int32)
        with pytest.raises(ValueError, match="bottom_level"):
            tke_vertical_mixing(
                u, v, T, S, rho, dz_half, None, tx, ty, dt=3600.0,
                cfg=TKEConfig(), rho_0=_RHO0, n_iterations=1,
                z_interface=z_int, bottom_level=bl)

    def test_bottom_level_zero_collides_with_nemo_z0_row0(self):
        """Degenerate-column regression: on a column where bottom_level=0
        (a single-interior-interface / 2-cell column — the bottom Dirichlet
        pin lands EXACTLY on interior interface 0), combined with
        surface_bc_level='nemo_z0' (which ALSO builds a coupling AT row 0
        to the virtual surface row), the bottom pin must survive intact —
        row 0 stays a pure identity row at the bottom_dirichlet value, not
        overridden by the surface-coupling addition."""
        shape = (1, 1, 8)
        T = jnp.asarray(20.0 - 2.0 * np.arange(8))[None, None, :] * jnp.ones(shape)
        S = jnp.full(shape, 35.0)
        rho = jnp.asarray(1026.0 + 0.2 * np.arange(8))[None, None, :] * jnp.ones(shape)
        u = jnp.zeros(shape)
        v = jnp.zeros(shape)
        dz_half = jnp.full((1, 1, 7), 25.0)
        z_int = -25.0 * jnp.arange(1, 8)
        tx = jnp.full((1, 1), 0.1)
        ty = jnp.zeros((1, 1))
        dz_surface = jnp.full((1, 1), 25.0)
        bl = jnp.asarray([[0]])
        bottom_val = jnp.full((1, 1), 3e-4)
        cfg = TKEConfig(surface_bc="nemo_dirichlet",
                        tke_surface_bc_level="nemo_z0", bottom_tke_bc=True)
        out = tke_vertical_mixing(
            u, v, T, S, rho, dz_half, None, tx, ty, dt=3600.0, cfg=cfg,
            rho_0=_RHO0, n_iterations=3, z_interface=z_int,
            dz_ref=jnp.full((dz_half.shape[-1] + 1,), 25.0),
            jacobian=jnp.ones((1, 1)),
            dz_surface=dz_surface, bottom_dirichlet=bottom_val,
            bottom_level=bl)
        assert float(out.tke_new[0, 0, 0]) == pytest.approx(3e-4, abs=1e-12)
        assert bool(np.all(np.isfinite(np.asarray(out.tke_new))))


class TestTridiagMixedPrecision:
    """Regression for the #1317 T3 acceptance blocker: mixed f32/f64 rows
    fed to tridiag_thomas crashed the lax.scan carry (float32 seed vs
    float64 body) on the first model step of the mixed-precision twin."""

    def test_mixed_dtype_inputs_solve_and_match_f64(self):
        import numpy as np
        import jax.numpy as jnp
        from legoesm.ocean.physics.vertical_mixing._shared import tridiag_thomas

        rng = np.random.default_rng(0)
        n = 12
        b64 = 2.0 + rng.random((5, n))
        a64 = -0.3 * rng.random((5, n)); a64[:, 0] = 0.0
        c64 = -0.3 * rng.random((5, n)); c64[:, -1] = 0.0
        d64 = rng.random((5, n))
        x_ref = tridiag_thomas(*(jnp.asarray(v, dtype=jnp.float64)
                                 for v in (a64, b64, c64, d64)))
        # f32 matrix, f64 rhs — the crashing combination
        x_mix = tridiag_thomas(
            jnp.asarray(a64, dtype=jnp.float32),
            jnp.asarray(b64, dtype=jnp.float32),
            jnp.asarray(c64, dtype=jnp.float32),
            jnp.asarray(d64, dtype=jnp.float64),
        )
        assert x_mix.dtype == jnp.float64
        np.testing.assert_allclose(np.asarray(x_mix), np.asarray(x_ref),
                                   rtol=2e-5)


# ---------------------------------------------------------------------------
# T15b: zdftke's OWN bottom velocity convention -- the wet-only SUM
# zmsku*(uu(ji)+uu(ji-1)), zmsku = 2 - umask(ji-1)*umask(ji), NO 0.5
# (zdftke.F90:282-287; contrast zdfgls.F90:196-197 which carries the 0.5).
# Exercises the PRODUCTION symbol `_tke_bottom_dirichlet` -- the reference is
# re-derived in plain numpy from the F90 line, independently of the model's
# own mask helpers, and the whole class FAILS if the wet-only masking or the
# doubling is reverted to the plain T-point average.
# ---------------------------------------------------------------------------


class TestNemoBottomTkeVelocityConvention:

    @staticmethod
    def _model_and_state():
        """Staircase bathymetry (three distinct bottom levels) so zmsku/zmskv
        are 2 somewhere, with u/v deliberately NON-ZERO below the face
        seafloor -- legoESM's prognostic u carries only the 2-D column mask
        and the barotropic correction adds a uniform-in-k increment, so
        sub-seafloor faces really do hold velocity (unlike NEMO's, which
        dynzdf.F90:121-150 umasks at every level)."""
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        from legoesm.ocean.init_latlon_cgrid import (
            rest_state_latlon_cgrid_ocean,
        )
        from legoesm.ocean.physics.combined import OceanPhysicsConfig
        from legoesm.ocean.physics.convection.config import (
            OceanConvectionConfig,
        )
        from legoesm.ocean.physics.vertical_mixing.config import (
            VerticalMixingConfig,
        )
        from legoesm.ocean.state import LatLonCGridOceanConfig
        from legoesm.ocean.vertical import (
            create_ocean_z_star, create_partial_cell_coordinate,
        )
        n_lat, n_lon, n_levels, H_max = 6, 8, 6, 3000.0
        grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
        z0c = create_ocean_z_star(n_levels=n_levels, H_max=H_max)
        H = np.full((n_lat, n_lon), H_max * 0.95)
        H[:, 3] = H_max * 0.45          # a zonal step
        H[2, :] = H_max * 0.65          # a meridional step
        H_bathy = jnp.asarray(H)
        z = create_partial_cell_coordinate(z0c, H_bathy)
        state = rest_state_latlon_cgrid_ocean(
            grid, z0c, T_water_init_C=10.0, T_deep=10.0, S_uniform=35.0,
            H_bathy_override=H_bathy)
        rng = np.random.default_rng(7)
        u = rng.normal(scale=0.2, size=state.u.data.shape)
        v = rng.normal(scale=0.2, size=state.v.data.shape)
        state = state._replace(
            u=state.u.replace(data=jnp.asarray(u)),
            v=state.v.replace(data=jnp.asarray(v)),
        )
        # rn_emin -> ~0 so the MAX() floor does NOT bind and the velocity
        # convention is actually observable (it is floor-bound in DINO).
        physics = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(
                scheme="tke",
                tke=TKEConfig(bottom_tke_bc=True,
                              tke_background=1.0e-30)),
            convection=OceanConvectionConfig(scheme="none"),
            lateral_mixing=type(OceanPhysicsConfig().lateral_mixing)(
                scheme="none"),
        )
        cfg = LatLonCGridOceanConfig.from_flat(
            A_v=1.0e-3, K_v=1.0e-4, bottom_drag_scheme="nemo_quadratic",
            bottom_drag_cd0=1.0e-3, bottom_drag_cdmax=0.1,
            bottom_drag_z0=3.0e-3, bottom_drag_ke0=2.5e-3,
            implicit_vertical_mixing=True, enable_runtime_checks=False,
            physics=physics,
        )
        return LatLonCGridOceanModel(grid, z, cfg), state, z

    @staticmethod
    def _f90_reference(state, z, cfg_tke, r_t):
        """zdftke.F90:285-288 re-derived in numpy from the T-mask alone."""
        tm = np.asarray(z.is_active).astype(float)          # tmask (y, x, k)
        bl = np.maximum(np.asarray(z.bottom_level), 0)
        ny, nx, _ = tm.shape
        u_f, v_f = np.asarray(state.u.data), np.asarray(state.v.data)
        # umask(ji,jj,jk) = tmask(ji)*tmask(ji+1) (dommsk.F90:150); x is
        # i-periodic, y walled.
        um = tm * np.roll(tm, 1, axis=1)                    # face j: (j-1, j)
        um = np.concatenate([um, um[:, :1, :]], axis=1)     # (ny, nx+1, k)
        vm = np.concatenate(
            [np.zeros((1, nx, tm.shape[2])), tm[:-1] * tm[1:],
             np.zeros((1, nx, tm.shape[2]))], axis=0)       # (ny+1, nx, k)
        j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        k = bl
        uw, ue = um[j, i, k], um[j, i + 1, k]
        vs, vn = vm[j, i, k], vm[j + 1, i, k]
        zmsku, zmskv = 2.0 - uw * ue, 2.0 - vs * vn
        u_sum = zmsku * (u_f[j, i, k] * uw + u_f[j, i + 1, k] * ue)
        v_sum = zmskv * (v_f[j, i, k] * vs + v_f[j + 1, i, k] * vn)
        zebot = 0.001875 * np.asarray(r_t) * np.hypot(u_sum, v_sum)
        ssmask = tm[j, i, k]
        return ssmask * np.maximum(zebot, cfg_tke.tke_background)

    @staticmethod
    def _drag_rate_and_plain_speed(state, z):
        """The model's own single-owner drag rate (NOT under test here -- the
        VELOCITY convention is) plus the plain T-point speed the pre-fix code
        used, so the two forms can be compared at identical ``r``."""
        from legoesm import constants
        from legoesm.ocean.dynamics.ocean_tendency_common import (
            nemo_effective_bottom_drag_r,
        )
        bl = np.maximum(np.asarray(z.bottom_level), 0)
        u_f, v_f = np.asarray(state.u.data), np.asarray(state.v.data)
        ny, nx = bl.shape
        j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        u_cc = 0.5 * (u_f[:, :-1, :] + u_f[:, 1:, :])
        v_cc = 0.5 * (v_f[:-1, :, :] + v_f[1:, :, :])
        u_bot, v_bot = u_cc[j, i, bl], v_cc[j, i, bl]
        r_t = nemo_effective_bottom_drag_r(
            jnp.asarray(u_bot), jnp.asarray(v_bot),
            jnp.asarray(np.asarray(z.h_partial)[j, i, bl]),
            scheme="nemo_quadratic", cd0=1.0e-3, cd_max=0.1, z0=3.0e-3,
            ke0=2.5e-3, von_karman=constants.kappa_von_karman)
        return r_t, np.hypot(u_bot, v_bot)

    def test_matches_f90_line_on_a_staircase(self):
        model, state, z = self._model_and_state()
        out = np.asarray(model._tke_bottom_dirichlet(state))
        r_t, _ = self._drag_rate_and_plain_speed(state, z)
        ref = self._f90_reference(
            state, z, model.config.physics.vertical_mixing.tke, r_t)
        assert np.any(ref > 1e-12), "fixture must not be floor-bound"
        np.testing.assert_allclose(out, ref, rtol=1e-12, atol=0.0)

    def test_differs_from_the_plain_average_form(self):
        """Synthetic-violation guard: the pre-fix plain-average form, built at
        the SAME drag rate, must disagree on most columns -- otherwise the
        F90 comparison above could pass vacuously."""
        model, state, z = self._model_and_state()
        out = np.asarray(model._tke_bottom_dirichlet(state))
        r_t, speed_plain = self._drag_rate_and_plain_speed(state, z)
        emin = model.config.physics.vertical_mixing.tke.tke_background
        pre_fix = np.maximum(0.001875 * np.asarray(r_t) * speed_plain, emin)
        live = out > 1e-12
        assert live.sum() > 0
        rel = np.abs(out[live] - pre_fix[live]) / np.abs(out[live])
        assert np.median(rel) > 0.4, (
            "the shipped form is indistinguishable from the plain average "
            f"(median rel diff {np.median(rel):.3e}) -- expected ~0.5 from "
            "the universal factor 2, more at step columns")
        assert rel.max() > 0.7, "no column shows the step-column amplification"

    def test_requires_raw_face_state(self):
        """Passing an already-collapsed (T-point) state must RAISE, never
        silently fall back to the plain average."""
        model, state, _ = self._model_and_state()
        u_cc = 0.5 * (state.u.data[:, :-1, :] + state.u.data[:, 1:, :])
        v_cc = 0.5 * (state.v.data[:-1, :, :] + state.v.data[1:, :, :])
        collapsed = state._replace(
            u=state.u.replace(data=u_cc), v=state.v.replace(data=v_cc))
        with pytest.raises(ValueError, match="RAW face-staggered"):
            model._tke_bottom_dirichlet(collapsed)
