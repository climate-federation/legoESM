"""NEMO zdftke closure-identity wiring for the DINO ``nemo_dino_kamm`` card.

Gate (a) for the "match the DINO TKE closure IDENTITY to NEMO zdftke" task:

1. ``tke_mxl_choice=3`` (NEMO ``nn_mxl=3``) no longer RAISES on the standard
   ``tke_vertical_mixing`` path when ``veros_dz_slots`` is OFF — the plumbing
   bug (``dz_cell`` was only derived behind ``veros_dz_slots``) is fixed by
   deriving the e3t cell thicknesses from the caller-threaded ``dz_ref`` /
   ``jacobian`` whenever choice 3 is selected. Still raises loudly when those
   are genuinely absent (no silent fallback).
2. ``dissipation_discretization="nemo_1p5_split"`` is now wired into the
   pre-mixing ``_solve_tke_backward_euler`` (previously only in the post-mixing
   Veros path) and changes the solved TKE vs ``backward_euler``; an unknown
   value raises (dispatch hardening).
3. The ``nemo_dino_kamm`` recipe assembles the full NEMO-faithful TKEConfig,
   and every OTHER recipe keeps the prior (byte-identical) TKE defaults.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
from legoesm.ocean.physics.vertical_mixing.tke import (
    _nemo_literal_tke_solve,
    _prandtl_number,
    _solve_tke_backward_euler,
    compute_mixing_lengths,
    tke_vertical_mixing,
)

_RHO0 = 1026.0


def _column(nlev=12):
    """A stably-stratified wind-forced column at cell centres (1 water col)."""
    rng = np.random.default_rng(0)
    T = jnp.asarray(np.linspace(18.0, 4.0, nlev))[None, None, :]
    S = jnp.asarray(np.linspace(35.4, 34.7, nlev))[None, None, :]
    u = jnp.asarray(0.1 * rng.standard_normal(nlev))[None, None, :]
    v = jnp.asarray(0.05 * rng.standard_normal(nlev))[None, None, :]
    # in-situ density ~ linear in T (stable): rho decreasing upward
    rho = jnp.asarray(1027.0 + 0.2 * np.linspace(1.0, 0.0, nlev))[None, None, :]
    dz_half = jnp.full((1, 1, nlev - 1), 10.0)
    dz_ref = jnp.full((nlev,), 10.0)
    jacobian = jnp.ones((1, 1))
    return T, S, u, v, rho, dz_half, dz_ref, jacobian


def _orchestrator_inputs(nlev=8):
    """(u, v, T, S, rho, dz_half, z_interface, tau_x, tau_y) on a small
    (3, 4) horizontal slab — mirrors test_tke_nemo_terms.py's helper of the
    same name (independent copy; a bare-call-level unit-test fixture, not
    production code, so this is not a doctrine violation)."""
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


def test_mxl_choice3_no_longer_raises_without_veros_dz_slots():
    """NEMO nn_mxl=3 runs on the standard path with veros_dz_slots OFF (the
    fixed plumbing bug): dz_cell is derived from dz_ref/jacobian."""
    T, S, u, v, rho, dz_half, dz_ref, jacobian = _column()
    cfg = TKEConfig(tke_mxl_choice=3, veros_dz_slots=False)
    out = tke_vertical_mixing(
        u, v, T, S, rho, dz_half, tke_old=None,
        tau_x_surface=jnp.array([[0.05]]), tau_y_surface=jnp.array([[0.0]]),
        dt=1800.0, cfg=cfg, n_iterations=1,
        dz_ref=dz_ref, jacobian=jacobian,
        surface_tmask=jnp.ones(T.shape[:-1]),
    )
    assert np.all(np.isfinite(np.asarray(out.K_M)))
    assert np.all(np.isfinite(np.asarray(out.tke_new)))
    assert np.all(np.asarray(out.K_M) >= 0.0)


def test_mxl_choice3_still_raises_when_cell_thickness_unavailable():
    """No silent fallback: choice 3 without dz_ref/jacobian raises."""
    T, S, u, v, rho, dz_half, _dz_ref, _J = _column()
    cfg = TKEConfig(tke_mxl_choice=3)
    with pytest.raises(ValueError, match="tke_mxl_choice=3"):
        tke_vertical_mixing(
            u, v, T, S, rho, dz_half, tke_old=None,
            tau_x_surface=None, tau_y_surface=None,
            dt=1800.0, cfg=cfg, n_iterations=1,
            dz_ref=None, jacobian=None,
            surface_tmask=jnp.ones(T.shape[:-1]),
        )


def _solve(disc):
    """One backward-Euler TKE solve with the given dissipation discretization."""
    rng = np.random.default_rng(1)
    n = 8
    e = jnp.asarray(1e-3 * np.abs(rng.standard_normal(n)) + 1e-4)[None, :]
    K = jnp.full((1, n), 1e-3)
    P_s = jnp.full((1, n), 1e-6)
    N2 = jnp.full((1, n), 1e-4)
    l_eps = jnp.full((1, n), 5.0)
    dz_half = jnp.full((1, n), 10.0)
    surf = jnp.asarray([1e-6])
    cfg = TKEConfig(dissipation_discretization=disc)
    return np.asarray(_solve_tke_backward_euler(
        e_old=e, K_M_old=K, K_H_old=K, P_s=P_s, N2=N2, l_eps=l_eps,
        dz_half=dz_half, surface_flux=surf, dt=3600.0, cfg=cfg))


def test_nemo_1p5_split_changes_result_and_stays_finite():
    """The NEMO 1.5/0.5 semi-implicit split differs from backward-Euler at
    large dt·diss yet stays finite and positive."""
    be = _solve("backward_euler")
    split = _solve("nemo_1p5_split")
    assert np.all(np.isfinite(split)) and np.all(split >= 0.0)
    assert not np.allclose(be, split), "split must differ from backward-Euler"


def test_nemo_1p5_split_matches_hand_factor_single_cell():
    """Isolated 1-interface decay: backward-Euler factor 1/(1+a) vs NEMO split
    (1+0.5a)/(1+1.5a) with a = dt·c_eps·sqrt(e)/l_eps (no diffusion/shear)."""
    e0 = 4e-4
    dt, c_eps, l_eps = 3600.0, 0.7, 2.0
    e = jnp.asarray([[e0]])
    zero = jnp.zeros((1, 1))
    dz = jnp.full((1, 1), 10.0)
    a = dt * c_eps * np.sqrt(e0) / l_eps
    common = dict(K_M_old=zero, K_H_old=zero, P_s=zero, N2=zero,
                  l_eps=jnp.full((1, 1), l_eps), dz_half=dz,
                  surface_flux=jnp.zeros((1,)), dt=dt)
    be = float(np.asarray(_solve_tke_backward_euler(
        e_old=e, cfg=TKEConfig(c_eps=c_eps, tke_background=0.0,
                               tke_surface_min=0.0), **common)).ravel()[0])
    split = float(np.asarray(_solve_tke_backward_euler(
        e_old=e, cfg=TKEConfig(c_eps=c_eps, tke_background=0.0,
                               tke_surface_min=0.0,
                               dissipation_discretization="nemo_1p5_split"),
        **common)).ravel()[0])
    assert np.isclose(be, e0 / (1.0 + a), rtol=1e-6)
    assert np.isclose(split, e0 * (1.0 + 0.5 * a) / (1.0 + 1.5 * a), rtol=1e-6)


def test_unknown_dissipation_discretization_raises():
    """Dispatch hardening: an unknown value fails loudly."""
    with pytest.raises(ValueError, match="dissipation_discretization"):
        _solve("bogus_scheme")


# ---------------------------------------------------------------------------
# T6 (Phase-2 #1317): buoyancy-sink discretization — nemo_explicit vs
# implicit_linearized. NEMO subtracts avt*rn2*dt fully EXPLICITLY using the
# PREVIOUS-step avt (zdftke.F90:417-418); legoESM's default linearises the
# stable-branch sink implicitly on the diagonal.
# ---------------------------------------------------------------------------


def _solve_buoyancy(disc, N2_val):
    """Isolated 1-interface decay under a buoyancy sink ONLY (no shear, no
    diffusion, ZERO dissipation via c_eps=0) — the exact regime
    zdftke.F90:417-418 targets."""
    e0 = 4e-4
    dt, K_H = 3600.0, 2.0e-3
    e = jnp.asarray([[e0]])
    zero = jnp.zeros((1, 1))
    K = jnp.full((1, 1), K_H)
    dz = jnp.full((1, 1), 10.0)
    cfg = TKEConfig(tke_background=0.0, tke_surface_min=0.0,
                    tke_buoyancy_sink=disc, c_eps=0.0)
    return float(np.asarray(_solve_tke_backward_euler(
        e_old=e, K_M_old=K, K_H_old=K, P_s=zero,
        N2=jnp.full((1, 1), N2_val), l_eps=jnp.full((1, 1), 1e6),
        dz_half=dz, surface_flux=jnp.zeros((1,)), dt=dt, cfg=cfg,
    )).ravel()[0])


def test_nemo_explicit_matches_hand_factor_single_cell():
    """Isolated 1-interface stable-stratification decay: implicit_linearized
    gives e0/(1+dt*K_H*N2/e0) (the rate linearised per unit e_new);
    nemo_explicit gives e0 - dt*K_H*N2 EXACTLY (the whole term explicit,
    zdftke.F90:417-418, using the previous-step K_H_old) — re-derived by
    hand here, not copy-pasted from the implementation under test."""
    e0, dt, K_H, N2 = 4e-4, 3600.0, 2.0e-3, 1e-5
    implicit = _solve_buoyancy("implicit_linearized", N2)
    explicit = _solve_buoyancy("nemo_explicit", N2)
    expected_implicit = e0 / (1.0 + dt * K_H * N2 / e0)
    expected_explicit = e0 - dt * K_H * N2
    assert np.isclose(implicit, expected_implicit, rtol=1e-10)
    assert np.isclose(explicit, expected_explicit, rtol=1e-10)
    # SELF-REVIEW sign check: under stable stratification (N2>0) BOTH
    # discretizations must DAMP TKE (e_new < e_old) — a buoyancy sink that
    # grows TKE under stable stratification would be a sign-convention bug.
    assert implicit < e0
    assert explicit < e0


def test_nemo_explicit_changes_result_and_stays_finite():
    implicit = _solve_buoyancy("implicit_linearized", 1e-4)
    explicit = _solve_buoyancy("nemo_explicit", 1e-4)
    assert np.isfinite(explicit)
    assert not np.isclose(implicit, explicit)


def test_nemo_explicit_floor_catches_large_sink_overshoot():
    """NEMO relies on the POST-solve floor MAX(en,rn_emin) to catch a large
    explicit sink driving en negative (no implicit clamp on this branch,
    unlike implicit_linearized's diagonal safety) — verify the floor
    actually engages (this function's tail, unconditional)."""
    e0, dt, K_H, N2 = 1e-6, 3600.0, 1.0, 1.0   # sink >> e0 -> would go deeply negative
    cfg = TKEConfig(tke_buoyancy_sink="nemo_explicit")
    out = float(np.asarray(_solve_tke_backward_euler(
        e_old=jnp.asarray([[e0]]), K_M_old=jnp.full((1, 1), K_H),
        K_H_old=jnp.full((1, 1), K_H), P_s=jnp.zeros((1, 1)),
        N2=jnp.full((1, 1), N2), l_eps=jnp.full((1, 1), 1e6),
        dz_half=jnp.full((1, 1), 10.0), surface_flux=jnp.zeros((1,)),
        dt=dt, cfg=cfg,
    )).ravel()[0])
    # The tail floors at tke_background everywhere, THEN re-floors the
    # surface interface (index 0, which this single-cell array IS) at the
    # (larger) tke_surface_min — so the surface value clamps there.
    assert out == pytest.approx(cfg.tke_surface_min)
    assert out >= 0.0


def test_unknown_buoyancy_sink_raises():
    """Dispatch hardening: an unknown value fails loudly."""
    with pytest.raises(ValueError, match="tke_buoyancy_sink"):
        _solve_buoyancy("bogus_scheme", 1e-5)


def test_buoyancy_sink_default_is_byte_identical():
    """implicit_linearized (default) reproduces the prior (pre-T6) closure
    exactly — dropping tke_buoyancy_sink entirely is bit-identical."""
    base = _solve("backward_euler")   # uses the TKEConfig default throughout
    explicit_default = np.asarray(_solve_tke_backward_euler(
        e_old=jnp.asarray(1e-3 * np.abs(
            np.random.default_rng(1).standard_normal(8)) + 1e-4)[None, :],
        K_M_old=jnp.full((1, 8), 1e-3), K_H_old=jnp.full((1, 8), 1e-3),
        P_s=jnp.full((1, 8), 1e-6), N2=jnp.full((1, 8), 1e-4),
        l_eps=jnp.full((1, 8), 5.0), dz_half=jnp.full((1, 8), 10.0),
        surface_flux=jnp.asarray([1e-6]), dt=3600.0,
        cfg=TKEConfig(tke_buoyancy_sink="implicit_linearized"),
    ))
    np.testing.assert_array_equal(base, explicit_default)


def test_nemo_dino_kamm_recipe_assembles_faithful_tke():
    """The nemo_dino_kamm(+_mlf) card produces the NEMO closure identity; every
    other DINO recipe keeps the prior (byte-identical) TKE defaults.

    Phase-2 #1317 extends this pin with the Tier A/B/C axes: T3 surface BC
    placement, T23 background composition, T6 buoyancy-sink discretization,
    T5 step-entry N² + T25 two-level EVD trigger, T8 NEMO-exact Prandtl,
    T18/T19 mixing-length floors, T15 bottom TKE BC."""
    from legoesm.ocean.experiments.dino import (
        DINO_RECIPES, _dino_vertical_mixing_config, dino_config_for_recipe,
    )
    from legoesm.ocean.physics.vertical_mixing.tke import _mixing_length_floor
    for card in ("nemo_dino_kamm", "nemo_dino_kamm_mlf"):
        cfg = dino_config_for_recipe(card)
        vm = _dino_vertical_mixing_config(cfg)
        t = vm.tke
        assert t.prognostic is True
        assert t.tke_mxl_choice == 3
        assert t.surface_bc == "nemo_dirichlet"
        assert t.dissipation_discretization == "nemo_1p5_split"
        assert t.kappa_convention == "veros_sqrte"
        assert t.alpha_tke == 1.0
        assert t.n2_mode == "nemo_bn2"
        expected_floor = np.float64(1.0e-6) / (
            np.float64(t.c_k) * np.sqrt(np.float64(t.tke_background)))
        assert np.float64(_mixing_length_floor(t)).view(np.uint64) == \
            expected_floor.view(np.uint64)
        assert _mixing_length_floor(t) != 0.01
        assert abs(t.c_k - 0.1) < 1e-12            # rn_ediff
        assert abs(t.c_eps - 0.7) < 1e-12         # rn_ediss
        # Phase-2 #1317 Tier A (the ranked structural suspects).
        assert t.tke_surface_bc_level == "nemo_z0"          # T3
        assert vm.vmix_background_mode == "nemo_max_floor"  # T23
        assert t.tke_buoyancy_sink == "nemo_explicit"       # T6
        # Phase-2 #1317 Tier B (one-line card flips).
        assert t.n2_before_advection is True                # T5
        assert cfg.convection_two_level_trigger is True     # T25 (on DINOConfig)
        # Phase-2 #1317 Tier C.
        assert t.prandtl_mode == "nemo_ri"                  # T8 (NEMO-exact, not Veros)
        assert t.bottom_tke_bc is True                      # T15
        # Gap-closure (final): T21 — no avm ceiling on either kamm card.
        assert t.kappaM_max == float("inf")                 # T21
    # T4/T8/T13 — MLF-only axes (require the carried leap-frog before-state):
    # the FE card keeps the fidelity ceiling, the MLF card enables them.
    # #1226 sh2_walk.py Candidate E/F: the MLF card now selects the FULL
    # face-native transcription (supersedes the earlier now×before-only
    # "nemo_burchard" fix -- see the DINO_RECIPES["nemo_dino_kamm_mlf"]
    # comment in dino.py).
    fe = _dino_vertical_mixing_config(dino_config_for_recipe("nemo_dino_kamm")).tke
    mlf = _dino_vertical_mixing_config(
        dino_config_for_recipe("nemo_dino_kamm_mlf")).tke
    assert fe.tke_shear_production == "squared_centered"
    assert fe.tke_n2_time_level == "step_entry"
    assert mlf.tke_shear_production == "nemo_face_native"
    assert mlf.tke_n2_time_level == "nemo_before"
    # #1455: avm INSIDE the zdfsh2 face sum (p_avm(ji+1)+p_avm(ji),
    # zdfsh2.F90:80). MLF-only (needs "nemo_face_native"), so the FE card
    # keeps "tpoint". Measured owner of the pdlr row's residual: legoESM's
    # Z = p_avm/(p_sh2+rn_bshear) ran a FLAT median 2.0114x NEMO's under
    # "tpoint", 1.0000x under "nemo_face".
    assert fe.tke_shear_avm_weighting == "tpoint"
    assert mlf.tke_shear_avm_weighting == "nemo_face"
    # #1317 S17 -- zdfevd trigger arms at NEMO's own time levels (rn2 on Nnn
    # tracers, rn2b on Nbb tracers, BOTH on Nnn geometry: zdfevd.F90:93-94 /
    # :119-120 fed by MY_SRC/stpmlf.F90:186-187). MLF-only (it consumes the
    # genuine Nbb level that tke_n2_time_level="nemo_before" threads), and
    # asserted on the ASSEMBLED EnhancedDiffusionConfig, not the card dict.
    from legoesm.ocean.experiments.dino import (
        dino_lat_lon_grid, dino_lat_lon_model_config,
    )

    def _ed_of(card):
        c = dino_config_for_recipe(card)
        _, phys = dino_lat_lon_model_config(dino_lat_lon_grid(c, n_lon=8), c)
        return phys.convection.enhanced_diffusion

    assert _ed_of("nemo_dino_kamm_mlf").evd_n2_time_level == "nemo_now_before"
    assert _ed_of("nemo_dino_kamm_mlf").two_level_trigger is True
    assert _ed_of("nemo_dino_kamm").evd_n2_time_level == "solver_state"
    for card in set(DINO_RECIPES) - {"nemo_dino_kamm_mlf"}:
        assert dino_config_for_recipe(card).convection_evd_n2_time_level == \
            "solver_state", card
    for card in set(DINO_RECIPES) - {"nemo_dino_kamm", "nemo_dino_kamm_mlf"}:
        vm = _dino_vertical_mixing_config(dino_config_for_recipe(card))
        if getattr(vm, "tke", None) is None:
            continue
        t = vm.tke
        assert t.tke_mxl_choice == 2
        assert t.surface_bc == "veros_flux"
        assert t.dissipation_discretization == "backward_euler"
        assert t.kappa_convention == "gaspar_sqrt2e"
        assert t.prognostic is False
        # Every non-kamm recipe stays byte-identical on the Phase-2 axes too.
        assert t.tke_surface_bc_level == "interior_pinned"
        assert vm.vmix_background_mode == "additive"
        assert t.tke_buoyancy_sink == "implicit_linearized"
        assert t.n2_before_advection is False
        assert t.bottom_tke_bc is False
        assert abs(t.mxl_min - 1e-8) < 1e-12
        assert t.kappaM_max == 100.0                 # T21: unchanged elsewhere
        assert t.tke_shear_production == "squared_centered"
        assert t.tke_n2_time_level == "step_entry"


# ---------------------------------------------------------------------------
# T21 (Phase-2 #1317): NEMO's tke_avn has NO avm ceiling — kappaM_max=inf
# leaves K_M unbounded above (formula-level test; the card wiring is pinned
# in test_nemo_dino_kamm_recipe_assembles_faithful_tke above).
# ---------------------------------------------------------------------------


class TestNoAvmCeiling:
    def test_kappaM_max_inf_leaves_K_M_unbounded(self):
        from legoesm.ocean.physics.vertical_mixing.tke import (
            compute_K_from_tke,
        )
        cfg_finite = TKEConfig(prandtl_mode="constant", kappaM_max=100.0,
                               kappaM_min=1e-6, kappaH_min=1e-6)
        cfg_inf = cfg_finite._replace(kappaM_max=float("inf"))
        e = jnp.asarray([1e8])       # deliberately huge TKE
        l_k = jnp.asarray([1e4])     # deliberately huge mixing length
        N2 = jnp.asarray([1e-5])
        shear_sq = jnp.asarray([1e-4])
        K_M_finite, _ = compute_K_from_tke(
            e, l_k, cfg_finite, N2=N2, shear_sq=shear_sq)
        K_M_inf, _ = compute_K_from_tke(
            e, l_k, cfg_inf, N2=N2, shear_sq=shear_sq)
        assert float(K_M_finite[0]) == pytest.approx(100.0)   # ceiling bound
        assert float(K_M_inf[0]) > 1e6                        # unbounded
        assert bool(np.isfinite(np.asarray(K_M_inf)[0]))

    def test_default_kappaM_max_unchanged(self):
        assert TKEConfig().kappaM_max == 100.0


# ---------------------------------------------------------------------------
# T4 (Phase-2 #1317): Burchard now×before shear (zdfsh2.F90:44-92).
# ---------------------------------------------------------------------------


class TestBurchardShear:
    def test_formula_matches_hand_derivation(self):
        """sh2 = (du_now/dz)(du_before/dz) + (dv_now/dz)(dv_before/dz) —
        re-derived by hand from zdfsh2.F90, not copy-pasted."""
        from legoesm.ocean.physics.vertical_mixing._shared import (
            vertical_shear_burchard,
        )
        u_now = jnp.asarray([1.0, 2.0, 4.0])[None, None, :]
        v_now = jnp.asarray([0.0, 0.0, 0.0])[None, None, :]
        u_before = jnp.asarray([1.0, 1.5, 3.0])[None, None, :]
        v_before = jnp.asarray([0.0, 1.0, 2.0])[None, None, :]
        dz_half = jnp.full((1, 1, 2), 2.0)
        out = np.asarray(vertical_shear_burchard(
            u_now, v_now, u_before, v_before, dz_half))
        du_now = np.asarray([1.0, 2.0]) / 2.0
        dv_now = np.asarray([0.0, 0.0]) / 2.0
        du_before = np.asarray([0.5, 1.5]) / 2.0
        dv_before = np.asarray([1.0, 1.0]) / 2.0
        expected = du_now * du_before + dv_now * dv_before
        np.testing.assert_allclose(out[0, 0], expected, rtol=1e-12)

    def test_can_be_negative_unlike_squared_form(self):
        """Genuine feature of the energy-conserving cross term: opposite-sign
        now/before gradients give a NEGATIVE shear production."""
        from legoesm.ocean.physics.vertical_mixing._shared import (
            vertical_shear_burchard,
        )
        u_now = jnp.asarray([0.0, 1.0])[None, None, :]     # du/dz > 0
        u_before = jnp.asarray([1.0, 0.0])[None, None, :]  # du/dz < 0
        v = jnp.zeros((1, 1, 2))
        dz_half = jnp.full((1, 1, 1), 1.0)
        out = float(np.asarray(
            vertical_shear_burchard(u_now, v, u_before, v, dz_half))[0, 0, 0])
        assert out < 0.0

    def test_zero_shear_zero_production(self):
        from legoesm.ocean.physics.vertical_mixing._shared import (
            vertical_shear_burchard,
        )
        u = jnp.full((1, 1, 4), 2.0)
        v = jnp.full((1, 1, 4), -1.0)
        dz_half = jnp.full((1, 1, 3), 5.0)
        out = np.asarray(vertical_shear_burchard(u, v, u, v, dz_half))
        np.testing.assert_allclose(out, 0.0, atol=1e-15)



    def test_tke_shear_production_dispatch_and_gates(self):
        """Wired into tke_vertical_mixing: nemo_burchard changes K_M vs
        squared_centered (a strongly-sheared now/before pair, checked
        through K_M rather than tke_new so the comparison is not swamped by
        the tke_background/tke_surface_min floors), requires
        u_before_cell/v_before_cell, and an unknown value / silent-no-op
        both raise."""
        u, v, T, S, rho, dz_half, z_int, tx, ty = _orchestrator_inputs()
        # LARGE velocities (not the tiny 0.05 m/s noise) with an
        # opposite-signed now/before shear, and tke_old seeded well above
        # the background floor, so the Burchard cross term genuinely
        # differs from the squared form in the FINAL solved TKE (not just
        # transiently within the Mode-B iteration).
        u = 2.0 * u
        v = 2.0 * v
        u_before = -u
        v_before = -v
        tke_seed = jnp.full(dz_half.shape, 1e-2)
        common = dict(
            u_cell=u, v_cell=v, T_cell=T, S_cell=S, rho_cell=rho,
            dz_half=dz_half, tke_old=tke_seed, tau_x_surface=tx,
            tau_y_surface=ty, dt=3600.0, rho_0=_RHO0, n_iterations=1,
            z_interface=z_int,
        )
        default_out = tke_vertical_mixing(cfg=TKEConfig(), **common)
        burchard_out = tke_vertical_mixing(
            cfg=TKEConfig(tke_shear_production="nemo_burchard"),
            u_before_cell=u_before, v_before_cell=v_before, **common)
        assert not np.allclose(
            np.asarray(default_out.tke_new), np.asarray(burchard_out.tke_new))
        assert bool(np.all(np.isfinite(np.asarray(burchard_out.tke_new))))

        with pytest.raises(ValueError, match="nemo_burchard"):
            tke_vertical_mixing(
                cfg=TKEConfig(tke_shear_production="nemo_burchard"), **common)
        with pytest.raises(ValueError, match="squared_centered"):
            tke_vertical_mixing(
                cfg=TKEConfig(), u_before_cell=u_before,
                v_before_cell=v_before, **common)
        with pytest.raises(ValueError, match="tke_shear_production"):
            tke_vertical_mixing(
                cfg=TKEConfig(tke_shear_production="bogus"), **common)

    def test_default_is_byte_identical(self):
        u, v, T, S, rho, dz_half, z_int, tx, ty = _orchestrator_inputs()
        common = dict(
            u_cell=u, v_cell=v, T_cell=T, S_cell=S, rho_cell=rho,
            dz_half=dz_half, tke_old=None, tau_x_surface=tx,
            tau_y_surface=ty, dt=3600.0, rho_0=_RHO0, n_iterations=3,
            z_interface=z_int,
        )
        base = tke_vertical_mixing(cfg=TKEConfig(), **common)
        explicit = tke_vertical_mixing(
            cfg=TKEConfig(tke_shear_production="squared_centered"), **common)
        np.testing.assert_array_equal(
            np.asarray(base.tke_new), np.asarray(explicit.tke_new))


# ---------------------------------------------------------------------------
# T8/T13 (Phase-2 #1317): rn2b (true leap-frog BEFORE/Nbb) for Prandtl zri
# and the Langmuir PE integral — genuinely different from rn2 (Nnow).
# ---------------------------------------------------------------------------


def _n2b_test_column():
    """A column with n2_mode='adiabatic' inputs (T_n2b/S_n2b are only
    consulted when N2 is a function of T/S — 'insitu' N2 depends on rho_cell
    ALONE and would make this a no-op test regardless of wiring)."""
    u, v, T, S, rho, dz_half, z_int, tx, ty = _orchestrator_inputs()
    nlev = T.shape[-1]
    p_cell = jnp.full(T.shape, 1.0e6)   # a representative interior pressure
    dz_ref = jnp.full((nlev,), 25.0)
    jacobian = jnp.ones(T.shape[:-1])
    return u, v, T, S, rho, dz_half, z_int, tx, ty, p_cell, dz_ref, jacobian


# ---------------------------------------------------------------------------
# #1226 sh2_walk.py Candidate E/F: face-native shear (zdfsh2.F90:78-94) —
# vertical_shear_face_native. The dominant gap the walk isolated: NEMO
# differences u/v at their NATIVE C-grid faces (one vertical diff PER face)
# and combines the two faces bracketing each T-point (wet-only
# coast-doubling weight) ONLY AFTER squaring/cross-multiplying, instead of
# collapsing u/v to the T-point BEFORE differencing.
# ---------------------------------------------------------------------------
def _nemo_zdfsh2_reference_loop(u_now, v_now, u_bef, v_bef, dz_half,
                                u_mask, v_mask):
    """Independent from-scratch pure-Python/NumPy transcription of
    zdfsh2.F90:78-94 (no-Stokes branch), adapted to legoESM's
    west-face-of-T(i)/south-face-of-T(j) C-grid convention (T(i) is
    bracketed by faces i (west) and i+1 (east) -- the mirror image of
    NEMO's literal (i-1, i) east-face-of-T(i) indexing).

    NOT a call into ``vertical_shear_face_native`` -- explicit triple
    Python loop over (j, i, k), re-deriving the formula term by term from
    the NEMO source quoted in that function's docstring, so this is an
    independent check of the vectorised implementation, not a tautology.
    """
    n_lat, n_lon, n_int = dz_half.shape
    nlev = n_int + 1
    dz_sq = dz_half * dz_half
    # Pad each T column's dz_sq onto its adjacent (n_lon+1 / n_lat+1) face
    # array by repeating the last column/row (see the function docstring's
    # "no separate u/v-face metric" scope note).
    dz_sq_u = np.concatenate([dz_sq, dz_sq[:, -1:, :]], axis=1)
    dz_sq_v = np.concatenate([dz_sq, dz_sq[-1:, :, :]], axis=0)
    wumask = u_mask[..., :-1] * u_mask[..., 1:]
    wvmask = v_mask[..., :-1] * v_mask[..., 1:]
    p_sh2 = np.zeros((n_lat, n_lon, n_int))
    for j in range(n_lat):
        for i in range(n_lon):
            for k in range(n_int):
                zsh2u = {}
                for ii in (i, i + 1):
                    du_now = u_now[j, ii, k] - u_now[j, ii, k + 1]
                    du_bef = u_bef[j, ii, k] - u_bef[j, ii, k + 1]
                    zsh2u[ii] = (du_now * du_bef / dz_sq_u[j, ii, k]
                                * wumask[j, ii, k])
                zsh2v = {}
                for jj in (j, j + 1):
                    dv_now = v_now[jj, i, k] - v_now[jj, i, k + 1]
                    dv_bef = v_bef[jj, i, k] - v_bef[jj, i, k + 1]
                    zsh2v[jj] = (dv_now * dv_bef / dz_sq_v[jj, i, k]
                                * wvmask[jj, i, k])
                coast_u = 2.0 - u_mask[j, i, k + 1] * u_mask[j, i + 1, k + 1]
                coast_v = 2.0 - v_mask[j, i, k + 1] * v_mask[j + 1, i, k + 1]
                # 0.5, NOT NEMO's literal 0.25: this reference drops NEMO's
                # ``avm(ji+1)+avm(ji)`` face SUM (= 2*mi(avm), zdfsh2.F90:80)
                # so that the caller can multiply by its own single K_M, and
                # the 0.25 at :93 is paired with that SUM. Keeping 0.25 here
                # applies the halving twice -- which is exactly the bug this
                # "independent" reference reproduced until 2026-08 (it was
                # written from :93 alone and so could never catch it; the
                # absolute-normalisation anchor below is what does).
                p_sh2[j, i, k] = 0.5 * (
                    (zsh2u[i] + zsh2u[i + 1]) * coast_u
                    + (zsh2v[j] + zsh2v[j + 1]) * coast_v
                )
    return p_sh2


def _face_masks_3d_reference(is_active):
    """Independent NumPy re-derivation of ``compute_face_masks_3d`` (a
    face is wet at level k iff BOTH adjacent T-cells are wet at that
    level; boundary faces are walls) -- used only to BUILD test fixtures,
    not to validate the production mask helper (that helper is exercised
    directly via ``compute_face_masks_3d`` in the wiring test below)."""
    a = is_active.astype(np.float64)
    u_interior = a * np.roll(a, 1, axis=1)
    u_mask = np.concatenate([u_interior, u_interior[:, 0:1, :]], axis=1)
    v_interior = a[:-1] * a[1:]
    south = np.zeros_like(a[:1])
    north = np.zeros_like(south)
    v_mask = np.concatenate([south, v_interior, north], axis=0)
    return u_mask, v_mask


class TestFaceNativeShear:
    def _synthetic_partial_cell_case(self, seed=0, n_lat=4, n_lon=5, nlev=6):
        """A domain with VARYING per-column bottom levels, so wumask/wvmask
        and the coast-doubling weight are genuinely non-trivial (exercises
        Test 2 -- the coast/mask path)."""
        rng = np.random.default_rng(seed)
        u_face = rng.standard_normal((n_lat, n_lon + 1, nlev))
        v_face = rng.standard_normal((n_lat + 1, n_lon, nlev))
        u_face_b = rng.standard_normal((n_lat, n_lon + 1, nlev))
        v_face_b = rng.standard_normal((n_lat + 1, n_lon, nlev))
        dz_half = np.abs(rng.uniform(5.0, 20.0, (n_lat, n_lon, nlev - 1)))
        bottom_level = rng.integers(2, nlev, size=(n_lat, n_lon))
        k = np.arange(nlev).reshape(1, 1, nlev)
        is_active = k <= bottom_level[:, :, None]
        u_mask, v_mask = _face_masks_3d_reference(is_active)
        return (u_face, v_face, u_face_b, v_face_b, dz_half, u_mask, v_mask,
                is_active)

    def test_uniform_shear_normalisation_matches_siblings(self):
        """ABSOLUTE-normalisation anchor (#1455 F2, 2026-08).

        The three ``tke_shear_production`` discretizations are alternative
        DISCRETIZATIONS of the SAME quantity, so on a horizontally-uniform
        column with a LINEAR vertical profile they must all return the
        analytic ``(du/dz)^2 + (dv/dz)^2`` -- there is no discretization
        freedom left to disagree about.  ``vertical_shear_face_native``
        returned exactly HALF of it (NEMO's literal 0.25 T-point prefactor
        kept while the ``avm(i+1)+avm(i)`` face SUM it is paired with was
        dropped), which silently halved BOTH the TKE shear source and the
        ``prandtl_mode="nemo_ri"`` denominator.

        NON-VACUOUS: reverting the prefactor to 0.25 makes the third
        assertion read 0.5 * analytic and this test fails.  The two
        reference-loop tests below CANNOT catch it -- their reference was
        transcribed from zdfsh2.F90:93 alone and reproduces the same
        omission (see ``_nemo_zdfsh2_reference_loop``).
        """
        from legoesm.ocean.physics.vertical_mixing._shared import (
            vertical_shear_squared, vertical_shear_burchard,
            vertical_shear_face_native, avm_weighted_shear_production,
        )
        n_lat, n_lon, nlev = 3, 4, 5
        dz, dudz, dvdz = 10.0, 0.03, -0.02
        z = np.arange(nlev) * dz
        u_cell = jnp.asarray(np.broadcast_to(
            dudz * z, (n_lat, n_lon, nlev)).copy())
        v_cell = jnp.asarray(np.broadcast_to(
            dvdz * z, (n_lat, n_lon, nlev)).copy())
        u_face = jnp.asarray(np.broadcast_to(
            dudz * z, (n_lat, n_lon + 1, nlev)).copy())
        v_face = jnp.asarray(np.broadcast_to(
            dvdz * z, (n_lat + 1, n_lon, nlev)).copy())
        dz_half = jnp.full((n_lat, n_lon, nlev - 1), dz)
        u_mask = jnp.ones((n_lat, n_lon + 1, nlev))
        v_mask = jnp.ones((n_lat + 1, n_lon, nlev))
        analytic = dudz ** 2 + dvdz ** 2

        sq = np.asarray(vertical_shear_squared(u_cell, v_cell, dz_half))
        bu = np.asarray(vertical_shear_burchard(
            u_cell, v_cell, u_cell, v_cell, dz_half))
        fn = np.asarray(vertical_shear_face_native(
            u_face, v_face, u_face, v_face, dz_half, u_mask, v_mask))
        np.testing.assert_allclose(sq, analytic, rtol=1e-12,
                                   err_msg="vertical_shear_squared")
        np.testing.assert_allclose(bu, analytic, rtol=1e-12,
                                   err_msg="vertical_shear_burchard")
        np.testing.assert_allclose(
            fn, analytic, rtol=1e-12,
            err_msg="vertical_shear_face_native is NOT normalised like its "
                    "two siblings in the same tke_shear_production dispatch "
                    "(a 0.25 prefactor here returns exactly half)")

        # ...and the avm-weighted form with UNIFORM K_M is that same
        # analytic shear times K_M -- pinning NEMO's own p_sh2/avm scale.
        K0 = 7.5
        p = np.asarray(avm_weighted_shear_production(
            u_face, v_face, u_face, v_face, dz_half, u_mask, v_mask,
            jnp.full((n_lat, n_lon, nlev - 1), K0)))
        np.testing.assert_allclose(p, K0 * analytic, rtol=1e-12)

    def test_face_native_matches_independent_reference_loop(self):
        """Test 1 (task spec): synthetic 3-D u/v field, asserted against an
        independent in-test transcription of zdfsh2.F90's formula (pure
        NumPy loops, NOT a call into the function under test)."""
        from legoesm.ocean.physics.vertical_mixing._shared import (
            vertical_shear_face_native,
        )
        (u_face, v_face, u_face_b, v_face_b, dz_half, u_mask, v_mask,
         _is_active) = self._synthetic_partial_cell_case()
        ref = _nemo_zdfsh2_reference_loop(
            u_face, v_face, u_face_b, v_face_b, dz_half, u_mask, v_mask)
        out = np.asarray(vertical_shear_face_native(
            jnp.asarray(u_face), jnp.asarray(v_face),
            jnp.asarray(u_face_b), jnp.asarray(v_face_b),
            jnp.asarray(dz_half), jnp.asarray(u_mask), jnp.asarray(v_mask)))
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref, rtol=0, atol=1e-12)

    def test_coast_doubling_exercised_and_matches_reference(self):
        """Test 2 (task spec): a column adjacent to land must exercise the
        "2 - mask*mask" doubling; assert it against the in-test reference.

        Builds a case with an EXPLICIT single-cell island (one T column
        dry at every level, all its neighbours wet) so the u-/v-faces
        touching the island have coast_u/coast_v == 2.0 (not 1.0) by
        direct construction -- not merely "some random mask happens to
        produce a coast somewhere".
        """
        from legoesm.ocean.physics.vertical_mixing._shared import (
            vertical_shear_face_native,
        )
        n_lat, n_lon, nlev = 4, 5, 3
        rng = np.random.default_rng(1)
        u_face = rng.standard_normal((n_lat, n_lon + 1, nlev))
        v_face = rng.standard_normal((n_lat + 1, n_lon, nlev))
        u_face_b = rng.standard_normal((n_lat, n_lon + 1, nlev))
        v_face_b = rng.standard_normal((n_lat + 1, n_lon, nlev))
        dz_half = np.full((n_lat, n_lon, nlev - 1), 10.0)
        is_active = np.ones((n_lat, n_lon, nlev), dtype=bool)
        island_j, island_i = 1, 2   # interior column -> 4 wet neighbours
        is_active[island_j, island_i, :] = False
        u_mask, v_mask = _face_masks_3d_reference(is_active)
        # Self-check: the island's own faces (west=island_i, east=
        # island_i+1) each border exactly one dry cell -> coast weight 2.0.
        assert u_mask[island_j, island_i, -1] == 0.0  # island west face dry
        coast_u_west = 2.0 - (u_mask[island_j, island_i - 1, -1]
                              * u_mask[island_j, island_i, -1])
        assert coast_u_west == 2.0, "island west face must hit the coast weight"

        ref = _nemo_zdfsh2_reference_loop(
            u_face, v_face, u_face_b, v_face_b, dz_half, u_mask, v_mask)
        out = np.asarray(vertical_shear_face_native(
            jnp.asarray(u_face), jnp.asarray(v_face),
            jnp.asarray(u_face_b), jnp.asarray(v_face_b),
            jnp.asarray(dz_half), jnp.asarray(u_mask), jnp.asarray(v_mask)))
        np.testing.assert_allclose(out, ref, rtol=0, atol=1e-12)
        # The neighbouring column (island_i - 1) must show the doubled
        # weight actually CHANGED its p_sh2 vs an all-wet control (else the
        # coast branch could be silently dead code that never fires).
        u_mask_allwet, v_mask_allwet = _face_masks_3d_reference(
            np.ones_like(is_active))
        ref_allwet = _nemo_zdfsh2_reference_loop(
            u_face, v_face, u_face_b, v_face_b, dz_half,
            u_mask_allwet, v_mask_allwet)
        assert not np.allclose(
            ref[island_j, island_i - 1], ref_allwet[island_j, island_i - 1]), (
            "coast-doubling weight must change p_sh2 next to the island "
            "(test would pass vacuously if the coast branch were dead)")

    def test_signed_like_burchard(self):
        """Same signed-cross-term feature as vertical_shear_burchard: an
        opposite-signed now/before gradient gives a NEGATIVE production."""
        from legoesm.ocean.physics.vertical_mixing._shared import (
            vertical_shear_face_native,
        )
        # n_lat=1, n_lon=1 T-grid -> u-face shape (1, 2, nlev), v-face
        # shape (2, 1, nlev). BOTH u-faces (west=0, east=1) get the SAME
        # now/before column so the T-combine (average of the two faces)
        # doesn't cancel the sign.
        nlev = 3
        u_col_now = jnp.asarray([0.0, 1.0, 1.0])      # du/dz > 0 at k=0
        u_col_before = jnp.asarray([1.0, 0.0, 0.0])   # du/dz < 0 at k=0
        u_now = jnp.stack([u_col_now, u_col_now])[None, :, :]      # (1,2,3)
        u_before = jnp.stack([u_col_before, u_col_before])[None, :, :]
        v = jnp.zeros((2, 1, nlev))
        dz_half = jnp.full((1, 1, nlev - 1), 1.0)
        mask_u = jnp.ones((1, 2, nlev))
        mask_v = jnp.ones((2, 1, nlev))
        out = np.asarray(vertical_shear_face_native(
            u_now, v, u_before, v, dz_half, mask_u, mask_v))
        assert out[0, 0, 0] < 0.0

    def test_dry_column_produces_zero(self):
        """A fully-dry face pair (both u-faces bracketing T dry) must give
        exactly zero shear production there, regardless of the velocity
        noise on the dry faces (masked, not merely small)."""
        from legoesm.ocean.physics.vertical_mixing._shared import (
            vertical_shear_face_native,
        )
        n_lat, n_lon, nlev = 2, 2, 3
        rng = np.random.default_rng(2)
        u_face = jnp.asarray(rng.standard_normal((n_lat, n_lon + 1, nlev)))
        v_face = jnp.asarray(rng.standard_normal((n_lat + 1, n_lon, nlev)))
        dz_half = jnp.full((n_lat, n_lon, nlev - 1), 10.0)
        u_mask = jnp.zeros((n_lat, n_lon + 1, nlev))
        v_mask = jnp.zeros((n_lat + 1, n_lon, nlev))
        out = np.asarray(vertical_shear_face_native(
            u_face, v_face, u_face, v_face, dz_half, u_mask, v_mask))
        np.testing.assert_allclose(out, 0.0, atol=0.0)


def _nemo_zdfsh2_avm_reference_loop(u_now, v_now, u_bef, v_bef, dz_half,
                                    u_mask, v_mask, kappaM_T):
    """Independent triple-Python-loop transcription of the AVM-WEIGHTED
    zdfsh2.F90:80-94 (#1455 sh2 chain-walk avm-weighting gap): ``avm``
    face-summed (NOT averaged -- NEMO's own :79 comment "2 x shear
    production... energy conserving form") INSIDE the per-face term, before
    the 0.25 T-point coast-doubled combine. NOT a call into
    ``avm_weighted_shear_production`` -- an independent re-derivation, same
    role as ``_nemo_zdfsh2_reference_loop`` above."""
    n_lat, n_lon, n_int = dz_half.shape
    dz_sq = dz_half * dz_half
    dz_sq_u = np.concatenate([dz_sq, dz_sq[:, -1:, :]], axis=1)
    dz_sq_v = np.concatenate([dz_sq, dz_sq[-1:, :, :]], axis=0)
    wumask = u_mask[..., :-1] * u_mask[..., 1:]
    wvmask = v_mask[..., :-1] * v_mask[..., 1:]
    p_sh2 = np.zeros((n_lat, n_lon, n_int))
    for j in range(n_lat):
        for i in range(n_lon):
            for k in range(n_int):
                zsh2u = {}
                for ii in (i, i + 1):
                    du_now = u_now[j, ii, k] - u_now[j, ii, k + 1]
                    du_bef = u_bef[j, ii, k] - u_bef[j, ii, k + 1]
                    # Zonal kappa pairing WRAPS periodically (#1455 review
                    # N1): seam face gets kappa[-1]+kappa[0], never
                    # 2*kappa[edge].
                    avm_l = kappaM_T[j, (ii - 1) % n_lon, k]
                    avm_r = kappaM_T[j, ii % n_lon, k]
                    zsh2u[ii] = ((avm_l + avm_r) * du_now * du_bef
                                / dz_sq_u[j, ii, k] * wumask[j, ii, k])
                zsh2v = {}
                for jj in (j, j + 1):
                    dv_now = v_now[jj, i, k] - v_now[jj, i, k + 1]
                    dv_bef = v_bef[jj, i, k] - v_bef[jj, i, k + 1]
                    avm_s = kappaM_T[jj - 1, i, k] if jj - 1 >= 0 else kappaM_T[jj, i, k]
                    avm_n = kappaM_T[jj, i, k] if jj < n_lat else kappaM_T[jj - 1, i, k]
                    zsh2v[jj] = ((avm_s + avm_n) * dv_now * dv_bef
                                / dz_sq_v[jj, i, k] * wvmask[jj, i, k])
                coast_u = 2.0 - u_mask[j, i, k + 1] * u_mask[j, i + 1, k + 1]
                coast_v = 2.0 - v_mask[j, i, k + 1] * v_mask[j + 1, i, k + 1]
                p_sh2[j, i, k] = 0.25 * (
                    (zsh2u[i] + zsh2u[i + 1]) * coast_u
                    + (zsh2v[j] + zsh2v[j + 1]) * coast_v
                )
    return p_sh2


class TestAvmWeightedShearProduction:
    """#1455 sh2 chain-walk avm-weighting unpark: synthetic-violation tests
    for ``_shared.avm_weighted_shear_production`` (the ``tke_shear_
    avm_weighting="nemo_face"`` option). Both directions per the task
    spec: uniform K_M gives the PREDICTED (not naively "bit-identical")
    relationship to the "tpoint" path, and a strong K_M gradient near a
    coast makes the two paths differ."""

    def _case(self, n_lat=4, n_lon=5, nlev=6, seed=7):
        rng = np.random.default_rng(seed)
        u_face = rng.standard_normal((n_lat, n_lon + 1, nlev)) * 0.1
        v_face = rng.standard_normal((n_lat + 1, n_lon, nlev)) * 0.1
        u_face_b = u_face + rng.standard_normal(u_face.shape) * 0.01
        v_face_b = v_face + rng.standard_normal(v_face.shape) * 0.01
        dz_half = np.full((n_lat, n_lon, nlev - 1), 10.0)
        bottom_level = rng.integers(2, nlev, size=(n_lat, n_lon))
        k = np.arange(nlev).reshape(1, 1, nlev)
        is_active = k <= bottom_level[:, :, None]
        u_mask, v_mask = _face_masks_3d_reference(is_active)
        return u_face, v_face, u_face_b, v_face_b, dz_half, u_mask, v_mask

    def test_matches_independent_reference_loop(self):
        """Vectorised implementation matches an independent from-scratch
        triple-Python-loop transcription (not a call into the function
        under test)."""
        from legoesm.ocean.physics.vertical_mixing._shared import (
            avm_weighted_shear_production,
        )
        u_face, v_face, u_face_b, v_face_b, dz_half, u_mask, v_mask = self._case()
        n_lat, n_lon, n_int = dz_half.shape
        rng = np.random.default_rng(11)
        kappaM_T = np.abs(rng.uniform(0.5, 5.0, (n_lat, n_lon, n_int)))
        ref = _nemo_zdfsh2_avm_reference_loop(
            u_face, v_face, u_face_b, v_face_b, dz_half, u_mask, v_mask, kappaM_T)
        out = np.asarray(avm_weighted_shear_production(
            jnp.asarray(u_face), jnp.asarray(v_face),
            jnp.asarray(u_face_b), jnp.asarray(v_face_b),
            jnp.asarray(dz_half), jnp.asarray(u_mask), jnp.asarray(v_mask),
            jnp.asarray(kappaM_T)))
        np.testing.assert_allclose(out, ref, rtol=0, atol=1e-10)

    def test_uniform_kappaM_matches_tpoint_exactly(self):
        """Synthetic self-consistency control: with SPATIALLY UNIFORM K_M
        the avm-weighted p_sh2 must equal EXACTLY K0*shear_sq_tpoint --
        avm-face-summing is a NO-OP when there is nothing to average.

        CORRECTED 2026-08 (this test previously asserted 2x, and PASSED,
        because ``vertical_shear_face_native`` carried NEMO's literal 0.25
        prefactor while dropping the ``avm(i+1)+avm(i)`` face SUM the 0.25
        is paired with -- i.e. it was half of NEMO.  The 2x was that bug,
        not a property of this function; asserting a RATIO between two
        implementations can never pin either one's absolute scale, which is
        why ``TestFaceNativeShear::
        test_uniform_shear_normalisation_matches_siblings`` now anchors it
        against an analytic profile.)

        Verified against BOTH the all-open-ocean case (coast weight 1.0
        everywhere) and a case with a coastal mask (coast weight 2.0
        exercised) -- the identity holds regardless of the coast-doubling
        weight, since that weight multiplies both formulas identically."""
        from legoesm.ocean.physics.vertical_mixing._shared import (
            vertical_shear_face_native, avm_weighted_shear_production,
        )
        for use_coast_mask in (False, True):
            u_face, v_face, u_face_b, v_face_b, dz_half, u_mask, v_mask = (
                self._case(seed=3))
            if not use_coast_mask:
                u_mask = np.ones_like(u_mask)
                v_mask = np.ones_like(v_mask)
            n_lat, n_lon, n_int = dz_half.shape
            K0 = 2.7
            kappaM_uniform = jnp.full((n_lat, n_lon, n_int), K0)
            shear_sq = vertical_shear_face_native(
                jnp.asarray(u_face), jnp.asarray(v_face),
                jnp.asarray(u_face_b), jnp.asarray(v_face_b),
                jnp.asarray(dz_half), jnp.asarray(u_mask), jnp.asarray(v_mask))
            p_sh2_tpoint = kappaM_uniform * shear_sq
            p_sh2_nemo_face = avm_weighted_shear_production(
                jnp.asarray(u_face), jnp.asarray(v_face),
                jnp.asarray(u_face_b), jnp.asarray(v_face_b),
                jnp.asarray(dz_half), jnp.asarray(u_mask), jnp.asarray(v_mask),
                kappaM_uniform)
            np.testing.assert_allclose(
                np.asarray(p_sh2_nemo_face), np.asarray(p_sh2_tpoint),
                rtol=0, atol=1e-10,
                err_msg=f"use_coast_mask={use_coast_mask}: uniform-K_M "
                        "self-consistency (nemo_face == tpoint) FAILED")

    def test_varying_kappaM_differs_from_tpoint_near_gradient(self):
        """With a STRONG K_M gradient at a coastal mask cell, nemo_face !=
        tpoint (the two formulas only coincide for spatially uniform
        K_M) -- and away from the gradient (flat K_M region) they DO
        coincide, confirming the difference is gradient-driven, not a
        blanket offset.  (Ratio corrected 2x -> 1x, 2026-08; see
        ``test_uniform_kappaM_matches_tpoint_exactly``.)"""
        from legoesm.ocean.physics.vertical_mixing._shared import (
            vertical_shear_face_native, avm_weighted_shear_production,
        )
        u_face, v_face, u_face_b, v_face_b, dz_half, u_mask, v_mask = (
            self._case(seed=5))
        n_lat, n_lon, n_int = dz_half.shape
        K_T = jnp.full((n_lat, n_lon, n_int), 1.0)
        # Sharp local spike + a flat region elsewhere.
        K_T = K_T.at[1, 2, :].set(50.0)
        K_T = K_T.at[1, 1, :].set(0.02)
        shear_sq = vertical_shear_face_native(
            jnp.asarray(u_face), jnp.asarray(v_face),
            jnp.asarray(u_face_b), jnp.asarray(v_face_b),
            jnp.asarray(dz_half), jnp.asarray(u_mask), jnp.asarray(v_mask))
        p_sh2_tpoint = K_T * shear_sq
        p_sh2_nemo_face = avm_weighted_shear_production(
            jnp.asarray(u_face), jnp.asarray(v_face),
            jnp.asarray(u_face_b), jnp.asarray(v_face_b),
            jnp.asarray(dz_half), jnp.asarray(u_mask), jnp.asarray(v_mask),
            K_T)
        diff = np.asarray(p_sh2_nemo_face) - np.asarray(p_sh2_tpoint)
        assert np.max(np.abs(diff)) > 1e-6, (
            "expected nemo_face to DIFFER from tpoint near the K_M "
            "gradient -- test would pass vacuously if they always matched")
        # Far from the gradient (e.g. column (3,4), away from the spike),
        # the two formulas should still coincide since K_M is locally flat
        # there and its neighbours are also flat.
        np.testing.assert_allclose(
            np.asarray(p_sh2_nemo_face)[3, 4, :],
            np.asarray(p_sh2_tpoint)[3, 4, :], rtol=0, atol=1e-10)

    def test_periodic_seam_face_wraps_kappa(self):
        """#1455 review N1: at an ALL-WET zonal-periodic seam, the seam
        u-face kappa pairing must WRAP -- kappa[-1]+kappa[0] -- not
        edge-repeat (2*kappa[0]). Analytic case: v-shear zeroed, uniform
        unit u-shear, kappa = 1+i in lon =>
        p_sh2[:, 0, :] = 0.25*bare*((k[-1]+k[0]) + (k[0]+k[1])).
        FAILS under edge-repeat padding (gives 2*k[0] at the seam)."""
        from legoesm.ocean.physics.vertical_mixing._shared import (
            avm_weighted_shear_production,
        )
        n_lat, n_lon, nlev = 2, 4, 4
        kk = np.arange(nlev)
        u_face = np.broadcast_to(-0.1 * kk, (n_lat, n_lon + 1, nlev)).copy()
        v_face = np.zeros((n_lat + 1, n_lon, nlev))
        dz_half = np.full((n_lat, n_lon, nlev - 1), 10.0)
        u_mask = np.ones((n_lat, n_lon + 1, nlev))
        v_mask = np.ones((n_lat + 1, n_lon, nlev))
        kappa = np.broadcast_to(
            (1.0 + np.arange(n_lon))[None, :, None],
            (n_lat, n_lon, nlev - 1)).copy()
        out = np.asarray(avm_weighted_shear_production(
            jnp.asarray(u_face), jnp.asarray(v_face),
            jnp.asarray(u_face), jnp.asarray(v_face),
            jnp.asarray(dz_half), jnp.asarray(u_mask), jnp.asarray(v_mask),
            jnp.asarray(kappa)))
        bare = (0.1 * 0.1) / (10.0 * 10.0)  # du_now*du_bef / dz_sq per face
        kap = 1.0 + np.arange(n_lon)
        for i in range(n_lon):
            expected = 0.25 * bare * (
                (kap[(i - 1) % n_lon] + kap[i])          # west face (wraps)
                + (kap[i] + kap[(i + 1) % n_lon]))       # east face (wraps)
            np.testing.assert_allclose(
                out[:, i, :], expected, rtol=0, atol=1e-12,
                err_msg=f"lon column {i}: seam/interior kappa wrap wrong")


def _face_native_orchestrator_inputs(n_lat=3, n_lon=4, nlev=6, seed=3):
    """Full ``tke_vertical_mixing``-level fixture for the wiring/dispatch
    tests: RAW C-grid face u/v (now + before), a per-level wet mask pair
    (varying bottom levels -> exercises wumask/wvmask + coast-doubling),
    plus the cell-centred T/S/rho/dz_half/tau the closure also needs."""
    rng = np.random.default_rng(seed)
    u_face = jnp.asarray(rng.standard_normal((n_lat, n_lon + 1, nlev)) * 0.3)
    v_face = jnp.asarray(rng.standard_normal((n_lat + 1, n_lon, nlev)) * 0.3)
    u_face_before = jnp.asarray(
        rng.standard_normal((n_lat, n_lon + 1, nlev)) * 0.3)
    v_face_before = jnp.asarray(
        rng.standard_normal((n_lat + 1, n_lon, nlev)) * 0.3)
    u_cell = 0.5 * (u_face[:, :-1, :] + u_face[:, 1:, :])
    v_cell = 0.5 * (v_face[:-1, :, :] + v_face[1:, :, :])
    u_cell_before = 0.5 * (u_face_before[:, :-1, :] + u_face_before[:, 1:, :])
    v_cell_before = 0.5 * (v_face_before[:-1, :, :] + v_face_before[1:, :, :])
    T = jnp.asarray(20.0 - 2.0 * np.arange(nlev))[None, None, :] * jnp.ones(
        (n_lat, n_lon, nlev))
    S = jnp.full((n_lat, n_lon, nlev), 35.0)
    rho = jnp.asarray(1026.0 + 0.2 * np.arange(nlev))[None, None, :] * jnp.ones(
        (n_lat, n_lon, nlev))
    dz_half = jnp.full((n_lat, n_lon, nlev - 1), 25.0)
    z_interface = -25.0 * jnp.arange(1, nlev)
    tau_x = jnp.full((n_lat, n_lon), 0.1)
    tau_y = jnp.zeros((n_lat, n_lon))
    bottom_level = rng.integers(2, nlev, size=(n_lat, n_lon))
    k = np.arange(nlev).reshape(1, 1, nlev)
    is_active = jnp.asarray(k <= bottom_level[:, :, None])
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d,
    )
    face_masks_3d = compute_face_masks_3d(is_active)
    return dict(
        u_cell=u_cell, v_cell=v_cell, T_cell=T, S_cell=S, rho_cell=rho,
        dz_half=dz_half, tau_x_surface=tau_x, tau_y_surface=tau_y,
        z_interface=z_interface,
        u_before_cell=u_cell_before, v_before_cell=v_cell_before,
        u_face_now=u_face, v_face_now=v_face,
        u_face_before=u_face_before, v_face_before=v_face_before,
        face_masks_3d=face_masks_3d,
    )


class TestFaceNativeShearWiring:
    """End-to-end ``tke_vertical_mixing`` dispatch (Test 3/4, task spec):
    bit-identical default, and unknown-value / silent-no-op / missing-input
    raises through the FULL orchestrator, not just the bare ``_shared.py``
    function (already covered by ``TestFaceNativeShear`` above)."""

    def test_nemo_face_native_changes_result_and_stays_finite(self):
        kwargs = _face_native_orchestrator_inputs()
        tke_seed = jnp.full(kwargs["dz_half"].shape, 1e-2)
        default_out = tke_vertical_mixing(
            cfg=TKEConfig(), tke_old=tke_seed, dt=3600.0, rho_0=_RHO0,
            n_iterations=1,
            **{k: v for k, v in kwargs.items()
               if k not in ("u_before_cell", "v_before_cell", "u_face_now",
                            "v_face_now", "u_face_before", "v_face_before",
                            "face_masks_3d")},
        )
        face_native_out = tke_vertical_mixing(
            cfg=TKEConfig(tke_shear_production="nemo_face_native"),
            tke_old=tke_seed, dt=3600.0, rho_0=_RHO0, n_iterations=1,
            **kwargs,
        )
        assert bool(np.all(np.isfinite(np.asarray(face_native_out.tke_new))))
        assert not np.allclose(
            np.asarray(default_out.tke_new),
            np.asarray(face_native_out.tke_new))

    def test_default_is_byte_identical_through_orchestrator(self):
        """Test 3 (task spec): default scheme value byte-identical pre/post
        -- passing the face-native inputs alongside 'squared_centered'
        must raise (silent-no-op guard), and 'squared_centered' alone must
        reproduce the untouched historical call exactly."""
        kwargs = _face_native_orchestrator_inputs()
        base_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("u_before_cell", "v_before_cell", "u_face_now",
                        "v_face_now", "u_face_before", "v_face_before",
                        "face_masks_3d")
        }
        base = tke_vertical_mixing(
            cfg=TKEConfig(), tke_old=None, dt=3600.0, rho_0=_RHO0,
            n_iterations=3, **base_kwargs)
        explicit = tke_vertical_mixing(
            cfg=TKEConfig(tke_shear_production="squared_centered"),
            tke_old=None, dt=3600.0, rho_0=_RHO0, n_iterations=3,
            **base_kwargs)
        np.testing.assert_array_equal(
            np.asarray(base.tke_new), np.asarray(explicit.tke_new))

    def test_unknown_value_raises(self):
        """Test 4 (task spec): unknown scheme value raises ValueError."""
        kwargs = _face_native_orchestrator_inputs()
        base_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("u_before_cell", "v_before_cell", "u_face_now",
                        "v_face_now", "u_face_before", "v_face_before",
                        "face_masks_3d")
        }
        with pytest.raises(ValueError, match="tke_shear_production"):
            tke_vertical_mixing(
                cfg=TKEConfig(tke_shear_production="bogus"),
                tke_old=None, dt=3600.0, rho_0=_RHO0, n_iterations=1,
                **base_kwargs)

    def test_missing_face_native_inputs_raises(self):
        kwargs = _face_native_orchestrator_inputs()
        base_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("u_before_cell", "v_before_cell", "u_face_now",
                        "v_face_now", "u_face_before", "v_face_before",
                        "face_masks_3d")
        }
        with pytest.raises(ValueError, match="nemo_face_native"):
            tke_vertical_mixing(
                cfg=TKEConfig(tke_shear_production="nemo_face_native"),
                tke_old=None, dt=3600.0, rho_0=_RHO0, n_iterations=1,
                **base_kwargs)

    def test_silent_no_op_guard_squared_centered_with_face_inputs(self):
        kwargs = _face_native_orchestrator_inputs()
        with pytest.raises(ValueError, match="nemo_face_native"):
            tke_vertical_mixing(
                cfg=TKEConfig(), tke_old=None, dt=3600.0, rho_0=_RHO0,
                n_iterations=1, **kwargs)


class TestAvmWeightingWiring:
    """#1455 sh2 chain-walk avm-weighting unpark: end-to-end
    ``tke_vertical_mixing`` dispatch for ``tke_shear_avm_weighting``
    ("tpoint" default | "nemo_face"). Wired on the ``nemo_dino_kamm_mlf``
    card (#1455 flip); the card-level assertion lives in
    :func:`test_kamm_card_selects_nemo_zdftke_identity` above."""

    def test_default_is_byte_identical(self):
        """tke_shear_avm_weighting="tpoint" (implicit default) must be
        BIT-IDENTICAL to a TKEConfig that never mentions the field."""
        kwargs = _face_native_orchestrator_inputs()
        base_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("u_before_cell", "v_before_cell", "u_face_now",
                        "v_face_now", "u_face_before", "v_face_before",
                        "face_masks_3d")
        }
        base = tke_vertical_mixing(
            cfg=TKEConfig(), tke_old=None, dt=3600.0, rho_0=_RHO0,
            n_iterations=3, **base_kwargs)
        explicit = tke_vertical_mixing(
            cfg=TKEConfig(tke_shear_avm_weighting="tpoint"),
            tke_old=None, dt=3600.0, rho_0=_RHO0, n_iterations=3,
            **base_kwargs)
        np.testing.assert_array_equal(
            np.asarray(base.tke_new), np.asarray(explicit.tke_new))

    def test_unknown_value_raises(self):
        kwargs = _face_native_orchestrator_inputs()
        base_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("u_before_cell", "v_before_cell", "u_face_now",
                        "v_face_now", "u_face_before", "v_face_before",
                        "face_masks_3d")
        }
        with pytest.raises(ValueError, match="tke_shear_avm_weighting"):
            tke_vertical_mixing(
                cfg=TKEConfig(tke_shear_avm_weighting="bogus"),
                tke_old=None, dt=3600.0, rho_0=_RHO0, n_iterations=1,
                **base_kwargs)

    def test_nemo_face_requires_nemo_face_native_shear(self):
        """'nemo_face' avm weighting without the matching face-native
        shear geometry must raise -- avm-weighting a T-collapsed shear_sq
        would double-apply the T-point combine."""
        kwargs = _face_native_orchestrator_inputs()
        base_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("u_before_cell", "v_before_cell", "u_face_now",
                        "v_face_now", "u_face_before", "v_face_before",
                        "face_masks_3d")
        }
        with pytest.raises(ValueError, match="tke_shear_production"):
            tke_vertical_mixing(
                cfg=TKEConfig(tke_shear_avm_weighting="nemo_face"),
                tke_old=None, dt=3600.0, rho_0=_RHO0, n_iterations=1,
                **base_kwargs)

    def test_nemo_face_changes_result_and_stays_finite(self):
        """"nemo_face" differs from "tpoint" exactly where K_M varies
        HORIZONTALLY, and coincides where it does not.

        Both halves matter (corrected 2026-08).  This test used to seed a
        spatially UNIFORM ``tke_old`` and still see a difference -- but that
        difference was the ``vertical_shear_face_native`` factor-2 halving,
        not the avm face-weighting.  With the prefactor fixed, a uniform K_M
        makes the two paths bit-identical (the avm SUM has nothing to
        average), so the "changes result" half now needs a real horizontal
        K_M gradient, which is what this option actually transcribes.
        """
        kwargs = _face_native_orchestrator_inputs()
        cfg_tpoint = TKEConfig(tke_shear_production="nemo_face_native",
                               prandtl_mode="nemo_ri")
        cfg_face = TKEConfig(tke_shear_production="nemo_face_native",
                             tke_shear_avm_weighting="nemo_face",
                             prandtl_mode="nemo_ri")

        def _run(cfg, tke_old):
            return np.asarray(tke_vertical_mixing(
                cfg=cfg, tke_old=tke_old, dt=3600.0, rho_0=_RHO0,
                n_iterations=1, **kwargs).tke_new)

        shape = kwargs["dz_half"].shape
        # (a) horizontally VARYING K_M (via a varying TKE seed) -> must differ.
        bump = np.ones(shape)
        bump[:, : shape[1] // 2, :] = 50.0     # strong zonal K_M contrast
        tke_varying = jnp.asarray(1e-2 * bump)
        out_tpoint = _run(cfg_tpoint, tke_varying)
        out_face = _run(cfg_face, tke_varying)
        assert bool(np.all(np.isfinite(out_face)))
        assert not np.allclose(out_tpoint, out_face), (
            "nemo_face must differ from tpoint under a horizontal K_M "
            "gradient -- that gradient is the entire content of the option")

        # (b) UNIFORM K_M -> identical to fp round-off (the corrected
        #     contract; the old 2x normalisation fails this by 2x).  NOT
        #     bit-identical: the two paths multiply by K_M at different
        #     points in the same sum, so fp non-associativity leaves a
        #     ~1e-15 relative residual (measured 1.5e-15 max) -- fifteen
        #     orders below the 2x this assertion is there to catch.
        tke_uniform = jnp.full(shape, 1e-2)
        np.testing.assert_allclose(
            _run(cfg_tpoint, tke_uniform), _run(cfg_face, tke_uniform),
            rtol=1e-12, atol=0,
            err_msg="uniform K_M: nemo_face must equal tpoint to round-off")

    def test_final_K_call_gets_face_weighted_override(self):
        """Regression (#1455 review B1): EVERY compute_K_from_tke call —
        including the post-loop one producing the RETURNED K_M/K_H — must
        receive the face-weighted p_sh2_override when 'nemo_face' is
        selected (spy idiom, same as TestRn2bTimeLevel)."""
        kwargs = _face_native_orchestrator_inputs()
        import legoesm.ocean.physics.vertical_mixing.tke as _tke_mod
        overrides_seen = []
        _orig = _tke_mod.compute_K_from_tke

        def _spy(*a, **kw):
            overrides_seen.append(kw.get("p_sh2_override"))
            return _orig(*a, **kw)

        _tke_mod.compute_K_from_tke = _spy
        try:
            tke_vertical_mixing(
                cfg=TKEConfig(tke_shear_production="nemo_face_native",
                              tke_shear_avm_weighting="nemo_face",
                              prandtl_mode="nemo_ri"),
                tke_old=jnp.full(kwargs["dz_half"].shape, 1e-2),
                dt=3600.0, rho_0=_RHO0, n_iterations=1, **kwargs)
        finally:
            _tke_mod.compute_K_from_tke = _orig
        # n_iterations=1 → in-loop call + final call = 2; both overridden.
        assert len(overrides_seen) >= 2
        assert all(o is not None for o in overrides_seen)

    def test_nemo_face_blocked_on_post_mixing_path(self):
        """Regression (#1455 review B2): 'nemo_face' with
        buoyancy_timing='post_mixing_veros' must raise, never silently
        keep the tpoint weighting on the tke_set_diffusivities path.
        Exercised via _validate_post_mixing_cfg — the guard shared by
        tke_set_diffusivities/tke_integrate_post_mixing (tke_vertical_mixing
        rejects post_mixing_veros earlier with its own message)."""
        import legoesm.ocean.physics.vertical_mixing.tke as _tke_mod
        with pytest.raises(
                ValueError,
                match="tke_shear_avm_weighting='nemo_face' is not supported"):
            _tke_mod._validate_post_mixing_cfg(
                TKEConfig(buoyancy_timing="post_mixing_veros",
                          tke_shear_avm_weighting="nemo_face"))
        # And the membership arm on the same path:
        with pytest.raises(ValueError,
                           match="Unknown TKEConfig.tke_shear_avm_weighting"):
            _tke_mod._validate_post_mixing_cfg(
                TKEConfig(tke_shear_avm_weighting="bogus"))


class TestRn2bTimeLevel:
    def test_langmuir_uses_n2b_not_n2(self):
        """With lc=True and n2_mode='adiabatic', tke_vertical_mixing feeds
        the Langmuir source N2b (falling back to N2 when T_n2b/S_n2b are
        None) — verified by SPYING on the real nemo_langmuir_tke_source
        call (the Axell h_lc threshold is a step function of N2, so a
        black-box tke_new/K_M comparison can accidentally land both calls on
        the SAME h_lc and look identical despite correct wiring; spying on
        the actual N2 argument is the robust check)."""
        (u, v, T, S, rho, dz_half, z_int, tx, ty, p_cell, dz_ref,
         jacobian) = _n2b_test_column()
        T_b = T + 3.0   # a materially different before-state contrast
        S_b = S
        tke_seed = jnp.full(dz_half.shape, 1e-2)
        common = dict(
            u_cell=u, v_cell=v, T_cell=T, S_cell=S, rho_cell=rho,
            dz_half=dz_half, tke_old=tke_seed, tau_x_surface=tx,
            tau_y_surface=ty, dt=3600.0, rho_0=_RHO0, n_iterations=1,
            z_interface=z_int, p_cell=p_cell, dz_ref=dz_ref, jacobian=jacobian,
            cfg=TKEConfig(lc=True, n2_mode="adiabatic"),
        )
        import legoesm.ocean.physics.vertical_mixing.tke as _tke_mod
        captured = {}
        _orig = _tke_mod.nemo_langmuir_tke_source

        def _spy(taum, N2, *a, **kw):
            captured["N2"] = np.asarray(N2).copy()
            return _orig(taum, N2, *a, **kw)

        _tke_mod.nemo_langmuir_tke_source = _spy
        try:
            tke_vertical_mixing(**common)
            n2_default = captured["N2"]
            tke_vertical_mixing(T_n2b=T_b, S_n2b=S_b, **common)
            n2_with_override = captured["N2"]
        finally:
            _tke_mod.nemo_langmuir_tke_source = _orig
        assert not np.allclose(n2_default, n2_with_override)

    def test_prandtl_uses_n2b_not_n2(self):
        """With prandtl_mode='nemo_ri' and n2_mode='adiabatic', feeding
        T_n2b/S_n2b changes K_H vs the default (zri's rn2b falls back to
        rn2)."""
        (u, v, T, S, rho, dz_half, z_int, tx, ty, p_cell, dz_ref,
         jacobian) = _n2b_test_column()
        # #1226 item 11: nemo_ri's zri denominator is now the AVM-WEIGHTED
        # p_sh2 (kappaM*shear_sq), not bare shear_sq -- the fixture's tiny
        # shear (~(0.05/25)^2) now saturates Pr at the 10-ceiling in BOTH
        # calls (physically correct: a near-zero-shear stratified column IS
        # abyssal-limited), masking the N2b sensitivity this test checks.
        # Scale up the velocity shear so zri sits off the ceiling.
        u = u * 20.0
        v = v * 20.0
        T_b = T + 3.0
        S_b = S
        common = dict(
            u_cell=u, v_cell=v, T_cell=T, S_cell=S, rho_cell=rho,
            dz_half=dz_half, tke_old=None, tau_x_surface=tx,
            tau_y_surface=ty, dt=3600.0, rho_0=_RHO0, n_iterations=3,
            z_interface=z_int, p_cell=p_cell, dz_ref=dz_ref, jacobian=jacobian,
            cfg=TKEConfig(prandtl_mode="nemo_ri", kappaM_min=1e-6,
                         n2_mode="adiabatic"),
        )
        default_out = tke_vertical_mixing(**common)
        n2b_out = tke_vertical_mixing(T_n2b=T_b, S_n2b=S_b, **common)
        assert not np.allclose(
            np.asarray(default_out.K_H), np.asarray(n2b_out.K_H))

    def test_requires_both_tracers_together(self):
        u, v, T, S, rho, dz_half, z_int, tx, ty = _orchestrator_inputs()
        with pytest.raises(ValueError, match="T_n2b and S_n2b"):
            tke_vertical_mixing(
                u, v, T, S, rho, dz_half, None, tx, ty, dt=3600.0,
                cfg=TKEConfig(), rho_0=_RHO0, n_iterations=1,
                z_interface=z_int, T_n2b=T + 1.0)

    def test_default_is_byte_identical(self):
        u, v, T, S, rho, dz_half, z_int, tx, ty = _orchestrator_inputs()
        base = tke_vertical_mixing(
            u, v, T, S, rho, dz_half, None, tx, ty, dt=3600.0,
            cfg=TKEConfig(lc=True), rho_0=_RHO0, n_iterations=3,
            z_interface=z_int)
        explicit_none = tke_vertical_mixing(
            u, v, T, S, rho, dz_half, None, tx, ty, dt=3600.0,
            cfg=TKEConfig(lc=True), rho_0=_RHO0, n_iterations=3,
            z_interface=z_int, T_n2b=None, S_n2b=None)
        np.testing.assert_array_equal(
            np.asarray(base.tke_new), np.asarray(explicit_none.tke_new))


class TestTkeN2TimeLevelDispatch:
    def test_unknown_value_raises(self):
        """Real dispatch guard in k_profiles._vmix_K_profiles, exercised
        through compute_vertical_K_profiles (not a re-implementation)."""
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.init_latlon_cgrid import (
            rest_state_latlon_cgrid_ocean,
        )
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.physics.vertical_mixing.config import (
            TKEConfig, VerticalMixingConfig,
        )
        from legoesm.ocean.physics.vertical_mixing.k_profiles import (
            compute_vertical_K_profiles,
        )

        grid = create_latlon_grid(n_lat=8, n_lon=12)
        z = create_ocean_z_star(n_levels=5, H_max=4000.0)
        state = rest_state_latlon_cgrid_ocean(
            grid, z, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0)
        from legoesm.ocean.physics.combined import OceanPhysicsConfig
        physics = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(
                scheme="tke", tke=TKEConfig(tke_n2_time_level="bogus")))
        with pytest.raises(ValueError, match="tke_n2_time_level"):
            compute_vertical_K_profiles(state, z, None, physics)

    def test_nemo_before_without_tracers_raises(self):
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.init_latlon_cgrid import (
            rest_state_latlon_cgrid_ocean,
        )
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.physics.vertical_mixing.config import (
            TKEConfig, VerticalMixingConfig,
        )
        from legoesm.ocean.physics.vertical_mixing.k_profiles import (
            compute_vertical_K_profiles,
        )
        from legoesm.ocean.physics.combined import OceanPhysicsConfig

        grid = create_latlon_grid(n_lat=8, n_lon=12)
        z = create_ocean_z_star(n_levels=5, H_max=4000.0)
        state = rest_state_latlon_cgrid_ocean(
            grid, z, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0)
        physics = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(
                scheme="tke", tke=TKEConfig(tke_n2_time_level="nemo_before")))
        with pytest.raises(ValueError, match="n2_tracers_before"):
            compute_vertical_K_profiles(state, z, None, physics)


class TestLeapfrogConstructionGuards:
    """T4/T8/T13 construction-time raises (dispatch hardening): these axes
    require outer_integrator='leapfrog' — the before-state does not exist
    otherwise."""

    def _model(self, outer_integrator, tke_cfg):
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.state import LatLonCGridOceanConfig
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        from legoesm.ocean.physics.combined import OceanPhysicsConfig
        from legoesm.ocean.physics.vertical_mixing.config import (
            VerticalMixingConfig,
        )
        from legoesm.ocean.physics.convection.config import (
            OceanConvectionConfig,
        )
        grid = create_latlon_grid(8, 16)
        z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
        _lateral_mixing_none = type(OceanPhysicsConfig().lateral_mixing)(
            scheme="none")
        physics = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme="tke", tke=tke_cfg),
            convection=OceanConvectionConfig(scheme="none"),
            lateral_mixing=_lateral_mixing_none,
        )
        cfg_kw = dict(
            A_h=2.0e4, bottom_drag_r=1.0e-3, n_barotropic_substeps=8,
            enable_runtime_checks=False, implicit_vertical_mixing=True,
            outer_integrator=outer_integrator, physics=physics,
        )
        if outer_integrator == "leapfrog":
            cfg_kw.update(coriolis_scheme="explicit_ab2",
                         vorticity_scheme="een_total")
        cfg = LatLonCGridOceanConfig.from_flat(**cfg_kw)
        return LatLonCGridOceanModel(grid, z_coord, cfg)

    def test_nemo_burchard_requires_leapfrog(self):
        from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
        with pytest.raises(ValueError, match="nemo_burchard"):
            self._model("forward_euler",
                        TKEConfig(tke_shear_production="nemo_burchard"))

    def test_nemo_before_requires_leapfrog(self):
        from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
        with pytest.raises(ValueError, match="nemo_before"):
            self._model("forward_euler",
                        TKEConfig(tke_n2_time_level="nemo_before"))

    def test_both_construct_cleanly_under_leapfrog(self):
        """Under outer_integrator='leapfrog' (which DOES carry the
        before-state) both axes construct without raising — a regression
        guard for the new construction-time checks."""
        from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
        self._model("leapfrog", TKEConfig(
            tke_shear_production="nemo_burchard",
            tke_n2_time_level="nemo_before"))

    def test_n2_nemo_before_tracers_raises_loudly_when_unpopulated(self):
        """#1317: outer_integrator='leapfrog' construction guarantees
        T_before/S_before EXIST as NamedTuple slots, but a state bridged
        straight from a NEMO restart (a twin's step-0 entry state, before
        the model's own Euler-start populates them) still has them as
        None. _n2_nemo_before_tracers must raise ValueError (not the old
        AttributeError: 'NoneType' object has no attribute 'data', and not
        a silent fallback to entry_state.T/.S)."""
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.init_latlon_cgrid import (
            rest_state_latlon_cgrid_ocean,
        )
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.physics.vertical_mixing.config import TKEConfig

        model = self._model("leapfrog", TKEConfig(tke_n2_time_level="nemo_before"))
        grid = create_latlon_grid(8, 16)
        z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0)
        assert state.T_before is None   # the rest-state / bridged-twin default

        with pytest.raises(ValueError, match="before-level tracers"):
            model._n2_nemo_before_tracers(state)

    def test_n2_nemo_before_tracers_returns_populated_before_state(self):
        """Sibling to the above: once T_before/S_before ARE populated (e.g.
        via kamm_twin_90d --bridge-before, or after the model's own
        Euler-start), the predicate returns them — not None, not a raise."""
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.init_latlon_cgrid import (
            rest_state_latlon_cgrid_ocean,
        )
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.physics.vertical_mixing.config import TKEConfig

        model = self._model("leapfrog", TKEConfig(tke_n2_time_level="nemo_before"))
        grid = create_latlon_grid(8, 16)
        z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0)
        state = state._replace(T_before=state.T, S_before=state.S)

        out = model._n2_nemo_before_tracers(state)
        assert out is not None
        t_before, s_before = out
        assert jnp.array_equal(t_before, state.T.data)
        assert jnp.array_equal(s_before, state.S.data)


class TestNemoBurchardPreCenteredCcState:
    """#1317 regression: ``_apply_implicit_vertical_mixing``'s fallback path
    builds a ``cc_state`` with u/v ALREADY cell-centered (``state._replace(
    u=u_cell, v=v_cell)``) but leaves ``u_before``/``v_before`` untouched at
    their ORIGINAL face-staggered shape (the model never builds a
    before-level cc_state). ``k_profiles.py``'s old ``_staggered`` flag was
    derived from ``u_data.shape`` (already centered by the caller) and reused
    for ``u_before_data`` too, so it stayed False and skipped centering
    ``u_before_data`` — the (n_lat, n_lon+1, nlev) face array then hit
    ``_vertical_shear_burchard``'s ``(du_now)*(du_before)`` product against a
    (n_lat, n_lon, nlev) ``du_now``, a broadcasting TypeError caught running
    the #1317 --bridge-before acceptance twin. Fixed by checking
    ``u_before_data.shape`` independently of the (possibly pre-centered)
    ``u_data.shape``."""

    def test_precentered_u_v_with_staggered_before_does_not_crash(self):
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.init_latlon_cgrid import (
            rest_state_latlon_cgrid_ocean,
        )
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.physics.vertical_mixing.config import (
            TKEConfig, VerticalMixingConfig,
        )
        from legoesm.ocean.physics.vertical_mixing.k_profiles import (
            compute_vertical_K_profiles,
        )
        from legoesm.ocean.physics.combined import OceanPhysicsConfig

        grid = create_latlon_grid(n_lat=8, n_lon=12)
        z = create_ocean_z_star(n_levels=5, H_max=4000.0)
        state = rest_state_latlon_cgrid_ocean(
            grid, z, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0)

        # Simulate the caller's cc_state: u/v pre-centered to T's width
        # (n_lon), u_before/v_before left at the ORIGINAL face width
        # (n_lon+1) -- the exact mismatch the model's own fallback path
        # produces.
        n_lat, n_lon_face, nlev = state.u.data.shape
        u_cell = 0.5 * (state.u.data[:, :-1, :] + state.u.data[:, 1:, :])
        v_cell = 0.5 * (state.v.data[:-1, :, :] + state.v.data[1:, :, :])
        rng = np.random.default_rng(0)
        u_before_face = jnp.asarray(
            rng.normal(scale=0.1, size=state.u.data.shape))
        v_before_face = jnp.asarray(
            rng.normal(scale=0.1, size=state.v.data.shape))
        cc_state = state._replace(
            u=state.u.replace(data=u_cell),
            v=state.v.replace(data=v_cell),
            u_before=state.u.replace(data=u_before_face),
            v_before=state.v.replace(data=v_before_face),
            T_before=state.T, S_before=state.S,
        )
        assert cc_state.u.data.shape[1] != cc_state.u_before.data.shape[1]

        # Mode B (diagnostic, prognostic=False -- the TKEConfig default):
        # exercises the SAME cc_state shape mismatch on the diagnostic path,
        # which had its own separate bug (u_before_cell/v_before_cell were
        # not threaded to tke_vertical_mixing at all in Mode B -- fixed
        # alongside the shape bug).
        physics = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(
                scheme="tke",
                tke=TKEConfig(tke_shear_production="nemo_burchard")))
        K_v, A_v = compute_vertical_K_profiles(cc_state, z, None, physics)
        assert bool(jnp.isfinite(K_v).all())
        assert bool(jnp.isfinite(A_v).all())

        # Mode A (prognostic=True -- nemo_dino_kamm_mlf's actual runtime
        # config, the path the #1317 acceptance twin exercises): same
        # cc_state shape mismatch, the ORIGINAL crash site
        # (_vertical_shear_burchard's du_now*du_before broadcasting
        # TypeError, (n_lat,n_lon+1,nlev-1) vs (n_lat,n_lon,nlev-1)).
        physics_prog = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(
                scheme="tke",
                tke=TKEConfig(tke_shear_production="nemo_burchard",
                              prognostic=True)))
        K_v_p, A_v_p, tke_new = compute_vertical_K_profiles(
            cc_state, z, None, physics_prog, dt_tke=2700.0, return_tke=True)
        assert bool(jnp.isfinite(K_v_p).all())
        assert bool(jnp.isfinite(A_v_p).all())
        assert bool(jnp.isfinite(tke_new).all())


# ---------------------------------------------------------------------------
# #1226 item 11: nemo_ri zri transcription bug fix. Ground truth is an
# independent loop-port of zdftke.F90:381-401 (the nn_pdl==1 branch) +
# zdfsh2.F90:80-94 (the p_sh2 avm-weighted shear-production term the
# denominator actually consumes) -- NOT a call into legoESM's own
# _prandtl_number, so the test is non-vacuous against a re-introduced bug.
# ---------------------------------------------------------------------------


def _nemo_zri_pdlr_reference(rn2b, p_avm, shear_sq, rn_bshear, ri_cri):
    """Independent NumPy loop-port of zdftke.F90:381-401 + zdfsh2's p_sh2.

    ``p_sh2`` (zdfsh2.F90:80-94, the no-Stokes branch) is, per interface,
    ``avm-weighted`` shear production [m^2/s^3] -- on legoESM's cell-centred
    single-K_M grid (no u-/v-point face-averaging to port), the faithful
    per-interface form is ``p_avm * shear_sq``. Returns (zri, pdlr) with the
    SAME branch structure as the Fortran (rn2b<=0 -> zri=0; zdiv==0 exact-
    zero guard -> divide by rn_bshear alone; else divide by zdiv).
    """
    rn2b = np.asarray(rn2b, dtype=np.float64)
    p_avm = np.asarray(p_avm, dtype=np.float64)
    shear_sq = np.asarray(shear_sq, dtype=np.float64)
    p_sh2 = p_avm * shear_sq
    zri = np.zeros_like(rn2b)
    for idx in np.ndindex(rn2b.shape):
        if rn2b[idx] <= 0.0:
            zri[idx] = 0.0
            continue
        zdiv = p_sh2[idx] + rn_bshear
        if zdiv == 0.0:
            zri[idx] = rn2b[idx] * p_avm[idx] / rn_bshear
        else:
            zri[idx] = rn2b[idx] * p_avm[idx] / zdiv
    pdlr = np.maximum(0.1, ri_cri / np.maximum(ri_cri, zri))
    return zri, pdlr


class TestNemoRiZriTranscription:
    """Ground-truth transcription test for #1226 item 11.

    The bug: legoESM's ``zri`` divided by the bare gradient ``shear_sq``
    [1/s^2] instead of NEMO's avm-weighted ``p_sh2`` [m^2/s^3] -- a
    dimensional error of order ``kappaM`` (1e-5..1e-1) that collapsed zri
    far below the Pr=1 clamp. This test builds the reference zri/pdlr by an
    INDEPENDENT loop-port (not a call into ``_prandtl_number``) and checks
    ``_prandtl_number``'s ``Pr = 1/pdlr`` against it.
    """

    RI_CRI = 2.0 / (2.0 + 5.0 / 10.0)  # NEMO default rn_ediss=5, rn_ediff=10

    def _run(self, N2, shear_sq, kappaM, bshear=1.0e-20):
        cfg = TKEConfig(prandtl_mode="nemo_ri", bshear_floor=bshear,
                         prandtl_ri_coeff=1.0 / self.RI_CRI)
        Pr = _prandtl_number(jnp.asarray(N2), jnp.asarray(shear_sq),
                              jnp.asarray(kappaM), cfg)
        return np.asarray(Pr)

    def test_literal_nemo_association_is_bit_exact_and_old_paths_are_red(self):
        """Matched-step scalar whose algebraically equivalent paths differ.

        The hexadecimal inputs make this a stable hand-computed fp64 case.
        NEMO evaluates ``(rn2b*p_avm)/zdiv`` and then constructs ``pdlr``
        literally before taking its reciprocal.  Multiplying by ``1/zdiv``
        or collapsing the pdlr expression to a clipped product is one ulp
        different on this case, so either historical shortcut turns red.
        """
        rn2b = np.asarray([
            np.float64.fromhex("0x1.04fa0afaa6d02p-14"),
            np.float64.fromhex("0x1.ffc95566895cap-15"),
        ])
        p_avm = np.asarray([
            np.float64.fromhex("0x1.95141a74aca4ap-4"),
            np.float64.fromhex("0x1.7d1cb254e6652p-4"),
        ])
        p_sh2 = np.asarray([
            np.float64.fromhex("0x1.693ab8013e477p-16"),
            np.float64.fromhex("0x1.9c874ac1cebd0p-16"),
        ])
        rn_bshear = np.float64(1.0e-20)
        ri_cri = np.float64(2.0) / (
            np.float64(2.0) + np.float64(0.7) / np.float64(0.1))

        zdiv = p_sh2 + rn_bshear
        zri = (rn2b * p_avm) / zdiv
        pdlr = np.maximum(
            np.float64(0.1), ri_cri / np.maximum(ri_cri, zri))
        expected = np.float64(1.0) / pdlr

        cfg = TKEConfig(
            prandtl_mode="nemo_ri", bshear_floor=float(rn_bshear),
            prandtl_ri_coeff=float(np.float64(1.0) / ri_cri),
            tke_n2_evaluation_stage="step_entry")
        got = np.asarray(_prandtl_number(
            jnp.asarray(rn2b), jnp.ones(2), jnp.asarray(p_avm), cfg,
            p_sh2_override=jnp.asarray(p_sh2)))
        np.testing.assert_array_equal(got, expected)

        reciprocal_first = rn2b * p_avm * (np.float64(1.0) / zdiv)
        old_direct_pr = np.maximum(
            np.float64(1.0), np.minimum(
                np.float64(10.0),
                (np.float64(1.0) / ri_cri) * zri))
        old_legacy_pr = np.maximum(
            np.float64(1.0), np.minimum(
                np.float64(10.0),
                (np.float64(1.0) / ri_cri) * reciprocal_first))
        assert np.any(reciprocal_first != zri)
        assert np.any(old_direct_pr != expected)
        assert np.any(old_legacy_pr != expected)

        legacy = cfg._replace(tke_n2_evaluation_stage="implicit_solve_state")
        legacy_got = np.asarray(_prandtl_number(
            jnp.asarray(rn2b), jnp.ones(2), jnp.asarray(p_avm), legacy,
            p_sh2_override=jnp.asarray(p_sh2)))
        np.testing.assert_array_equal(legacy_got, old_legacy_pr)

    def test_matches_independent_loop_port_turbulent_column(self):
        rng = np.random.default_rng(42)
        n = 25
        rn2b = rng.uniform(1e-6, 5e-4, n)          # stratified (>0)
        p_avm = rng.uniform(1e-5, 1e-2, n)          # realistic K_M range
        shear_sq = rng.uniform(1e-6, 1e-3, n)       # realistic shear^2
        zri_ref, pdlr_ref = _nemo_zri_pdlr_reference(
            rn2b, p_avm, shear_sq, rn_bshear=1.0e-20, ri_cri=self.RI_CRI)
        Pr_ref = 1.0 / pdlr_ref

        Pr_lego = self._run(rn2b, shear_sq, p_avm)
        np.testing.assert_allclose(Pr_lego, Pr_ref, rtol=1e-10)

    def test_old_unweighted_formula_fails_this_case(self):
        """Non-vacuous: the OLD (buggy) ``zri = N2*kappaM/shear_sq`` formula
        must NOT reproduce the reference on a case where kappaM is far from
        1 (the two formulas only coincide when kappaM == 1)."""
        rn2b = np.array([2.0e-4])
        p_avm = np.array([1.0e-3])          # realistic abyssal K_M, << 1
        shear_sq = np.array([5.0e-5])
        zri_ref, pdlr_ref = _nemo_zri_pdlr_reference(
            rn2b, p_avm, shear_sq, rn_bshear=1.0e-20, ri_cri=self.RI_CRI)
        Pr_ref = float(1.0 / pdlr_ref[0])

        # OLD buggy transcription (dimensionally wrong: divides by bare
        # shear_sq, no avm weighting in the denominator).
        zri_old = rn2b[0] * p_avm[0] / max(shear_sq[0] + 1.0e-20, 1e-30)
        pdlr_old = max(0.1, self.RI_CRI / max(self.RI_CRI, zri_old))
        Pr_old = 1.0 / pdlr_old

        Pr_lego = float(self._run(rn2b, shear_sq, p_avm)[0])
        assert Pr_lego == pytest.approx(Pr_ref, rel=1e-10)
        assert Pr_old != pytest.approx(Pr_ref, rel=1e-6), (
            "old formula should NOT match the faithful reference here")
        assert Pr_lego != pytest.approx(Pr_old, rel=1e-6), (
            "fixed _prandtl_number should differ from the old buggy value")

    def test_rn2b_non_positive_gives_ri_zero_pr_floor(self):
        """zdftke.F90:384-385: rn2b<=0 -> zri=0 -> pdlr=max(0.1, ri_cri/
        max(ri_cri,0))=1 -> Pr=1/pdlr=1 (the convective branch: K_H tracks
        the full convective K_M, matching the "richardson" mode's Ri<0 ->
        Pr=1 behaviour)."""
        for rn2b_val in (0.0, -1.0e-4):
            Pr = self._run(np.array([rn2b_val]), np.array([1.0e-4]),
                            np.array([1.0e-2]))
            np.testing.assert_allclose(Pr, 1.0, rtol=1e-10)

    def test_pr_saturates_at_10_in_strongly_stratified_column(self):
        """zdftke.F90:399: pdlr = max(0.1, ri_cri/max(ri_cri,zri)) -> as
        zri -> large, pdlr -> 0.1 -> Pr = 1/pdlr -> 10 (the abyssal
        saturation ceiling)."""
        rn2b = np.array([1.0])            # strongly stratified
        p_avm = np.array([1.0])
        shear_sq = np.array([1.0e-8])     # vanishing shear -> huge zri
        Pr = self._run(rn2b, shear_sq, p_avm)
        np.testing.assert_allclose(Pr, 10.0, rtol=1e-10)

    def test_bshear_floor_now_in_avm_weighted_units(self):
        """#1226 item 11: bshear_floor is added to the AVM-WEIGHTED p_sh2
        [m^2/s^3], not to bare shear_sq [1/s^2] -- so with kappaM << 1 and
        shear_sq below the OLD floor's scale, the floor binds at a
        different point than it would under the (buggy) unweighted form.
        Concretely: zero shear_sq with a large bshear_floor must give
        zri = rn2b*kappaM/bshear_floor (the floor alone dominates the
        denominator), matching the reference port exactly."""
        rn2b = np.array([1.0e-4])
        p_avm = np.array([1.0e-2])
        shear_sq = np.array([0.0])
        bshear = 1.0e-6
        zri_ref, pdlr_ref = _nemo_zri_pdlr_reference(
            rn2b, p_avm, shear_sq, rn_bshear=bshear, ri_cri=self.RI_CRI)
        Pr_ref = float(1.0 / pdlr_ref[0])
        Pr_lego = float(self._run(rn2b, shear_sq, p_avm, bshear=bshear)[0])
        assert Pr_lego == pytest.approx(Pr_ref, rel=1e-10)


class TestNemoRiNegativeZdivSign:
    """#1226 zdftke_chain_walk (STAGE 2 finding, tke.py:1324 pre-fix): a
    genuinely negative ``zdiv = p_sh2 + rn_bshear`` (from float-noise-negative
    shear production, zdftke.F90:459-476) must give ``pdlr=1.0`` (Pr=1), NOT
    ``pdlr=0.1`` (Pr=10) -- the OLD ``jnp.maximum(p_sh2+bshear, 1e-30)`` code
    flipped the sign of a negative denominator, landing on the WRONG end of
    the same clamp. Reference = the file's OWN independent loop-port
    (:func:`_nemo_zri_pdlr_reference`), not a call into ``_prandtl_number``.
    """

    RI_CRI = 2.0 / (2.0 + 5.0 / 10.0)

    def _cfg(self, bshear=1.0e-20):
        return TKEConfig(prandtl_mode="nemo_ri", bshear_floor=bshear,
                          prandtl_ri_coeff=1.0 / self.RI_CRI)

    def test_sh2_negative_gives_pdlr_one_not_pointone(self):
        """rn2b>0 (stratified) with kappaM*shear_sq a tiny NEGATIVE value
        (|p_sh2| > rn_bshear=1e-20) -> zdiv<0 -> NEMO's zri<0 -> pdlr=1.0
        (Pr=1). This is the exact failure mode from the walk's 11 flagged
        cells (all sh2<0)."""
        rn2b = np.array([2.0e-4])
        kappaM = np.array([1.0e-2])
        shear_sq = np.array([-1.0e-12])          # p_sh2 = -1e-14, |.|>1e-20
        bshear = 1.0e-20

        zri_ref, pdlr_ref = _nemo_zri_pdlr_reference(
            rn2b, kappaM, shear_sq, rn_bshear=bshear, ri_cri=self.RI_CRI)
        assert zri_ref[0] < 0.0, "test setup must exercise the zdiv<0 branch"
        Pr_ref = float(1.0 / pdlr_ref[0])
        assert Pr_ref == pytest.approx(1.0, rel=1e-10)

        cfg = self._cfg(bshear=bshear)
        Pr_lego = _prandtl_number(jnp.asarray(rn2b), jnp.asarray(shear_sq),
                                   jnp.asarray(kappaM), cfg)
        assert float(Pr_lego[0]) == pytest.approx(1.0, rel=1e-10)
        assert float(Pr_lego[0]) == pytest.approx(Pr_ref, rel=1e-10)

    def test_old_buggy_sign_flip_would_give_pointone(self):
        """Non-vacuous: demonstrate the OLD formula (``jnp.maximum(zdiv,
        1e-30)``) gives pdlr=0.1 (Pr=10) on this exact input -- the opposite
        of the correct Pr=1 -- so this test would have FAILED against the
        pre-fix ``_prandtl_number``."""
        rn2b = np.array([2.0e-4])
        kappaM = np.array([1.0e-2])
        shear_sq = np.array([-1.0e-12])
        bshear = 1.0e-20
        p_sh2 = kappaM[0] * shear_sq[0]
        zri_old = rn2b[0] * kappaM[0] / max(p_sh2 + bshear, 1e-30)
        pdlr_old = max(0.1, self.RI_CRI / max(self.RI_CRI, zri_old))
        Pr_old = 1.0 / pdlr_old
        assert Pr_old == pytest.approx(10.0, rel=1e-6), (
            "sanity: the OLD sign-flipped formula saturates at Pr=10 here")

        cfg = self._cfg(bshear=bshear)
        Pr_lego = _prandtl_number(jnp.asarray(rn2b), jnp.asarray(shear_sq),
                                   jnp.asarray(kappaM), cfg)
        assert float(Pr_lego[0]) != pytest.approx(Pr_old, rel=1e-3), (
            "fixed _prandtl_number must NOT reproduce the old sign-flipped "
            "Pr=10 value on this input"
        )

    def test_grad_finite_across_sign_boundary(self):
        """AD safety: jax.grad of a scalar reduction through the nemo_ri
        branch stays finite at and around the zdiv sign boundary (the
        where-NaN-grad trap the double-where idiom guards against)."""
        cfg = self._cfg()
        N2 = jnp.array([2.0e-4])
        kappaM = jnp.array([1.0e-2])

        def loss(shear_sq):
            Pr = _prandtl_number(N2, shear_sq, kappaM, cfg)
            return jnp.sum(Pr)

        for s0 in (-1.0e-12, 0.0, 1.0e-12, -1.0e-18, 1.0e-18):
            g = jax.grad(loss)(jnp.array([s0]))
            assert bool(jnp.isfinite(g).all()), (
                f"grad not finite at shear_sq={s0!r}: {g}")

        # exact zdiv==0 boundary: p_sh2 + bshear == 0 -> shear_sq == -bshear/kappaM
        s_zero = jnp.array([-1.0e-20 / 1.0e-2])
        g_zero = jax.grad(loss)(s_zero)
        assert bool(jnp.isfinite(g_zero).all()), (
            f"grad not finite AT zdiv==0: {g_zero}")


class TestMxlChoice3LdownSeed:
    """#1226 zdftke_chain_walk (STAGE 4 finding, tke.py:698-701 pre-fix): the
    ``tke_mxl_choice==3`` bottom-up ``ldown`` scan must seed its carry from
    ``cfg.mxl_min`` (NEMO's ``rmxl_min``, zdftke.F90:678-679), NOT from the
    RAW buoyancy length at the deepest carried row. Ground truth is an
    INDEPENDENT NumPy loop-port of the NEMO ``nn_mxl=3`` lup/ldown recurrence
    (zdftke.F90:775-793), not a call into ``compute_mixing_lengths``.
    """

    def _nemo_mxl3_reference(self, l_sfc, l_int_raw, e3t_body, e3t_bottom,
                              mxl_min):
        """Independent loop-port of zdftke.F90:775-793 (nn_mxl=3 CASE(3)).

        ``l_int_raw`` = raw buoyancy length at interior W-levels jk=2..jpkm1
        (length ``n``). ``e3t_body`` = e3t(jk=1..jpkm1) (length ``n+1``,
        e3t_body[0]=e3t(1)). ``e3t_bottom`` = e3t(jpk) (the TRUE bottommost
        T-cell, the row the ORIGINAL bug's seed step needs but the legacy
        ``dz_cell`` contract cannot supply).
        """
        # jk convention: zmxlm/zmxld index r (0-based, r=0..n) <-> Fortran
        # jk=r+1 (r=0 is the surface, r=1..n are the interior W-levels
        # jk=2..jpkm1). ``e3t_body[k]`` (0-based, k=0..n) <-> ``e3t(k+1)``
        # (Fortran), i.e. e3t_body[0]=e3t(1) .. e3t_body[n]=e3t(jpkm1).
        n = len(l_int_raw)
        zmxlm = np.concatenate([[l_sfc], np.asarray(l_int_raw, dtype=np.float64)])
        zmxld = np.zeros(n + 1)
        zmxld[0] = l_sfc
        # lup: jk=2..jpkm1 (0-based idx 1..n), forward.
        # zmxld(jk) = min(zmxld(jk-1)+e3t(jk-1), zmxlm(jk))
        #           = min(zmxld[idx-1]+e3t_body[idx-1], zmxlm[idx])
        for idx in range(1, n + 1):
            zmxld[idx] = min(zmxld[idx - 1] + e3t_body[idx - 1], zmxlm[idx])
        # ldown: jk=jpkm1..2 (0-based idx n..1), backward.
        # zmxlm(jk) = min(zmxlm(jk+1)+e3t(jk+1), zmxlm(jk)).
        # For idx=n (jk=jpkm1): e3t(jk+1)=e3t(jpk)=e3t_bottom.
        # For idx<n (jk=idx+1): e3t(jk+1)=e3t(idx+2)=e3t_body[idx+1].
        for idx in range(n, 0, -1):
            e3_next = e3t_bottom if idx == n else e3t_body[idx + 1]
            new_val = min(zmxlm[idx + 1] + e3_next, zmxlm[idx]) if idx < n \
                else min(mxl_min + e3_next, zmxlm[idx])
            zmxlm[idx] = new_val
        zemlm = np.minimum(zmxld[1:], zmxlm[1:])
        zemlp = np.sqrt(zmxld[1:] * zmxlm[1:])
        return zemlm, zemlp   # (l_k, l_eps) at the n interior rows

    def test_near_neutral_deep_column_ldown_bound(self):
        """The walk's pathological case: a near-neutral deep column
        (rn2b~-1e-10, clamped to a tiny positive floor upstream) gives a
        HUGE raw buoyancy length at the deepest interior row. Assert the
        FIXED function's bottom-interface ``l_k`` against the independent
        NEMO transcription (using the caller's ``dz_cell`` widened by the
        true bottommost e3t row, so the comparison is NEMO-exact, not
        proxy-bounded)."""
        rng = np.random.default_rng(11)
        n = 20
        e3t_body = rng.uniform(50.0, 700.0, n + 1)      # e3t(1)..e3t(jpkm1)
        e3t_bottom = 617.46228681330                     # e3t(jpk), walk's column
        mxl_min = 1.0e-6 / (10.0 * np.sqrt(1.0e-10))      # rmxl_min formula
        l_sfc = 0.0672399578

        # Near-neutral deep column: N2 tiny positive (post-floor) -> huge raw
        # buoyancy length except at the LAST row, matched to the walk's
        # actual pathological numbers so the bound is the one that matters.
        en = rng.uniform(1e-5, 6e-5, n)
        N2_floor = 1.0e-12
        l_int_raw = np.maximum(mxl_min, np.sqrt(2.0 * en / N2_floor))
        assert l_int_raw.max() > 1.0e4, (
            "test setup must produce a raw buoyancy length far above any "
            "physically bounded value, to exercise the ldown bound")

        l_k_ref, l_eps_ref = self._nemo_mxl3_reference(
            l_sfc, l_int_raw, e3t_body, e3t_bottom, mxl_min)

        cfg = TKEConfig(tke_mxl_choice=3, mxl_min=mxl_min)
        e = jnp.asarray(en)[None, None, :]
        N2 = jnp.full((1, 1, n), N2_floor)
        dz_cell_widened = jnp.asarray(
            np.concatenate([e3t_body, [e3t_bottom]]))[None, None, :]
        l_sfc_arr = jnp.asarray([[l_sfc]])
        l_k, l_eps = compute_mixing_lengths(
            e, N2, jnp.zeros_like(e), cfg,
            dz_cell=dz_cell_widened, l_surface_anchor=l_sfc_arr,
        )
        l_k = np.asarray(l_k)[0, 0]
        l_eps = np.asarray(l_eps)[0, 0]

        # Bottom interface is the one the seed bug corrupts (last row).
        np.testing.assert_allclose(l_k[-1], l_k_ref[-1], rtol=1e-10)
        np.testing.assert_allclose(l_eps[-1], l_eps_ref[-1], rtol=1e-10)
        np.testing.assert_allclose(l_k, l_k_ref, rtol=1e-8)
        np.testing.assert_allclose(l_eps, l_eps_ref, rtol=1e-8)

    def test_old_seed_would_give_unbounded_length(self):
        """Non-vacuous: reproduce the OLD (buggy) seed behaviour by hand
        (seed = the raw buoyancy length at the deepest row, instead of
        ``mxl_min``) and show it does NOT match the independent NEMO
        reference on this pathological column -- i.e. this test would have
        FAILED against the pre-fix code."""
        rng = np.random.default_rng(11)
        n = 20
        e3t_body = rng.uniform(50.0, 700.0, n + 1)
        e3t_bottom = 617.46228681330
        mxl_min = 1.0e-6 / (10.0 * np.sqrt(1.0e-10))
        l_sfc = 0.0672399578
        en = rng.uniform(1e-5, 6e-5, n)
        N2_floor = 1.0e-12
        l_int_raw = np.maximum(mxl_min, np.sqrt(2.0 * en / N2_floor))

        l_k_ref, _ = self._nemo_mxl3_reference(
            l_sfc, l_int_raw, e3t_body, e3t_bottom, mxl_min)

        # OLD buggy seed: carry starts from l_int_raw[-1] (the RAW length),
        # not mxl_min, and the deepest row is never re-bounded at all.
        old_ldn_deepest = l_int_raw[-1]     # unbounded, per the pre-fix bug
        assert old_ldn_deepest != pytest.approx(l_k_ref[-1], rel=1e-3), (
            "sanity: the old seed's unbounded deepest-row value must differ "
            "sharply from the correct NEMO-bounded value"
        )
        assert old_ldn_deepest > 5.0 * l_k_ref[-1], (
            "the old seed's value should be wildly larger than the correct "
            "bound (reproducing the walk's 3454.8m vs 617.5m finding)"
        )

    def test_choice2_untouched_shape_and_values(self):
        """Bit-identity guard: ``tke_mxl_choice==2`` (the Veros/default path,
        NOT touched by this fix) must be completely unaffected."""
        rng = np.random.default_rng(5)
        n = 9
        e = jnp.asarray(rng.uniform(1e-6, 1e-2, (2, 3, n)))
        N2 = jnp.asarray(rng.uniform(1e-8, 1e-4, (2, 3, n)))
        dz_half = jnp.asarray(rng.uniform(5, 50, (2, 3, n - 1)))
        cfg = TKEConfig(tke_mxl_choice=2, mxl_min=0.01)
        l_k, l_eps = compute_mixing_lengths(e, N2, dz_half, cfg)
        assert bool(jnp.isfinite(l_k).all())
        assert bool(jnp.isfinite(l_eps).all())
        assert l_k.shape == (2, 3, n)
        assert l_eps.shape == (2, 3, n)



# ---------------------------------------------------------------------------
# WHY THERE IS NO TestFaceNativeUnderRK3 CLASS HERE (codex 9406102)
# ---------------------------------------------------------------------------
# One was added on 2026-08-14 asserting that nemo_face_native may construct off
# the leap-frog family, on the argument that NEMO's RK3 step calls
# `zdf_phy( kstp, Nbb, Nbb, Nrhs )` (stprk3.F90:165) so the now x before
# product is a square on one time level. The source observation is correct and
# both ORCA1 oracle runs are RK3 builds. The conclusion did not follow, for
# three reasons, and the whole change was reverted:
#
#   1. legoESM's "RK3" is momentum_time_integrator, NOT outer_integrator --
#      outer_integrator has no RK3 value at all (forward_euler, ab2, leapfrog,
#      nemo_mlf). So "we are an RK3 model like NEMO" was a category error.
#
#   2. NEMO evaluates zdf_phy BEFORE stage 1, from the step-entry Nbb state.
#      legoESM calls _apply_implicit_vertical_mixing(state_new, ...) AFTER the
#      RK3 momentum stages and the barotropic solve, so its "now" is a
#      post-stage value, not NEMO's Nbb. Feeding it as both factors is
#      therefore NOT an Nbb x Nbb transcription.
#
#   3. The k_profiles branch written to support it was UNREACHABLE: the
#      _needs_before raise fires for nemo_face_native before the else-branch
#      is ever considered. The construction test added alongside it passed
#      while proving nothing about the runtime path -- it exercised the
#      constructor, which was the only thing the change had actually altered.
#
# Codex also showed the sibling rn2b argument was wrong in the other
# direction: stprk3.F90:156 sets `rn2 = rn2b` before zdf_phy, so under RK3
# they ARE equal, and tke_n2_time_level="nemo_before" is leapfrog-only for the
# same missing-entry-state reason, not because RK3 keeps the levels distinct.
#
# Closing this properly means splitting the closure call in two: a step-entry
# (NEMO Nbb) state for the face shear and rn2b, and the post-stage state for
# the implicit vertical solve. That is a real refactor, not a guard tweak, and
# until it exists the constructor guard is correct as written.


class TestNemoLiteralSolveCoversJpkm1:
    """The deepest carried TKE row is NEMO's ``jpkm1`` and must be SOLVED.

    legoESM carries ``n_levels-1`` interior W-interfaces plus one prepended
    z=0 row, so the assembled system runs from NEMO's ``jk=1`` to ``jk=jpkm1``
    INCLUSIVE and has no ``jk=jpk`` slot -- ``en(jpk)`` is read by nothing
    (``zdftke.f90:468`` seeds the back-substitution without the
    ``zd_up(jpkm1)*en(jpk)`` term, and ``tke_avn`` loops ``jk=1,jpkm1`` at
    ``zdftke.f90:681-687``).

    Until 2026-09-11 the solver computed ``jpkm1 = len-2`` and returned that
    deepest row as its RAW right-hand side.  On the GYRE identity card that
    right-hand side carries ``zdftke.f90:422-425``'s EXPLICIT half of the
    semi-implicit dissipation split, ``+rn_Dt*0.5*rn_ediss*dissl*en``, with no
    ``zdftke.f90:419`` diagonal against it, so that one row ran
    ``e(n+1) = e(n) + 0.5*rn_ediss*rn_Dt/L * e(n)**1.5`` and the model went
    non-finite at step 48 of the year.
    """

    @staticmethod
    def _system(n=30):
        a = jnp.full((n,), -0.1).at[0].set(0.0)
        b = jnp.full((n,), 2.0).at[0].set(1.0)
        c = jnp.full((n,), -0.1).at[0].set(0.0)
        rhs = jnp.arange(1.0, n + 1.0)
        return a, b, c, rhs

    @staticmethod
    def _dense_reference(a, b, c, rhs):
        """NEMO's system written out: the last row has no ``jk+1`` coupling."""
        n = a.shape[-1]
        A = np.zeros((n, n))
        for k in range(n):
            A[k, k] = float(b[k])
            if k:
                A[k, k - 1] = float(a[k])
            if k < n - 1:
                A[k, k + 1] = float(c[k])
        return np.linalg.solve(A, np.asarray(rhs))

    def test_deepest_row_is_solved_not_returned_as_its_rhs(self):
        a, b, c, rhs = self._system()
        got = np.asarray(_nemo_literal_tke_solve(
            a, b, c, rhs, jnp.asarray(1.0),
            jnp.ones((a.shape[-1] - 1,)), 1e-30))
        reference = self._dense_reference(a, b, c, rhs)[1:]
        np.testing.assert_allclose(got, reference, rtol=0, atol=1e-14)
        # The decisive assertion: the deepest row is NOT its own right-hand
        # side.  That equality WAS the defect, and it is what this pins.
        assert float(got[-1]) != float(rhs[-1])

    def test_previous_len_minus_two_convention_is_red(self):
        """Synthetic violation: the discarded ``jpkm1 = len-2`` convention.

        Re-implemented here (not called) so the test above cannot pass
        vacuously -- if someone restores it, the reference below is what the
        solver would return, and the assertion says it must not.
        """
        a, b, c, rhs = self._system()
        n = a.shape[-1]
        jpkm1_old = n - 2
        diag = np.empty(n)
        diag[0] = 1.0 / 1.0
        for k in range(1, jpkm1_old + 1):
            diag[k] = float(b[k]) - float(a[k]) * float(c[k - 1]) / diag[k - 1]
        work = np.empty(n)
        work[0] = 1.0
        for k in range(1, jpkm1_old + 1):
            work[k] = float(rhs[k]) - float(a[k]) / diag[k - 1] * work[k - 1]
        old = np.empty(n)
        old[jpkm1_old] = work[jpkm1_old] / diag[jpkm1_old]
        for k in range(jpkm1_old - 1, 0, -1):
            old[k] = (work[k] - float(c[k]) * old[k + 1]) / diag[k]
        old[jpkm1_old + 1:] = np.asarray(rhs)[jpkm1_old + 1:]
        old_interior = old[1:]
        # The discarded convention hands back the raw RHS at the deepest row.
        assert old_interior[-1] == float(rhs[-1])
        got = np.asarray(_nemo_literal_tke_solve(
            a, b, c, rhs, jnp.asarray(1.0),
            jnp.ones((n - 1,)), 1e-30))
        assert not np.allclose(got, old_interior)

    def test_dry_rows_below_the_seafloor_are_identity_rows(self):
        """A column whose deepest rows are dry must carry ``en`` unchanged.

        NEMO gets this from ``zcof = zfact1*tmask`` and the trailing
        ``*wmask`` (``zdftke.f90:409,419,425``): below the seafloor every
        coefficient vanishes, ``zdiag = 1`` and the right-hand side is the
        untouched ``en``.  legoESM's ``w_active`` must reproduce it, and the
        final ``MAX(en,rn_emin)*wmask`` (``zdftke.f90:473-475``) must then
        zero those rows.
        """
        n = 30
        a, b, c, rhs = self._system(n)
        a = a.at[-3:].set(0.0)
        b = b.at[-3:].set(1.0)
        c = c.at[-4:-1].set(0.0)
        w = jnp.ones((n - 1,)).at[-3:].set(0.0)
        got = np.asarray(_nemo_literal_tke_solve(
            a, b, c, rhs, jnp.asarray(1.0), w, 1e-30))
        np.testing.assert_array_equal(got[-3:], np.zeros(3))
        assert np.all(np.isfinite(got))


class TestNemoLiteralMatrixRejectsImplicitBuoyancy:
    """``zdftke.f90:419`` has no stratification term on the diagonal.

    The literal diagonal is therefore built without ``buoy_sink_rate``; an
    implicit-linearised selection would have its sink silently DELETED, not
    moved.  Every shipped literal card selects ``nemo_explicit``, so this must
    raise rather than run a configuration nothing chose.
    """

    def test_implicit_linearized_raises_under_the_literal_matrix(self):
        n = 5
        cfg = TKEConfig(
            tke_matrix_evaluation="nemo_literal",
            tke_solver_evaluation="nemo_literal",
            dissipation_discretization="nemo_1p5_split",
            tke_surface_bc_level="nemo_z0",
            tke_buoyancy_sink="implicit_linearized",
            positivity="floor")
        with pytest.raises(ValueError, match="nemo_explicit"):
            _solve_tke_backward_euler(
                e_old=jnp.full((n,), 1e-6),
                K_M_old=jnp.full((n,), 1e-4),
                K_H_old=jnp.full((n,), 1e-5),
                P_s=jnp.zeros((n,)), N2=jnp.full((n,), 1e-5),
                l_eps=jnp.full((n,), 10.0),
                dz_half=jnp.full((n,), 100.0),
                surface_flux=jnp.asarray(0.0), dt=14400.0, cfg=cfg,
                dz_cell=jnp.full((n + 1,), 100.0),
                dz_surface=jnp.asarray(50.0),
                surface_dirichlet=jnp.asarray(1e-4),
                surface_bc_level="nemo_z0",
                K_M_surface=jnp.asarray(1e-4),
                w_active=jnp.ones((n,)),
                nemo_e3t=jnp.full((n + 1,), 100.0),
                dissl_old=jnp.full((n,), 1e-4))
