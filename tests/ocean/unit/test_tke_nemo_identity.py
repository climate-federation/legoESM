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

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
from legoesm.ocean.physics.vertical_mixing.tke import (
    _prandtl_number,
    _solve_tke_backward_euler,
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
        assert abs(t.mxl0_min_m - 0.01) < 1e-12    # NEMO rmxl_min (T19; was 0.04)
        assert abs(t.mxl_min - 0.01) < 1e-12       # NEMO rmxl_min (T18; was 1e-8)
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
    fe = _dino_vertical_mixing_config(dino_config_for_recipe("nemo_dino_kamm")).tke
    mlf = _dino_vertical_mixing_config(
        dino_config_for_recipe("nemo_dino_kamm_mlf")).tke
    assert fe.tke_shear_production == "squared_centered"
    assert fe.tke_n2_time_level == "step_entry"
    assert mlf.tke_shear_production == "nemo_burchard"
    assert mlf.tke_n2_time_level == "nemo_before"
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
        assert abs(t.mxl0_min_m - 0.04) < 1e-12
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
