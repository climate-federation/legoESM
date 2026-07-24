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
    _solve_tke_backward_euler,
    tke_vertical_mixing,
)


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
