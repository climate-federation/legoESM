"""Unit tests for the TKE Veros vertical-metric slots (``veros_dz_slots``).

The ACC_Basic regression diagnosis (.physics-validator/accbasic_regression/)
found legoESM's TKE chain mixing vertical-metric conventions vs Veros's own
reference (veros/core/tke.py + thermodynamics.py:99). Four consumption-site
fixes, each gated behind ``TKEConfig.veros_dz_slots`` (default False ⇒
BIT-IDENTICAL legacy):

1. adiabatic N² over the caller's ``dz_half`` (Veros dzw) instead of the
   midpoint reconstruction (eos.compute_buoyancy_frequency_adiabatic);
2. TKE-diffusion face gradients over the cell thickness ``dzt`` and
   per-interface control volumes = ``dz_half`` (Veros tke.py:193-222);
3. buoyancy-length growth allowance = ``dzt`` with the Veros pass order
   (tke.py:54-65);
4. surface injection over ``0.5·dzw_top = -z_full_ref[0]·J``
   (tke.py:225).

Each test manufactures a column where the Veros-slot answer is
hand-computable (independent numpy reference), plus default-path
equivalence, validation errors, k_profiles wiring and AD checks.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.ocean.eos import compute_buoyancy_frequency_adiabatic
from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
from legoesm.ocean.physics.vertical_mixing.tke import (
    _solve_tke_backward_euler,
    _veros_buoyancy_length,
    tke_vertical_mixing,
)

jax.config.update("jax_enable_x64", True)

# A stretched u_centered-like column (the ACC top levels): dz_half does NOT
# equal the midpoint spacing, so the slots are genuinely distinguishable.
DZ_CELL = np.array([20.0, 28.0, 40.0, 56.0])          # dzt, top-down
DZ_HALF_UC = np.array([12.0, 44.0, 36.0])             # dzw interior (ACC-like)
DZ_SURFACE = 14.0                                     # 0.5*dzw_top = -z_top


def _linear_eos(alpha=0.2, gamma=4.0e-6):
    """rho(T, S, p) = 1000 - alpha*T + gamma*p — pressure-dependent so the
    adiabatic displacement is non-trivial but exactly hand-computable."""
    def eos_fn(T, S, p):
        return 1000.0 - alpha * T + gamma * p
    return eos_fn


# ---------------------------------------------------------------------------
# Fix 1 — adiabatic N² over the caller's dz_half (Veros dzw slot)
# ---------------------------------------------------------------------------


def test_adiabatic_n2_dz_half_slot_hand_computed():
    nlev = 4
    T = jnp.asarray([18.0, 14.0, 9.0, 5.0])[None, None, :]
    S = jnp.full((1, 1, nlev), 35.0)
    p = jnp.asarray([1.0e5, 3.0e5, 6.0e5, 1.0e6])[None, None, :]
    dz = jnp.asarray(DZ_CELL)
    J = jnp.ones((1, 1))
    dz_half = jnp.asarray(DZ_HALF_UC)[None, None, :]
    eos_fn = _linear_eos()
    g, rho0 = 9.81, 1024.0

    N2 = compute_buoyancy_frequency_adiabatic(
        T, S, p, dz, J, eos_fn=eos_fn, rho_ref=rho0, g=g, dz_half=dz_half)

    # Hand-computed: both parcels at the UPPER cell pressure p[k]; divided
    # by the GIVEN dz_half (the Veros dzw slot), NOT the midpoint spacing.
    Tn, pn = np.asarray(T)[0, 0], np.asarray(p)[0, 0]
    alpha = 0.2
    for k in range(nlev - 1):
        rho_up = 1000.0 - alpha * Tn[k] + 4.0e-6 * pn[k]
        rho_dn = 1000.0 - alpha * Tn[k + 1] + 4.0e-6 * pn[k]
        expect = -(g / rho0) * (rho_up - rho_dn) / DZ_HALF_UC[k]
        np.testing.assert_allclose(float(N2[0, 0, k]), expect, rtol=1e-13)


def test_adiabatic_n2_default_is_midpoint_reconstruction():
    """dz_half=None (every pre-existing caller) keeps the midpoint slot."""
    nlev = 4
    T = jnp.asarray([18.0, 14.0, 9.0, 5.0])[None, None, :]
    S = jnp.full((1, 1, nlev), 35.0)
    p = jnp.asarray([1.0e5, 3.0e5, 6.0e5, 1.0e6])[None, None, :]
    dz = jnp.asarray(DZ_CELL)
    J = jnp.ones((1, 1))
    eos_fn = _linear_eos()

    N2_default = compute_buoyancy_frequency_adiabatic(
        T, S, p, dz, J, eos_fn=eos_fn)
    dz_mid = 0.5 * (DZ_CELL[:-1] + DZ_CELL[1:])
    N2_mid = compute_buoyancy_frequency_adiabatic(
        T, S, p, dz, J, eos_fn=eos_fn,
        dz_half=jnp.asarray(dz_mid)[None, None, :])
    np.testing.assert_array_equal(np.asarray(N2_default), np.asarray(N2_mid))

    # And the dzw slot differs on this stretched column (the bug surface).
    N2_uc = compute_buoyancy_frequency_adiabatic(
        T, S, p, dz, J, eos_fn=eos_fn,
        dz_half=jnp.asarray(DZ_HALF_UC)[None, None, :])
    assert not np.allclose(np.asarray(N2_uc), np.asarray(N2_mid))


# ---------------------------------------------------------------------------
# Fix 2 — tridiagonal: face gradients over dzt, control volumes = dzw
# Fix 4 — surface injection over 0.5*dzw_top
# ---------------------------------------------------------------------------


def _dense_solve(a, b, c, d):
    """Reference dense solve of the tridiagonal system (numpy)."""
    n = len(b)
    A = np.zeros((n, n))
    for k in range(n):
        A[k, k] = b[k]
        if k > 0:
            A[k, k - 1] = a[k]
        if k < n - 1:
            A[k, k + 1] = c[k]
    return np.linalg.solve(A, d)


def test_solver_veros_slots_hand_assembled_tridiagonal():
    """One backward-Euler step against an independently hand-assembled
    Veros tridiagonal (delta over dzt[k+1], volumes = dzw, tke.py:185-222),
    on a 3-interface stretched column with diffusion + dissipation +
    sources + surface injection all active."""
    N = 3                                   # nlev - 1 interfaces
    dt = 4800.0
    cfg = TKEConfig(veros_dz_slots=True)
    e_old = np.array([4.0e-3, 1.0e-3, 5.0e-4])
    K_M = np.array([2.0e-2, 8.0e-3, 3.0e-3])
    K_H = K_M / 10.0
    P_s = np.array([1.0e-7, 4.0e-8, 1.0e-8])
    N2 = np.array([6.0e-6, 2.0e-6, -1.0e-6])    # incl. one unstable level
    l_eps = np.array([3.0, 5.0, 2.0])
    flux = 2.5e-7

    sh = (1, 1, N)
    e_new = _solve_tke_backward_euler(
        e_old=jnp.asarray(e_old)[None, None, :],
        K_M_old=jnp.asarray(K_M)[None, None, :],
        K_H_old=jnp.asarray(K_H)[None, None, :],
        P_s=jnp.asarray(P_s)[None, None, :],
        N2=jnp.asarray(N2)[None, None, :],
        l_eps=jnp.asarray(l_eps)[None, None, :],
        dz_half=jnp.asarray(DZ_HALF_UC)[None, None, :],
        surface_flux=jnp.full(sh[:-1], flux),
        dt=dt, cfg=cfg,
        dz_cell=jnp.asarray(DZ_CELL)[None, None, :],
        dz_surface=jnp.full(sh[:-1], DZ_SURFACE),
    )

    # ---- independent numpy reference (Veros slots, legoESM closure) ----
    # face k between interfaces k and k+1 crosses CELL k+1 (Veros
    # delta = dt/dzt[k+1] * alpha/2 * (kM[k]+kM[k+1]), tke.py:193).
    K_face = cfg.alpha_tke * 0.5 * (K_M[:-1] + K_M[1:])
    delta = dt * K_face / DZ_CELL[1:N]      # dzt slot
    vol = DZ_HALF_UC                        # dzw control volumes
    a = np.zeros(N)
    b_diff = np.zeros(N)
    c = np.zeros(N)
    a[1:] = -delta / vol[1:]
    c[:-1] = -delta / vol[:-1]
    b_diff = -(a + c)
    diss = cfg.c_eps * np.sqrt(np.maximum(e_old, cfg.tke_background)) \
        / np.maximum(l_eps, cfg.mxl_min)
    e_safe = np.maximum(e_old, cfg.tke_background)
    sink = K_H * np.maximum(N2, 0.0) / e_safe
    src = -K_H * np.minimum(N2, 0.0)
    b = 1.0 + dt * (diss + sink) + b_diff
    d = e_old + dt * (P_s + src)
    d[0] += dt * flux / DZ_SURFACE          # 0.5*dzw_top injection slot
    expect = _dense_solve(a, b, c, d)
    expect = np.maximum(expect, cfg.tke_background)
    expect[0] = max(expect[0], cfg.tke_surface_min)

    np.testing.assert_allclose(np.asarray(e_new)[0, 0], expect, rtol=1e-12)


def test_solver_legacy_slots_unchanged_hand_assembled():
    """Default path (no dz_cell/dz_surface) still uses the legacy slots:
    face gradients over dz_half, avg-of-faces control volumes, injection
    over dz_half[0] — pinned by an independent numpy reference so the
    veros branch cannot silently leak into the default."""
    N = 3
    dt = 4800.0
    cfg = TKEConfig()
    assert cfg.veros_dz_slots is False
    e_old = np.array([4.0e-3, 1.0e-3, 5.0e-4])
    K_M = np.array([2.0e-2, 8.0e-3, 3.0e-3])
    K_H = np.maximum(K_M, cfg.kappaH_min)
    P_s = np.array([1.0e-7, 4.0e-8, 1.0e-8])
    N2 = np.array([6.0e-6, 2.0e-6, 1.0e-6])
    l_eps = np.array([3.0, 5.0, 2.0])
    flux = 2.5e-7

    sh = (1, 1, N)
    e_new = _solve_tke_backward_euler(
        e_old=jnp.asarray(e_old)[None, None, :],
        K_M_old=jnp.asarray(K_M)[None, None, :],
        K_H_old=jnp.asarray(K_H)[None, None, :],
        P_s=jnp.asarray(P_s)[None, None, :],
        N2=jnp.asarray(N2)[None, None, :],
        l_eps=jnp.asarray(l_eps)[None, None, :],
        dz_half=jnp.asarray(DZ_HALF_UC)[None, None, :],
        surface_flux=jnp.full(sh[:-1], flux),
        dt=dt, cfg=cfg,
    )

    dz_face = DZ_HALF_UC[:N - 1]            # legacy slot
    K_face = cfg.alpha_tke * 0.5 * (K_M[:-1] + K_M[1:])
    coef = K_face / dz_face
    # legacy dz_int_eff: [dz_face[0], dz_face[0], dz_face[1]]
    dz_int_eff = np.array([dz_face[0], dz_face[0], dz_face[1]])
    a = np.zeros(N)
    c = np.zeros(N)
    a[1:] = -dt * coef / dz_int_eff[1:]
    c[:-1] = -dt * coef / dz_int_eff[:-1]
    b_diff = -(a + c)
    diss = cfg.c_eps * np.sqrt(np.maximum(e_old, cfg.tke_background)) \
        / np.maximum(l_eps, cfg.mxl_min)
    sink = K_H * N2 / np.maximum(e_old, cfg.tke_background)
    b = 1.0 + dt * (diss + sink) + b_diff
    d = e_old + dt * P_s
    d[0] += dt * flux / DZ_HALF_UC[0]       # legacy injection slot
    expect = _dense_solve(a, b, c, d)
    expect = np.maximum(expect, cfg.tke_background)
    expect[0] = max(expect[0], cfg.tke_surface_min)

    np.testing.assert_allclose(np.asarray(e_new)[0, 0], expect, rtol=1e-12)


def test_solver_dz_cell_without_dz_surface_raises():
    N = 3
    args = dict(
        e_old=jnp.ones((1, 1, N)) * 1e-3,
        K_M_old=jnp.ones((1, 1, N)) * 1e-2,
        K_H_old=jnp.ones((1, 1, N)) * 1e-3,
        P_s=jnp.zeros((1, 1, N)),
        N2=jnp.zeros((1, 1, N)),
        l_eps=jnp.ones((1, 1, N)),
        dz_half=jnp.asarray(DZ_HALF_UC)[None, None, :],
        surface_flux=jnp.zeros((1, 1)),
        dt=100.0, cfg=TKEConfig(),
    )
    with pytest.raises(ValueError, match="dz_cell requires dz_surface"):
        _solve_tke_backward_euler(
            **args, dz_cell=jnp.asarray(DZ_CELL)[None, None, :])


# ---------------------------------------------------------------------------
# Item 5 — Veros positivity (explicit P_b, negative interior TKE,
# surface-only zero clamp; tke.py:224-245)
# ---------------------------------------------------------------------------


def test_solver_veros_positivity_negative_interior_hand_assembled():
    """Strong stratification sink drives the interior TKE NEGATIVE (the
    Veros energy debt); only the surface level is clamped at zero. The
    solve must equal the hand-assembled system with the buoyancy work
    fully explicit and sqrt(max(0,e)) dissipation linearisation."""
    N = 3
    dt = 4800.0
    cfg = TKEConfig(positivity="veros_surface_correction",
                    veros_dz_slots=True)
    e_old = np.array([5.0e-5, -2.0e-4, 1.0e-5])     # carried debt at k=1
    K_M = np.array([2.0e-4, 2.0e-4, 2.0e-4])
    K_H = np.array([2.0e-5, 2.0e-5, 2.0e-5])
    P_s = np.zeros(N)
    N2 = np.array([5.0e-4, 5.0e-4, 5.0e-4])         # strong stable sink
    l_eps = np.array([1.0e-8, 1.0e-8, 1.0e-8])

    sh = (1, 1, N)
    e_new = _solve_tke_backward_euler(
        e_old=jnp.asarray(e_old)[None, None, :],
        K_M_old=jnp.asarray(K_M)[None, None, :],
        K_H_old=jnp.asarray(K_H)[None, None, :],
        P_s=jnp.asarray(P_s)[None, None, :],
        N2=jnp.asarray(N2)[None, None, :],
        l_eps=jnp.asarray(l_eps)[None, None, :],
        dz_half=jnp.asarray(DZ_HALF_UC)[None, None, :],
        surface_flux=jnp.zeros(sh[:-1]),
        dt=dt, cfg=cfg,
        dz_cell=jnp.asarray(DZ_CELL)[None, None, :],
        dz_surface=jnp.full(sh[:-1], DZ_SURFACE),
    )

    # Independent reference: explicit P_b (both signs), sqrt(max(0,e))
    # dissipation, Veros metric slots, NO floors, surface clamp only.
    K_face = cfg.alpha_tke * 0.5 * (K_M[:-1] + K_M[1:])
    delta = dt * K_face / DZ_CELL[1:N]
    vol = DZ_HALF_UC
    a = np.zeros(N)
    c = np.zeros(N)
    a[1:] = -delta / vol[1:]
    c[:-1] = -delta / vol[:-1]
    b_diff = -(a + c)
    diss = cfg.c_eps * np.sqrt(np.maximum(e_old, 0.0)) \
        / np.maximum(l_eps, cfg.mxl_min)
    b = 1.0 + dt * diss + b_diff
    d = e_old + dt * (P_s - K_H * N2)
    expect = _dense_solve(a, b, c, d)
    expect[0] = max(expect[0], 0.0)

    got = np.asarray(e_new)[0, 0]
    np.testing.assert_allclose(got, expect, rtol=1e-12)
    assert got[1] < 0.0 and got[2] < 0.0      # the debt is carried
    assert got[0] >= 0.0                       # surface clamp


def test_orchestrator_veros_positivity_finite_no_nan():
    """End-to-end orchestrator with the faithful positivity stays finite
    (the debt flows through sqrt(max(0,e)) everywhere)."""
    cfg, kw = _column_inputs()
    cfg = cfg._replace(positivity="veros_surface_correction")
    kw["tke_old"] = kw["tke_old"].at[..., 1].set(-3.0e-4)   # carried debt
    out = tke_vertical_mixing(cfg=cfg, **kw)
    for arr in (out.K_M, out.K_H, out.tke_new, out.l_eps):
        assert bool(jnp.all(jnp.isfinite(arr)))
    # Floors did NOT fire: K stays at the scheme minima where in debt.
    assert float(jnp.min(out.K_M)) >= cfg.kappaM_min - 1e-15


def test_positivity_validation():
    cfg, kw = _column_inputs()
    with pytest.raises(ValueError, match="positivity"):
        tke_vertical_mixing(cfg=cfg._replace(positivity="bogus"), **kw)
    # veros_surface_correction requires the signed-N² adiabatic chain
    # (both mxl_choice ∈ {1,2} are debt-safe on it); the in-situ branch raises:
    with pytest.raises(ValueError, match="veros_surface_correction"):
        tke_vertical_mixing(
            cfg=cfg._replace(positivity="veros_surface_correction",
                             n2_mode="insitu"), **kw)


# ---------------------------------------------------------------------------
# Fix 3 — buoyancy-length growth allowance over dzt, Veros pass order
# ---------------------------------------------------------------------------


def _veros_mxl_reference(e, N2, dzt, mxl_min):
    """Numpy port of Veros tke.py:30-65 for the legoESM interior interfaces
    (top-down index k; interface k sits below cell k). The surface anchor
    only constrains the surface W point, which legoESM does not carry."""
    mxl = np.sqrt(2.0) * np.sqrt(np.maximum(0.0, e)) \
        / np.sqrt(np.maximum(1e-12, N2))
    M = len(e)
    for k in range(1, M):                   # downward, limited by above
        mxl[k] = min(mxl[k], mxl[k - 1] + dzt[k])
    for k in range(M - 2, -1, -1):          # upward, limited by below
        mxl[k] = min(mxl[k], mxl[k + 1] + dzt[k + 1])
    return np.maximum(mxl, mxl_min)


def test_veros_buoyancy_length_dzt_allowance_hand_computed():
    """A spiked column where the limiter binds in both directions:
    the JAX result must equal the numpy Veros-order/dzt reference, and
    differ from the legacy dz_half-allowance result on a stretched grid."""
    e = jnp.asarray([1.0e-4, 4.0e-1, 1.0e-4])      # big TKE spike at k=1
    N2 = jnp.asarray([1.0e-5, 1.0e-7, 1.0e-5])
    mxl_min = 1.0e-8
    dz_cell = jnp.asarray(DZ_CELL)

    out = _veros_buoyancy_length(
        e[None, None, :], N2[None, None, :],
        jnp.asarray(DZ_HALF_UC)[None, None, :], mxl_min,
        dz_cell=dz_cell[None, None, :])
    expect = _veros_mxl_reference(
        np.asarray(e), np.asarray(N2), DZ_CELL, mxl_min)
    np.testing.assert_allclose(np.asarray(out)[0, 0], expect, rtol=1e-13)

    # Sanity: the limiter actually bound (the spike leaks to neighbours
    # with exactly the dzt allowance).
    raw0 = np.sqrt(2.0) * np.sqrt(1e-4) / np.sqrt(1e-5)
    assert expect[0] < expect[1]                    # spike dominates
    np.testing.assert_allclose(expect[0], min(raw0, expect[1] + DZ_CELL[1]),
                               rtol=1e-13)

    # Legacy slot (dz_half allowance) differs on this stretched column.
    legacy = _veros_buoyancy_length(
        e[None, None, :], N2[None, None, :],
        jnp.asarray(DZ_HALF_UC)[None, None, :], mxl_min)
    assert not np.allclose(np.asarray(legacy), np.asarray(out))


def test_veros_buoyancy_length_legacy_branch_untouched():
    """dz_cell=None reproduces the historical up-then-down dz_half sweeps
    (pinned against an inline numpy port of the legacy branch)."""
    e = jnp.asarray([1.0e-4, 4.0e-1, 1.0e-4, 2.0e-3])
    N2 = jnp.asarray([1.0e-5, 1.0e-7, 1.0e-5, -1.0e-6])
    dz_half = jnp.asarray([12.0, 44.0, 36.0])
    mxl_min = 1.0e-8

    out = _veros_buoyancy_length(
        e[None, None, :], N2[None, None, :],
        dz_half[None, None, :], mxl_min)

    m = np.sqrt(2.0) * np.sqrt(np.maximum(0.0, np.asarray(e))) \
        / np.sqrt(np.maximum(1e-12, np.asarray(N2)))
    dz_step = np.asarray(dz_half)
    M = len(m)
    for k in range(M - 2, -1, -1):          # legacy backward (by below)
        m[k] = min(m[k], m[k + 1] + dz_step[min(k, M - 2)])
    for k in range(1, M):                   # legacy forward (by above)
        m[k] = min(m[k], m[k - 1] + dz_step[min(k - 1, M - 2)])
    m = np.maximum(m, mxl_min)
    np.testing.assert_allclose(np.asarray(out)[0, 0], m, rtol=1e-13)


# ---------------------------------------------------------------------------
# Orchestrator + wiring + AD
# ---------------------------------------------------------------------------


def _column_inputs(nlev=4, veros_slots=True):
    cfg = TKEConfig(
        n2_mode="adiabatic", prandtl_mode="constant", prognostic=True,
        veros_dz_slots=veros_slots)
    sh = (2, 3)
    u = jnp.broadcast_to(jnp.asarray([0.1, 0.05, 0.02, 0.0]), sh + (nlev,))
    v = jnp.zeros(sh + (nlev,))
    T = jnp.broadcast_to(jnp.asarray([18.0, 14.0, 9.0, 5.0]), sh + (nlev,))
    S = jnp.full(sh + (nlev,), 35.0)
    rho = jnp.broadcast_to(jnp.asarray([1024.0, 1025.0, 1026.0, 1027.0]),
                           sh + (nlev,))
    p = jnp.broadcast_to(jnp.asarray([1.0e5, 3.0e5, 6.0e5, 1.0e6]),
                         sh + (nlev,))
    dz_half = jnp.broadcast_to(jnp.asarray(DZ_HALF_UC), sh + (nlev - 1,))
    tke0 = jnp.full(sh + (nlev - 1,), 1.0e-4)
    tau = jnp.full(sh, 0.1)
    return cfg, dict(
        u_cell=u, v_cell=v, T_cell=T, S_cell=S, rho_cell=rho,
        dz_half=dz_half, tke_old=tke0, tau_x_surface=tau, tau_y_surface=None,
        dt=4800.0, rho_0=1024.0, g=9.81, n_iterations=1,
        p_cell=p, dz_ref=jnp.asarray(DZ_CELL), jacobian=jnp.ones(sh),
        eos_fn=_linear_eos(), dz_surface=jnp.full(sh, DZ_SURFACE))


def test_orchestrator_veros_slots_requires_dz_surface():
    cfg, kw = _column_inputs()
    kw["dz_surface"] = None
    with pytest.raises(ValueError, match="veros_dz_slots"):
        tke_vertical_mixing(cfg=cfg, **kw)


def test_orchestrator_veros_slots_finite_and_differs_from_legacy():
    cfg, kw = _column_inputs()
    out = tke_vertical_mixing(cfg=cfg, **kw)
    for arr in (out.K_M, out.K_H, out.tke_new, out.l_eps):
        assert bool(jnp.all(jnp.isfinite(arr)))
    cfg_legacy, kw2 = _column_inputs(veros_slots=False)
    out_legacy = tke_vertical_mixing(cfg=cfg_legacy, **kw2)
    # On the stretched u_centered column the slot fix changes the answer.
    assert not np.allclose(np.asarray(out.tke_new),
                           np.asarray(out_legacy.tke_new))


def test_orchestrator_veros_slots_grads_finite():
    """Reverse-mode AD flows finite through every rewired slot."""
    cfg, kw = _column_inputs()

    def loss(u, T, dzh):
        out = tke_vertical_mixing(
            cfg=cfg, **{**kw, "u_cell": u, "T_cell": T, "dz_half": dzh})
        return (jnp.sum(out.tke_new ** 2) + jnp.sum(out.K_M)
                + jnp.sum(out.K_H))

    gu, gT, gdz = jax.grad(loss, argnums=(0, 1, 2))(
        kw["u_cell"], kw["T_cell"], kw["dz_half"])
    for gr in (gu, gT, gdz):
        assert bool(jnp.all(jnp.isfinite(gr)))
    assert float(jnp.max(jnp.abs(gu))) > 0.0
    assert float(jnp.max(jnp.abs(gT))) > 0.0


def test_k_profiles_wiring_veros_slots():
    """compute_vertical_K_profiles passes dz_surface = -z_full_ref[0]·J on
    the prognostic-TKE path when the config opts in (finite K, carried TKE
    differs from the legacy-slot run on a stretched coordinate)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.surface_forcing.config import (
        SurfaceForcingConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.config import (
        VerticalMixingConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.k_profiles import (
        compute_vertical_K_profiles,
    )
    from legoesm.ocean.vertical import OceanZStarCoordinate

    # Stretched u_centered-style coordinate (centres NOT at midpoints).
    dz_ref = jnp.asarray(DZ_CELL, dtype=jnp.float64)
    z_half = jnp.concatenate([jnp.zeros(1), -jnp.cumsum(dz_ref)])
    z_full = jnp.asarray([-14.0, -26.0, -62.0, -98.0])  # stretched centres
    z_coord = OceanZStarCoordinate(
        n_levels=4, H_max=float(jnp.sum(dz_ref)),
        z_full_ref=z_full, z_half_ref=z_half, dz_ref=dz_ref,
        dz_half_ref=jnp.abs(z_full[:-1] - z_full[1:]))

    grid = create_latlon_grid(n_lat=4, n_lon=8)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=18.0, T_deep=4.0, S_uniform=35.0,
        H_max=float(jnp.sum(dz_ref)))
    rng = np.random.default_rng(1)
    state = state._replace(
        u=state.u.replace(data=jnp.asarray(
            0.1 * rng.standard_normal(state.u.data.shape))),
        T=state.T.replace(data=state.T.data
                          + 0.1 * rng.standard_normal(state.T.data.shape)))

    def run(veros_slots):
        tke_cfg = TKEConfig(
            n2_mode="adiabatic", prandtl_mode="constant", prognostic=True,
            veros_dz_slots=veros_slots)
        physics = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme="tke", tke=tke_cfg),
            lateral_mixing=LateralMixingConfig(scheme="none"),
            surface_forcing=SurfaceForcingConfig(scheme="none"),
            bottom_drag=BottomDragConfig(scheme="none"),
            convection=OceanConvectionConfig(scheme="none"),
            shortwave_penetration=None)
        tke0 = jnp.full(state.T.data.shape[:-1] + (3,), 1.0e-4)
        return compute_vertical_K_profiles(
            state, z_coord, None, physics,
            tke_old=tke0, dt_tke=4800.0, return_tke=True)

    K_on, A_on, tke_on = run(True)
    K_off, A_off, tke_off = run(False)
    for arr in (K_on, A_on, tke_on):
        assert bool(jnp.all(jnp.isfinite(arr)))
    assert not np.allclose(np.asarray(tke_on), np.asarray(tke_off))


def test_kappa_convention_veros_sqrte():
    """Veros tke.py:73: kappaM = c_k·mxl·sqrt(max(0,e)) — the legacy Gaspar
    √(2e) double-counts the √2 already inside the Veros buoyancy length."""
    from legoesm.ocean.physics.vertical_mixing.tke import (
        compute_K_from_tke,
        compute_mixing_lengths,
    )
    e = jnp.asarray([1.0e-3, 5.0e-4, -2.0e-4])[None, None, :]
    N2 = jnp.full((1, 1, 3), 1.0e-5)
    dzh = jnp.asarray(DZ_HALF_UC)[None, None, :]
    dzc = jnp.asarray(DZ_CELL)[None, None, :]
    shear = jnp.full((1, 1, 3), 1.0e-6)

    cfg_v = TKEConfig(n2_mode="adiabatic", prandtl_mode="constant",
                      veros_dz_slots=True, kappa_convention="veros_sqrte")
    l_k, _ = compute_mixing_lengths(e, N2, dzh, cfg_v, signed_n2=True,
                                    dz_cell=dzc)
    K_M, K_H = compute_K_from_tke(e, l_k, cfg_v, N2=N2, shear_sq=shear)
    # Hand-computed Veros chain: kappaM = min(max, c_k·mxl·sqrt(max(0,e)))
    # then floored; K_H = max(kappaH_min, K_M/Pr).
    mxl = np.asarray(l_k)[0, 0]
    en = np.asarray(e)[0, 0]
    expect = np.minimum(cfg_v.kappaM_max,
                        cfg_v.c_k * mxl * np.sqrt(np.maximum(en, 0.0)))
    expect = np.maximum(expect, cfg_v.kappaM_min)
    np.testing.assert_allclose(np.asarray(K_M)[0, 0], expect, rtol=1e-13)
    np.testing.assert_allclose(
        np.asarray(K_H)[0, 0],
        np.maximum(cfg_v.kappaH_min, expect / cfg_v.Prandtl_tke0), rtol=1e-13)
    # Negative-TKE debt: sqrttke = 0 -> K_M at the floor.
    assert float(K_M[0, 0, 2]) == cfg_v.kappaM_min

    # Legacy default is the √(2e) Gaspar form (×√2 on this path).
    cfg_g = cfg_v._replace(kappa_convention="gaspar_sqrt2e")
    K_Mg, _ = compute_K_from_tke(e, l_k, cfg_g, N2=N2, shear_sq=shear)
    np.testing.assert_allclose(
        float(K_Mg[0, 0, 0]) / float(K_M[0, 0, 0]), np.sqrt(2.0), rtol=1e-6)

    with pytest.raises(ValueError, match="kappa_convention"):
        compute_K_from_tke(e, l_k, cfg_v._replace(kappa_convention="x"),
                           N2=N2, shear_sq=shear)


def test_recipe_configs_opt_in():
    """The Veros-faithful recipes carry the metric fix; acc_basic also
    recycles K_diss_bot (Veros tke.py:178, the no-idemix branch)."""
    from legoesm.ocean.fidelity.veros_acc_basic_recipe import (
        ACC_BASIC_TKE_CONFIG,
    )
    from legoesm.ocean.fidelity.veros_acc_recipe import ACC_TKE_CONFIG

    assert ACC_TKE_CONFIG.veros_dz_slots is True
    assert ACC_BASIC_TKE_CONFIG.veros_dz_slots is True
    assert ACC_TKE_CONFIG.positivity == "veros_surface_correction"
    assert ACC_BASIC_TKE_CONFIG.positivity == "veros_surface_correction"
    assert ACC_TKE_CONFIG.kappa_convention == "veros_sqrte"
    assert ACC_BASIC_TKE_CONFIG.kappa_convention == "veros_sqrte"
    assert ACC_BASIC_TKE_CONFIG.source_bottom_drag_diss is True
    assert ACC_BASIC_TKE_CONFIG.source_eke_diss is False
    # Defaults stay off (bit-identity doctrine).
    assert TKEConfig().veros_dz_slots is False
    assert TKEConfig().positivity == "floor"
    assert TKEConfig().kappa_convention == "gaspar_sqrt2e"


def test_veros_buoyancy_length_upward_pass_binds():
    """Mutation-hole closure (review finding): a column where the UPWARD
    (surface->bottom in lego orientation) min-plus pass binds decisively —
    the 3-level hand-computed test never exercised it, so a corrupted
    ``dz_cell[k+1] -> dz_cell[k]`` allowance slot survived mutation testing.
    Construction lifted from the reviewer's sweep-equivalence probe
    (.physics-validator/tke_fix_review/sweep_equiv.py): random raw mxl with a
    spike, stretched dz — the staged faithful branch must equal the numpy
    Veros backwards+forwards reference exactly (the two monotone min-plus
    passes commute, so pass ORDER is free but the ALLOWANCE slots are not).
    """
    import numpy as np
    from legoesm.ocean.physics.vertical_mixing import tke as tke_mod

    rng = np.random.default_rng(3)
    nlev = 8
    n_int = nlev - 1
    mxl_min = 1e-8
    mxl_raw = np.abs(rng.standard_normal(n_int)) * 200.0 + 10.0
    mxl_raw[3] = 5000.0                       # spike binds the down pass
    mxl_raw[0] = 1.0                          # tiny surface value: the UP
    #                                           pass binds at k=1..(allowance
    #                                           chain from the surface)
    dz_cell = np.array([20.0, 40.0, 70.0, 110.0, 170.0, 260.0, 400.0, 600.0])

    def veros_reference(mxl, dzc):
        m = mxl.copy()
        # Veros backwards pass in lego orientation (bottom -> surface):
        for k in range(n_int - 2, -1, -1):
            m[k] = min(m[k], m[k + 1] + dzc[k + 1])
        # Veros forwards pass (surface -> bottom): allowance dzt[k]
        for k in range(1, n_int):
            m[k] = min(m[k], m[k - 1] + dzc[k])
        return np.maximum(m, mxl_min)

    ref = veros_reference(mxl_raw, dz_cell)
    # Assert the up pass genuinely binds (the mutation-hole condition):
    no_up = mxl_raw.copy()
    for k in range(n_int - 2, -1, -1):
        no_up[k] = min(no_up[k], no_up[k + 1] + dz_cell[k + 1])
    assert np.any(ref < np.maximum(no_up, mxl_min) - 1e-12), \
        "construction failed: upward pass never binds"

    # Drive the staged faithful branch: mxl_raw = sqrt(2)*sqrt(e)/sqrt(N2)
    # with e = 0.5 => mxl_raw = 1/sqrt(N2).
    e = jnp.full((n_int,), 0.5)
    N2 = jnp.asarray(1.0 / mxl_raw**2)
    dz_int = jnp.asarray(0.5 * (dz_cell[:-1] + dz_cell[1:]))
    out = tke_mod._veros_buoyancy_length(
        e, N2, dz_int, mxl_min, dz_cell=jnp.asarray(dz_cell))
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-12)


def test_negative_tke_gradients_are_finite():
    """AD-safety pin for the negative-TKE energy debt (#22 follow-on fix).

    ``sqrt(max(0, e))`` has a NaN derivative wherever e <= 0 (``d sqrt`` at 0
    is inf, the ``max`` tangent there is 0, and 0*inf = NaN) — with Veros's
    ``positivity="veros_surface_correction"`` the carried TKE goes negative,
    so EVERY parameter/state gradient across >= 2 model steps was NaN-poisoned
    (found while threading the kappa-scale c_k through the ACC recipe;
    .physics-validator/gm_adjoint_stab/RESULTS.md).  The double-``where`` fix
    keeps the primal BIT-IDENTICAL and the debt-branch derivative exactly 0.

    Pins: (a) grad/jvp of the three fixed sites are FINITE with negative e;
    (b) the primal equals the sqrt(max(0,e)) form bit-for-bit; (c) the raw
    pattern itself is NaN (non-vacuity: if the double-where is reverted to
    sqrt(max(0,e)), (a) goes red exactly like the raw pattern).
    """
    from legoesm.ocean.physics.vertical_mixing.tke import (
        _veros_buoyancy_length,
        compute_K_from_tke,
        compute_mixing_lengths,
    )

    # (c) non-vacuity: the unguarded pattern IS NaN under AD.
    raw = jax.grad(lambda x: jnp.sum(jnp.sqrt(jnp.maximum(0.0, x))))(
        jnp.asarray([-1.0e-4, 1.0e-3]))
    assert not bool(jnp.all(jnp.isfinite(raw)))

    e = jnp.asarray([1.0e-3, -2.0e-4, 0.0])[None, None, :]
    N2 = jnp.full((1, 1, 3), 1.0e-5)
    dzh = jnp.asarray(DZ_HALF_UC)[None, None, :]
    dzc = jnp.asarray(DZ_CELL)[None, None, :]
    shear = jnp.full((1, 1, 3), 1.0e-6)
    cfg = TKEConfig(n2_mode="adiabatic", prandtl_mode="constant",
                    veros_dz_slots=True, kappa_convention="veros_sqrte",
                    positivity="veros_surface_correction")

    # (b) primal bit-identity with the sqrt(max(0,e)) form.
    l_buoy = _veros_buoyancy_length(e, N2, dzh, cfg.mxl_min, dz_cell=dzc)
    sqrttke_ref = jnp.sqrt(jnp.maximum(0.0, e))
    mxl_ref = jnp.sqrt(2.0) * sqrttke_ref / jnp.sqrt(jnp.maximum(1e-12, N2))
    # reproduce the two limiter sweeps on the reference start value
    import numpy as _np
    m = _np.asarray(mxl_ref)[0, 0].copy()
    dzc_n = _np.asarray(dzc)[0, 0]
    for k in range(1, 3):
        m[k] = min(m[k], m[k - 1] + dzc_n[k])
    for k in range(1, -1, -1):
        m[k] = min(m[k], m[k + 1] + dzc_n[k + 1])
    m = _np.maximum(m, cfg.mxl_min)
    _np.testing.assert_array_equal(_np.asarray(l_buoy)[0, 0], m)

    # (a) finite gradients through the three fixed sites.
    def loss_len(ee):
        return jnp.sum(_veros_buoyancy_length(
            ee, N2, dzh, cfg.mxl_min, dz_cell=dzc))

    g1 = jax.grad(loss_len)(e)
    assert bool(jnp.all(jnp.isfinite(g1)))

    def loss_K(ee):
        l_k, _ = compute_mixing_lengths(ee, N2, dzh, cfg, signed_n2=True,
                                        dz_cell=dzc)
        K_M, K_H = compute_K_from_tke(ee, l_k, cfg, N2=N2, shear_sq=shear)
        return jnp.sum(K_M) + jnp.sum(K_H)

    g2 = jax.grad(loss_K)(e)
    assert bool(jnp.all(jnp.isfinite(g2)))
    jv = jax.jvp(loss_K, (e,), (jnp.ones_like(e),))[1]
    assert bool(jnp.isfinite(jv))
