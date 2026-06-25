"""Tests for the Gaspar 1990 / Burchard 2002 TKE closure (Phase G.1a).

Tests target the closure module
:mod:`legoesm.ocean.physics.vertical_mixing.tke` directly. The
integration into ``_apply_implicit_vertical_mixing`` (with prognostic
TKE state on ``LatLonCGridOceanState.tke``) is deferred to a follow-up
commit; see ``docs/ocean/fidelity/phase_g_veros_recipe_audit.md`` G.1a.

Test surface:

1. **Mixing length** — Bougeault-Lacarrere asymmetric form gives
   plausible lengths (positive, bounded, increases with sqrt(TKE)
   for fixed N²).
2. **Shear production** — zero shear → zero production; uniform
   shear → analytic value.
3. **N²** — neutral stratification → zero; stable → positive.
4. **K_M / K_H** — clamped to ``kappaM_min`` / ``kappaH_min``;
   ``K_H == K_M`` in canonical form (Pr_t=1).
5. **Backward-Euler step** — positivity preservation (TKE never goes
   negative even with zero source).
6. **End-to-end orchestrator** — runs without NaN on a non-trivial
   state; converges toward steady state when iterated (Mode B);
   prognostic mode (Mode A) carries TKE forward consistently.
7. **Wallace surface flux** — wind stress drives TKE up; calm sea
   gives no surface contribution.
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
    TKEOutput,
    _compute_N2,
    _vertical_shear_squared,
    compute_K_from_tke,
    compute_mixing_lengths,
    tke_vertical_mixing,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# 1. Mixing length
# ---------------------------------------------------------------------------


def test_mixing_length_positive_and_bounded():
    """For positive TKE and positive N², l_k and l_eps are positive
    and bounded by the upper limit (2 * dz_half)."""
    nlev = 8
    dz = 100.0
    e = jnp.full((nlev - 1,), 1e-4)
    N2 = jnp.full((nlev - 1,), 1e-6)
    dz_half = jnp.full((nlev - 1,), dz)
    cfg = TKEConfig()
    l_k, l_eps = compute_mixing_lengths(e, N2, dz_half, cfg)
    assert jnp.all(l_k > 0)
    assert jnp.all(l_eps > 0)
    assert jnp.all(l_k <= 2 * dz + 1e-9)
    assert jnp.all(l_eps <= 2 * dz + 1e-9)


def test_mixing_length_grows_with_tke():
    """l_k ∝ sqrt(2e/N²) in the closed-form approximation. Doubling e
    grows l_k by sqrt(2) (modulo the upper-bound clip). Requires
    ≥3 interfaces so the Bougeault-Lacarrère ``dz_int`` slice is
    non-empty."""
    n_int = 4
    dz_half = jnp.full((n_int,), 10000.0)   # very large so bound is not tight
    N2 = jnp.full((n_int,), 1e-4)
    cfg = TKEConfig(mxl_min=1e-8)
    e1 = jnp.full((n_int,), 1e-4)
    e2 = e1 * 2
    l_k1, _ = compute_mixing_lengths(e1, N2, dz_half, cfg)
    l_k2, _ = compute_mixing_lengths(e2, N2, dz_half, cfg)
    # Take an interior interface so it's not affected by edge handling.
    ratio = float(l_k2[1] / l_k1[1])
    assert abs(ratio - float(jnp.sqrt(2.0))) < 0.05


def test_mixing_length_choice1_negative_tke_is_finite():
    """``tke_mxl_choice=1`` must stay FINITE when the prognostic TKE carries
    Veros's interior NEGATIVE-energy debt (``e < 0``).

    Regression for the global_1deg south-wall step-2 blowup: the choice=1
    length scale used to be ``sqrt(2·e / max(N², eps))`` — with the RAW
    (possibly negative) ``e`` INSIDE the sqrt, so ``sqrt(negative) = NaN``
    the first step the prognostic TKE went negative.  The faithful Veros form
    (tke.py:30,34) clamps ``sqrttke = sqrt(max(0, e))`` BEFORE the division.
    Choice=2 already clamped (via ``_veros_buoyancy_length``), so the 4° /
    flexible / ACC recipes never hit it; only the choice=1 global_1deg path
    did.

    NON-VACUOUS: with the pre-fix ``sqrt(2·e/N²)`` body this assertion FAILS
    (the negative-``e`` interfaces are NaN); the manufactured column mirrors
    the real blowup cell (mixed-sign interior TKE, small positive N²)."""
    n_int = 6
    # Mixed-sign carried TKE — exactly the Veros interior debt that triggered
    # the blowup (s1.tke had min ≈ -1.7e-5 at the south wall).
    e = jnp.array([1.0e-4, -1.7e-5, 2.0e-4, -5.0e-6, 8.0e-5, 1.0e-6])
    N2 = jnp.full((n_int,), 5.0e-6)            # finite, statically stable
    dz_half = jnp.full((n_int,), 50.0)
    dz_cell = jnp.full((n_int + 1,), 50.0)
    cfg = TKEConfig(tke_mxl_choice=1, mxl_min=1.0e-8)

    l_k, l_eps = compute_mixing_lengths(
        e, N2, dz_half, cfg, signed_n2=True, dz_cell=dz_cell)
    assert jnp.all(jnp.isfinite(l_k)), l_k
    assert jnp.all(jnp.isfinite(l_eps)), l_eps
    # Negative-TKE interfaces collapse to the floor (sqrttke = 0 there).
    assert float(l_k[1]) == pytest.approx(cfg.mxl_min)
    assert float(l_k[3]) == pytest.approx(cfg.mxl_min)
    # Positive-TKE interfaces follow the Veros buoyancy length sqrt(2e/N²).
    expected = float(jnp.sqrt(2.0 * e[0] / N2[0]))
    assert float(l_k[0]) == pytest.approx(expected, rel=1e-6)

    # AD-safe: the double-``where`` must give a FINITE gradient through the
    # negative-TKE debt (a plain sqrt(max(0,e)) has a NaN derivative at e<=0).
    def _loss(ev):
        lk, _ = compute_mixing_lengths(
            ev, N2, dz_half, cfg, signed_n2=True, dz_cell=dz_cell)
        return jnp.sum(lk)
    grad = jax.grad(_loss)(e)
    assert jnp.all(jnp.isfinite(grad)), grad


# ---------------------------------------------------------------------------
# 2. Shear production
# ---------------------------------------------------------------------------


def test_zero_shear_zero_production():
    nlev = 5
    u = jnp.full((nlev,), 0.1)
    v = jnp.full((nlev,), -0.05)
    dz_half = jnp.full((nlev - 1,), 50.0)
    S2 = _vertical_shear_squared(u, v, dz_half)
    np.testing.assert_allclose(np.asarray(S2), 0.0, atol=1e-14)


def test_uniform_shear_known_value():
    """u(z) linear, du/dz constant ⇒ |du/dz|² = (du/dz)²."""
    nlev = 5
    dz = 50.0
    du_dz = 0.002   # 0.1 m/s over 50 m
    u_vals = jnp.arange(nlev, dtype=jnp.float64) * dz * du_dz
    v_vals = jnp.zeros(nlev)
    dz_half = jnp.full((nlev - 1,), dz)
    S2 = _vertical_shear_squared(u_vals, v_vals, dz_half)
    np.testing.assert_allclose(
        np.asarray(S2), np.full(nlev - 1, du_dz ** 2), rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# 3. N²
# ---------------------------------------------------------------------------


def test_neutral_stratification_zero_n2():
    """Uniform rho ⇒ drho/dz = 0 ⇒ N² = 0."""
    nlev = 6
    rho = jnp.full((nlev,), 1025.0)
    dz_half = jnp.full((nlev - 1,), 50.0)
    N2 = _compute_N2(rho, dz_half, 1025.0)
    np.testing.assert_allclose(np.asarray(N2), 0.0, atol=1e-14)


def test_stable_stratification_positive_n2():
    """rho increases with depth ⇒ N² > 0."""
    nlev = 6
    # rho[k=0] = lightest (surface), rho[k=nlev-1] = densest (bottom)
    rho = 1024.0 + 1.0 * jnp.arange(nlev, dtype=jnp.float64) / (nlev - 1)
    dz_half = jnp.full((nlev - 1,), 50.0)
    N2 = _compute_N2(rho, dz_half, 1025.0)
    assert jnp.all(N2 > 0)


# ---------------------------------------------------------------------------
# 4. K from TKE
# ---------------------------------------------------------------------------


def test_K_clamped_at_minima():
    """When TKE is at background floor, K_M and K_H should hit
    kappaM_min / kappaH_min and not go below."""
    cfg = TKEConfig(
        c_k=0.1, kappaM_min=2e-4, kappaH_min=2e-5,
        tke_background=1e-12,
    )
    e = jnp.full((3,), cfg.tke_background)
    l_k = jnp.full((3,), 1.0)
    K_M, K_H = compute_K_from_tke(e, l_k, cfg)
    assert jnp.all(K_M >= cfg.kappaM_min - 1e-15)
    assert jnp.all(K_H >= cfg.kappaH_min - 1e-15)


def test_K_proportional_to_l_k_sqrt_e():
    """K_M = c_k * l_k * sqrt(2 e). Verify the scaling."""
    cfg = TKEConfig(c_k=0.1, kappaM_min=0.0, kappaH_min=0.0)
    e = jnp.array([1e-3])
    l_k = jnp.array([5.0])
    K_M, _ = compute_K_from_tke(e, l_k, cfg)
    expected = cfg.c_k * l_k * jnp.sqrt(2.0 * e)
    np.testing.assert_allclose(np.asarray(K_M), np.asarray(expected),
                                rtol=1e-12)


# ---------------------------------------------------------------------------
# 5. Backward-Euler positivity
# ---------------------------------------------------------------------------


def test_tke_backward_euler_stays_positive():
    """Even with zero shear and stable stratification (i.e., only
    sink terms acting), backward-Euler must not drive TKE negative."""
    cfg = TKEConfig()
    nlev = 10
    u = jnp.zeros(nlev)
    v = jnp.zeros(nlev)
    # Stable stratification
    rho = 1024.0 + 2.0 * jnp.arange(nlev, dtype=jnp.float64) / (nlev - 1)
    # Cell-centre arrays
    T_dummy = jnp.zeros_like(u)
    S_dummy = jnp.full_like(u, 35.0)
    dz_half = jnp.full((nlev - 1,), 50.0)
    tke_old = jnp.full((nlev - 1,), 1e-3)
    out = tke_vertical_mixing(
        u, v, T_dummy, S_dummy, rho, dz_half,
        tke_old=tke_old, tau_x_surface=None, tau_y_surface=None,
        dt=3600.0, cfg=cfg, n_iterations=1,
    )
    assert jnp.all(out.tke_new >= cfg.tke_background - 1e-15)


# ---------------------------------------------------------------------------
# 6. End-to-end
# ---------------------------------------------------------------------------


def test_orchestrator_runs_on_realistic_state():
    """Run the orchestrator on a perturbed-from-rest state and verify
    the output is finite and has the right shapes."""
    cfg = TKEConfig()
    nlev = 12
    rng = np.random.default_rng(0)
    u = jnp.asarray(0.05 * rng.standard_normal(nlev))
    v = jnp.asarray(0.05 * rng.standard_normal(nlev))
    rho = jnp.asarray(1024.0 + 2.0 * np.linspace(0, 1, nlev))
    T = jnp.asarray(15.0 - 12.0 * np.linspace(0, 1, nlev))
    S = jnp.full((nlev,), 35.0)
    dz_half = jnp.full((nlev - 1,), 100.0)
    out = tke_vertical_mixing(
        u, v, T, S, rho, dz_half,
        tke_old=None, tau_x_surface=None, tau_y_surface=None,
        dt=3600.0, cfg=cfg, n_iterations=3,
    )
    assert out.K_M.shape == (nlev - 1,)
    assert out.K_H.shape == (nlev - 1,)
    assert out.tke_new.shape == (nlev - 1,)
    assert jnp.all(jnp.isfinite(out.K_M))
    assert jnp.all(jnp.isfinite(out.K_H))
    assert jnp.all(jnp.isfinite(out.tke_new))
    assert jnp.all(out.tke_new >= cfg.tke_background - 1e-15)


def test_orchestrator_prognostic_mode_carries_state():
    """Mode A (prognostic): passing the previous step's TKE drives the
    closure forward, and after enough steps the TKE converges (within
    a tolerance) to the same quasi-steady-state as Mode B's iterated
    diagnostic computation."""
    cfg = TKEConfig()
    nlev = 12
    rng = np.random.default_rng(1)
    u = jnp.asarray(0.05 * rng.standard_normal(nlev))
    v = jnp.asarray(0.05 * rng.standard_normal(nlev))
    rho = jnp.asarray(1024.0 + 2.0 * np.linspace(0, 1, nlev))
    T = jnp.full((nlev,), 10.0)
    S = jnp.full((nlev,), 35.0)
    dz_half = jnp.full((nlev - 1,), 100.0)
    dt = 3600.0

    # Mode B: 5 iterations from background.
    out_b = tke_vertical_mixing(
        u, v, T, S, rho, dz_half,
        tke_old=None, tau_x_surface=None, tau_y_surface=None,
        dt=dt, cfg=cfg, n_iterations=5,
    )
    # Mode A: 5 single-step prognostic calls.
    tke_prog = jnp.full((nlev - 1,), cfg.tke_background)
    for _ in range(5):
        out_a = tke_vertical_mixing(
            u, v, T, S, rho, dz_half,
            tke_old=tke_prog, tau_x_surface=None, tau_y_surface=None,
            dt=dt, cfg=cfg, n_iterations=1,
        )
        tke_prog = out_a.tke_new
    # The two modes should converge to the same TKE — Mode B iterates
    # the SAME backward-Euler step, while Mode A advances dt at a time.
    # They are mathematically identical sequences.
    np.testing.assert_allclose(
        np.asarray(out_b.tke_new), np.asarray(tke_prog),
        rtol=1e-12, atol=1e-12,
    )


# ---------------------------------------------------------------------------
# 7. Wallace surface flux
# ---------------------------------------------------------------------------


def test_wind_stress_drives_surface_tke_up():
    """With strong wind stress and minimal shear, surface TKE should
    rise above the calm-sea baseline."""
    cfg = TKEConfig()
    nlev = 8
    u = jnp.zeros(nlev)
    v = jnp.zeros(nlev)
    rho = jnp.full((nlev,), 1025.0)   # neutral stratification
    T = jnp.full((nlev,), 10.0)
    S = jnp.full((nlev,), 35.0)
    dz_half = jnp.full((nlev - 1,), 50.0)

    # Calm sea — no wind.
    out_calm = tke_vertical_mixing(
        u, v, T, S, rho, dz_half,
        tke_old=None, tau_x_surface=None, tau_y_surface=None,
        dt=3600.0, cfg=cfg, n_iterations=3,
    )

    # Strong wind — 0.2 N/m^2 zonal.
    tau_x = jnp.asarray(0.2)
    tau_y = jnp.zeros(())
    out_wind = tke_vertical_mixing(
        u, v, T, S, rho, dz_half,
        tke_old=None, tau_x_surface=tau_x, tau_y_surface=tau_y,
        dt=3600.0, cfg=cfg, n_iterations=3,
    )

    # Wind-driven case has higher surface (top interface) TKE.
    assert float(out_wind.tke_new[0]) > float(out_calm.tke_new[0])
