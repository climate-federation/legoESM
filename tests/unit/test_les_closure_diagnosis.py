"""Unit tests for :mod:`legoesm.atmosphere.dynamics.les_closure_diagnosis`.

Stage 6 of ``docs/COMPARE_REANALYSIS.md``: diagnose closure coefficients by
inverting LES-resolved fluxes.  Analytic checks: K recovered exactly from a
down-gradient flux on a linear profile; ill-posed (zero-gradient / counter-
gradient) flagged invalid; Prandtl mixing length round-trip; entrainment
velocity from a crafted buoyancy-flux minimum + inversion jump.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.les_closure_diagnosis import (
    eddy_diffusivity_from_flux,
    entrainment_velocity_from_buoyancy_flux,
    mean_gradient_at_interfaces,
    mixing_length_from_momentum_diffusivity,
)


def test_mean_gradient_linear_profile():
    z = jnp.array([0.0, 100.0, 200.0, 300.0])
    phi = 2.0 + 0.5 * z  # slope 0.5
    grad = mean_gradient_at_interfaces(phi, z)
    assert grad.shape == (3,)
    np.testing.assert_allclose(np.asarray(grad), 0.5, rtol=1e-12)


def test_eddy_diffusivity_recovers_K0():
    """flux = -K0 · ∂φ/∂z on a linear profile ⇒ diagnosed K == K0."""
    z = jnp.linspace(0.0, 400.0, 5)
    slope = -0.01  # K/m (stable θ gradient)
    phi = 290.0 + slope * z
    K0 = 5.0
    grad = mean_gradient_at_interfaces(phi, z)
    flux = -K0 * grad  # exact down-gradient flux at interfaces
    diag = eddy_diffusivity_from_flux(flux, phi, z)
    np.testing.assert_allclose(np.asarray(diag.K), K0, rtol=1e-12)
    assert bool(jnp.all(diag.valid))


def test_eddy_diffusivity_zero_gradient_invalid():
    z = jnp.linspace(0.0, 400.0, 5)
    phi = jnp.full((5,), 290.0)  # zero gradient everywhere
    flux = jnp.array([0.1, -0.2, 0.05, 0.0])
    diag = eddy_diffusivity_from_flux(flux, phi, z)
    assert not bool(jnp.any(diag.valid))
    np.testing.assert_array_equal(np.asarray(diag.K), np.zeros(4))


def test_eddy_diffusivity_countergradient_invalid():
    """A flux with the SAME sign as the gradient ⇒ K<0 ⇒ invalid (zeroed)."""
    z = jnp.linspace(0.0, 400.0, 5)
    phi = 290.0 + 0.01 * z  # positive gradient
    flux = mean_gradient_at_interfaces(phi, z) * 3.0  # same sign -> K=-3 <0
    diag = eddy_diffusivity_from_flux(flux, phi, z)
    assert not bool(jnp.any(diag.valid))
    np.testing.assert_array_equal(np.asarray(diag.K), np.zeros(4))


def test_eddy_diffusivity_ad_safe_at_zero_gradient():
    z = jnp.linspace(0.0, 400.0, 5)

    def loss(scale):
        phi = jnp.full((5,), 290.0)  # zero gradient -> masked branch
        flux = jnp.array([0.1, -0.2, 0.05, 0.0]) * scale
        return jnp.sum(eddy_diffusivity_from_flux(flux, phi, z).K)

    g = jax.grad(loss)(1.0)
    assert jnp.isfinite(g)


def test_mixing_length_round_trip():
    """K_m = ℓ0²·|shear| ⇒ diagnosed ℓ == ℓ0."""
    shear = jnp.array([0.02, 0.05, 0.1])
    ell0 = 50.0
    K_m = ell0**2 * jnp.abs(shear)
    ell, valid = mixing_length_from_momentum_diffusivity(K_m, shear)
    np.testing.assert_allclose(np.asarray(ell), ell0, rtol=1e-12)
    assert bool(jnp.all(valid))


def test_mixing_length_grad_safe_at_zero_Km():
    """sqrt(0) AD hazard: grad must be finite when K_m hits exactly zero."""
    shear = jnp.array([0.05, 0.05, 0.05])

    def loss(scale):
        K_m = jnp.array([0.0, 1.0, 2.0]) * scale  # first element exactly 0
        ell, _ = mixing_length_from_momentum_diffusivity(K_m, shear)
        return jnp.sum(ell)

    g = jax.grad(loss)(1.0)
    assert jnp.isfinite(g)


def test_mixing_length_zero_shear_invalid():
    shear = jnp.array([0.0, 1e-12, 0.0])
    K_m = jnp.array([10.0, 10.0, 10.0])
    ell, valid = mixing_length_from_momentum_diffusivity(K_m, shear)
    assert not bool(jnp.any(valid))
    np.testing.assert_array_equal(np.asarray(ell), np.zeros(3))


def test_entrainment_velocity_from_crafted_inversion():
    # Buoyancy flux: surface-positive, negative minimum at interface k=2.
    w_thetav = jnp.array([0.03, 0.01, -0.02, 0.0])  # (nlev-1,) min at idx 2
    # theta_v jumps +2 K across interface 2 (full levels nlev=5).
    thetav_full = jnp.array([300.0, 300.5, 301.0, 303.0, 303.5])
    delta = thetav_full[1:] - thetav_full[:-1]  # [0.5,0.5,2.0,0.5]
    z_iface = jnp.array([100.0, 300.0, 600.0, 900.0])
    diag = entrainment_velocity_from_buoyancy_flux(w_thetav, thetav_full, z_iface)
    assert int(diag.inversion_index) == 2
    assert bool(diag.valid)
    # w_e = -(-0.02)/2.0 = 0.01 m/s
    # fp32-safe tolerances: this file lives in CI's fp32-by-default unit tier (no
    # JAX_ENABLE_X64), where these analytic inverses recover to ~1e-8 relative.
    # rel/rtol=1e-5 is deterministic at fp32 AND x64 yet still catches any real
    # (≥0.001%) bug; the prior 1e-12/1e-9 passed ONLY via the session-wide x64 leak
    # from a test_correction_loop import (order-dependent / xdist-fragile).
    assert float(diag.w_entrainment) == pytest.approx(0.01, rel=1e-5)
    assert float(diag.delta_thetav) == pytest.approx(2.0, rel=1e-5)
    assert float(diag.z_inversion) == pytest.approx(600.0)


def test_entrainment_invalid_when_no_stable_jump():
    # Minimum buoyancy flux sits where the layer is well-mixed (no jump).
    w_thetav = jnp.array([-0.01, -0.02, 0.0])
    thetav_full = jnp.array([300.0, 300.0, 300.0, 300.0])  # zero jumps
    z_iface = jnp.array([100.0, 300.0, 600.0])
    diag = entrainment_velocity_from_buoyancy_flux(w_thetav, thetav_full, z_iface)
    assert not bool(diag.valid)
    assert float(diag.w_entrainment) == 0.0
    assert jnp.isnan(diag.z_inversion)


def test_entrainment_invalid_when_flux_positive_at_min():
    # All buoyancy flux positive ⇒ "min" still >=0 ⇒ not an entrainment flux.
    w_thetav = jnp.array([0.03, 0.01, 0.02])
    thetav_full = jnp.array([300.0, 301.0, 302.0, 305.0])  # stable jumps
    z_iface = jnp.array([100.0, 300.0, 600.0])
    diag = entrainment_velocity_from_buoyancy_flux(w_thetav, thetav_full, z_iface)
    assert not bool(diag.valid)


def test_entrainment_boundary_minimum_invalid():
    """A buoyancy-flux minimum on a boundary interface is not a capping
    inversion (e.g. a surface-flux minimum) and must be rejected."""
    # Most negative flux at index 0 (the lowest interface).
    w_thetav = jnp.array([-0.05, -0.01, 0.0, 0.02])
    thetav_full = jnp.array([300.0, 303.0, 303.5, 304.0, 304.5])  # stable jumps
    z_iface = jnp.array([100.0, 300.0, 600.0, 900.0])
    diag = entrainment_velocity_from_buoyancy_flux(w_thetav, thetav_full, z_iface)
    assert int(diag.inversion_index) == 0
    assert not bool(diag.valid)
    assert float(diag.w_entrainment) == 0.0


def test_entrainment_jit_and_grad():
    w_thetav = jnp.array([0.03, 0.01, -0.02, 0.0])
    thetav_full = jnp.array([300.0, 300.5, 301.0, 303.0, 303.5])
    z_iface = jnp.array([100.0, 300.0, 600.0, 900.0])

    out = jax.jit(
        lambda w: entrainment_velocity_from_buoyancy_flux(w, thetav_full, z_iface)
    )(w_thetav)
    assert jnp.isfinite(out.w_entrainment)

    def loss(w):
        return entrainment_velocity_from_buoyancy_flux(
            w, thetav_full, z_iface
        ).w_entrainment

    g = jax.grad(loss)(w_thetav)
    assert g.shape == w_thetav.shape
    assert bool(jnp.all(jnp.isfinite(g)))


# --- momentum diffusivity (shear-projected) + dimensionless C_K -----------
from legoesm.atmosphere.dynamics.les_closure_diagnosis import (  # noqa: E402
    clubb_coefficient_from_diffusivity,
    momentum_diffusivity_from_fluxes,
)


def test_momentum_diffusivity_recovers_km():
    # u = S0*z (constant shear), v=0, down-gradient w'u' = -Km0*S0 ⇒ recover Km0.
    z = jnp.array([0.0, 100.0, 200.0, 300.0])
    s0, km0 = 0.01, 5.0
    u = s0 * z
    v = jnp.zeros_like(z)
    w_u = jnp.full((3,), -km0 * s0)
    Km, valid = momentum_diffusivity_from_fluxes(w_u, jnp.zeros((3,)), u, v, z)
    assert bool(jnp.all(valid))
    np.testing.assert_allclose(np.asarray(Km), km0, rtol=1e-5)


def test_momentum_diffusivity_projects_misaligned_flux():
    # Shear purely in u; a v-flux component is cross-shear and must be projected
    # out (K_m depends only on the along-shear flux w'u').
    z = jnp.array([0.0, 100.0, 200.0])
    u = 0.01 * z
    v = jnp.zeros_like(z)
    w_u = jnp.full((2,), -0.05)            # along-shear
    Km_a, _ = momentum_diffusivity_from_fluxes(w_u, jnp.zeros((2,)), u, v, z)
    Km_b, _ = momentum_diffusivity_from_fluxes(w_u, jnp.full((2,), 9.9), u, v, z)
    np.testing.assert_allclose(np.asarray(Km_a), np.asarray(Km_b))  # v-flux ignored


def test_momentum_diffusivity_countergradient_invalid():
    z = jnp.array([0.0, 100.0, 200.0])
    u = 0.01 * z
    Km, valid = momentum_diffusivity_from_fluxes(
        jnp.full((2,), +0.05), jnp.zeros((2,)), u, jnp.zeros_like(z), z)  # up-gradient
    assert not bool(jnp.any(valid))


def test_momentum_diffusivity_zero_shear_invalid_and_ad_safe():
    z = jnp.array([0.0, 100.0, 200.0])
    flat = jnp.zeros((3,))                  # no shear
    w_u = jnp.full((2,), -0.05)

    def loss(wu):
        Km, _ = momentum_diffusivity_from_fluxes(wu, jnp.zeros((2,)), flat, flat, z)
        return jnp.sum(Km ** 2)

    _, valid = momentum_diffusivity_from_fluxes(w_u, jnp.zeros((2,)), flat, flat, z)
    assert not bool(jnp.any(valid))
    g = jax.grad(loss)(w_u)
    assert bool(jnp.all(jnp.isfinite(g)))   # no NaN gradient at zero shear


def test_clubb_coefficient_dimensionless_value():
    # C_K = Km/(l*sqrt(wp2)) = 5/(50*0.5) = 0.2.
    Km = jnp.full((3,), 5.0)
    CK, valid = clubb_coefficient_from_diffusivity(
        Km, jnp.ones((3,), bool), jnp.full((3,), 50.0), jnp.full((3,), 0.25))
    assert bool(jnp.all(valid))
    np.testing.assert_allclose(np.asarray(CK), 0.2, rtol=1e-5)


def test_clubb_coefficient_exact_inverse_of_gcm_forward():
    """The C_K diagnosis is the EXACT inverse of the GCM closure (the iter-46
    claim that underpins 'parameters estimated from LES'): injecting a known C_K
    through the REAL GCM forward Km = broadcast_column_param(C_K, l_mix)·l_mix·
    √wp2 (exactly clubb_lite.py) and inverting recovers C_K to machine precision —
    so the diagnosed coefficient IS the one the GCM would use (modulo the
    documented LES-vs-GCM-wp2 caveat). Validated for BOTH the scalar (production
    column-constant) AND the per-column-array (LES-corrected) usage of C_K."""
    from legoesm.atmosphere.physics._shared import broadcast_column_param

    l_mix = jnp.array([50.0, 100.0, 150.0, 200.0])
    wp2 = jnp.array([0.5, 1.0, 1.5, 2.0])

    # (1) Scalar C_K — the production, vertically-constant coefficient.
    ck_true = 0.37
    km = broadcast_column_param(ck_true, l_mix) * l_mix * jnp.sqrt(wp2)   # GCM forward
    ck_rec, valid = clubb_coefficient_from_diffusivity(
        km, jnp.ones_like(km, bool), l_mix, wp2)
    assert bool(jnp.all(valid))
    np.testing.assert_allclose(np.asarray(ck_rec), ck_true, rtol=1e-5)

    # (2) Per-column array C_K — the LES-informed per-column correction (2 columns).
    l2 = jnp.broadcast_to(l_mix, (2, 4))
    w2 = jnp.broadcast_to(wp2, (2, 4))
    ck_col = jnp.array([0.25, 0.6])                                      # (ncol,)
    km2 = broadcast_column_param(ck_col, l2) * l2 * jnp.sqrt(w2)         # (2,4) forward
    ck_rec2, valid2 = clubb_coefficient_from_diffusivity(
        km2, jnp.ones_like(km2, bool), l2, w2)
    assert bool(jnp.all(valid2))
    # Each column's recovered C_K equals its injected per-column value at every level.
    np.testing.assert_allclose(
        np.asarray(ck_rec2),
        np.broadcast_to(np.asarray(ck_col)[:, None], (2, 4)), rtol=1e-5)


def test_clubb_coefficient_low_wp2_invalid_and_ad_safe():
    Km = jnp.full((3,), 5.0)
    wp2 = jnp.array([0.25, 1.0e-12, 0.25])   # middle below the floor
    CK, valid = clubb_coefficient_from_diffusivity(
        Km, jnp.ones((3,), bool), jnp.full((3,), 50.0), wp2)
    assert bool(valid[0]) and not bool(valid[1]) and bool(valid[2])
    assert float(CK[1]) == 0.0

    def loss(w):
        CK, _ = clubb_coefficient_from_diffusivity(
            Km, jnp.ones((3,), bool), jnp.full((3,), 50.0), w)
        return jnp.sum(CK ** 2)

    g = jax.grad(loss)(wp2)
    assert bool(jnp.all(jnp.isfinite(g)))    # sqrt(wp2) double-where is AD-safe


def test_prandtl_number_ratio():
    # Pr_t = Km/Kh = 4/5 = 0.8.
    from legoesm.atmosphere.dynamics.les_closure_diagnosis import (
        prandtl_number_from_diffusivities,
    )
    Pr, valid = prandtl_number_from_diffusivities(
        jnp.full((3,), 4.0), jnp.ones((3,), bool),
        jnp.full((3,), 5.0), jnp.ones((3,), bool))
    assert bool(jnp.all(valid))
    np.testing.assert_allclose(np.asarray(Pr), 0.8, rtol=1e-5)


def test_prandtl_number_low_kh_invalid_and_ad_safe():
    from legoesm.atmosphere.dynamics.les_closure_diagnosis import (
        prandtl_number_from_diffusivities,
    )
    Km = jnp.full((3,), 4.0)
    Kh = jnp.array([5.0, 1.0e-9, 5.0])         # middle below the floor
    Pr, valid = prandtl_number_from_diffusivities(
        Km, jnp.ones((3,), bool), Kh, jnp.ones((3,), bool))
    assert bool(valid[0]) and not bool(valid[1]) and bool(valid[2])
    assert float(Pr[1]) == 0.0

    def loss(kh):
        Pr, _ = prandtl_number_from_diffusivities(
            Km, jnp.ones((3,), bool), kh, jnp.ones((3,), bool))
        return jnp.sum(Pr ** 2)

    g = jax.grad(loss)(Kh)
    assert bool(jnp.all(jnp.isfinite(g)))      # masked-denominator division is AD-safe


def test_prandtl_number_requires_both_diffusivities_valid():
    from legoesm.atmosphere.dynamics.les_closure_diagnosis import (
        prandtl_number_from_diffusivities,
    )
    _, valid = prandtl_number_from_diffusivities(
        jnp.full((2,), 4.0), jnp.array([True, False]),
        jnp.full((2,), 5.0), jnp.array([True, True]))
    assert bool(valid[0]) and not bool(valid[1])   # invalid K_m ⇒ Pr_t invalid


# --- C_eps (wp2-dissipation) from the steady-state budget -------------------
from legoesm.atmosphere.dynamics.les_closure_diagnosis import (  # noqa: E402
    c_eps_from_budget,
)


def test_c_eps_from_budget_value():
    # C_eps = (Km*S2 - Kh*N2)*l/wp2^1.5; unstable N2<0 ⇒ buoyancy produces.
    # P = 5*1e-4 - 7*(-1e-4) = 1.2e-3; wp2^1.5 = 0.125; C_eps = 1.2e-3*50/0.125 = 0.48
    ce, valid = c_eps_from_budget(
        jnp.full((3,), 5.0), jnp.ones((3,), bool),
        jnp.full((3,), 7.0), jnp.ones((3,), bool),
        jnp.full((3,), 1.0e-4), jnp.full((3,), -1.0e-4),
        jnp.full((3,), 50.0), jnp.full((3,), 0.25))
    assert bool(jnp.all(valid))
    np.testing.assert_allclose(np.asarray(ce), 0.48, rtol=1e-5)


def test_c_eps_negative_production_invalid():
    # Strongly stable (N2>0, weak shear) ⇒ net production P<0 ⇒ invalid.
    _, valid = c_eps_from_budget(
        jnp.full((3,), 5.0), jnp.ones((3,), bool),
        jnp.full((3,), 7.0), jnp.ones((3,), bool),
        jnp.full((3,), 1.0e-4), jnp.full((3,), 1.0e-3),   # N2>0, buoy destroys
        jnp.full((3,), 50.0), jnp.full((3,), 0.25))
    assert not bool(jnp.any(valid))


def test_c_eps_blowup_invalid_and_ad_safe():
    # Tiny wp2 + large production ⇒ raw C_eps explodes past the sanity ceiling
    # ⇒ flagged invalid (not averaged in via the clamp).
    ce, valid = c_eps_from_budget(
        jnp.full((3,), 5.0), jnp.ones((3,), bool),
        jnp.full((3,), 7.0), jnp.ones((3,), bool),
        jnp.full((3,), 1.0), jnp.full((3,), -1.0e-4),
        jnp.full((3,), 50.0), jnp.full((3,), 1.0e-3))
    assert not bool(jnp.any(valid))

    def loss(w):
        ce, _ = c_eps_from_budget(
            jnp.full((3,), 5.0), jnp.ones((3,), bool),
            jnp.full((3,), 7.0), jnp.ones((3,), bool),
            jnp.full((3,), 1.0e-4), jnp.full((3,), -1.0e-4),
            jnp.full((3,), 50.0), w)
        return jnp.sum(ce ** 2)

    g = jax.grad(loss)(jnp.array([0.25, 1.0e-9, 0.25]))   # one below the wp2 floor
    assert bool(jnp.all(jnp.isfinite(g)))                 # wp2^1.5 double-where AD-safe
