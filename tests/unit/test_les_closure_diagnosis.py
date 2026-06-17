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
    assert float(diag.w_entrainment) == pytest.approx(0.01, rel=1e-12)
    assert float(diag.delta_thetav) == pytest.approx(2.0, rel=1e-12)
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
