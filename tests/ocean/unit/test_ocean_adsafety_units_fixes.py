"""AD-safety + units regression tests for ocean vertical-mixing fixes.

Covers three confirmed bugs in ``ocean/physics/vertical_mixing/`` (oracle
review 2026-07-07):

* ``tke._safe_stress_modulus`` — the surface wind-work TKE flux
  ``(|tau|/rho_0)**1.5`` used a plain ``jnp.sqrt(tx*tx + ty*ty)`` whose
  reverse-mode gradient is ``0*inf = NaN`` at zero stress (``tx=ty=0``),
  even though the analytic limit of the flux and its gradient is 0.  The
  double-``where`` helper keeps the primal identical and yields a finite
  0 gradient at zero stress.
* ``internal_wave_mixing`` — ``sqrt_reb = jnp.sqrt(jnp.maximum(reb, 0.0))``
  (only reached when ``cfg.mevar=True``) has a NaN gradient at ``reb=0``
  (land / zero-forcing cells).  Same double-``where`` fix.
* ``tidal.synthetic_baroclinic_tide_energy_from_bathy`` — a UNITS error:
  the Jayne & St. Laurent (2001) / Simmons et al. (2004) conversion
  ``E = 0.5*rho0*kappa_h*<h^2>*N_b*<u^2>`` takes ``kappa_h`` LINEARLY; the
  code squared it, yielding W/m^3 (~477x too weak) instead of W/m^2.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # Metal has no float64

import jax  # noqa: E402
from jax import config as jax_config  # noqa: E402

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.ocean.physics.vertical_mixing import (  # noqa: E402
    internal_wave_mixing as iwm,
)
from legoesm.ocean.physics.vertical_mixing import tidal, tke


# ---------------------------------------------------------------------------
# BUG 1 — tke surface stress modulus: AD-safe sqrt at zero wind stress.
# ---------------------------------------------------------------------------
def test_safe_stress_modulus_primal_matches_plain_sqrt():
    """Primal is BIT-IDENTICAL to the plain sqrt away from zero stress."""
    tx = jnp.array([3.0, -0.5, 1.0])
    ty = jnp.array([4.0, 1.2, 0.0])
    got = tke._safe_stress_modulus(tx, ty)
    expected = jnp.sqrt(tx * tx + ty * ty)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_tke_surface_flux_grad_finite_at_zero_stress():
    """(|tau|/rho_0)^{3/2} has a finite (zero) gradient at tx=ty=0."""
    rho_0 = constants.rho_ocean

    def surface_flux(tx, ty):
        taum = tke._safe_stress_modulus(tx, ty)
        return jnp.sum((taum / rho_0) ** 1.5)

    # Primal at zero stress is exactly 0.
    assert float(surface_flux(jnp.array(0.0), jnp.array(0.0))) == 0.0

    g = jax.grad(surface_flux, argnums=(0, 1))(jnp.array(0.0), jnp.array(0.0))
    for gi in g:
        assert np.isfinite(np.asarray(gi)).all()
        np.testing.assert_allclose(np.asarray(gi), 0.0, atol=0.0)

    # A non-zero-stress gradient is finite too (sanity, not a NaN elsewhere).
    g_nz = jax.grad(surface_flux, argnums=(0, 1))(
        jnp.array(0.03), jnp.array(0.04))
    for gi in g_nz:
        assert np.isfinite(np.asarray(gi)).all()


# ---------------------------------------------------------------------------
# BUG 2 — internal_wave_mixing: AD-safe sqrt(reb) at reb=0.
# ---------------------------------------------------------------------------
def test_iwm_grad_finite_at_zero_reb():
    """d/dpower of the mevar K_wave is finite when reb collapses to 0.

    With zero N^2-scaled power the local dissipation ``zemx`` (hence
    ``reb``) is 0 at every interface; ``reb ∝ ensq`` so a gradient wrt the
    power flows straight into the ``sqrt_reb`` node — the exact zero-stress
    NaN class the double-``where`` fix closes.
    """
    cfg = iwm.IWMConfig(enabled=True, mevar=True)
    dtype = jnp.float64
    shape = (1,)
    nlev = 3

    # Positive, stably stratified column (N^2 > 0) so the reb denominator is
    # non-degenerate; reb is 0 solely because the forcing power is 0.
    depth_cell = jnp.asarray([[100.0, 300.0, 600.0]], dtype=dtype)
    dz_w = jnp.asarray([[200.0, 300.0]], dtype=dtype)
    H = jnp.asarray([1000.0], dtype=dtype)
    N2 = jnp.asarray([[1.0e-4, 4.0e-5]], dtype=dtype)

    def k_wave_sum(ensq_power):
        zero = jnp.zeros(shape, dtype=dtype)
        forcing = iwm.IWMForcing(
            ebot=zero,
            ecri=zero,
            ensq=jnp.full(shape, ensq_power, dtype=dtype),
            esho=zero,
            hbot=jnp.full(shape, cfg.scale_bot_m, dtype=dtype),
            hcri_inv=jnp.full(shape, 1.0 / cfg.scale_cri_m, dtype=dtype),
        )
        k_wave, _ = iwm.compute_iwm_diffusivity(
            forcing, depth_cell, dz_w, H, N2, cfg=cfg, rho_0=constants.rho_ocean)
        return jnp.sum(k_wave)

    # Primal is finite at zero power (reb == 0 everywhere).
    assert np.isfinite(float(k_wave_sum(jnp.array(0.0, dtype=dtype))))

    grad = jax.grad(k_wave_sum)(jnp.array(0.0, dtype=dtype))
    assert np.isfinite(np.asarray(grad)).all(), (
        "sqrt(reb) NaN gradient at reb=0 not fixed")


# ---------------------------------------------------------------------------
# BUG 3 — tidal synthetic E_BT: kappa_h LINEAR => W/m^2 of the right order.
# ---------------------------------------------------------------------------
def test_synthetic_tide_energy_order_of_magnitude_wm2():
    """Deep-ocean E_BT is ~1e-2 W/m^2 (W/m^2), not the ~5e-5 W/m^3 bug value."""
    H = jnp.asarray([2000.0])  # deep => full depth ramp
    E = tidal.synthetic_baroclinic_tide_energy_from_bathy(H)
    E_val = float(E[0])

    # Analytic reference with kappa_h LINEAR (Jayne & St. Laurent 2001).
    kappa_h = 2.0 * np.pi / 3000.0
    ref = (0.5 * constants.rho_ocean * kappa_h
           * 250.0 ** 2 * 1.0e-3 * 0.02 ** 2)
    np.testing.assert_allclose(E_val, ref, rtol=1e-6)

    # Right order of magnitude: ~1e-2 W/m^2 (the squared-kappa bug gave ~5e-5).
    assert 1.0e-3 < E_val < 1.0e-1, f"E_BT={E_val} outside the W/m^2 band"


def test_synthetic_tide_energy_scales_linearly_in_kappa_h():
    """Halving roughness_scale_m doubles kappa_h => ~2x E (linear, not 4x)."""
    H = jnp.asarray([2000.0])
    E_base = float(
        tidal.synthetic_baroclinic_tide_energy_from_bathy(
            H, roughness_scale_m=3000.0)[0])
    E_half = float(
        tidal.synthetic_baroclinic_tide_energy_from_bathy(
            H, roughness_scale_m=1500.0)[0])
    # kappa_h doubled -> E doubled (a squared dependence would give 4x).
    np.testing.assert_allclose(E_half / E_base, 2.0, rtol=1e-6)
