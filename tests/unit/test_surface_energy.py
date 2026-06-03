"""Unit tests for legoesm.coupler.surface_energy (UNTESTED-LIVE).

Exercises every public function directly, covering output shapes/dtypes,
Stefan-Boltzmann magnitudes, sign conventions, albedo limits, and
differentiability via jax.grad.

Sign convention (from docstring):
  sw_net  = (1 - alpha) * sw_down          [positive = absorbed by surface]
  lw_up   = emissivity * sigma * T^4
            + (1 - emissivity) * lw_down    [positive = leaving surface]
  lw_net  = emissivity * lw_down
            - emissivity * sigma * T^4      [positive = warming surface]
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.coupler.surface_energy import surface_radiation_fluxes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SHAPE = (3, 4, 4)  # small but non-trivial


def _call(
    sw_down=200.0,
    lw_down=300.0,
    T_surface=288.0,
    alpha=0.1,
    emissivity=0.95,
    shape=SHAPE,
):
    """Call surface_radiation_fluxes with scalar or broadcast values."""
    return surface_radiation_fluxes(
        jnp.full(shape, sw_down),
        jnp.full(shape, lw_down),
        jnp.full(shape, T_surface),
        jnp.full(shape, alpha),
        emissivity,
    )


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------


def test_output_shapes_match_input():
    """All three returned arrays have the same shape as inputs."""
    sw_net, lw_net, lw_up = _call()
    assert sw_net.shape == SHAPE
    assert lw_net.shape == SHAPE
    assert lw_up.shape == SHAPE


def test_output_dtypes_are_float():
    """Returned arrays are floating-point (not int or complex)."""
    sw_net, lw_net, lw_up = _call()
    for arr in (sw_net, lw_net, lw_up):
        assert jnp.issubdtype(arr.dtype, jnp.floating)


def test_all_outputs_finite_at_realistic_inputs():
    """No NaN or Inf under realistic surface conditions."""
    sw_net, lw_net, lw_up = _call()
    for arr in (sw_net, lw_net, lw_up):
        assert jnp.all(jnp.isfinite(arr))


# ---------------------------------------------------------------------------
# Stefan-Boltzmann: lw_up magnitude
# ---------------------------------------------------------------------------


def test_lw_up_magnitude_matches_stefan_boltzmann():
    """lw_up at emissivity=1, lw_down=0 equals sigma*T^4 exactly."""
    T = 288.0
    expected = constants.sigma_sb * T ** 4
    _, _, lw_up = surface_radiation_fluxes(
        jnp.zeros(SHAPE),          # sw_down — irrelevant
        jnp.zeros(SHAPE),          # lw_down = 0 → no reflected term
        jnp.full(SHAPE, T),
        jnp.zeros(SHAPE),          # alpha — irrelevant for lw
        emissivity=1.0,
    )
    assert jnp.allclose(lw_up, expected, rtol=1e-7), (
        f"lw_up = {float(lw_up.mean()):.4f} W/m², "
        f"expected sigma*T^4 = {expected:.4f} W/m²"
    )


def test_lw_up_monotone_with_temperature():
    """lw_up strictly increases with T_surface (Stefan-Boltzmann T^4 law)."""
    temps = jnp.array([250.0, 270.0, 288.0, 300.0, 320.0])
    for i in range(len(temps) - 1):
        _, _, lw_up_lo = surface_radiation_fluxes(
            jnp.array(0.0), jnp.array(0.0), temps[i], jnp.array(0.1), 0.95
        )
        _, _, lw_up_hi = surface_radiation_fluxes(
            jnp.array(0.0), jnp.array(0.0), temps[i + 1], jnp.array(0.1), 0.95
        )
        assert float(lw_up_hi) > float(lw_up_lo), (
            f"lw_up not monotone: T={float(temps[i])} → {float(lw_up_lo):.2f}, "
            f"T={float(temps[i+1])} → {float(lw_up_hi):.2f}"
        )


def test_lw_up_scales_with_emissivity():
    """Higher emissivity → larger thermal emission component of lw_up."""
    # With lw_down=0, lw_up = emissivity * sigma * T^4, so scales linearly.
    _, _, lw_up_lo = surface_radiation_fluxes(
        jnp.array(0.0), jnp.array(0.0), jnp.array(288.0), jnp.array(0.1), 0.70
    )
    _, _, lw_up_hi = surface_radiation_fluxes(
        jnp.array(0.0), jnp.array(0.0), jnp.array(288.0), jnp.array(0.1), 0.97
    )
    assert float(lw_up_hi) > float(lw_up_lo)


def test_lw_up_linear_in_emissivity_at_zero_lw_down():
    """At lw_down=0: lw_up = emissivity * sigma * T^4 (linear in emissivity)."""
    T = 300.0
    eps1, eps2 = 0.8, 0.95
    sigma_T4 = constants.sigma_sb * T ** 4
    _, _, lw_up1 = surface_radiation_fluxes(
        jnp.array(0.0), jnp.array(0.0), jnp.array(T), jnp.array(0.0), eps1
    )
    _, _, lw_up2 = surface_radiation_fluxes(
        jnp.array(0.0), jnp.array(0.0), jnp.array(T), jnp.array(0.0), eps2
    )
    assert jnp.isclose(lw_up1, eps1 * sigma_T4, rtol=1e-7), (
        f"lw_up1={float(lw_up1):.4f}, expected {eps1 * sigma_T4:.4f}"
    )
    assert jnp.isclose(lw_up2, eps2 * sigma_T4, rtol=1e-7), (
        f"lw_up2={float(lw_up2):.4f}, expected {eps2 * sigma_T4:.4f}"
    )


# ---------------------------------------------------------------------------
# Net longwave: sign convention
# ---------------------------------------------------------------------------


def test_lw_net_positive_when_lw_down_exceeds_emission():
    """lw_net > 0 (surface warming) when incoming LW exceeds surface emission."""
    # At T=0 K (hypothetically), emission is zero; any lw_down > 0 gives lw_net > 0.
    # Use a very cold surface so emission is small.
    T_cold = 100.0  # K — emission = 0.95 * 5.67e-8 * 1e8 ≈ 5.4 W/m²
    lw_down_strong = 400.0  # W/m² >> emission at 100 K
    _, lw_net, _ = surface_radiation_fluxes(
        jnp.array(0.0),
        jnp.array(lw_down_strong),
        jnp.array(T_cold),
        jnp.array(0.0),
        emissivity=0.95,
    )
    assert float(lw_net) > 0.0, (
        f"lw_net should be positive when lw_down >> emission; got {float(lw_net):.4f}"
    )


def test_lw_net_negative_when_emission_exceeds_lw_down():
    """lw_net < 0 (surface cooling) when surface emission exceeds incoming LW."""
    T_hot = 320.0  # emission ≈ 0.95 * 5.67e-8 * 320^4 ≈ 558 W/m²
    lw_down_weak = 100.0
    _, lw_net, _ = surface_radiation_fluxes(
        jnp.array(0.0),
        jnp.array(lw_down_weak),
        jnp.array(T_hot),
        jnp.array(0.0),
        emissivity=0.95,
    )
    assert float(lw_net) < 0.0, (
        f"lw_net should be negative (surface cooling); got {float(lw_net):.4f}"
    )


def test_lw_net_formula_explicit():
    """lw_net == emissivity*lw_down - emissivity*sigma*T^4 by definition."""
    eps = 0.97
    T = 288.0
    lw_down = 350.0
    _, lw_net, _ = surface_radiation_fluxes(
        jnp.array(0.0), jnp.array(lw_down), jnp.array(T), jnp.array(0.0), eps
    )
    expected = eps * lw_down - eps * constants.sigma_sb * T ** 4
    assert jnp.isclose(lw_net, expected, rtol=1e-7), (
        f"lw_net = {float(lw_net):.4f}, expected {expected:.4f}"
    )


# ---------------------------------------------------------------------------
# Upward LW: full gray-radiation surface BC decomposition
# ---------------------------------------------------------------------------


def test_lw_up_full_formula():
    """lw_up = emissivity*sigma*T^4 + (1-emissivity)*lw_down."""
    eps = 0.85
    T = 280.0
    lw_down = 300.0
    _, _, lw_up = surface_radiation_fluxes(
        jnp.array(0.0), jnp.array(lw_down), jnp.array(T), jnp.array(0.0), eps
    )
    expected = eps * constants.sigma_sb * T ** 4 + (1.0 - eps) * lw_down
    assert jnp.isclose(lw_up, expected, rtol=1e-7), (
        f"lw_up = {float(lw_up):.4f}, expected {expected:.4f}"
    )


def test_lw_net_plus_lw_up_equals_lw_down():
    """Energy closure: lw_net + lw_up = lw_down (no net loss in the decomposition)."""
    # lw_net = eps*lw_down - eps*sigma*T^4
    # lw_up  = eps*sigma*T^4 + (1-eps)*lw_down
    # sum    = eps*lw_down - eps*sigma*T^4 + eps*sigma*T^4 + (1-eps)*lw_down
    #        = eps*lw_down + lw_down - eps*lw_down = lw_down  ✓
    lw_down = 320.0
    _, lw_net, lw_up = surface_radiation_fluxes(
        jnp.array(0.0), jnp.full(SHAPE, lw_down),
        jnp.full(SHAPE, 285.0), jnp.zeros(SHAPE), 0.95,
    )
    assert jnp.allclose(lw_net + lw_up, lw_down, rtol=1e-7), (
        f"lw_net + lw_up should equal lw_down ({lw_down}); "
        f"got sum = {float(jnp.mean(lw_net + lw_up)):.6f}"
    )


# ---------------------------------------------------------------------------
# Shortwave: albedo limits
# ---------------------------------------------------------------------------


def test_sw_net_zero_when_albedo_one():
    """Perfect reflector (alpha=1): all SW reflected, sw_net=0."""
    sw_net, _, _ = surface_radiation_fluxes(
        jnp.full(SHAPE, 500.0), jnp.zeros(SHAPE),
        jnp.full(SHAPE, 288.0), jnp.ones(SHAPE), 0.95,
    )
    assert jnp.allclose(sw_net, 0.0, atol=1e-10), (
        f"albedo=1 should give sw_net=0; got max {float(jnp.max(jnp.abs(sw_net))):.2e}"
    )


def test_sw_net_equals_sw_down_when_albedo_zero():
    """Perfect absorber (alpha=0): all SW absorbed, sw_net=sw_down."""
    sw_down_val = 400.0
    sw_net, _, _ = surface_radiation_fluxes(
        jnp.full(SHAPE, sw_down_val), jnp.zeros(SHAPE),
        jnp.full(SHAPE, 288.0), jnp.zeros(SHAPE), 0.95,
    )
    assert jnp.allclose(sw_net, sw_down_val, rtol=1e-7), (
        f"albedo=0 should give sw_net=sw_down={sw_down_val}; "
        f"got {float(sw_net.mean()):.4f}"
    )


def test_sw_net_formula_explicit():
    """sw_net = (1 - alpha) * sw_down by formula."""
    alpha = 0.3
    sw_down_val = 250.0
    sw_net, _, _ = surface_radiation_fluxes(
        jnp.full(SHAPE, sw_down_val), jnp.zeros(SHAPE),
        jnp.full(SHAPE, 288.0), jnp.full(SHAPE, alpha), 0.95,
    )
    expected = (1.0 - alpha) * sw_down_val
    assert jnp.allclose(sw_net, expected, rtol=1e-7)


def test_sw_net_decreases_with_albedo():
    """sw_net monotonically decreases as albedo increases from 0 to 1."""
    sw_down_val = 300.0
    alphas = [0.0, 0.1, 0.3, 0.6, 1.0]
    prev = float("inf")
    for alpha in alphas:
        sw_net, _, _ = surface_radiation_fluxes(
            jnp.array(sw_down_val), jnp.array(0.0),
            jnp.array(288.0), jnp.array(alpha), 0.95,
        )
        assert float(sw_net) <= prev + 1e-9, (
            f"sw_net not monotone at alpha={alpha}"
        )
        prev = float(sw_net)


def test_sw_net_nonneg_for_nonneg_inputs():
    """sw_net >= 0 for any alpha in [0,1] and sw_down >= 0."""
    sw_net, _, _ = surface_radiation_fluxes(
        jnp.full(SHAPE, 200.0), jnp.full(SHAPE, 300.0),
        jnp.full(SHAPE, 288.0), jnp.full(SHAPE, 0.5), 0.95,
    )
    assert jnp.all(sw_net >= 0.0)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


def test_grad_lw_up_wrt_T_sfc_is_finite_and_positive():
    """d(lw_up)/d(T_sfc) is finite and positive (Stefan-Boltzmann T^4 law).

    Analytically: d(eps*sigma*T^4)/dT = 4*eps*sigma*T^3 > 0.
    """
    eps = 0.95
    T0 = 288.0

    def lw_up_scalar(T_sfc):
        _, _, lw_up = surface_radiation_fluxes(
            jnp.array(0.0), jnp.array(300.0), T_sfc, jnp.array(0.1), eps
        )
        return jnp.sum(lw_up)

    grad = jax.grad(lw_up_scalar)(jnp.array(T0))
    assert jnp.isfinite(grad), f"Gradient is not finite: {grad}"
    assert float(grad) > 0.0, f"d(lw_up)/dT should be positive; got {float(grad)}"

    # Check magnitude against analytical 4*eps*sigma*T^3
    expected_grad = 4.0 * eps * constants.sigma_sb * T0 ** 3
    assert jnp.isclose(grad, expected_grad, rtol=1e-6), (
        f"grad = {float(grad):.6e}, expected 4*eps*sigma*T^3 = {expected_grad:.6e}"
    )


def test_grad_lw_net_wrt_T_sfc_is_negative():
    """d(lw_net)/d(T_sfc) < 0: warmer surface → more emission → less net warming.

    lw_net = eps*lw_down - eps*sigma*T^4, so derivative = -4*eps*sigma*T^3 < 0.
    """
    eps = 0.97

    def lw_net_scalar(T_sfc):
        _, lw_net, _ = surface_radiation_fluxes(
            jnp.array(0.0), jnp.array(350.0), T_sfc, jnp.array(0.0), eps
        )
        return jnp.sum(lw_net)

    grad = jax.grad(lw_net_scalar)(jnp.array(288.0))
    assert jnp.isfinite(grad), f"Gradient is not finite: {grad}"
    assert float(grad) < 0.0, f"d(lw_net)/dT should be negative; got {float(grad)}"


def test_grad_sw_net_wrt_alpha_is_negative():
    """d(sw_net)/d(alpha) < 0: higher albedo → less absorbed SW."""
    sw_down_val = 300.0

    def sw_net_scalar(alpha):
        sw_net, _, _ = surface_radiation_fluxes(
            jnp.array(sw_down_val), jnp.array(0.0),
            jnp.array(288.0), alpha, 0.95,
        )
        return jnp.sum(sw_net)

    grad = jax.grad(sw_net_scalar)(jnp.array(0.3))
    assert jnp.isfinite(grad), f"Gradient is not finite: {grad}"
    assert float(grad) < 0.0, f"d(sw_net)/d(alpha) should be negative; got {float(grad)}"

    # Analytical: d((1-alpha)*sw_down)/d(alpha) = -sw_down
    assert jnp.isclose(grad, -sw_down_val, rtol=1e-7)


def test_grad_wrt_emissivity_lw_up_is_positive_when_T_warm():
    """d(lw_up)/d(emissivity) > 0 when T_surface is warm (emission > lw_down).

    lw_up = eps*sigma*T^4 + (1-eps)*lw_down
    d(lw_up)/d(eps) = sigma*T^4 - lw_down > 0 when emission > lw_down.
    """
    T = 350.0  # sigma*T^4 ≈ 850 W/m² >> lw_down=200
    lw_down_val = 200.0

    def lw_up_sum(eps):
        _, _, lw_up = surface_radiation_fluxes(
            jnp.array(0.0), jnp.array(lw_down_val), jnp.array(T), jnp.array(0.0), eps
        )
        return jnp.sum(lw_up)

    grad = jax.grad(lw_up_sum)(jnp.array(0.9))
    assert jnp.isfinite(grad)
    assert float(grad) > 0.0, (
        f"d(lw_up)/d(eps) should be positive for hot surface; got {float(grad)}"
    )


def test_full_output_differentiable_via_reduction():
    """sum(sw_net + lw_net + lw_up) is differentiable wrt T_surface over a 3D array."""
    def total_flux(T_sfc):
        sw_net, lw_net, lw_up = surface_radiation_fluxes(
            jnp.full(SHAPE, 200.0),
            jnp.full(SHAPE, 300.0),
            T_sfc,
            jnp.full(SHAPE, 0.1),
            0.95,
        )
        return jnp.sum(sw_net + lw_net + lw_up)

    T_sfc = jnp.full(SHAPE, 288.0)
    grad = jax.grad(total_flux)(T_sfc)
    assert grad.shape == SHAPE
    assert jnp.all(jnp.isfinite(grad))
    # Analytical: d(sw_net)/dT = 0, d(lw_net)/dT = -4*eps*sigma*T^3,
    # d(lw_up)/dT = +4*eps*sigma*T^3 → they cancel → total grad ≈ 0.
    assert jnp.allclose(grad, 0.0, atol=1e-6), (
        "sw_net + lw_net + lw_up gradient wrt T should be ~0 "
        f"(lw_net and lw_up terms cancel); max |grad| = {float(jnp.max(jnp.abs(grad))):.2e}"
    )


# ---------------------------------------------------------------------------
# Grey-body: emissivity=0 edge case
# ---------------------------------------------------------------------------


def test_emissivity_zero_is_perfect_reflector_lw():
    """emissivity=0: surface reflects all LW; lw_up = lw_down, lw_net = 0."""
    lw_down_val = 350.0
    _, lw_net, lw_up = surface_radiation_fluxes(
        jnp.array(0.0), jnp.array(lw_down_val),
        jnp.array(288.0), jnp.array(0.0), emissivity=0.0,
    )
    assert jnp.isclose(lw_net, 0.0, atol=1e-10), (
        f"emissivity=0 → lw_net should be 0; got {float(lw_net):.2e}"
    )
    assert jnp.isclose(lw_up, lw_down_val, rtol=1e-10), (
        f"emissivity=0 → lw_up should equal lw_down={lw_down_val}; got {float(lw_up):.4f}"
    )


def test_emissivity_one_no_reflected_lw():
    """emissivity=1: no reflected component; lw_up = sigma*T^4 only."""
    T = 288.0
    lw_down_val = 300.0
    _, _, lw_up = surface_radiation_fluxes(
        jnp.array(0.0), jnp.array(lw_down_val),
        jnp.array(T), jnp.array(0.0), emissivity=1.0,
    )
    expected = constants.sigma_sb * T ** 4
    assert jnp.isclose(lw_up, expected, rtol=1e-7), (
        f"emissivity=1 → lw_up = sigma*T^4 = {expected:.4f}; got {float(lw_up):.4f}"
    )
