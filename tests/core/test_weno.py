"""Tests for legoesm.core.weno — WENO-Z reconstruction kernels.

Tests cover:
  1. Polynomial exactness (smooth data convergence at design order)
  2. Left/right symmetry on symmetric stencils
  3. AD correctness via Taylor test (jax.grad through WENO kernels)
  4. {phi; psi} smoothness-optimised variant
  5. Float32 epsilon safety
  6. weno_upwind sign selection
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.weno import (
    weno5_z,
    weno7_z,
    weno9_z,
    weno_reconstruct_split,
    weno_upwind,
)


# ---------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------

def _cell_avg_sin(x, dx):
    """Exact cell average of sin(x) over [x - dx/2, x + dx/2]."""
    return (2.0 / dx) * np.sin(x) * np.sin(dx / 2)


def _cell_avg_poly(x, dx, degree):
    """Exact cell average of x^degree over [x - dx/2, x + dx/2]."""
    # int_{a}^{b} t^n dt = (b^{n+1} - a^{n+1}) / (n+1)
    n = degree
    a = x - dx / 2
    b = x + dx / 2
    return (b ** (n + 1) - a ** (n + 1)) / ((n + 1) * dx)


def _make_stencil(f_array, i, half_width, N):
    """Extract periodic stencil of length 2*half_width from f_array at face i+1/2."""
    width = 2 * half_width
    return [jnp.array(f_array[(i - half_width + 1 + j) % N]) for j in range(width)]


def _convergence_rate(errors):
    """Compute log2 convergence rates between successive resolutions."""
    rates = []
    for i in range(len(errors) - 1):
        if errors[i + 1] > 0 and errors[i] > 0:
            rates.append(np.log2(errors[i] / errors[i + 1]))
        else:
            rates.append(np.inf)
    return rates


# ---------------------------------------------------------------
#  1. Polynomial exactness / convergence order
# ---------------------------------------------------------------

class TestConvergenceOrder:
    """Verify that WENO reconstructions converge at design order on smooth data."""

    Ns = [32, 64, 128, 256]

    def _run_convergence(self, weno_fn, half_width, min_order):
        """Run convergence test and assert minimum order is achieved."""
        errs_plus, errs_minus = [], []
        for N in self.Ns:
            dx = 2 * np.pi / N
            x = np.array([dx * (i + 0.5) for i in range(N)])
            f = _cell_avg_sin(x, dx)
            i = N // 4
            exact = np.sin(dx * (i + 1))  # sin at face i+1/2
            stencil = _make_stencil(f, i, half_width, N)
            fp, fm = weno_fn(stencil)
            errs_plus.append(float(abs(fp - exact)))
            errs_minus.append(float(abs(fm - exact)))

        rates_p = _convergence_rate(errs_plus)
        rates_m = _convergence_rate(errs_minus)
        # Use the rate from the middle resolutions (avoid machine precision)
        avg_rate_p = np.mean(rates_p[:2])
        avg_rate_m = np.mean(rates_m[:2])
        assert avg_rate_p >= min_order - 0.5, (
            f"Left-biased rate {avg_rate_p:.1f} < {min_order - 0.5}")
        assert avg_rate_m >= min_order - 0.5, (
            f"Right-biased rate {avg_rate_m:.1f} < {min_order - 0.5}")

    def test_weno5_convergence(self):
        self._run_convergence(weno5_z, half_width=3, min_order=5)

    def test_weno7_convergence(self):
        self._run_convergence(weno7_z, half_width=4, min_order=7)

    def test_weno9_convergence(self):
        self._run_convergence(weno9_z, half_width=5, min_order=9)


# ---------------------------------------------------------------
#  2. Polynomial exactness: exact for degree < order
# ---------------------------------------------------------------

class TestPolynomialExactness:
    """WENO should reconstruct polynomials of degree < order exactly."""

    @pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
    def test_weno5_exact_polynomial(self, degree):
        # Use N=256 for better floating-point behaviour on higher-degree polys.
        N, dx = 256, 2 * np.pi / 256
        x = np.array([dx * (i + 0.5) for i in range(N)])
        f = _cell_avg_poly(x, dx, degree)
        i = N // 4
        exact = (dx * (i + 1)) ** degree
        stencil = _make_stencil(f, i, 3, N)
        fp, fm = weno5_z(stencil)
        tol = 1e-10 if degree <= 2 else 1e-6
        assert abs(fp - exact) < tol, f"degree {degree}: left err = {abs(fp - exact)}"
        assert abs(fm - exact) < tol, f"degree {degree}: right err = {abs(fm - exact)}"

    @pytest.mark.parametrize("degree", [0, 1, 2, 3, 4, 5, 6])
    def test_weno7_exact_polynomial(self, degree):
        N, dx = 256, 2 * np.pi / 256
        x = np.array([dx * (i + 0.5) for i in range(N)])
        f = _cell_avg_poly(x, dx, degree)
        i = N // 4
        exact = (dx * (i + 1)) ** degree
        stencil = _make_stencil(f, i, 4, N)
        fp, fm = weno7_z(stencil)
        tol = 1e-10 if degree <= 3 else 1e-6
        assert abs(fp - exact) < tol, f"degree {degree}: left err = {abs(fp - exact)}"
        assert abs(fm - exact) < tol, f"degree {degree}: right err = {abs(fm - exact)}"

    @pytest.mark.parametrize("degree", [0, 1, 2, 3, 4, 5, 6, 7, 8])
    def test_weno9_exact_polynomial(self, degree):
        N, dx = 256, 2 * np.pi / 256
        x = np.array([dx * (i + 0.5) for i in range(N)])
        f = _cell_avg_poly(x, dx, degree)
        i = N // 4
        exact = (dx * (i + 1)) ** degree
        stencil = _make_stencil(f, i, 5, N)
        fp, fm = weno9_z(stencil)
        tol = 1e-10 if degree <= 4 else 1e-5
        assert abs(fp - exact) < tol, f"degree {degree}: left err = {abs(fp - exact)}"
        assert abs(fm - exact) < tol, f"degree {degree}: right err = {abs(fm - exact)}"


class TestPointToCellAvgConversion:
    """point_to_cellavg_periodic must convert point values to cell averages
    exactly for polynomials below its formal order (4/6/8)."""

    @pytest.mark.parametrize("order,max_deg", [(4, 2), (6, 4), (8, 6)])
    def test_periodic_exact_for_polynomials(self, order, max_deg):
        from legoesm.core.weno import point_to_cellavg_periodic
        N, dx = 64, 2 * np.pi / 64
        xc = np.array([dx * (i + 0.5) for i in range(N)])  # cell centres
        for deg in range(0, max_deg + 1):
            f_pt = xc ** deg                                # point values at centres
            # True cell average over [xc-dx/2, xc+dx/2] of x^deg.
            xl, xr = xc - dx / 2, xc + dx / 2
            true_avg = (xr ** (deg + 1) - xl ** (deg + 1)) / ((deg + 1) * dx)
            got = np.asarray(point_to_cellavg_periodic(
                jnp.asarray(f_pt), axis=0, order=order))
            # Periodic wrap pollutes the boundary cells; check the interior.
            sl = slice(order, N - order)
            err = float(np.max(np.abs(got[sl] - true_avg[sl])))
            assert err < 1e-9, f"order {order} deg {deg}: err {err}"

    def test_order8_beats_order6_on_smooth(self):
        """The 7-point (order-8) conversion is strictly more accurate than the
        5-point (order-6) one on a smooth field — the property W9V relies on."""
        from legoesm.core.weno import point_to_cellavg_periodic
        N, dx = 48, 2 * np.pi / 48
        xc = np.array([dx * (i + 0.5) for i in range(N)])
        f_pt = np.sin(xc)
        xl, xr = xc - dx / 2, xc + dx / 2
        true_avg = (-np.cos(xr) + np.cos(xl)) / dx
        e6 = float(np.max(np.abs(np.asarray(point_to_cellavg_periodic(
            jnp.asarray(f_pt), axis=0, order=6)) - true_avg)))
        e8 = float(np.max(np.abs(np.asarray(point_to_cellavg_periodic(
            jnp.asarray(f_pt), axis=0, order=8)) - true_avg)))
        assert e8 < e6, f"order-8 err {e8} not better than order-6 {e6}"


# ---------------------------------------------------------------
#  3. AD Taylor test
# ---------------------------------------------------------------

class TestADTaylorTest:
    """Verify jax.grad through WENO kernels gives correct gradients."""

    def _taylor_test(self, weno_fn, stencil_width):
        """Run Taylor test: |f(x+h*d) - f(x) - h*df(x)*d| = O(h^2)."""
        rng = np.random.RandomState(42)
        x0 = jnp.array(rng.randn(stencil_width) + 5.0)  # smooth, positive
        d = jnp.array(rng.randn(stencil_width))

        def scalar_fn(x):
            stencil = [x[i] for i in range(stencil_width)]
            fp, fm = weno_fn(stencil)
            return jnp.sum(fp + fm)

        f0 = scalar_fn(x0)
        grad_f = jax.grad(scalar_fn)(x0)
        directional_deriv = jnp.dot(grad_f, d)

        hs = [1e-2, 1e-3, 1e-4, 1e-5]
        errors = []
        for h in hs:
            f_perturbed = scalar_fn(x0 + h * d)
            first_order = abs(f_perturbed - f0 - h * directional_deriv)
            errors.append(float(first_order))

        # Check O(h^2) convergence
        rates = _convergence_rate(errors)
        avg_rate = np.mean(rates[:2])
        assert avg_rate > 1.8, f"Taylor test rate {avg_rate:.2f} < 1.8 (expected ~2.0)"

    def test_weno5_taylor(self):
        self._taylor_test(weno5_z, 6)

    def test_weno7_taylor(self):
        self._taylor_test(weno7_z, 8)

    def test_weno9_taylor(self):
        self._taylor_test(weno9_z, 10)

    def test_weno5_split_taylor(self):
        """Taylor test for {phi; psi} variant."""
        rng = np.random.RandomState(123)
        phi0 = jnp.array(rng.randn(6) + 3.0)
        psi0 = jnp.array(rng.randn(6) + 3.0)
        d_phi = jnp.array(rng.randn(6))
        d_psi = jnp.array(rng.randn(6))

        def scalar_fn(phi, psi):
            fp, fm = weno_reconstruct_split(
                [phi[i] for i in range(6)],
                [psi[i] for i in range(6)],
                order=5,
            )
            return jnp.sum(fp + fm)

        f0 = scalar_fn(phi0, psi0)
        grad_phi, grad_psi = jax.grad(scalar_fn, argnums=(0, 1))(phi0, psi0)
        dd = jnp.dot(grad_phi, d_phi) + jnp.dot(grad_psi, d_psi)

        hs = [1e-2, 1e-3, 1e-4, 1e-5]
        errors = []
        for h in hs:
            f_p = scalar_fn(phi0 + h * d_phi, psi0 + h * d_psi)
            errors.append(float(abs(f_p - f0 - h * dd)))

        rates = _convergence_rate(errors)
        assert np.mean(rates[:2]) > 1.8


# ---------------------------------------------------------------
#  4. {phi; psi} variant tests
# ---------------------------------------------------------------

class TestSplitSmoothnessVariant:
    """Test the {phi; psi} smoothness-optimised variant."""

    def test_split_reduces_to_standard_when_phi_eq_psi(self):
        """When phi == psi, the split variant should match the standard WENO."""
        rng = np.random.RandomState(7)
        stencil = [jnp.array(v) for v in rng.randn(6) + 5.0]
        fp_std, fm_std = weno5_z(stencil)
        fp_split, fm_split = weno_reconstruct_split(stencil, stencil, order=5)
        np.testing.assert_allclose(fp_split, fp_std, atol=1e-14)
        np.testing.assert_allclose(fm_split, fm_std, atol=1e-14)

    @pytest.mark.parametrize("order", [5, 7, 9])
    def test_split_convergence(self, order):
        """Split variant should converge at design order on smooth data."""
        k = (order + 1) // 2
        hw = k
        Ns = [32, 64, 128]
        errs = []
        for N in Ns:
            dx = 2 * np.pi / N
            x = np.array([dx * (i + 0.5) for i in range(N)])
            phi_f = _cell_avg_sin(x, dx)
            psi_f = np.cos(x)  # pointwise (smoother proxy)
            i = N // 4
            exact = np.sin(dx * (i + 1))
            phi_s = _make_stencil(phi_f, i, hw, N)
            psi_s = _make_stencil(psi_f, i, hw, N)
            fp, _ = weno_reconstruct_split(phi_s, psi_s, order=order)
            errs.append(float(abs(fp - exact)))
        rate = np.log2(errs[0] / errs[1])
        assert rate >= order - 1.0, f"Split order {order}: rate {rate:.1f}"


# ---------------------------------------------------------------
#  5. Float32 safety
# ---------------------------------------------------------------

class TestFloat32Safety:
    """WENO kernels should produce finite results in float32."""

    @pytest.mark.parametrize("weno_fn,width", [
        (weno5_z, 6), (weno7_z, 8), (weno9_z, 10)])
    def test_float32_no_nan(self, weno_fn, width):
        stencil = [jnp.array(v, dtype=jnp.float32)
                   for v in np.random.RandomState(0).randn(width) + 3.0]
        fp, fm = weno_fn(stencil)
        assert jnp.isfinite(fp), "Left-biased contains NaN/Inf"
        assert jnp.isfinite(fm), "Right-biased contains NaN/Inf"

    @pytest.mark.parametrize("weno_fn,width", [
        (weno5_z, 6), (weno7_z, 8), (weno9_z, 10)])
    def test_float32_near_constant_field(self, weno_fn, width):
        """Near-constant fields stress epsilon regularisation."""
        val = 1e3
        stencil = [jnp.array(val + 1e-6 * i, dtype=jnp.float32)
                   for i in range(width)]
        fp, fm = weno_fn(stencil)
        assert jnp.isfinite(fp)
        assert jnp.isfinite(fm)

    @pytest.mark.parametrize("weno_fn,width", [
        (weno5_z, 6), (weno7_z, 8), (weno9_z, 10)])
    def test_float32_gradient_finite(self, weno_fn, width):
        """Gradients through WENO should be finite in float32."""
        x = jnp.array(np.random.RandomState(1).randn(width) + 5.0, dtype=jnp.float32)

        def fn(x):
            stencil = [x[i] for i in range(width)]
            fp, fm = weno_fn(stencil)
            return fp + fm

        g = jax.grad(fn)(x)
        assert jnp.all(jnp.isfinite(g)), f"Non-finite gradients: {g}"


# ---------------------------------------------------------------
#  6. weno_upwind
# ---------------------------------------------------------------

class TestWenoUpwind:

    def test_positive_velocity_selects_plus(self):
        fp = jnp.array(1.0)
        fm = jnp.array(2.0)
        assert weno_upwind(fp, fm, jnp.array(0.5)) == fp

    def test_negative_velocity_selects_minus(self):
        fp = jnp.array(1.0)
        fm = jnp.array(2.0)
        assert weno_upwind(fp, fm, jnp.array(-0.5)) == fm

    def test_zero_velocity_selects_plus(self):
        fp = jnp.array(1.0)
        fm = jnp.array(2.0)
        assert weno_upwind(fp, fm, jnp.array(0.0)) == fp

    def test_array_velocity(self):
        fp = jnp.array([1.0, 1.0])
        fm = jnp.array([2.0, 2.0])
        vel = jnp.array([1.0, -1.0])
        result = weno_upwind(fp, fm, vel)
        np.testing.assert_array_equal(result, jnp.array([1.0, 2.0]))
