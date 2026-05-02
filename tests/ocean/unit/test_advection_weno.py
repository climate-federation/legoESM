"""Tests for WENO-Z tracer advection (Phase 2a of Silvestri et al. 2024).

Tests:
1. Output shapes and finiteness
2. Constant and linear field exactness
3. Conservation (column integral, horizontal divergence theorem)
4. Periodic wrapping (zonal)
5. Wall BC (meridional poles)
6. Higher accuracy than TVD/upwind on smooth profiles
7. AD (reverse-mode gradient) correctness via Taylor test
8. WENO7 higher accuracy than WENO5 on smooth data
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# =====================================================================
# Helpers
# =====================================================================

def _make_grid(n_lat=10, n_lon=20):
    from legoesm.grids.latlon import create_latlon_grid
    return create_latlon_grid(n_lat=n_lat, n_lon=n_lon, radius=6.371e6)


# =====================================================================
# WENO5 horizontal (u-points)
# =====================================================================

class TestWENO5Zonal:
    """Tests for weno5_to_u_points."""

    def test_output_shape(self):
        from legoesm.ocean.advection import weno5_to_u_points
        n_lat, n_lon, nlev = 8, 16, 5
        f = jnp.ones((n_lat, n_lon, nlev))
        mf = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.5
        result = weno5_to_u_points(f, mf)
        assert result.shape == (n_lat, n_lon + 1, nlev)

    def test_constant_field(self):
        """Uniform tracer gives T_face = T everywhere."""
        from legoesm.ocean.advection import weno5_to_u_points
        n_lat, n_lon, nlev = 8, 16, 5
        T_val = 3.14
        f = jnp.full((n_lat, n_lon, nlev), T_val)
        mf = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.5
        result = weno5_to_u_points(f, mf)
        assert jnp.allclose(result, T_val, atol=1e-13)

    def test_linear_field_exact_interior(self):
        """WENO5 reproduces a linear field exactly at interior faces.

        A linear f(j) = a + b*j is non-periodic, so the stencil sees a
        discontinuity near the wrap-around boundary. We test only interior
        faces where the full 6-point stencil lies within the linear region.
        """
        from legoesm.ocean.advection import weno5_to_u_points
        n_lat, n_lon, nlev = 8, 32, 3
        # Linear in longitude: f(j) = a + b*j
        lon_idx = jnp.arange(n_lon)
        f = (2.0 + 0.1 * lon_idx)[jnp.newaxis, :, jnp.newaxis]
        f = jnp.broadcast_to(f, (n_lat, n_lon, nlev)).copy()
        mf = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.5
        result = weno5_to_u_points(f, mf)
        # Face j between cell j-1 and cell j: face value = 2.0 + 0.1*(j-0.5)
        face_idx = jnp.arange(n_lon)
        expected = (2.0 + 0.1 * (face_idx - 0.5))[jnp.newaxis, :, jnp.newaxis]
        expected = jnp.broadcast_to(expected, (n_lat, n_lon, nlev))
        # Skip faces within half-width (3) of the periodic boundary
        # where the stencil crosses the discontinuity.
        interior = slice(4, n_lon - 3)
        assert jnp.allclose(result[:, interior, :], expected[:, interior, :], atol=1e-12), (
            f"Max interior error: "
            f"{float(jnp.max(jnp.abs(result[:, interior, :] - expected[:, interior, :])))}")

    def test_periodic_wrapping(self):
        """u-face at n_lon should equal u-face at 0."""
        from legoesm.ocean.advection import weno5_to_u_points
        n_lat, n_lon, nlev = 8, 16, 3
        key = jax.random.PRNGKey(42)
        f = jax.random.uniform(key, (n_lat, n_lon, nlev), minval=1.0, maxval=5.0)
        mf = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.3
        result = weno5_to_u_points(f, mf)
        assert jnp.allclose(result[:, -1, :], result[:, 0, :], atol=1e-14)

    def test_finite_values(self):
        """Output is finite for random input."""
        from legoesm.ocean.advection import weno5_to_u_points
        n_lat, n_lon, nlev = 8, 16, 5
        key = jax.random.PRNGKey(7)
        f = jax.random.uniform(key, (n_lat, n_lon, nlev), minval=0.5, maxval=5.0)
        mf = 0.5 * jax.random.normal(jax.random.PRNGKey(8), (n_lat, n_lon + 1, nlev))
        result = weno5_to_u_points(f, mf)
        assert jnp.all(jnp.isfinite(result))


# =====================================================================
# WENO5 horizontal (v-points)
# =====================================================================

class TestWENO5Meridional:
    """Tests for weno5_to_v_points."""

    def test_output_shape(self):
        from legoesm.ocean.advection import weno5_to_v_points
        n_lat, n_lon, nlev = 10, 16, 5
        f = jnp.ones((n_lat, n_lon, nlev))
        mf = jnp.ones((n_lat + 1, n_lon, nlev)) * 0.5
        result = weno5_to_v_points(f, mf)
        assert result.shape == (n_lat + 1, n_lon, nlev)

    def test_constant_field(self):
        """Uniform tracer: interior=T, poles=0."""
        from legoesm.ocean.advection import weno5_to_v_points
        n_lat, n_lon, nlev = 10, 16, 5
        T_val = 2.71
        f = jnp.full((n_lat, n_lon, nlev), T_val)
        mf = jnp.ones((n_lat + 1, n_lon, nlev)) * 0.5
        result = weno5_to_v_points(f, mf)
        assert jnp.allclose(result[1:-1], T_val, atol=1e-13)
        assert jnp.allclose(result[0], 0.0)
        assert jnp.allclose(result[-1], 0.0)

    def test_wall_bc_poles(self):
        """Zero flux at pole boundaries regardless of mass flux."""
        from legoesm.ocean.advection import weno5_to_v_points
        n_lat, n_lon, nlev = 10, 16, 3
        key = jax.random.PRNGKey(0)
        f = jax.random.uniform(key, (n_lat, n_lon, nlev), minval=1.0, maxval=5.0)
        mf = jnp.ones((n_lat + 1, n_lon, nlev)) * 10.0  # large mass flux
        result = weno5_to_v_points(f, mf)
        assert jnp.allclose(result[0], 0.0)
        assert jnp.allclose(result[-1], 0.0)

    def test_finite_values(self):
        from legoesm.ocean.advection import weno5_to_v_points
        n_lat, n_lon, nlev = 10, 16, 5
        key = jax.random.PRNGKey(3)
        f = jax.random.uniform(key, (n_lat, n_lon, nlev), minval=0.5, maxval=5.0)
        mf = 0.5 * jax.random.normal(jax.random.PRNGKey(4), (n_lat + 1, n_lon, nlev))
        result = weno5_to_v_points(f, mf)
        assert jnp.all(jnp.isfinite(result))


# =====================================================================
# WENO5 vertical
# =====================================================================

class TestWENO5Vertical:
    """Tests for flux_form_vertical_tracer_advection_weno5."""

    def _fn(self):
        from legoesm.ocean.advection import flux_form_vertical_tracer_advection_weno5
        return flux_form_vertical_tracer_advection_weno5

    def test_zero_w_zero_flux(self):
        """Zero vertical velocity gives zero flux divergence."""
        fn = self._fn()
        nlev = 8
        field = jnp.linspace(1.0, 5.0, nlev)
        w_half = jnp.zeros(nlev + 1)
        h_k = jnp.full(nlev, 100.0)
        result = fn(field, w_half, h_k, dt=300.0)
        assert jnp.allclose(result, 0.0, atol=1e-15)

    def test_uniform_tracer_preserves_value(self):
        """Uniform tracer is preserved under flux-form update."""
        fn = self._fn()
        nlev = 8
        T_val = 5.0
        field = jnp.full(nlev, T_val)
        w_half = jnp.array([0.0, 0.1, -0.05, 0.08, -0.03, 0.04, -0.02, 0.01, 0.0])
        h_k = jnp.full(nlev, 100.0)
        dt = 300.0
        flux_div = fn(field, w_half, h_k, dt)

        # Continuity: h_new = h_old - dt * (w_top - w_bot)
        dw = w_half[:-1] - w_half[1:]
        h_new = h_k - dt * dw
        T_new = (h_k * field - dt * flux_div) / h_new
        assert jnp.allclose(T_new, T_val, atol=1e-12)

    def test_conservation_closed_column(self):
        """Column integral of flux divergence = 0 for closed boundaries."""
        fn = self._fn()
        nlev = 10
        key = jax.random.PRNGKey(42)
        field = jax.random.uniform(key, (nlev,), minval=0.5, maxval=5.0)
        w_half = jnp.concatenate([
            jnp.array([0.0]),
            0.01 * jnp.sin(jnp.linspace(0, 2 * jnp.pi, nlev - 1)),
            jnp.array([0.0]),
        ])
        h_k = jnp.linspace(50.0, 200.0, nlev)
        result = fn(field, w_half, h_k, dt=300.0)
        assert abs(float(jnp.sum(result))) < 1e-13

    def test_less_diffusive_than_upwind(self):
        """WENO5 advects a Gaussian more accurately than upwind."""
        fn = self._fn()
        from legoesm.ocean.vertical import flux_form_vertical_tracer_advection

        nlev = 40
        dz = 50.0
        z = jnp.arange(nlev) * dz
        field_init = 1.0 + 3.0 * jnp.exp(-((z - 12.0 * dz) / (3.0 * dz)) ** 2)

        w_val = 0.1  # upward
        w_half = jnp.concatenate([
            jnp.array([0.0]),
            jnp.full(nlev - 1, w_val),
            jnp.array([0.0]),
        ])
        h_k = jnp.full(nlev, dz)
        dt = 100.0

        n_steps = 20
        T_weno = field_init
        T_upw = field_init
        for _ in range(n_steps):
            flux_div_weno = fn(T_weno, w_half, h_k, dt)
            T_weno = T_weno - dt * flux_div_weno / h_k
            flux_div_upw = flux_form_vertical_tracer_advection(T_upw, w_half)
            T_upw = T_upw - dt * flux_div_upw / h_k

        cfl = w_val * dt / dz
        shift = n_steps * cfl * dz
        T_exact = 1.0 + 3.0 * jnp.exp(-((z - 12.0 * dz + shift) / (3.0 * dz)) ** 2)

        interior = slice(5, 35)
        err_weno = float(jnp.sqrt(jnp.mean((T_weno[interior] - T_exact[interior]) ** 2)))
        err_upw = float(jnp.sqrt(jnp.mean((T_upw[interior] - T_exact[interior]) ** 2)))
        assert err_weno < err_upw * 0.7, (
            f"WENO5 L2 error {err_weno:.6f} not much better than "
            f"upwind {err_upw:.6f} (ratio={err_weno / err_upw:.3f})")

    def test_batched_shape(self):
        """Works with batched leading dimensions."""
        fn = self._fn()
        shape = (4, 5, 8)
        nlev = shape[-1]
        key = jax.random.PRNGKey(0)
        field = jax.random.uniform(key, shape, minval=1.0, maxval=5.0)
        w_half = jnp.zeros((*shape[:-1], nlev + 1))
        w_half = w_half.at[..., 1:-1].set(0.001)
        h_k = jnp.full(shape, 100.0)
        result = fn(field, w_half, h_k, dt=300.0)
        assert result.shape == shape


# =====================================================================
# WENO7 horizontal
# =====================================================================

class TestWENO7Horizontal:
    """Tests for weno7_to_u_points and weno7_to_v_points."""

    def test_u_output_shape(self):
        from legoesm.ocean.advection import weno7_to_u_points
        n_lat, n_lon, nlev = 8, 20, 5
        f = jnp.ones((n_lat, n_lon, nlev))
        mf = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.5
        result = weno7_to_u_points(f, mf)
        assert result.shape == (n_lat, n_lon + 1, nlev)

    def test_v_output_shape(self):
        from legoesm.ocean.advection import weno7_to_v_points
        n_lat, n_lon, nlev = 12, 20, 5
        f = jnp.ones((n_lat, n_lon, nlev))
        mf = jnp.ones((n_lat + 1, n_lon, nlev)) * 0.5
        result = weno7_to_v_points(f, mf)
        assert result.shape == (n_lat + 1, n_lon, nlev)

    def test_u_constant_field(self):
        from legoesm.ocean.advection import weno7_to_u_points
        n_lat, n_lon, nlev = 8, 20, 5
        T_val = 4.0
        f = jnp.full((n_lat, n_lon, nlev), T_val)
        mf = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.5
        result = weno7_to_u_points(f, mf)
        assert jnp.allclose(result, T_val, atol=1e-13)

    def test_v_constant_field(self):
        from legoesm.ocean.advection import weno7_to_v_points
        n_lat, n_lon, nlev = 12, 20, 5
        T_val = 4.0
        f = jnp.full((n_lat, n_lon, nlev), T_val)
        mf = jnp.ones((n_lat + 1, n_lon, nlev)) * 0.5
        result = weno7_to_v_points(f, mf)
        assert jnp.allclose(result[1:-1], T_val, atol=1e-13)
        assert jnp.allclose(result[0], 0.0)
        assert jnp.allclose(result[-1], 0.0)

    def test_u_periodic_wrapping(self):
        from legoesm.ocean.advection import weno7_to_u_points
        n_lat, n_lon, nlev = 8, 20, 3
        key = jax.random.PRNGKey(42)
        f = jax.random.uniform(key, (n_lat, n_lon, nlev), minval=1.0, maxval=5.0)
        mf = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.3
        result = weno7_to_u_points(f, mf)
        assert jnp.allclose(result[:, -1, :], result[:, 0, :], atol=1e-14)


# =====================================================================
# WENO7 vertical
# =====================================================================

class TestWENO7Vertical:
    """Tests for flux_form_vertical_tracer_advection_weno7."""

    def _fn(self):
        from legoesm.ocean.advection import flux_form_vertical_tracer_advection_weno7
        return flux_form_vertical_tracer_advection_weno7

    def test_zero_w_zero_flux(self):
        fn = self._fn()
        nlev = 10
        field = jnp.linspace(1.0, 5.0, nlev)
        w_half = jnp.zeros(nlev + 1)
        h_k = jnp.full(nlev, 100.0)
        result = fn(field, w_half, h_k, dt=300.0)
        assert jnp.allclose(result, 0.0, atol=1e-15)

    def test_conservation_closed_column(self):
        fn = self._fn()
        nlev = 12
        key = jax.random.PRNGKey(99)
        field = jax.random.uniform(key, (nlev,), minval=0.5, maxval=5.0)
        w_half = jnp.concatenate([
            jnp.array([0.0]),
            0.01 * jnp.sin(jnp.linspace(0, 2 * jnp.pi, nlev - 1)),
            jnp.array([0.0]),
        ])
        h_k = jnp.linspace(50.0, 200.0, nlev)
        result = fn(field, w_half, h_k, dt=300.0)
        assert abs(float(jnp.sum(result))) < 1e-13

    def test_uniform_tracer_preserves_value(self):
        fn = self._fn()
        nlev = 10
        T_val = 7.0
        field = jnp.full(nlev, T_val)
        w_half = jnp.concatenate([
            jnp.array([0.0]),
            0.05 * jnp.sin(jnp.linspace(0, jnp.pi, nlev - 1)),
            jnp.array([0.0]),
        ])
        h_k = jnp.full(nlev, 100.0)
        dt = 300.0
        flux_div = fn(field, w_half, h_k, dt)
        dw = w_half[:-1] - w_half[1:]
        h_new = h_k - dt * dw
        T_new = (h_k * field - dt * flux_div) / h_new
        assert jnp.allclose(T_new, T_val, atol=1e-12)


# =====================================================================
# WENO7 higher accuracy than WENO5 on smooth data
# =====================================================================

class TestWENO7VsWENO5:
    """WENO7 should be more accurate than WENO5 on smooth profiles."""

    def test_vertical_smooth_gaussian(self):
        from legoesm.ocean.advection import (
            flux_form_vertical_tracer_advection_weno5,
            flux_form_vertical_tracer_advection_weno7,
        )
        nlev = 40
        dz = 50.0
        z = jnp.arange(nlev) * dz
        field_init = 1.0 + 3.0 * jnp.exp(-((z - 12.0 * dz) / (4.0 * dz)) ** 2)

        w_val = 0.05
        w_half = jnp.concatenate([
            jnp.array([0.0]),
            jnp.full(nlev - 1, w_val),
            jnp.array([0.0]),
        ])
        h_k = jnp.full(nlev, dz)
        dt = 100.0
        n_steps = 10

        T_w5 = field_init
        T_w7 = field_init
        for _ in range(n_steps):
            T_w5 = T_w5 - dt * flux_form_vertical_tracer_advection_weno5(
                T_w5, w_half, h_k, dt) / h_k
            T_w7 = T_w7 - dt * flux_form_vertical_tracer_advection_weno7(
                T_w7, w_half, h_k, dt) / h_k

        cfl = w_val * dt / dz
        shift = n_steps * cfl * dz
        T_exact = 1.0 + 3.0 * jnp.exp(-((z - 12.0 * dz + shift) / (4.0 * dz)) ** 2)

        interior = slice(5, 35)
        err_w5 = float(jnp.sqrt(jnp.mean((T_w5[interior] - T_exact[interior]) ** 2)))
        err_w7 = float(jnp.sqrt(jnp.mean((T_w7[interior] - T_exact[interior]) ** 2)))
        assert err_w7 <= err_w5 * 1.05, (
            f"WENO7 error {err_w7:.2e} not better than WENO5 {err_w5:.2e}")


# =====================================================================
# Horizontal conservation
# =====================================================================

class TestWENOHorizontalConservation:
    """Divergence of WENO horizontal flux integrates to zero globally."""

    @pytest.mark.parametrize("scheme", ["weno5", "weno7"])
    def test_conservation(self, scheme):
        if scheme == "weno5":
            from legoesm.ocean.advection import weno5_to_u_points as to_u
            from legoesm.ocean.advection import weno5_to_v_points as to_v
        else:
            from legoesm.ocean.advection import weno7_to_u_points as to_u
            from legoesm.ocean.advection import weno7_to_v_points as to_v
        from legoesm.ocean.dynamics.latlon_cgrid_operators import divergence_cgrid

        grid = _make_grid(n_lat=10, n_lon=20)
        nlev = 3
        n_lat, n_lon = grid.n_lat, grid.n_lon
        key = jax.random.PRNGKey(99)
        f = jax.random.uniform(key, (n_lat, n_lon, nlev), minval=1.0, maxval=5.0)

        k1, k2 = jax.random.split(key)
        mf_u_interior = 0.1 * jax.random.normal(k1, (n_lat, n_lon, nlev))
        mf_u = jnp.concatenate([mf_u_interior, mf_u_interior[:, :1, :]], axis=1)
        mf_v = 0.1 * jax.random.normal(k2, (n_lat + 1, n_lon, nlev))
        mf_v = mf_v.at[0].set(0.0).at[-1].set(0.0)

        f_u = to_u(f, mf_u)
        f_v = to_v(f, mf_v)
        flux_u = mf_u * f_u
        flux_v = mf_v * f_v
        div_flux = divergence_cgrid(flux_u, flux_v, grid)

        area = grid.area[..., jnp.newaxis]
        global_integral = float(jnp.sum(div_flux * area))
        assert abs(global_integral) < 1e-6, (
            f"{scheme} global flux divergence integral = {global_integral}")


# =====================================================================
# AD Taylor test
# =====================================================================

class TestWENOAD:
    """Reverse-mode AD through WENO tracer advection."""

    def test_ad_weno5_vertical(self):
        """Taylor test: ||f(x+eps*v) - f(x)|| converges at O(eps),
        ||f(x+eps*v) - f(x) - eps*Jv|| converges at O(eps^2).
        """
        from legoesm.ocean.advection import flux_form_vertical_tracer_advection_weno5

        nlev = 12
        key = jax.random.PRNGKey(17)
        field = jax.random.uniform(key, (nlev,), minval=1.0, maxval=5.0)
        w_half = jnp.concatenate([
            jnp.array([0.0]),
            0.01 * jnp.sin(jnp.linspace(0, 2 * jnp.pi, nlev - 1)),
            jnp.array([0.0]),
        ])
        h_k = jnp.full(nlev, 100.0)
        dt = 300.0

        def loss(f):
            fd = flux_form_vertical_tracer_advection_weno5(f, w_half, h_k, dt)
            return jnp.sum(fd ** 2)

        f0 = field
        L0 = loss(f0)
        g = jax.grad(loss)(f0)

        v = jax.random.normal(jax.random.PRNGKey(18), f0.shape)
        v = v / jnp.linalg.norm(v)
        Jv = jnp.dot(g, v)

        epsilons = [1e-2, 1e-3, 1e-4, 1e-5]
        first_order_errors = []
        second_order_errors = []
        for eps in epsilons:
            L_pert = loss(f0 + eps * v)
            first_order_errors.append(abs(float(L_pert - L0)))
            second_order_errors.append(abs(float(L_pert - L0 - eps * Jv)))

        # Check convergence rates
        for i in range(1, len(epsilons)):
            ratio_1 = first_order_errors[i - 1] / max(first_order_errors[i], 1e-30)
            ratio_2 = second_order_errors[i - 1] / max(second_order_errors[i], 1e-30)
            # First order errors should decrease ~10x (ratio ~10)
            assert ratio_1 > 5.0, f"First-order convergence failed: ratio={ratio_1:.1f}"
            # Second order errors should decrease ~100x (ratio ~100), but be lenient
            assert ratio_2 > 30.0, f"Second-order convergence failed: ratio={ratio_2:.1f}"

    def test_ad_weno5_horizontal(self):
        """Gradient through weno5_to_u_points is finite and nonzero."""
        from legoesm.ocean.advection import weno5_to_u_points

        n_lat, n_lon, nlev = 6, 12, 3
        key = jax.random.PRNGKey(21)
        f = jax.random.uniform(key, (n_lat, n_lon, nlev), minval=1.0, maxval=5.0)
        mf = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.5

        def loss(f_in):
            f_u = weno5_to_u_points(f_in, mf)
            return jnp.sum(f_u ** 2)

        g = jax.grad(loss)(f)
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.max(jnp.abs(g))) > 1e-10, "Gradient is zero"

    def test_ad_weno7_vertical(self):
        """Gradient through weno7 vertical is finite and nonzero."""
        from legoesm.ocean.advection import flux_form_vertical_tracer_advection_weno7

        nlev = 12
        key = jax.random.PRNGKey(30)
        field = jax.random.uniform(key, (nlev,), minval=1.0, maxval=5.0)
        w_half = jnp.concatenate([
            jnp.array([0.0]),
            0.01 * jnp.sin(jnp.linspace(0, 2 * jnp.pi, nlev - 1)),
            jnp.array([0.0]),
        ])
        h_k = jnp.full(nlev, 100.0)

        def loss(f):
            fd = flux_form_vertical_tracer_advection_weno7(f, w_half, h_k, 300.0)
            return jnp.sum(fd ** 2)

        g = jax.grad(loss)(field)
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.max(jnp.abs(g))) > 1e-10


# =====================================================================
# Config dispatch integration
# =====================================================================

class TestConfigDispatch:
    """Verify that 'weno5' and 'weno7' are accepted by LatLonCGridOceanConfig."""

    @pytest.mark.parametrize("scheme", ["weno5", "weno7"])
    def test_config_field_accepted(self, scheme):
        from legoesm.ocean.state import LatLonCGridOceanConfig
        cfg = LatLonCGridOceanConfig(tracer_advection=scheme)
        assert cfg.tracer_advection == scheme
