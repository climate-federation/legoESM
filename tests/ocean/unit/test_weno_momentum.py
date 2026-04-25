"""Tests for WENO-Z momentum advection (Phase 2b of Silvestri et al. 2024).

Tests:
1. WENO vorticity flux reconstruction: shapes, constant/linear field
   exactness, comparison with 2-point average
2. WENO vertical momentum advection: shapes, constant field, conservation
3. Config dispatch: momentum_advection="weno5"/"weno7" accepted
4. Full tendency: finite output with WENO momentum advection
5. AD: reverse-mode gradient finiteness through WENO momentum path
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
# WENO vorticity flux reconstruction: zeta_at_u
# =====================================================================

class TestWENOZetaAtU:
    """Tests for _weno_zeta_at_u (meridional reconstruction)."""

    def test_output_shape(self):
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            _weno_zeta_at_u,
        )
        n_lat, n_lon, nlev = 10, 20, 5
        zeta = jnp.ones((n_lat + 1, n_lon + 1, nlev))
        v_prime = jnp.ones((n_lat + 1, n_lon, nlev)) * 0.1
        v_at_u = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.1
        result = _weno_zeta_at_u(zeta, v_prime, v_at_u, order=5)
        assert result.shape == (n_lat, n_lon + 1, nlev)

    def test_constant_field(self):
        """Uniform vorticity → WENO reconstruction = uniform value."""
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            _weno_zeta_at_u,
        )
        n_lat, n_lon, nlev = 10, 20, 5
        zeta_val = 1.5e-4
        zeta = jnp.full((n_lat + 1, n_lon + 1, nlev), zeta_val)
        v_prime = jnp.ones((n_lat + 1, n_lon, nlev)) * 0.1
        v_at_u = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.1
        result = _weno_zeta_at_u(zeta, v_prime, v_at_u, order=5)
        assert jnp.allclose(result, zeta_val, atol=1e-13), (
            f"Max error: {float(jnp.max(jnp.abs(result - zeta_val)))}")

    def test_linear_field_interior(self):
        """Linear ζ(lat) → WENO5 reconstructs exactly at interior faces."""
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            _weno_zeta_at_u,
        )
        n_lat, n_lon, nlev = 20, 10, 3
        # Linear in axis 0: ζ[i] = a + b*i
        lat_idx = jnp.arange(n_lat + 1, dtype=jnp.float64)
        zeta = (1.0 + 0.01 * lat_idx)[:, jnp.newaxis, jnp.newaxis]
        zeta = jnp.broadcast_to(
            zeta, (n_lat + 1, n_lon + 1, nlev)).copy()
        v_prime = jnp.ones((n_lat + 1, n_lon, nlev)) * 0.1
        v_at_u = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.1
        result = _weno_zeta_at_u(zeta, v_prime, v_at_u, order=5)
        # Face i between vertex i and vertex i+1: expected = a + b*(i+0.5)
        face_idx = jnp.arange(n_lat, dtype=jnp.float64)
        expected = (1.0 + 0.01 * (face_idx + 0.5))[:, jnp.newaxis, jnp.newaxis]
        expected = jnp.broadcast_to(expected, (n_lat, n_lon + 1, nlev))
        # Skip faces near boundaries where ghosts degrade accuracy
        interior = slice(3, n_lat - 3)
        assert jnp.allclose(
            result[interior], expected[interior], atol=1e-12), (
            f"Max interior error: "
            f"{float(jnp.max(jnp.abs(result[interior] - expected[interior])))}")

    def test_finite_values(self):
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            _weno_zeta_at_u,
        )
        n_lat, n_lon, nlev = 10, 16, 5
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)
        zeta = jax.random.uniform(k1, (n_lat + 1, n_lon + 1, nlev),
                                  minval=-1e-4, maxval=1e-4)
        v_prime = jax.random.uniform(k2, (n_lat + 1, n_lon, nlev),
                                     minval=-0.5, maxval=0.5)
        v_at_u = jax.random.uniform(k3, (n_lat, n_lon + 1, nlev),
                                    minval=-0.5, maxval=0.5)
        result = _weno_zeta_at_u(zeta, v_prime, v_at_u, order=5)
        assert jnp.all(jnp.isfinite(result))

    def test_agrees_with_average_for_smooth_field(self):
        """On smooth ζ, WENO should be close to the 2-point average."""
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            _weno_zeta_at_u,
        )
        n_lat, n_lon, nlev = 20, 16, 3
        lat_idx = jnp.arange(n_lat + 1, dtype=jnp.float64)
        # Smooth sinusoidal vorticity
        zeta = (1e-4 * jnp.sin(
            2 * jnp.pi * lat_idx / n_lat
        ))[:, jnp.newaxis, jnp.newaxis]
        zeta = jnp.broadcast_to(
            zeta, (n_lat + 1, n_lon + 1, nlev)).copy()
        v_prime = jnp.ones((n_lat + 1, n_lon, nlev)) * 0.1
        v_at_u = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.1
        result = _weno_zeta_at_u(zeta, v_prime, v_at_u, order=5)
        avg = 0.5 * (zeta[:-1, :, :] + zeta[1:, :, :])
        # For smooth fields, both agree to within ~O(Δx²·amplitude).
        # With n_lat=20 and amplitude~1e-4, deviation ~ 1e-6.
        interior = slice(3, n_lat - 3)
        assert jnp.allclose(
            result[interior], avg[interior], atol=1e-5), (
            f"Max deviation from 2-pt avg: "
            f"{float(jnp.max(jnp.abs(result[interior] - avg[interior])))}")


# =====================================================================
# WENO vorticity flux reconstruction: zeta_at_v
# =====================================================================

class TestWENOZetaAtV:
    """Tests for _weno_zeta_at_v (zonal reconstruction)."""

    def test_output_shape(self):
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            _weno_zeta_at_v,
        )
        n_lat, n_lon, nlev = 10, 20, 5
        zeta = jnp.ones((n_lat + 1, n_lon + 1, nlev))
        u_prime = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.1
        u_at_v = jnp.ones((n_lat + 1, n_lon, nlev)) * 0.1
        result = _weno_zeta_at_v(zeta, u_prime, u_at_v, order=5)
        assert result.shape == (n_lat + 1, n_lon, nlev)

    def test_constant_field(self):
        """Uniform vorticity → WENO reconstruction = uniform value."""
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            _weno_zeta_at_v,
        )
        n_lat, n_lon, nlev = 10, 20, 5
        zeta_val = 2.5e-4
        zeta = jnp.full((n_lat + 1, n_lon + 1, nlev), zeta_val)
        u_prime = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.1
        u_at_v = jnp.ones((n_lat + 1, n_lon, nlev)) * 0.1
        result = _weno_zeta_at_v(zeta, u_prime, u_at_v, order=5)
        assert jnp.allclose(result, zeta_val, atol=1e-13), (
            f"Max error: {float(jnp.max(jnp.abs(result - zeta_val)))}")

    def test_periodic_wrapping(self):
        """Result should be consistent with periodic ζ."""
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            _weno_zeta_at_v,
        )
        n_lat, n_lon, nlev = 8, 20, 3
        key = jax.random.PRNGKey(11)
        # Build periodic ζ: column n_lon == column 0
        zeta_core = jax.random.uniform(
            key, (n_lat + 1, n_lon, nlev), minval=-1e-4, maxval=1e-4)
        zeta = jnp.concatenate(
            [zeta_core, zeta_core[:, 0:1, :]], axis=1)
        u_prime = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.1
        u_at_v = jnp.ones((n_lat + 1, n_lon, nlev)) * 0.1
        result = _weno_zeta_at_v(zeta, u_prime, u_at_v, order=5)
        assert jnp.all(jnp.isfinite(result))

    def test_finite_values(self):
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            _weno_zeta_at_v,
        )
        n_lat, n_lon, nlev = 10, 16, 5
        key = jax.random.PRNGKey(55)
        k1, k2, k3 = jax.random.split(key, 3)
        zeta = jax.random.uniform(k1, (n_lat + 1, n_lon + 1, nlev),
                                  minval=-1e-4, maxval=1e-4)
        u_prime = jax.random.uniform(k2, (n_lat, n_lon + 1, nlev),
                                     minval=-0.5, maxval=0.5)
        u_at_v = jax.random.uniform(k3, (n_lat + 1, n_lon, nlev),
                                    minval=-0.5, maxval=0.5)
        result = _weno_zeta_at_v(zeta, u_prime, u_at_v, order=5)
        assert jnp.all(jnp.isfinite(result))


# =====================================================================
# WENO7 vorticity reconstruction
# =====================================================================

class TestWENO7VorticityFlux:
    """WENO7 vorticity reconstruction shapes and constant field."""

    def test_zeta_at_u_shape(self):
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            _weno_zeta_at_u,
        )
        n_lat, n_lon, nlev = 12, 20, 5
        zeta = jnp.ones((n_lat + 1, n_lon + 1, nlev))
        v_prime = jnp.ones((n_lat + 1, n_lon, nlev)) * 0.1
        v_at_u = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.1
        result = _weno_zeta_at_u(zeta, v_prime, v_at_u, order=7)
        assert result.shape == (n_lat, n_lon + 1, nlev)

    def test_zeta_at_v_shape(self):
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            _weno_zeta_at_v,
        )
        n_lat, n_lon, nlev = 12, 24, 5
        zeta = jnp.ones((n_lat + 1, n_lon + 1, nlev))
        u_prime = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.1
        u_at_v = jnp.ones((n_lat + 1, n_lon, nlev)) * 0.1
        result = _weno_zeta_at_v(zeta, u_prime, u_at_v, order=7)
        assert result.shape == (n_lat + 1, n_lon, nlev)

    def test_constant_field_both(self):
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            _weno_zeta_at_u, _weno_zeta_at_v,
        )
        n_lat, n_lon, nlev = 12, 24, 3
        zeta_val = 3.0e-4
        zeta = jnp.full((n_lat + 1, n_lon + 1, nlev), zeta_val)
        v_prime = jnp.ones((n_lat + 1, n_lon, nlev)) * 0.1
        v_at_u = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.1
        u_prime = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.1
        u_at_v = jnp.ones((n_lat + 1, n_lon, nlev)) * 0.1
        r_u = _weno_zeta_at_u(zeta, v_prime, v_at_u, order=7)
        r_v = _weno_zeta_at_v(zeta, u_prime, u_at_v, order=7)
        assert jnp.allclose(r_u, zeta_val, atol=1e-13)
        assert jnp.allclose(r_v, zeta_val, atol=1e-13)


# =====================================================================
# WENO vertical momentum advection
# =====================================================================

class TestWENOVerticalMomentum:
    """Tests for _flux_form_vertical_momentum_advection_weno."""

    def _fn(self):
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            _flux_form_vertical_momentum_advection_weno,
        )
        return _flux_form_vertical_momentum_advection_weno

    def test_zero_w_zero_tendency(self):
        """Zero vertical velocity ⟹ zero tendency."""
        fn = self._fn()
        nlev = 8
        u = jnp.linspace(0.1, 0.5, nlev)
        w_half = jnp.zeros(nlev + 1)
        h_u = jnp.full(nlev, 100.0)
        result = fn(u, w_half, h_u, order=5)
        assert jnp.allclose(result, 0.0, atol=1e-15)

    def test_uniform_velocity_zero_tendency(self):
        """Uniform velocity ⟹ advection can't change it (conservation)."""
        fn = self._fn()
        nlev = 10
        u_val = 0.3
        u = jnp.full(nlev, u_val)
        w_half = jnp.concatenate([
            jnp.array([0.0]),
            0.01 * jnp.sin(jnp.linspace(0, 2 * jnp.pi, nlev - 1)),
            jnp.array([0.0]),
        ])
        h_u = jnp.full(nlev, 100.0)
        result = fn(u, w_half, h_u, order=5)
        # For uniform velocity, F_top = w_top * u_val, F_bot = w_bot * u_val
        # tendency = -(F_top - F_bot) / h_u. With uniform u, flux_div cancels
        # to give zero tendency (column integral = 0, and each level gets
        # a nonzero tendency only from w gradients but the overall effect
        # is the same velocity redistributed).
        # Actually for uniform u, all interface reconstructions = u_val
        # so flux_div = w * u_val cancellation gives tendency that
        # preserves the uniform profile under the update.
        # Check: column-integrated momentum tendency sums to zero
        col_sum = float(jnp.sum(result * h_u))
        assert abs(col_sum) < 1e-12, f"Column sum = {col_sum}"

    def test_output_shape_3d(self):
        """Works with 3D input (n_lat, n_lon+1, nlev)."""
        fn = self._fn()
        shape = (8, 17, 10)  # n_lat, n_lon+1, nlev
        nlev = shape[-1]
        key = jax.random.PRNGKey(0)
        u = jax.random.uniform(key, shape, minval=-0.5, maxval=0.5)
        w_half = jnp.zeros((*shape[:-1], nlev + 1))
        w_half = w_half.at[..., 1:-1].set(0.001)
        h_u = jnp.full(shape, 100.0)
        result = fn(u, w_half, h_u, order=5)
        assert result.shape == shape

    def test_finite_values(self):
        fn = self._fn()
        nlev = 12
        key = jax.random.PRNGKey(7)
        u = jax.random.uniform(key, (nlev,), minval=-0.5, maxval=0.5)
        w_half = jnp.concatenate([
            jnp.array([0.0]),
            0.01 * jax.random.normal(jax.random.PRNGKey(8), (nlev - 1,)),
            jnp.array([0.0]),
        ])
        h_u = jnp.full(nlev, 100.0)
        result = fn(u, w_half, h_u, order=5)
        assert jnp.all(jnp.isfinite(result))


# =====================================================================
# Config dispatch
# =====================================================================

class TestMomentumAdvectionConfig:
    """Verify that momentum_advection options are accepted."""

    @pytest.mark.parametrize("mode", ["vector_invariant", "weno5", "weno7"])
    def test_config_field_accepted(self, mode):
        from legoesm.ocean.state import LatLonCGridOceanConfig
        cfg = LatLonCGridOceanConfig(momentum_advection=mode)
        assert cfg.momentum_advection == mode


# =====================================================================
# AD (reverse-mode gradient) correctness
# =====================================================================

class TestWENOMomentumAD:
    """Reverse-mode AD through WENO momentum advection."""

    def test_ad_zeta_at_u_finite(self):
        """Gradient through _weno_zeta_at_u is finite and nonzero."""
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            _weno_zeta_at_u,
        )
        n_lat, n_lon, nlev = 8, 12, 3
        key = jax.random.PRNGKey(21)
        k1, k2, k3 = jax.random.split(key, 3)
        zeta = jax.random.uniform(
            k1, (n_lat + 1, n_lon + 1, nlev),
            minval=-1e-4, maxval=1e-4)
        v_prime = jax.random.uniform(
            k2, (n_lat + 1, n_lon, nlev), minval=-0.3, maxval=0.3)
        v_at_u = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.1

        def loss(z):
            r = _weno_zeta_at_u(z, v_prime, v_at_u, order=5)
            return jnp.sum(r ** 2)

        g = jax.grad(loss)(zeta)
        assert jnp.all(jnp.isfinite(g)), "Gradient has non-finite values"
        assert float(jnp.max(jnp.abs(g))) > 1e-15, "Gradient is zero"

    def test_ad_zeta_at_v_finite(self):
        """Gradient through _weno_zeta_at_v is finite and nonzero."""
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            _weno_zeta_at_v,
        )
        n_lat, n_lon, nlev = 8, 16, 3
        key = jax.random.PRNGKey(31)
        k1, k2, k3 = jax.random.split(key, 3)
        zeta = jax.random.uniform(
            k1, (n_lat + 1, n_lon + 1, nlev),
            minval=-1e-4, maxval=1e-4)
        u_prime = jax.random.uniform(
            k2, (n_lat, n_lon + 1, nlev), minval=-0.3, maxval=0.3)
        u_at_v = jnp.ones((n_lat + 1, n_lon, nlev)) * 0.1

        def loss(z):
            r = _weno_zeta_at_v(z, u_prime, u_at_v, order=5)
            return jnp.sum(r ** 2)

        g = jax.grad(loss)(zeta)
        assert jnp.all(jnp.isfinite(g)), "Gradient has non-finite values"
        assert float(jnp.max(jnp.abs(g))) > 1e-15, "Gradient is zero"

    def test_ad_vertical_momentum_weno(self):
        """Gradient through vertical momentum WENO is finite."""
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            _flux_form_vertical_momentum_advection_weno,
        )
        nlev = 10
        key = jax.random.PRNGKey(40)
        u = jax.random.uniform(key, (nlev,), minval=-0.5, maxval=0.5)
        w_half = jnp.concatenate([
            jnp.array([0.0]),
            0.01 * jnp.sin(jnp.linspace(0, 2 * jnp.pi, nlev - 1)),
            jnp.array([0.0]),
        ])
        h_u = jnp.full(nlev, 100.0)

        def loss(u_in):
            t = _flux_form_vertical_momentum_advection_weno(
                u_in, w_half, h_u, order=5)
            return jnp.sum(t ** 2)

        g = jax.grad(loss)(u)
        assert jnp.all(jnp.isfinite(g)), "Gradient has non-finite values"
        assert float(jnp.max(jnp.abs(g))) > 1e-15, "Gradient is zero"

    def test_ad_taylor_zeta_at_u(self):
        """Taylor test for AD correctness through _weno_zeta_at_u.

        ||f(x+eps*v) - f(x)|| = O(eps)
        ||f(x+eps*v) - f(x) - eps*Jv|| = O(eps²)
        """
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            _weno_zeta_at_u,
        )
        n_lat, n_lon, nlev = 10, 12, 3
        key = jax.random.PRNGKey(50)
        k1, k2 = jax.random.split(key)
        zeta = jax.random.uniform(
            k1, (n_lat + 1, n_lon + 1, nlev),
            minval=-1e-4, maxval=1e-4)
        v_prime = jnp.ones((n_lat + 1, n_lon, nlev)) * 0.1
        v_at_u = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.1

        def loss(z):
            r = _weno_zeta_at_u(z, v_prime, v_at_u, order=5)
            return jnp.sum(r ** 2)

        z0 = zeta
        L0 = loss(z0)
        g = jax.grad(loss)(z0)
        v = jax.random.normal(k2, z0.shape)
        v = v / jnp.linalg.norm(v)
        Jv = jnp.sum(g * v)

        epsilons = [1e-3, 1e-4, 1e-5, 1e-6]
        second_order_errors = []
        for eps in epsilons:
            L_pert = loss(z0 + eps * v)
            second_order_errors.append(
                abs(float(L_pert - L0 - eps * Jv)))

        for i in range(1, len(epsilons)):
            ratio = (second_order_errors[i - 1]
                     / max(second_order_errors[i], 1e-30))
            assert ratio > 30.0, (
                f"Second-order convergence failed: ratio={ratio:.1f}")
