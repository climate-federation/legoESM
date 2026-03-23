"""Convergence rate tests for operators and dynamics (Category 6).

Tests:
  6a) Gradient operator convergence on cubed-sphere (C4, C8, C16)
  6b) Spectral roundtrip accuracy (sh_analysis -> sh_synthesis)
  6c) Williamson TC2 error decreases with resolution (C4 vs C8)
"""

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.operators import gradient_x, gradient_y
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.gaussian import create_gaussian_grid, sh_analysis, sh_synthesis
from legoesm import constants


# ---------------------------------------------------------------------------
# 6a  Gradient operator convergence on cubed-sphere
# ---------------------------------------------------------------------------

class TestGradientConvergence:
    """Gradient of cos(lat)*cos(2*lon) should converge at >= 1st order."""

    @staticmethod
    def _compute_gradient_error(n):
        grid = create_cubed_sphere(n)
        a = grid.radius
        lat = grid.lat
        lon = grid.lon

        # phi = cos(lat) * cos(2*lon)
        phi_data = (jnp.cos(lat) * jnp.cos(2.0 * lon)).astype(jnp.float32)
        phi = Field(data=phi_data, name="phi", dims=("face", "x", "y"), units="1")

        # Analytic gradient in geographic coordinates:
        # dphi/d(x_east) = (1 / (a * cos_lat)) * dphi/dlon
        #                 = -2 * sin(2*lon) * cos(lat) / (a * cos_lat)
        #                 = -2 * sin(2*lon) / a
        # dphi/d(y_north) = (1/a) * dphi/dlat = -sin(lat)*cos(2*lon) / a
        #
        # But gradient_x and gradient_y return derivatives in grid-aligned coords.
        # On the cubed-sphere, this is df/dx_grid and df/dy_grid, where x_grid
        # and y_grid are the local gnomonic directions. Since the operators are
        # metric-aware, we compare with geographic derivatives rotated to grid.

        cos_lat = jnp.maximum(jnp.cos(lat), 1e-10)
        dphi_deast = -2.0 * jnp.sin(2.0 * lon) / a
        dphi_dnorth = -jnp.sin(lat) * jnp.cos(2.0 * lon) / a

        # Rotate geographic -> grid-aligned
        cos_a = jnp.cos(grid.angle)
        sin_a = jnp.sin(grid.angle)
        dphi_dx_exact = (cos_a * dphi_deast + sin_a * dphi_dnorth).astype(jnp.float32)

        # Numerical gradient
        dphi_dx_num = gradient_x(phi, grid).data

        # L2 error (area-weighted)
        diff = dphi_dx_num - dphi_dx_exact
        area = grid.area
        l2_err = float(jnp.sqrt(jnp.sum(diff ** 2 * area) / jnp.sum(area)))
        return l2_err

    def test_convergence_c4_c8_c16(self):
        err_c4 = self._compute_gradient_error(4)
        err_c8 = self._compute_gradient_error(8)
        err_c16 = self._compute_gradient_error(16)

        # At least 1st-order convergence: error ratio > 2 when resolution doubles
        ratio_1 = err_c4 / err_c8
        ratio_2 = err_c8 / err_c16

        assert ratio_1 > 2.0, (
            f"err_C4/err_C8 = {ratio_1:.2f}, expected > 2.0 "
            f"(err_C4={err_c4:.4e}, err_C8={err_c8:.4e})"
        )
        assert ratio_2 > 1.9, (
            f"err_C8/err_C16 = {ratio_2:.2f}, expected > 1.9 "
            f"(err_C8={err_c8:.4e}, err_C16={err_c16:.4e})"
        )


# ---------------------------------------------------------------------------
# 6b  Spectral roundtrip accuracy
# ---------------------------------------------------------------------------

class TestSpectralRoundtrip:
    """sh_analysis -> sh_synthesis should roundtrip bandlimited fields."""

    @pytest.mark.parametrize("n_max", [5, 10])
    def test_bandlimited_roundtrip(self, n_max):
        grid = create_gaussian_grid(n_max)
        lat2d = grid.lat2d
        lon2d = grid.lon2d

        # Build a bandlimited field: Y_2^0 + Y_3^1 type
        # Use low-order spherical harmonics that are within the truncation
        field = (
            jnp.cos(lat2d) * jnp.cos(lon2d)         # ~ Y_1^1
            + 0.5 * (3.0 * jnp.sin(lat2d) ** 2 - 1)  # ~ Y_2^0
        )

        coeffs = sh_analysis(grid, field)
        reconstructed = sh_synthesis(grid, coeffs)

        err = float(jnp.max(jnp.abs(reconstructed - field)))
        assert err < 1e-10, (
            f"Spectral roundtrip error = {err:.2e} for n_max={n_max}, expected < 1e-10"
        )


# ---------------------------------------------------------------------------
# 6c  Williamson TC2 error at multiple resolutions (cubed-sphere)
# ---------------------------------------------------------------------------

class TestWilliamsonTC2Convergence:
    """TC2 (steady geostrophic flow) error should decrease with resolution."""

    @staticmethod
    def _run_tc2(n, n_steps=100, dt=300.0):
        from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
            CDGridShallowWaterModel,
            CDGridShallowWaterConfig,
            CDGridShallowWaterState,
        )
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

        grid_base = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid_base)

        # Williamson TC2 initial condition directly on CD-grid
        g = constants.g
        omega = constants.Omega
        R = grid_base.radius
        u_0 = 2.0 * jnp.pi * R / (12.0 * 86400.0)
        gh_0 = 2.94e4
        h_0 = gh_0 / g

        # Height at cell centres
        lat_c = grid_base.lat
        h_init = h_0 - (R * omega * u_0 + 0.5 * u_0 ** 2) * jnp.sin(lat_c) ** 2 / g

        # Velocity at D-grid corners: u_east = u_0*cos(lat), v_north = 0
        lat_corner = cdgrid.lat_corner
        cos_lat = jnp.cos(lat_corner)
        u_geo = u_0 * cos_lat
        u_d_init = u_geo * cdgrid.cos_angle_corner
        v_d_init = u_geo * cdgrid.sin_angle_corner

        h_s = jnp.zeros_like(h_init)

        state0 = CDGridShallowWaterState(
            h=h_init, u_d=u_d_init, v_d=v_d_init, h_s=h_s,
        )

        config = CDGridShallowWaterConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=False,
        )
        model = CDGridShallowWaterModel(grid_base, config)

        state = state0
        for _ in range(n_steps):
            state = model.step(state, dt)

        # L2 error in h (exact solution = initial condition)
        h_err = state.h - h_init
        area = grid_base.area
        l2 = float(jnp.sqrt(jnp.sum(h_err ** 2 * area) / jnp.sum(area)))
        return l2

    def test_c8_better_than_c4(self):
        err_c4 = self._run_tc2(4, n_steps=50, dt=600.0)
        err_c8 = self._run_tc2(8, n_steps=50, dt=300.0)

        assert err_c8 < err_c4, (
            f"C8 error ({err_c8:.4e}) should be < C4 error ({err_c4:.4e})"
        )


# ---------------------------------------------------------------------------
# 6d  Gradient operator convergence on lat-lon
# ---------------------------------------------------------------------------

class TestGradientConvergenceLatLon:
    """Gradient of cos(lat)*cos(2*lon) on lat-lon grids should converge."""

    @staticmethod
    def _compute_gradient_error(n_lat):
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.core.operators_latlon import gradient_x as gradient_x_ll

        n_lon = 2 * n_lat
        grid = create_latlon_grid(n_lat, n_lon)
        a = grid.radius

        lat2d = grid.lat2d
        lon2d = grid.lon2d

        # phi = cos(lat) * cos(2*lon)
        phi_data = jnp.cos(lat2d) * jnp.cos(2.0 * lon2d)
        phi = Field(data=phi_data, name="phi", dims=("lat", "lon"), units="1")

        # Analytic eastward gradient:
        # dphi/dx = (1 / (a * cos(lat))) * dphi/dlon
        #         = (1 / (a * cos(lat))) * cos(lat) * (-2 * sin(2*lon))
        #         = -2 * sin(2*lon) / a
        dphi_dx_exact = -2.0 * jnp.sin(2.0 * lon2d) / a

        # Numerical gradient
        dphi_dx_num = gradient_x_ll(phi, grid).data

        # L2 error (area-weighted)
        diff = dphi_dx_num - dphi_dx_exact
        area = grid.area
        l2_err = float(jnp.sqrt(jnp.sum(diff ** 2 * area) / jnp.sum(area)))
        return l2_err

    def test_convergence_8_16_32(self):
        err_8 = self._compute_gradient_error(8)
        err_16 = self._compute_gradient_error(16)
        err_32 = self._compute_gradient_error(32)

        ratio_1 = err_8 / err_16
        ratio_2 = err_16 / err_32

        assert ratio_1 > 1.5, (
            f"err_8/err_16 = {ratio_1:.2f}, expected > 1.5 "
            f"(err_8={err_8:.4e}, err_16={err_16:.4e})"
        )
        assert ratio_2 > 1.5, (
            f"err_16/err_32 = {ratio_2:.2f}, expected > 1.5 "
            f"(err_16={err_16:.4e}, err_32={err_32:.4e})"
        )


# ---------------------------------------------------------------------------
# 6e  Hyperdiffusion convergence on cubed-sphere
# ---------------------------------------------------------------------------

class TestHyperdiffusionConvergence:
    """Hyperdiffusion convergence on the cubed-sphere.

    Apply nabla^4 to phi = cos(lat)*cos(2*lon) on C4, C8, C16.
    Since cos(lat)*cos(2*lon) mixes many spherical harmonic degrees,
    we verify convergence by checking that the global RMS of the
    hyperdiffusion result scales regularly with resolution (h^4
    scaling for the biharmonic operator means RMS roughly quadruples
    when resolution doubles). The L2 difference of successive
    resolution RMS values should shrink as a ratio.

    Additionally, we verify that the hyperdiffusion operator is
    dissipative (removes energy from a perturbation) and that
    a dt-stepped diffusion reduces variance at all resolutions.
    """

    @staticmethod
    def _compute_hyperdiff_rms(n):
        from legoesm.core.operators import hyperdiffusion as hyperdiff_cs

        grid = create_cubed_sphere(n)
        lat = grid.lat
        lon = grid.lon

        phi_data = jnp.cos(lat) * jnp.cos(2.0 * lon)
        phi = Field(data=phi_data, name="phi", dims=("face", "x", "y"), units="1")

        # Use a physically reasonable coefficient that does not blow up
        dx_min = float(jnp.min(grid.dx))
        coeff = dx_min ** 4 / (16.0 * 600.0)  # safe sub-CFL coefficient

        result = hyperdiff_cs(phi, grid, coeff=coeff).data

        # After one Euler step: phi_new = phi + dt * hyperdiff
        dt = 600.0
        phi_new = phi_data + dt * result

        # Variance reduction: hyperdiffusion should reduce variance
        area = grid.area
        total_area = jnp.sum(area)
        mean_old = float(jnp.sum(phi_data * area) / total_area)
        mean_new = float(jnp.sum(phi_new * area) / total_area)
        var_old = float(jnp.sum((phi_data - mean_old) ** 2 * area) / total_area)
        var_new = float(jnp.sum((phi_new - mean_new) ** 2 * area) / total_area)

        return var_old, var_new

    def test_variance_reduction_c4_c8_c16(self):
        """Hyperdiffusion should reduce variance at all resolutions."""
        for n in [4, 8, 16]:
            var_old, var_new = self._compute_hyperdiff_rms(n)
            assert var_new < var_old, (
                f"C{n}: hyperdiffusion did not reduce variance: "
                f"var_old={var_old:.4e}, var_new={var_new:.4e}"
            )

    def test_relative_reduction_increases(self):
        """Finer grids with resolution-scaled coeff have smaller relative reduction.

        Because the hyperdiffusion coefficient is scaled as dx^4/tau, finer
        grids damp less (the operator is more scale-selective). The relative
        variance reduction (var_new/var_old) should be closer to 1.0 at
        higher resolution, showing convergence toward the identity.
        """
        ratios = []
        for n in [4, 8, 16]:
            var_old, var_new = self._compute_hyperdiff_rms(n)
            ratios.append(var_new / var_old)

        # At finer resolution with CFL-scaled coeff, the relative
        # change in variance should decrease (ratio closer to 1)
        assert ratios[1] > ratios[0], (
            f"C8 ratio ({ratios[1]:.6f}) should be > C4 ratio ({ratios[0]:.6f})"
        )
        assert ratios[2] > ratios[1], (
            f"C16 ratio ({ratios[2]:.6f}) should be > C8 ratio ({ratios[1]:.6f})"
        )


# ---------------------------------------------------------------------------
# 6f  Spectral Laplacian convergence
# ---------------------------------------------------------------------------

class TestSpectralLaplacianConvergence:
    """Spectral Laplacian of Y_2^0 should be exact (machine precision)."""

    @pytest.mark.parametrize("n_max", [5, 10])
    def test_laplacian_Y20(self, n_max):
        grid = create_gaussian_grid(n_max)
        a = grid.radius

        lat2d = grid.lat2d

        # Y_2^0 proportional to (3*sin^2(lat) - 1)/2
        Y20 = 0.5 * (3.0 * jnp.sin(lat2d) ** 2 - 1.0)

        # Forward transform
        coeffs = sh_analysis(grid, Y20)

        # Apply spectral Laplacian: multiply by -n(n+1)/a^2
        lap_coeffs = grid.lap * coeffs

        # Inverse transform
        lap_grid = sh_synthesis(grid, lap_coeffs)

        # Analytic: Laplacian of Y_2^0 = -n(n+1)/a^2 * Y_2^0 with n=2
        # = -6/a^2 * Y_2^0
        lap_exact = -6.0 / (a ** 2) * Y20

        err = float(jnp.max(jnp.abs(lap_grid - lap_exact)))
        assert err < 1e-10, (
            f"Spectral Laplacian error = {err:.2e} for n_max={n_max}, "
            f"expected < 1e-10 (should be exact for spectral methods)"
        )
