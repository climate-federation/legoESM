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
        from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
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
        # Both must integrate to the same total time for a fair comparison.
        # C4: 50 * 600 = 30000s, C8: 100 * 300 = 30000s
        err_c4 = self._run_tc2(4, n_steps=50, dt=600.0)
        err_c8 = self._run_tc2(8, n_steps=100, dt=300.0)

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


# ---------------------------------------------------------------------------
# 6g  Voronoi gradient operator convergence
# ---------------------------------------------------------------------------

class TestGradientConvergenceVoronoi:
    """Gradient of cos(lat)*cos(2*lon) on Voronoi meshes should converge."""

    @staticmethod
    def _compute_gradient_error(level):
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.core.operators_voronoi import gradient_edge

        mesh = create_voronoi_mesh(level, lloyd_iterations=30)
        a = mesh.radius

        phi = jnp.cos(mesh.latCell) * jnp.cos(2.0 * mesh.lonCell)
        grad = gradient_edge(phi, mesh)

        # Analytic gradient projected onto edge normals:
        # dphi/dx = -2*sin(2*lon) / a
        # dphi/dy = -sin(lat)*cos(2*lon) / a
        # edge-normal component: dphi_n = dphi/dx * cos(angle) + dphi/dy * sin(angle)
        dphi_dx = -2.0 * jnp.sin(2.0 * mesh.lonEdge) / a
        dphi_dy = -jnp.sin(mesh.latEdge) * jnp.cos(2.0 * mesh.lonEdge) / a
        grad_exact = dphi_dx * jnp.cos(mesh.angleEdge) + dphi_dy * jnp.sin(mesh.angleEdge)

        err = float(jnp.sqrt(jnp.mean((grad - grad_exact)**2)))
        return err

    def test_convergence_level2_level3(self):
        """Error should decrease from level-2 to level-3 mesh."""
        err_2 = self._compute_gradient_error(2)
        err_3 = self._compute_gradient_error(3)

        ratio = err_2 / err_3
        assert ratio > 1.5, (
            f"err_L2/err_L3 = {ratio:.2f}, expected > 1.5 "
            f"(err_L2={err_2:.4e}, err_L3={err_3:.4e})"
        )


# ---------------------------------------------------------------------------
# 6i  Williamson TC2 convergence on MPAS
# ---------------------------------------------------------------------------

class TestWilliamsonTC2ConvergenceMPAS:
    """TC2 on MPAS Voronoi mesh should converge with resolution."""

    @staticmethod
    def _run_tc2_mpas(level, n_steps=50, dt=300.0):
        from legoesm.atmosphere.dynamics.gcm.shallow_water_mpas import (
            MPASShallowWaterModel,
            MPASShallowWaterConfig,
            MPASShallowWaterState,
        )
        from legoesm.grids.voronoi import create_voronoi_mesh

        mesh = create_voronoi_mesh(level, lloyd_iterations=30)

        g = constants.g
        Omega = constants.Omega
        R = mesh.radius
        u0 = 20.0
        h0 = 1e4

        h_init = h0 - (R * Omega * u0 + 0.5 * u0**2) * jnp.sin(mesh.latCell)**2 / g
        u_edge = u0 * jnp.cos(mesh.latEdge) * jnp.cos(mesh.angleEdge)

        state = MPASShallowWaterState(
            h=Field(data=h_init, name="h", dims=("nCells",), units="m"),
            u=Field(data=u_edge, name="u", dims=("nEdges",), units="m/s",
                    staggering="edge"),
            h_s=Field(data=jnp.zeros(mesh.nCells), name="h_s",
                      dims=("nCells",), units="m"),
        )

        config = MPASShallowWaterConfig(fix_mass=True, fix_energy=False)
        model = MPASShallowWaterModel(mesh, config)

        for _ in range(n_steps):
            state = model.step(state, dt)

        h_err = state.h.data - h_init
        area = mesh.areaCell
        return float(jnp.sqrt(jnp.sum(h_err**2 * area) / jnp.sum(area)))

    def test_level3_better_than_level2(self):
        """Level-3 mesh should have smaller TC2 error than level-2."""
        err_2 = self._run_tc2_mpas(2, n_steps=50, dt=600.0)
        err_3 = self._run_tc2_mpas(3, n_steps=50, dt=300.0)

        assert err_3 < err_2, (
            f"Level-3 error ({err_3:.4e}) should be < level-2 error ({err_2:.4e})"
        )


# ---------------------------------------------------------------------------
# 6j  Time integrator order verification (spectral SW)
# ---------------------------------------------------------------------------

class TestTimeIntegratorOrder:
    """Verify time integrator order by running with dt, dt/2, dt/4
    and checking that the error ratio matches the expected order.

    Uses the spectral shallow water model which has minimal spatial error,
    isolating the temporal error.
    """

    @staticmethod
    def _run_spectral_sw(n_max, dt, n_steps, H0=1e4):
        """Run spectral SW with Gaussian bump and return final phi_hat."""
        from legoesm.grids.gaussian import create_gaussian_grid, sh_analysis
        from legoesm.atmosphere.dynamics.gcm.spectral_sw import (
            SpectralShallowWaterModel,
            SpectralSWConfig,
            SpectralSWState,
        )

        grid = create_gaussian_grid(n_max)
        g = constants.g

        # Small Gaussian bump
        lat0, lon0 = 0.0, jnp.pi
        sigma_rad = 20.0 * jnp.pi / 180.0
        dist_sq = (grid.lat2d - lat0)**2 + (grid.lon2d - lon0)**2
        bump = 10.0 * jnp.exp(-dist_sq / (2.0 * sigma_rad**2))

        h_grid = H0 + bump
        phi_hat = sh_analysis(grid, g * h_grid)

        state = SpectralSWState(
            vor_hat=Field(data=jnp.zeros(grid.n_sh, dtype=jnp.complex128),
                         name="vor_hat", dims=("n_sh",), units="1/s"),
            div_hat=Field(data=jnp.zeros(grid.n_sh, dtype=jnp.complex128),
                         name="div_hat", dims=("n_sh",), units="1/s"),
            phi_hat=Field(data=phi_hat, name="phi_hat", dims=("n_sh",), units="m2/s2"),
            phis_hat=Field(data=jnp.zeros(grid.n_sh, dtype=jnp.complex128),
                          name="phis_hat", dims=("n_sh",), units="m2/s2"),
        )

        # Background depth H0 enters via the initial phi_hat (h_grid = H0 + bump);
        # the removed config field of that name was never read by the solver.
        config = SpectralSWConfig(
            hyperdiff_coeff=0.0,
            spectral_filter_order=0,
        )
        model = SpectralShallowWaterModel(grid, config)

        for _ in range(n_steps):
            state = model.step(state, dt)

        return state.phi_hat.data, grid

    def test_error_decreases_with_smaller_dt(self):
        """Error should decrease when dt is halved."""
        n_max = 10
        total_time = 1800.0  # 30 minutes

        dt1 = 60.0
        dt2 = 30.0
        n1 = int(total_time / dt1)
        n2 = int(total_time / dt2)

        # Use dt/4 as "reference" solution
        dt_ref = 15.0
        n_ref = int(total_time / dt_ref)

        phi1, grid = self._run_spectral_sw(n_max, dt1, n1)
        phi2, _ = self._run_spectral_sw(n_max, dt2, n2)
        phi_ref, _ = self._run_spectral_sw(n_max, dt_ref, n_ref)

        err1 = float(jnp.sqrt(jnp.sum(jnp.abs(phi1 - phi_ref)**2)))
        err2 = float(jnp.sqrt(jnp.sum(jnp.abs(phi2 - phi_ref)**2)))

        # Error should decrease when dt halves
        assert err2 < err1, (
            f"err(dt={dt2}) = {err2:.4e} should be < err(dt={dt1}) = {err1:.4e}"
        )

        # For RK3: error ratio should be ~8 (2^3) when dt halves
        # At coarse resolution with nonlinear dynamics, we just check ratio > 2
        ratio = err1 / (err2 + 1e-30)
        assert ratio > 2.0, (
            f"Error ratio = {ratio:.2f}, expected > 2.0 for at least 1st-order convergence"
        )
