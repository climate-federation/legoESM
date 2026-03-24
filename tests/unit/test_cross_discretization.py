"""Cross-discretization tests (Category 7).

Tests:
  7a) Spectral roundtrip with supergrid content — truncation fidelity
  7b) Global diagnostics agreement across grids (lat-lon vs MPAS)
"""

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState, MPASShallowWaterState
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.grids.gaussian import create_gaussian_grid, sh_analysis, sh_synthesis
from legoesm.atmosphere.dynamics.shallow_water_fv_latlon import (
    FVShallowWaterLatLonModel,
    FVShallowWaterLatLonConfig,
)
from legoesm.atmosphere.dynamics.shallow_water_mpas import (
    MPASShallowWaterModel,
    MPASShallowWaterConfig,
)
from legoesm import constants


# ---------------------------------------------------------------------------
# 7a  Spectral roundtrip with supergrid content
# ---------------------------------------------------------------------------

class TestSpectralTruncation:
    """After analysis->synthesis, content beyond n_max should be removed."""

    def test_supergrid_content_removed(self):
        n_max = 5
        grid = create_gaussian_grid(n_max)
        lat2d = grid.lat2d
        lon2d = grid.lon2d

        # Build a field with content within n_max (low-order modes)
        low = jnp.cos(lat2d) * jnp.cos(lon2d)  # ~ Y_1^1

        # Add high-wavenumber content that exceeds n_max
        # n_max=5 means modes 0..5 are retained; wavenumber 10 is beyond that.
        high = 0.5 * jnp.cos(10.0 * lon2d)   # zonal wavenumber 10 >> n_max=5

        field = low + high

        # Roundtrip
        coeffs = sh_analysis(grid, field)
        reconstructed = sh_synthesis(grid, coeffs)

        # The reconstructed field should match only the low-order part,
        # not the original field (which includes the high modes).
        err_vs_original = float(jnp.max(jnp.abs(reconstructed - field)))
        err_vs_low = float(jnp.max(jnp.abs(reconstructed - low)))

        # The high-mode content (amplitude ~0.5) should be gone
        assert err_vs_original > 0.1, (
            f"Roundtrip matches original too well ({err_vs_original:.2e}); "
            "high-frequency content was not removed"
        )
        # The reconstructed field should be close to the bandlimited part
        assert err_vs_low < 1e-8, (
            f"Roundtrip does not match bandlimited part: err={err_vs_low:.2e}"
        )


# ---------------------------------------------------------------------------
# 7b  Global diagnostics agreement across grids
# ---------------------------------------------------------------------------

class TestCrossGridDiagnostics:
    """A gravity wave test on lat-lon and MPAS should give similar global means."""

    @pytest.fixture(scope="class")
    def latlon_result(self):
        """Run Gaussian bump on lat-lon 16x32."""
        n_lat, n_lon = 16, 32
        grid = create_latlon_grid(n_lat, n_lon)
        config = FVShallowWaterLatLonConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=True,
            fix_mass=True,
            fix_energy=False,
            use_polar_filter=True,
        )
        model = FVShallowWaterLatLonModel(grid, config, dt=60.0)

        lat2d = grid.lat2d
        lon2d = grid.lon2d
        H0 = constants.H_MEAN
        sigma = 15.0 * jnp.pi / 180.0  # 15 degrees
        # Gaussian bump centred at (0, 0)
        dist2 = lat2d ** 2 + lon2d ** 2
        # Handle periodicity in lon: use minimum of lon and 2pi-lon
        lon_dist = jnp.minimum(lon2d, 2 * jnp.pi - lon2d)
        dist2 = lat2d ** 2 + lon_dist ** 2
        h_data = H0 + 100.0 * jnp.exp(-dist2 / (2.0 * sigma ** 2))
        u_data = jnp.zeros_like(lat2d)
        v_data = jnp.zeros_like(lat2d)

        dims = ("lat", "lon")
        state = ShallowWaterState(
            h=Field(data=h_data, name="h", dims=dims, units="m"),
            u=Field(data=u_data, name="u", dims=dims, units="m/s"),
            v=Field(data=v_data, name="v", dims=dims, units="m/s"),
            h_s=Field(data=jnp.zeros_like(h_data), name="h_s", dims=dims, units="m"),
        )

        dt = 60.0
        for _ in range(30):
            state = model.step(state, dt)

        area = grid.area
        h = state.h.data
        mean_h = float(jnp.sum(h * area) / jnp.sum(area))
        # Total energy = 0.5*g*h^2 + 0.5*h*(u^2+v^2)
        g = constants.g
        ke = 0.5 * h * (state.u.data ** 2 + state.v.data ** 2)
        pe = 0.5 * g * h ** 2
        total_energy = float(jnp.sum((ke + pe) * area))

        return mean_h, total_energy

    @pytest.fixture(scope="class")
    def mpas_result(self):
        """Run Gaussian bump on MPAS level-2 mesh."""
        mesh = create_voronoi_mesh(2, lloyd_iterations=30)
        config = MPASShallowWaterConfig(
            fix_mass=True,
            fix_energy=False,
            time_integrator="rk4",
        )
        model = MPASShallowWaterModel(mesh, config)

        H0 = constants.H_MEAN
        sigma = 15.0 * jnp.pi / 180.0
        lat = mesh.latCell
        lon = mesh.lonCell
        lon_dist = jnp.minimum(lon, 2 * jnp.pi - lon)
        dist2 = lat ** 2 + lon_dist ** 2
        h_data = H0 + 100.0 * jnp.exp(-dist2 / (2.0 * sigma ** 2))

        # Zero initial velocity
        u_edge = jnp.zeros(mesh.nEdges)

        state = MPASShallowWaterState(
            h=Field(data=h_data, name="h", dims=("nCells",), units="m"),
            u=Field(data=u_edge, name="u", dims=("nEdges",), units="m/s",
                    staggering="edge"),
            h_s=Field(data=jnp.zeros_like(h_data), name="h_s",
                      dims=("nCells",), units="m"),
        )

        dt = 60.0
        for _ in range(30):
            state = model.step(state, dt)

        area = mesh.areaCell
        h = state.h.data
        mean_h = float(jnp.sum(h * area) / jnp.sum(area))

        # KE from edge-normal velocity (TRiSK: KE_cell = 0.5 * sum of
        # edge contributions). Use the operator for consistency.
        from legoesm.core.operators_voronoi import kinetic_energy_cell
        ke_cell = kinetic_energy_cell(state.u.data, mesh)
        g = constants.g
        ke = 0.5 * h * ke_cell  # per-cell KE weighted by h (not exactly
        # same definition as lat-lon, but for zero-velocity start they match)
        # Actually: KE = h * ke_cell where ke_cell = 0.5*|u|^2
        # So total KE = sum(ke_cell * h * area). PE = 0.5*g*h^2.
        total_ke = float(jnp.sum(ke_cell * h * area))
        total_pe = float(jnp.sum(0.5 * g * h ** 2 * area))
        total_energy = total_ke + total_pe

        return mean_h, total_energy

    def test_mean_h_agreement(self, latlon_result, mpas_result):
        """Global mean h should agree within 1%."""
        mean_ll, _ = latlon_result
        mean_mpas, _ = mpas_result

        rel_diff = abs(mean_ll - mean_mpas) / abs(mean_ll)
        assert rel_diff < 0.01, (
            f"Mean h disagrees: lat-lon={mean_ll:.2f}, MPAS={mean_mpas:.2f}, "
            f"rel_diff={rel_diff:.4e}"
        )

    def test_total_energy_agreement(self, latlon_result, mpas_result):
        """Total energy should agree within 10%."""
        _, energy_ll = latlon_result
        _, energy_mpas = mpas_result

        rel_diff = abs(energy_ll - energy_mpas) / abs(energy_ll)
        assert rel_diff < 0.10, (
            f"Total energy disagrees: lat-lon={energy_ll:.4e}, "
            f"MPAS={energy_mpas:.4e}, rel_diff={rel_diff:.4e}"
        )


# ---------------------------------------------------------------------------
# 7c  Gravity wave cross-grid comparison (lat-lon vs cubed-sphere CDGrid)
# ---------------------------------------------------------------------------

class TestGravityWaveCrossGrid:
    """Run a Gaussian height bump on lat-lon and cubed-sphere shallow water
    models and compare global diagnostics after 30 steps."""

    @pytest.fixture(scope="class")
    def latlon_result(self):
        """Run Gaussian bump on lat-lon 16x32."""
        n_lat, n_lon = 16, 32
        grid = create_latlon_grid(n_lat, n_lon)
        config = FVShallowWaterLatLonConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=True,
            fix_mass=True,
            fix_energy=False,
            use_polar_filter=True,
        )
        model = FVShallowWaterLatLonModel(grid, config, dt=60.0)

        lat2d = grid.lat2d
        lon2d = grid.lon2d
        H0 = constants.H_MEAN
        sigma = 15.0 * jnp.pi / 180.0
        lon_dist = jnp.minimum(lon2d, 2 * jnp.pi - lon2d)
        dist2 = lat2d ** 2 + lon_dist ** 2
        h_data = H0 + 100.0 * jnp.exp(-dist2 / (2.0 * sigma ** 2))
        u_data = jnp.zeros_like(lat2d)
        v_data = jnp.zeros_like(lat2d)

        dims = ("lat", "lon")
        state = ShallowWaterState(
            h=Field(data=h_data, name="h", dims=dims, units="m"),
            u=Field(data=u_data, name="u", dims=dims, units="m/s"),
            v=Field(data=v_data, name="v", dims=dims, units="m/s"),
            h_s=Field(data=jnp.zeros_like(h_data), name="h_s",
                      dims=dims, units="m"),
        )

        dt = 60.0
        for _ in range(30):
            state = model.step(state, dt)

        area = grid.area
        h = state.h.data
        mean_h = float(jnp.sum(h * area) / jnp.sum(area))
        max_u = float(jnp.max(jnp.abs(state.u.data)))

        return mean_h, max_u

    @pytest.fixture(scope="class")
    def cdgrid_result(self):
        """Run Gaussian bump on cubed-sphere C8 (CDGrid)."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
            CDGridShallowWaterModel,
            CDGridShallowWaterConfig,
            CDGridShallowWaterState,
        )

        grid_base = create_cubed_sphere(8)
        cdgrid = create_cubed_sphere_cdgrid(grid_base)

        H0 = constants.H_MEAN
        sigma = 15.0 * jnp.pi / 180.0
        lat_c = grid_base.lat
        lon_c = grid_base.lon
        lon_dist = jnp.minimum(lon_c, 2 * jnp.pi - lon_c)
        dist2 = lat_c ** 2 + lon_dist ** 2
        h_init = H0 + 100.0 * jnp.exp(-dist2 / (2.0 * sigma ** 2))

        # Zero initial D-grid winds
        n = grid_base.n
        u_d_init = jnp.zeros((6, n + 1, n + 1))
        v_d_init = jnp.zeros((6, n + 1, n + 1))
        h_s = jnp.zeros_like(h_init)

        state = CDGridShallowWaterState(
            h=h_init, u_d=u_d_init, v_d=v_d_init, h_s=h_s,
        )

        config = CDGridShallowWaterConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=True,
        )
        model = CDGridShallowWaterModel(grid_base, config)

        dt = 60.0
        for _ in range(30):
            state = model.step(state, dt)

        area = grid_base.area
        h = state.h
        mean_h = float(jnp.sum(h * area) / jnp.sum(area))

        # Estimate max velocity from D-grid winds
        max_u = float(jnp.max(jnp.sqrt(state.u_d ** 2 + state.v_d ** 2)))

        return mean_h, max_u

    def test_mean_h_agreement(self, latlon_result, cdgrid_result):
        """Global mean h should agree within 1%."""
        mean_ll, _ = latlon_result
        mean_cs, _ = cdgrid_result

        rel_diff = abs(mean_ll - mean_cs) / abs(mean_ll)
        assert rel_diff < 0.01, (
            f"Mean h disagrees: lat-lon={mean_ll:.2f}, CS={mean_cs:.2f}, "
            f"rel_diff={rel_diff:.4e}"
        )

    def test_max_velocity_same_order(self, latlon_result, cdgrid_result):
        """Max|u| should be within a factor of 3."""
        _, max_u_ll = latlon_result
        _, max_u_cs = cdgrid_result

        ratio = max(max_u_ll, max_u_cs) / max(min(max_u_ll, max_u_cs), 1e-30)
        assert ratio < 3.0, (
            f"Max velocity ratio too large: lat-lon={max_u_ll:.4e}, "
            f"CS={max_u_cs:.4e}, ratio={ratio:.2f}"
        )


# ---------------------------------------------------------------------------
# 7d  Spectral roundtrip exact test
# ---------------------------------------------------------------------------

class TestSpectralRoundtripExact:
    """Create Y_3^2 in spectral space, synthesize and re-analyze.

    For a single spectral coefficient, synthesis -> analysis should
    recover the exact coefficient to machine precision.
    """

    def test_y32_roundtrip(self):
        from legoesm.grids.gaussian import _sh_idx

        grid = create_gaussian_grid(10)
        n_sh = grid.n_sh
        idx = _sh_idx(3, 2)

        # Create spectral array with only Y_3^2 coefficient = 1.0
        coeffs = jnp.zeros(n_sh, dtype=jnp.complex128)
        coeffs = coeffs.at[idx].set(1.0 + 0.0j)

        # Roundtrip: synthesis -> analysis
        field = sh_synthesis(grid, coeffs)
        coeffs_back = sh_analysis(grid, field)

        # The recovered (3,2) coefficient should match to < 1e-12
        recovered = coeffs_back[idx]
        err_target = abs(complex(recovered) - (1.0 + 0.0j))
        assert err_target < 1e-12, (
            f"Recovered Y_3^2 coefficient error = {err_target:.2e}, "
            f"expected < 1e-12. Got {complex(recovered)}"
        )

        # All other coefficients should be < 1e-12
        other_coeffs = coeffs_back.at[idx].set(0.0)
        max_other = float(jnp.max(jnp.abs(other_coeffs)))
        assert max_other < 1e-12, (
            f"Max non-target coefficient = {max_other:.2e}, expected < 1e-12"
        )


# ---------------------------------------------------------------------------
# 7e  Balanced jet cross-discretization (lat-lon FV vs MPAS)
# ---------------------------------------------------------------------------

class TestBalancedJetCrossGrid:
    """Williamson TC2 (balanced zonal flow) on lat-lon FV and MPAS.

    Both grids should maintain the geostrophic balance for 50 steps
    and produce similar global mean h.
    """

    @pytest.fixture(scope="class")
    def latlon_tc2_result(self):
        """Run Williamson TC2 on lat-lon 16x32 for 50 steps."""
        from tests.atmosphere.shallow_water.test_cases.williamson_latlon import (
            williamson_test2_latlon,
        )

        grid = create_latlon_grid(16, 32)
        config = FVShallowWaterLatLonConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=True,
            fix_mass=True,
            fix_energy=False,
            use_polar_filter=True,
        )
        model = FVShallowWaterLatLonModel(grid, config, dt=300.0)

        state = williamson_test2_latlon(grid)

        dt = 300.0
        for _ in range(50):
            state = model.step(state, dt)

        area = grid.area
        mean_h = float(jnp.sum(state.h.data * area) / jnp.sum(area))
        return mean_h

    @pytest.fixture(scope="class")
    def mpas_tc2_result(self):
        """Run Williamson TC2 on MPAS level-2 mesh for 50 steps."""
        from tests.atmosphere.shallow_water.test_cases.williamson_mpas import (
            williamson_test2_mpas,
        )

        mesh = create_voronoi_mesh(2, lloyd_iterations=30)
        config = MPASShallowWaterConfig(
            fix_mass=True,
            fix_energy=False,
            time_integrator="rk4",
        )
        model = MPASShallowWaterModel(mesh, config)

        state = williamson_test2_mpas(mesh)

        dt = 300.0
        for _ in range(50):
            state = model.step(state, dt)

        area = mesh.areaCell
        mean_h = float(jnp.sum(state.h.data * area) / jnp.sum(area))
        return mean_h

    def test_mean_h_agreement(self, latlon_tc2_result, mpas_tc2_result):
        """Global mean h should agree within 1%."""
        mean_ll = latlon_tc2_result
        mean_mpas = mpas_tc2_result

        rel_diff = abs(mean_ll - mean_mpas) / abs(mean_ll)
        assert rel_diff < 0.01, (
            f"TC2 mean h disagrees: lat-lon={mean_ll:.2f}, MPAS={mean_mpas:.2f}, "
            f"rel_diff={rel_diff:.4e}"
        )


# ---------------------------------------------------------------------------
# 7f  Spectral SW vs lat-lon gravity wave comparison
# ---------------------------------------------------------------------------

class TestSpectralVsLatLonGravityWave:
    """Run a Gaussian bump on spectral and lat-lon SW models.
    Compare global diagnostics after 30 steps."""

    @pytest.fixture(scope="class")
    def spectral_result(self):
        """Run Gaussian bump on spectral T10 shallow water."""
        from legoesm.grids.gaussian import create_gaussian_grid, sh_analysis, sh_synthesis
        from legoesm.atmosphere.dynamics.spectral_sw import (
            SpectralShallowWaterModel,
            SpectralSWConfig,
            SpectralSWState,
        )
        from legoesm.core.field import Field

        grid = create_gaussian_grid(10)
        g = constants.g
        H0 = constants.H_MEAN

        lat0, lon0 = 0.0, 0.0
        sigma = 15.0 * jnp.pi / 180.0
        lon_dist = jnp.minimum(grid.lon2d, 2 * jnp.pi - grid.lon2d)
        dist2 = grid.lat2d**2 + lon_dist**2
        h_grid = H0 + 100.0 * jnp.exp(-dist2 / (2.0 * sigma**2))

        phi_hat = sh_analysis(grid, g * h_grid)
        n_sh = grid.n_sh
        zeros = jnp.zeros(n_sh, dtype=jnp.complex128)

        state = SpectralSWState(
            vor_hat=Field(data=zeros, name="vor_hat", dims=("n_sh",), units="1/s"),
            div_hat=Field(data=zeros, name="div_hat", dims=("n_sh",), units="1/s"),
            phi_hat=Field(data=phi_hat, name="phi_hat", dims=("n_sh",), units="m2/s2"),
            phis_hat=Field(data=zeros, name="phis_hat", dims=("n_sh",), units="m2/s2"),
        )

        config = SpectralSWConfig(
            mean_depth=H0,
            hyperdiff_coeff=0.0,
            spectral_filter_order=0,
        )
        model = SpectralShallowWaterModel(grid, config)

        dt = 60.0
        for _ in range(30):
            state = model.step(state, dt)

        # Recover grid-space h
        phi_grid = sh_synthesis(grid, state.phi_hat.data)
        h = phi_grid / g
        weights = grid.weights
        n_lon = grid.n_lon
        # Area-weighted mean h using Gaussian quadrature
        mean_h = float(jnp.sum(jnp.sum(h, axis=1) * weights) / (n_lon * jnp.sum(weights)))

        return mean_h

    @pytest.fixture(scope="class")
    def latlon_gw_result(self):
        """Run Gaussian bump on lat-lon 16x32."""
        grid = create_latlon_grid(16, 32)
        config = FVShallowWaterLatLonConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=True,
            fix_mass=True,
            fix_energy=False,
            use_polar_filter=True,
        )
        model = FVShallowWaterLatLonModel(grid, config, dt=60.0)

        H0 = constants.H_MEAN
        sigma = 15.0 * jnp.pi / 180.0
        lon_dist = jnp.minimum(grid.lon2d, 2 * jnp.pi - grid.lon2d)
        dist2 = grid.lat2d**2 + lon_dist**2
        h_data = H0 + 100.0 * jnp.exp(-dist2 / (2.0 * sigma**2))

        dims = ("lat", "lon")
        state = ShallowWaterState(
            h=Field(data=h_data, name="h", dims=dims, units="m"),
            u=Field(data=jnp.zeros_like(h_data), name="u", dims=dims, units="m/s"),
            v=Field(data=jnp.zeros_like(h_data), name="v", dims=dims, units="m/s"),
            h_s=Field(data=jnp.zeros_like(h_data), name="h_s", dims=dims, units="m"),
        )

        dt = 60.0
        for _ in range(30):
            state = model.step(state, dt)

        area = grid.area
        mean_h = float(jnp.sum(state.h.data * area) / jnp.sum(area))
        return mean_h

    def test_mean_h_agreement(self, spectral_result, latlon_gw_result):
        """Global mean h should agree within 1%."""
        mean_sp = spectral_result
        mean_ll = latlon_gw_result

        rel_diff = abs(mean_sp - mean_ll) / abs(mean_ll)
        assert rel_diff < 0.01, (
            f"Mean h disagrees: spectral={mean_sp:.2f}, lat-lon={mean_ll:.2f}, "
            f"rel_diff={rel_diff:.4e}"
        )


# ---------------------------------------------------------------------------
# 7g  PE cross-discretization: lat-lon vs C-D grid (hydrostatic at rest)
# ---------------------------------------------------------------------------

class TestPECrossDiscretization:
    """Run an isothermal atmosphere at rest on both lat-lon and C-D grid PE.
    Both should preserve the resting state (near-zero winds and T drift).
    This tests that different discretizations agree on the simplest PE solution."""

    @pytest.fixture(scope="class")
    def latlon_pe_result(self):
        """Run PE at rest on lat-lon for 10 steps."""
        from legoesm.atmosphere.dynamics.primitive_eq_latlon import (
            LatLonPrimitiveEquationModel,
            LatLonPrimitiveEquationConfig,
        )
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.core.state import HydrostaticState

        grid = create_latlon_grid(8, 16)
        sigma = create_sigma_coordinate(5)
        T0, ps0, nlev = 250.0, 1e5, 5

        state = HydrostaticState(
            u=Field(data=jnp.zeros((8, 16, nlev)), name="u",
                    dims=("lat", "lon", "level"), units="m/s"),
            v=Field(data=jnp.zeros((8, 16, nlev)), name="v",
                    dims=("lat", "lon", "level"), units="m/s"),
            T=Field(data=jnp.full((8, 16, nlev), T0), name="T",
                    dims=("lat", "lon", "level"), units="K"),
            p_s=Field(data=jnp.full((8, 16), ps0), name="p_s",
                      dims=("lat", "lon"), units="Pa"),
            phis=Field(data=jnp.zeros((8, 16)), name="phis",
                       dims=("lat", "lon"), units="m^2/s^2"),
        )

        config = LatLonPrimitiveEquationConfig(
            hyperdiff_coeff=0.0, A_h=0.0,
            use_conservation_fixer=True, fix_mass=True,
            use_polar_filter=False, sponge_tau_sec=0.0,
        )
        model = LatLonPrimitiveEquationModel(grid, sigma, config, dt=60.0)

        for _ in range(10):
            state = model.step(state, 60.0)

        max_u = float(jnp.max(jnp.abs(state.u.data)))
        T_drift = float(jnp.max(jnp.abs(state.T.data - T0)))
        return max_u, T_drift

    @pytest.fixture(scope="class")
    def cdgrid_pe_result(self):
        """Run PE at rest on C-D grid for 10 steps."""
        from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel,
            CDGridPrimitiveEquationConfig,
        )
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.core.state import FV3HydrostaticState

        n, nlev = 4, 5
        T0, ps0 = 250.0, 1e5
        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        sigma = create_sigma_coordinate(nlev)

        d_shape = (6, n + 1, n + 1, nlev)
        c_shape = (6, n, n, nlev)
        s_shape = (6, n, n)

        state = FV3HydrostaticState(
            u_d=Field(data=jnp.zeros(d_shape), name="u_d",
                      dims=("face", "x", "y", "level"), units="m/s"),
            v_d=Field(data=jnp.zeros(d_shape), name="v_d",
                      dims=("face", "x", "y", "level"), units="m/s"),
            T=Field(data=jnp.full(c_shape, T0), name="T",
                    dims=("face", "x", "y", "level"), units="K"),
            p_s=Field(data=jnp.full(s_shape, ps0), name="p_s",
                      dims=("face", "x", "y"), units="Pa"),
            phis=Field(data=jnp.zeros(s_shape), name="phis",
                       dims=("face", "x", "y"), units="m^2/s^2"),
        )

        config = CDGridPrimitiveEquationConfig(
            hyperdiff_coeff=0.0, A_h=0.0,
            use_conservation_fixer=True, fix_mass=True,
            sponge_tau_sec=0.0,
        )
        model = CDGridPrimitiveEquationModel(grid, sigma, config)

        for _ in range(10):
            state = model.step(state, 60.0)

        max_u = float(jnp.max(jnp.abs(state.u_d.data)))
        T_drift = float(jnp.max(jnp.abs(state.T.data - T0)))
        return max_u, T_drift

    def test_both_preserve_rest_state(self, latlon_pe_result, cdgrid_pe_result):
        """Both discretizations should preserve the resting state."""
        max_u_ll, T_drift_ll = latlon_pe_result
        max_u_cs, T_drift_cs = cdgrid_pe_result

        # Both should have near-zero winds
        assert max_u_ll < 1e-3, f"Lat-lon PE: max|u| = {max_u_ll}"
        assert max_u_cs < 1e-3, f"C-D grid PE: max|u| = {max_u_cs}"

        # Both should have small T drift
        assert T_drift_ll < 0.1, f"Lat-lon PE: T drift = {T_drift_ll}"
        assert T_drift_cs < 0.1, f"C-D grid PE: T drift = {T_drift_cs}"

    def test_wind_magnitudes_similar(self, latlon_pe_result, cdgrid_pe_result):
        """Spurious wind magnitudes should be of similar order."""
        max_u_ll, _ = latlon_pe_result
        max_u_cs, _ = cdgrid_pe_result

        # Both should be very small; check they're within an order of magnitude
        if max_u_ll > 1e-15 and max_u_cs > 1e-15:
            ratio = max(max_u_ll, max_u_cs) / min(max_u_ll, max_u_cs)
            assert ratio < 100, (
                f"Wind magnitude ratio = {ratio:.1f}, expected < 100. "
                f"lat-lon={max_u_ll:.3e}, CS={max_u_cs:.3e}"
            )
