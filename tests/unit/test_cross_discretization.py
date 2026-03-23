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
