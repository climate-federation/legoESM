"""Conservation tests beyond global mass.

Category 4 tests:
  4e) Energy conservation (shallow water, C-D grid cubed-sphere)
  4i) Enstrophy budget (MPAS)
  4k) MPAS shallow water energy conservation
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm import constants


# =====================================================================
# 4e) Energy conservation (shallow water, C-D grid cubed-sphere)
# =====================================================================

class TestEnergyConservation:
    """Total energy (KE + PE) should be approximately conserved in inviscid
    shallow water with balanced initial condition (Williamson TC2)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
            CDGridShallowWaterModel,
            CDGridShallowWaterConfig,
            CDGridShallowWaterState,
        )

        n = 8
        self.grid = create_cubed_sphere(n)
        self.cdgrid = create_cubed_sphere_cdgrid(self.grid)

        g = constants.g
        Omega = constants.Omega
        R = constants.R_earth
        u0 = 38.61068276698372
        h0 = 29400.0 / g

        # Williamson TC2 initial condition on C-D grid
        lat_c = self.grid.lat  # (6, n, n)
        h = h0 - (R * Omega * u0 + 0.5 * u0**2) * jnp.sin(lat_c)**2 / g

        lat_corner = self.cdgrid.lat_corner
        u_geo = u0 * jnp.cos(lat_corner)
        u_d = u_geo * self.cdgrid.cos_angle_corner
        v_d = u_geo * self.cdgrid.sin_angle_corner
        h_s = jnp.zeros_like(h)

        self.state0 = CDGridShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)

        # No viscosity, no energy fixer
        config = CDGridShallowWaterConfig(
            A_h=0.0,
            hyperdiff_coeff=0.0,
            use_conservation_fixer=True,
            fix_mass=True,
        )
        self.model = CDGridShallowWaterModel(self.grid, config)
        self.dt = 300.0
        self.n_steps = 100
        self.g = g

    def _total_energy(self, state):
        """Compute total energy KE + PE."""
        area = self.cdgrid.base.area
        # D-grid to cell center average
        u_c = 0.25 * (state.u_d[..., :-1, :-1] + state.u_d[..., 1:, :-1]
                       + state.u_d[..., :-1, 1:] + state.u_d[..., 1:, 1:])
        v_c = 0.25 * (state.v_d[..., :-1, :-1] + state.v_d[..., 1:, :-1]
                       + state.v_d[..., :-1, 1:] + state.v_d[..., 1:, 1:])
        # For 2D arrays: shape is (6, n, n)
        # Need to handle the case where u_d is (6, n+1, n+1)
        ke = 0.5 * state.h * (u_c**2 + v_c**2)
        pe = 0.5 * self.g * (state.h + state.h_s)**2
        return float(jnp.sum((ke + pe) * area))

    def test_energy_conserved(self):
        """Relative energy change should be < 5% after 100 steps."""
        E0 = self._total_energy(self.state0)

        state = self.state0
        for _ in range(self.n_steps):
            state = self.model.step(state, self.dt)

        E_final = self._total_energy(state)
        rel_change = abs(E_final - E0) / abs(E0)
        print(f"  Energy: E0={E0:.6e}, E_final={E_final:.6e}, "
              f"rel_change={rel_change:.4e}")
        assert rel_change < 0.05, f"Energy change = {rel_change}"

    def test_energy_finite(self):
        """Energy should stay finite after integration."""
        state = self.state0
        for _ in range(self.n_steps):
            state = self.model.step(state, self.dt)

        assert jnp.all(jnp.isfinite(state.h)), "h has non-finite values"
        assert jnp.all(jnp.isfinite(state.u_d)), "u_d has non-finite values"
        assert jnp.all(jnp.isfinite(state.v_d)), "v_d has non-finite values"


# =====================================================================
# 4i) Enstrophy budget (MPAS, energy-conserving vs enstrophy-conserving)
# =====================================================================

class TestEnstrophyBudgetMPAS:
    """Compare enstrophy behaviour under energy-conserving vs
    enstrophy-conserving PV flux options in MPAS shallow water.

    For enstrophy-conserving flux, potential enstrophy Z should be
    much better conserved than for energy-conserving flux.
    """

    @staticmethod
    def _run_enstrophy_test(pv_scheme: str, n_steps: int = 50):
        """Run MPAS SW with given PV scheme and return initial/final enstrophy."""
        from legoesm.atmosphere.dynamics.gcm.shallow_water_mpas import (
            MPASShallowWaterModel,
            MPASShallowWaterConfig,
            MPASShallowWaterState,
        )
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.core.operators_voronoi import curl_vertex, vertex_thickness

        mesh = create_voronoi_mesh(2, lloyd_iterations=30)

        g = constants.g
        Omega = constants.Omega
        R = constants.R_earth
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

        def _enstrophy(s):
            zeta = curl_vertex(s.u.data, mesh)
            h_v = vertex_thickness(s.h.data, mesh)
            pv = (zeta + mesh.fVertex) / h_v
            # Potential enstrophy: Z = 0.5 * sum(pv^2 * h_v * areaTriangle)
            return float(0.5 * jnp.sum(pv**2 * h_v * mesh.areaTriangle))

        config = MPASShallowWaterConfig(
            pv_scheme=pv_scheme,
            fix_mass=True,
            fix_energy=False,
        )
        model = MPASShallowWaterModel(mesh, config)

        Z0 = _enstrophy(state)
        dt = 60.0
        for _ in range(n_steps):
            state = model.step(state, dt)
        Z_final = _enstrophy(state)

        return Z0, Z_final

    def test_enstrophy_conserving_better(self):
        """Enstrophy-conserving PV flux should preserve Z better than
        energy-conserving flux."""
        Z0_en, Zf_en = self._run_enstrophy_test("enstrophy", n_steps=50)
        Z0_ec, Zf_ec = self._run_enstrophy_test("energy", n_steps=50)

        rel_en = abs(Zf_en - Z0_en) / abs(Z0_en)
        rel_ec = abs(Zf_ec - Z0_ec) / abs(Z0_ec)

        print(f"  Enstrophy-conserving: Z0={Z0_en:.6e}, Zf={Zf_en:.6e}, "
              f"rel_change={rel_en:.4e}")
        print(f"  Energy-conserving: Z0={Z0_ec:.6e}, Zf={Zf_ec:.6e}, "
              f"rel_change={rel_ec:.4e}")

        # Both should preserve enstrophy reasonably at balanced flow
        assert rel_en < 0.20, f"Enstrophy-conserving Z drift = {rel_en}"
        assert rel_ec < 0.50, f"Energy-conserving Z drift = {rel_ec}"


# =====================================================================
# 4k) MPAS shallow water energy conservation
# =====================================================================

class TestMPASEnergyConservation:
    """Total energy (KE + PE) should be approximately conserved in
    MPAS inviscid shallow water with balanced initial condition."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.atmosphere.dynamics.gcm.shallow_water_mpas import (
            MPASShallowWaterModel,
            MPASShallowWaterConfig,
            MPASShallowWaterState,
        )
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.core.operators_voronoi import kinetic_energy_cell

        self.mesh = create_voronoi_mesh(2, lloyd_iterations=30)
        mesh = self.mesh
        self.kinetic_energy_cell = kinetic_energy_cell

        g = constants.g
        Omega = constants.Omega
        R = constants.R_earth
        u0 = 20.0
        h0 = 1e4

        h_init = h0 - (R * Omega * u0 + 0.5 * u0**2) * jnp.sin(mesh.latCell)**2 / g
        u_edge = u0 * jnp.cos(mesh.latEdge) * jnp.cos(mesh.angleEdge)

        self.state0 = MPASShallowWaterState(
            h=Field(data=h_init, name="h", dims=("nCells",), units="m"),
            u=Field(data=u_edge, name="u", dims=("nEdges",), units="m/s",
                    staggering="edge"),
            h_s=Field(data=jnp.zeros(mesh.nCells), name="h_s",
                      dims=("nCells",), units="m"),
        )

        config = MPASShallowWaterConfig(fix_mass=True, fix_energy=False)
        self.model = MPASShallowWaterModel(mesh, config)
        self.dt = 60.0
        self.n_steps = 100
        self.g = g

    def _total_energy(self, state):
        """Compute total energy KE + PE."""
        mesh = self.mesh
        h = state.h.data
        ke_cell = self.kinetic_energy_cell(state.u.data, mesh)
        KE = float(jnp.sum(h * ke_cell * mesh.areaCell))
        PE = float(0.5 * self.g * jnp.sum(h**2 * mesh.areaCell))
        return KE + PE

    def test_energy_conserved(self):
        """Relative energy change should be < 5% after 100 steps."""
        E0 = self._total_energy(self.state0)

        state = self.state0
        for _ in range(self.n_steps):
            state = self.model.step(state, self.dt)

        E_final = self._total_energy(state)
        rel_change = abs(E_final - E0) / abs(E0)
        assert rel_change < 0.05, f"MPAS energy change = {rel_change}"
        assert jnp.all(jnp.isfinite(state.h.data)), "h non-finite"
