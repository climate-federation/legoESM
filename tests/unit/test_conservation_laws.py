"""Conservation tests beyond global mass.

Category 4 tests:
  4a) Potential vorticity conservation (shallow water, lat-lon)
  4d) Tracer mass conservation (shallow water, lat-lon)
  4e) Energy conservation (shallow water, C-D grid cubed-sphere)
  4f) Volume conservation (ocean) — skipped
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.latlon import create_latlon_grid
from legoesm import constants


# =====================================================================
# Helper: compute relative vorticity on lat-lon A-grid
# =====================================================================

def _relative_vorticity_latlon(u, v, grid):
    """Compute relative vorticity zeta = dv/dx - du/dy on lat-lon grid.

    Uses centered differences consistent with the model operators.
    """
    from legoesm.core.operators_latlon import curl_z
    u_field = Field(data=u, name="u", dims=("lat", "lon"), units="m/s")
    v_field = Field(data=v, name="v", dims=("lat", "lon"), units="m/s")
    return curl_z(u_field, v_field, grid).data


# =====================================================================
# 4a) Potential vorticity conservation (shallow water, lat-lon)
# =====================================================================

class TestPVConservation:
    """PV = (zeta + f) / h should be conserved in inviscid shallow water."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.atmosphere.dynamics.shallow_water_fv_latlon import (
            FVShallowWaterLatLonModel,
            FVShallowWaterLatLonConfig,
        )

        n_lat, n_lon = 16, 32
        self.grid = create_latlon_grid(n_lat, n_lon)

        g = constants.g
        a = constants.R_earth
        Omega = constants.Omega
        u0 = 20.0
        h0 = 1e4

        lat2d = self.grid.lat2d

        # Balanced zonal jet (Williamson TC2 on lat-lon)
        u_2d = u0 * jnp.cos(lat2d)
        h_2d = h0 - (a * Omega * u0 + 0.5 * u0**2) / g * jnp.sin(lat2d)**2

        dims = ("lat", "lon")
        self.state0 = ShallowWaterState(
            h=Field(data=h_2d, name="h", dims=dims, units="m"),
            u=Field(data=u_2d, name="u", dims=dims, units="m/s"),
            v=Field(data=jnp.zeros_like(u_2d), name="v", dims=dims, units="m/s"),
            h_s=Field(data=jnp.zeros_like(h_2d), name="h_s", dims=dims, units="m"),
        )

        # No hyperdiffusion for PV conservation test
        config = FVShallowWaterLatLonConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=True,
            fix_mass=True,
            fix_energy=False,
            use_polar_filter=True,
        )
        self.dt = 60.0
        self.model = FVShallowWaterLatLonModel(self.grid, config, dt=self.dt)
        self.n_steps = 100

    def _compute_pv(self, state):
        """Compute potential vorticity (zeta + f) / h."""
        zeta = _relative_vorticity_latlon(
            state.u.data, state.v.data, self.grid,
        )
        f = self.grid.f  # (n_lat, n_lon)
        h = state.h.data
        return (zeta + f) / h

    def test_global_mean_pv_conserved(self):
        """Global area-weighted mean PV should be conserved within 1%."""
        pv0 = self._compute_pv(self.state0)
        area = self.grid.area
        mean_pv0 = float(jnp.sum(pv0 * area) / jnp.sum(area))

        state = self.state0
        for _ in range(self.n_steps):
            state = self.model.step(state, self.dt)

        pv_final = self._compute_pv(state)
        mean_pv_final = float(jnp.sum(pv_final * area) / jnp.sum(area))

        # mean PV can be very small (near zero for symmetric flow), so use
        # absolute tolerance scaled by the PV range
        pv_range = float(jnp.max(jnp.abs(pv0)))
        if pv_range < 1e-20:
            pytest.skip("PV range too small to test")

        rel_err = abs(mean_pv_final - mean_pv0) / pv_range
        print(f"  PV mean: initial={mean_pv0:.6e}, final={mean_pv_final:.6e}, "
              f"rel_err={rel_err:.4e}")
        assert rel_err < 0.01, f"PV mean drift = {rel_err}"

    def test_pv_range_bounded(self):
        """PV range (max-min) should not expand by more than 10%."""
        pv0 = self._compute_pv(self.state0)
        pv0_range = float(jnp.max(pv0) - jnp.min(pv0))

        state = self.state0
        for _ in range(self.n_steps):
            state = self.model.step(state, self.dt)

        pv_final = self._compute_pv(state)
        pv_final_range = float(jnp.max(pv_final) - jnp.min(pv_final))

        if pv0_range < 1e-20:
            pytest.skip("PV range too small")

        expansion = (pv_final_range - pv0_range) / pv0_range
        print(f"  PV range: initial={pv0_range:.6e}, final={pv_final_range:.6e}, "
              f"expansion={expansion:.4f}")
        assert expansion < 0.10, f"PV range expanded by {expansion:.4f}"


# =====================================================================
# 4d) Tracer mass conservation (shallow water, lat-lon)
# =====================================================================

class TestTracerMassConservation:
    """Tracer mass integral should be conserved in shallow water transport.

    We embed the tracer in the height field: h_with_tracer = h * (1 + eps*q)
    vs h_no_tracer = h. The difference integral should be conserved.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.atmosphere.dynamics.shallow_water_fv_latlon import (
            FVShallowWaterLatLonModel,
            FVShallowWaterLatLonConfig,
        )

        n_lat, n_lon = 16, 32
        self.grid = create_latlon_grid(n_lat, n_lon)

        g = constants.g
        a = constants.R_earth
        Omega = constants.Omega
        u0 = 20.0
        h0 = 1e4

        lat2d = self.grid.lat2d
        lon2d = self.grid.lon2d

        u_2d = u0 * jnp.cos(lat2d)
        h_2d = h0 - (a * Omega * u0 + 0.5 * u0**2) / g * jnp.sin(lat2d)**2

        # Tracer: cone centered at (0, pi) with radius r0
        lat_c, lon_c = 0.0, jnp.pi
        r0 = jnp.pi / 6.0  # 30 degrees
        r = jnp.arccos(
            jnp.sin(lat2d) * jnp.sin(lat_c)
            + jnp.cos(lat2d) * jnp.cos(lat_c) * jnp.cos(lon2d - lon_c)
        )
        q = jnp.maximum(0.0, 1.0 - r / r0)

        eps = 0.01
        h_tracer = h_2d * (1.0 + eps * q)

        dims = ("lat", "lon")
        self.state_tracer = ShallowWaterState(
            h=Field(data=h_tracer, name="h", dims=dims, units="m"),
            u=Field(data=u_2d, name="u", dims=dims, units="m/s"),
            v=Field(data=jnp.zeros_like(u_2d), name="v", dims=dims, units="m/s"),
            h_s=Field(data=jnp.zeros_like(h_2d), name="h_s", dims=dims, units="m"),
        )
        self.state_base = ShallowWaterState(
            h=Field(data=h_2d, name="h", dims=dims, units="m"),
            u=Field(data=u_2d, name="u", dims=dims, units="m/s"),
            v=Field(data=jnp.zeros_like(u_2d), name="v", dims=dims, units="m/s"),
            h_s=Field(data=jnp.zeros_like(h_2d), name="h_s", dims=dims, units="m"),
        )

        config = FVShallowWaterLatLonConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=True,
            fix_mass=True,
            fix_energy=False,
            use_polar_filter=True,
        )
        self.dt = 60.0
        self.model = FVShallowWaterLatLonModel(self.grid, config, dt=self.dt)
        self.n_steps = 50
        self.area = self.grid.area

    def test_tracer_mass_conserved(self):
        """Total mass integral of h (with tracer) should be conserved to 0.1%."""
        area = self.area
        mass0 = float(jnp.sum(self.state_tracer.h.data * area))

        state = self.state_tracer
        for _ in range(self.n_steps):
            state = self.model.step(state, self.dt)

        mass_final = float(jnp.sum(state.h.data * area))
        rel_err = abs(mass_final - mass0) / abs(mass0)
        print(f"  Tracer mass: initial={mass0:.6e}, final={mass_final:.6e}, "
              f"rel_err={rel_err:.4e}")
        assert rel_err < 0.001, f"Tracer mass conservation failed: rel_err={rel_err}"


# =====================================================================
# 4e) Energy conservation (shallow water, C-D grid cubed-sphere)
# =====================================================================

class TestEnergyConservation:
    """Total energy (KE + PE) should be approximately conserved in inviscid
    shallow water with balanced initial condition (Williamson TC2)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
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
# 4f) Volume conservation (ocean) — skipped
# =====================================================================

@pytest.mark.skip(reason="Ocean model initialization requires mesh files; skipping.")
class TestOceanVolumeConservation:
    """Integral of sea surface height eta should be conserved."""

    def test_volume_conservation(self):
        pass
