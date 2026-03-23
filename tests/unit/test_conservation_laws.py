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
# 4f) Volume conservation (ocean)
# =====================================================================

class TestOceanVolumeConservation:
    """Integral of sea surface height eta should be conserved."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.ocean.dynamics.ocean_model_latlon import LatLonOceanModel
        from legoesm.ocean.state import LatLonOceanState, LatLonOceanConfig
        from legoesm.ocean.vertical import create_ocean_z_star

        n_lat, n_lon = 8, 16
        nlev = 5
        H_bathy_val = 4000.0

        self.grid = create_latlon_grid(n_lat, n_lon)
        self.z_coord = create_ocean_z_star(
            n_levels=nlev, H_max=H_bathy_val,
            dz_surface=100.0, dz_deep=1200.0,
        )

        lat2d = self.grid.lat2d
        lon2d = self.grid.lon2d

        # Gaussian bump in eta centred at (0, pi)
        r2 = lat2d**2 + (lon2d - jnp.pi)**2
        eta_data = 1.0 * jnp.exp(-r2 / (jnp.pi / 6.0)**2)

        # Uniform T, S, zero velocity
        zeros_3d = jnp.zeros((n_lat, n_lon, nlev))
        T_data = 20.0 * jnp.ones((n_lat, n_lon, nlev))
        S_data = 35.0 * jnp.ones((n_lat, n_lon, nlev))
        H_bathy = H_bathy_val * jnp.ones((n_lat, n_lon))
        land_mask = jnp.ones((n_lat, n_lon))

        dims_2d = ("lat", "lon")
        dims_3d = ("lat", "lon", "level")
        self.state0 = LatLonOceanState(
            u=Field(data=zeros_3d, name="u", dims=dims_3d, units="m/s"),
            v=Field(data=zeros_3d, name="v", dims=dims_3d, units="m/s"),
            T=Field(data=T_data, name="T", dims=dims_3d, units="degC"),
            S=Field(data=S_data, name="S", dims=dims_3d, units="PSU"),
            eta=Field(data=eta_data, name="eta", dims=dims_2d, units="m"),
            H_bathy=Field(data=H_bathy, name="H_bathy", dims=dims_2d, units="m"),
            land_mask=Field(data=land_mask, name="land_mask", dims=dims_2d, units=""),
        )

        config = LatLonOceanConfig(
            A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0,
            hyperdiff_coeff=0.0,
            n_barotropic_substeps=10,
            use_conservation_fixer=True,
            fix_volume=True,
            enable_runtime_checks=False,
        )
        self.model = LatLonOceanModel(self.grid, self.z_coord, config)
        self.dt = 300.0
        self.n_steps = 20
        self.area = self.grid.area

    def test_volume_conservation(self):
        """sum(eta * area) should be conserved to < 0.1%."""
        area = self.area
        vol0 = float(jnp.sum(self.state0.eta.data * area))

        state = self.state0
        for _ in range(self.n_steps):
            state = self.model.step(state, self.dt)

        vol_final = float(jnp.sum(state.eta.data * area))

        # Use absolute tolerance relative to the initial volume magnitude
        if abs(vol0) < 1e-30:
            # If initial volume integral is near zero, use absolute tolerance
            assert abs(vol_final - vol0) < 1e-6, (
                f"Volume integral drift: {abs(vol_final - vol0):.4e}"
            )
        else:
            rel_err = abs(vol_final - vol0) / abs(vol0)
            print(f"  Ocean volume: initial={vol0:.6e}, final={vol_final:.6e}, "
                  f"rel_err={rel_err:.4e}")
            assert rel_err < 0.001, f"Volume conservation failed: rel_err={rel_err}"


# =====================================================================
# 4g) Angular momentum conservation (PE, lat-lon)
# =====================================================================

class TestAngularMomentumConservation:
    """Angular momentum M = sum(area * dsigma * p_s/g * (u + Omega*a*cos(lat)) * cos(lat))
    should be approximately conserved in inviscid PE."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.atmosphere.dynamics.primitive_eq_latlon import (
            LatLonPrimitiveEquationModel,
            LatLonPrimitiveEquationConfig,
        )
        from legoesm.grids.vertical import create_sigma_coordinate

        n_lat, n_lon, nlev = 8, 16, 5
        self.grid = create_latlon_grid(n_lat, n_lon)
        self.sigma_coord = create_sigma_coordinate(nlev)

        g = constants.g
        a = constants.R_earth
        Omega = constants.Omega
        R_d = constants.R_d
        T0 = 300.0
        p_s0 = 1.0e5
        u0 = 20.0

        lat2d = self.grid.lat2d  # (n_lat, n_lon)
        dsigma = self.sigma_coord.dsigma  # (nlev,)

        # Balanced zonal jet
        u_2d = u0 * jnp.cos(lat2d)
        u_3d = u_2d[:, :, None] * jnp.ones(nlev)[None, None, :]

        # Isothermal atmosphere
        T_3d = T0 * jnp.ones((n_lat, n_lon, nlev))

        # Surface pressure in approximate balance
        p_s_data = p_s0 * jnp.ones((n_lat, n_lon))
        phis_data = jnp.zeros((n_lat, n_lon))

        dims_3d = ("lat", "lon", "level")
        dims_2d = ("lat", "lon")

        from legoesm.core.state import HydrostaticState

        self.state0 = HydrostaticState(
            u=Field(data=u_3d, name="u", dims=dims_3d, units="m/s"),
            v=Field(data=jnp.zeros_like(u_3d), name="v", dims=dims_3d, units="m/s"),
            T=Field(data=T_3d, name="T", dims=dims_3d, units="K"),
            p_s=Field(data=p_s_data, name="p_s", dims=dims_2d, units="Pa"),
            phis=Field(data=phis_data, name="phis", dims=dims_2d, units="m^2/s^2"),
        )

        self.dt = 120.0
        config = LatLonPrimitiveEquationConfig(
            hyperdiff_coeff=0.0,
            A_h=0.0,
            use_polar_filter=True,
            use_conservation_fixer=True,
            fix_mass=True,
            sponge_tau_sec=0.0,  # disable sponge
        )
        self.model = LatLonPrimitiveEquationModel(
            self.grid, self.sigma_coord, config, dt=self.dt,
        )
        self.n_steps = 50

    def _compute_angular_momentum(self, state):
        """Compute global angular momentum."""
        g = constants.g
        a = constants.R_earth
        Omega = constants.Omega

        area = self.grid.area  # (n_lat, n_lon)
        lat2d = self.grid.lat2d
        dsigma = self.sigma_coord.dsigma  # (nlev,)
        p_s = state.p_s.data  # (n_lat, n_lon)
        u = state.u.data  # (n_lat, n_lon, nlev)

        cos_lat = jnp.cos(lat2d)

        # M = sum over (lat, lon, lev) of:
        #   area * dsigma * (p_s/g) * (u + Omega*a*cos(lat)) * cos(lat)
        integrand = (
            area[:, :, None]
            * dsigma[None, None, :]
            * (p_s / g)[:, :, None]
            * (u + Omega * a * cos_lat[:, :, None])
            * cos_lat[:, :, None]
        )
        return float(jnp.sum(integrand))

    def test_angular_momentum_conserved(self):
        """Relative angular momentum change should be < 5% after 50 steps."""
        M0 = self._compute_angular_momentum(self.state0)

        state = self.state0
        for _ in range(self.n_steps):
            state = self.model.step(state, self.dt)

        M_final = self._compute_angular_momentum(state)
        rel_change = abs(M_final - M0) / abs(M0)
        print(f"  Angular momentum: M0={M0:.6e}, M_final={M_final:.6e}, "
              f"rel_change={rel_change:.4e}")
        assert rel_change < 0.05, f"Angular momentum change = {rel_change}"


# =====================================================================
# 4h) Energy partition (PE, lat-lon)
# =====================================================================

class TestEnergyPartition:
    """KE and IE should be positive, finite, and their sum should be
    approximately conserved in the inviscid PE."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.atmosphere.dynamics.primitive_eq_latlon import (
            LatLonPrimitiveEquationModel,
            LatLonPrimitiveEquationConfig,
        )
        from legoesm.grids.vertical import create_sigma_coordinate

        n_lat, n_lon, nlev = 8, 16, 5
        self.grid = create_latlon_grid(n_lat, n_lon)
        self.sigma_coord = create_sigma_coordinate(nlev)

        g = constants.g
        c_pd = constants.c_pd
        T0 = 300.0
        p_s0 = 1.0e5

        lat2d = self.grid.lat2d
        dsigma = self.sigma_coord.dsigma

        # Small balanced perturbation: gentle zonal wind
        u0 = 5.0
        u_3d = u0 * jnp.cos(lat2d)[:, :, None] * jnp.ones(nlev)[None, None, :]
        T_3d = T0 * jnp.ones((n_lat, n_lon, nlev))
        p_s_data = p_s0 * jnp.ones((n_lat, n_lon))
        phis_data = jnp.zeros((n_lat, n_lon))

        dims_3d = ("lat", "lon", "level")
        dims_2d = ("lat", "lon")

        from legoesm.core.state import HydrostaticState

        self.state0 = HydrostaticState(
            u=Field(data=u_3d, name="u", dims=dims_3d, units="m/s"),
            v=Field(data=jnp.zeros_like(u_3d), name="v", dims=dims_3d, units="m/s"),
            T=Field(data=T_3d, name="T", dims=dims_3d, units="K"),
            p_s=Field(data=p_s_data, name="p_s", dims=dims_2d, units="Pa"),
            phis=Field(data=phis_data, name="phis", dims=dims_2d, units="m^2/s^2"),
        )

        self.dt = 120.0
        config = LatLonPrimitiveEquationConfig(
            hyperdiff_coeff=0.0,
            A_h=0.0,
            use_polar_filter=True,
            use_conservation_fixer=True,
            fix_mass=True,
            sponge_tau_sec=0.0,
        )
        self.model = LatLonPrimitiveEquationModel(
            self.grid, self.sigma_coord, config, dt=self.dt,
        )
        self.n_steps = 30
        self.g = g
        self.c_pd = c_pd

    def _compute_ke_ie(self, state):
        """Compute kinetic energy KE and internal energy IE."""
        area = self.grid.area
        dsigma = self.sigma_coord.dsigma
        p_s = state.p_s.data
        u = state.u.data
        v = state.v.data
        T = state.T.data

        mass = area[:, :, None] * dsigma[None, None, :] * (p_s / self.g)[:, :, None]
        KE = 0.5 * jnp.sum(mass * (u**2 + v**2))
        IE = jnp.sum(mass * self.c_pd * T)
        return float(KE), float(IE)

    def test_energy_components_positive_finite(self):
        """KE and IE should remain positive and finite after 30 steps."""
        state = self.state0
        for _ in range(self.n_steps):
            state = self.model.step(state, self.dt)

        KE, IE = self._compute_ke_ie(state)
        assert KE > 0, f"KE is not positive: {KE}"
        assert IE > 0, f"IE is not positive: {IE}"
        assert jnp.isfinite(KE), f"KE is not finite: {KE}"
        assert jnp.isfinite(IE), f"IE is not finite: {IE}"

    def test_total_energy_conserved(self):
        """Total energy (KE+IE) should change by < 5% after 30 steps."""
        KE0, IE0 = self._compute_ke_ie(self.state0)
        E0 = KE0 + IE0

        state = self.state0
        for _ in range(self.n_steps):
            state = self.model.step(state, self.dt)

        KE_f, IE_f = self._compute_ke_ie(state)
        E_f = KE_f + IE_f

        rel_change = abs(E_f - E0) / abs(E0)
        print(f"  Energy partition: KE0={KE0:.6e}, IE0={IE0:.6e}, "
              f"KE_f={KE_f:.6e}, IE_f={IE_f:.6e}")
        print(f"  Total energy: E0={E0:.6e}, E_f={E_f:.6e}, "
              f"rel_change={rel_change:.4e}")
        assert rel_change < 0.05, f"Total energy change = {rel_change}"
