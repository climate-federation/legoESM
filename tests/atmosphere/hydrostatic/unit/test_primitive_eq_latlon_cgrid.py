"""Unit and integration tests for the lat-lon C-grid hydrostatic PE model."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate, make_hybrid_levels
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonPrimitiveEquationModel,
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonHydrostaticState,
    cgrid_latlon_hydrostatic_tendencies,
    hydrostatic_to_cgrid,
    cgrid_to_hydrostatic,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import (
    held_suarez_init_latlon,
    held_suarez_forcing_latlon,
)
from tests.test_cases.baroclinic_wave import baroclinic_wave_init_latlon
from legoesm import constants


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(scope="module")
def grid():
    return create_latlon_grid(n_lat=32, radius=constants.R_earth, omega=constants.Omega)


@pytest.fixture(scope="module")
def sigma():
    return create_sigma_coordinate(n_levels=10)


@pytest.fixture(scope="module")
def model(grid, sigma):
    config = CGridLatLonPrimitiveEquationConfig(fix_mass=True)
    return CGridLatLonPrimitiveEquationModel(grid, sigma, config)


def _make_rest_state(grid, sigma):
    """Isothermal rest state on the C-grid."""
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = sigma.n_levels
    return CGridLatLonHydrostaticState(
        u=jnp.zeros((n_lat, n_lon + 1, nlev)),
        v=jnp.zeros((n_lat + 1, n_lon, nlev)),
        T=jnp.full((n_lat, n_lon, nlev), 300.0),
        p_s=jnp.full((n_lat, n_lon), 1.0e5),
        phis=jnp.zeros((n_lat, n_lon)),
    )


# ==============================================================================
# Shape tests
# ==============================================================================

class TestShapes:

    def test_state_shapes(self, grid, sigma):
        state = _make_rest_state(grid, sigma)
        n_lat, n_lon = grid.n_lat, grid.n_lon
        nlev = sigma.n_levels
        assert state.u.shape == (n_lat, n_lon + 1, nlev)
        assert state.v.shape == (n_lat + 1, n_lon, nlev)
        assert state.T.shape == (n_lat, n_lon, nlev)
        assert state.p_s.shape == (n_lat, n_lon)

    def test_tendency_shapes(self, grid, sigma):
        state = _make_rest_state(grid, sigma)
        du, dv, dT, dps, _ = cgrid_latlon_hydrostatic_tendencies(state, grid, sigma)
        n_lat, n_lon = grid.n_lat, grid.n_lon
        nlev = sigma.n_levels
        assert du.shape == (n_lat, n_lon + 1, nlev)
        assert dv.shape == (n_lat + 1, n_lon, nlev)
        assert dT.shape == (n_lat, n_lon, nlev)
        assert dps.shape == (n_lat, n_lon)


# ==============================================================================
# Rest state tests
# ==============================================================================

class TestRestState:

    def test_tendencies_finite(self, grid, sigma):
        state = _make_rest_state(grid, sigma)
        du, dv, dT, dps, _ = cgrid_latlon_hydrostatic_tendencies(state, grid, sigma)
        assert jnp.all(jnp.isfinite(du))
        assert jnp.all(jnp.isfinite(dv))
        assert jnp.all(jnp.isfinite(dT))
        assert jnp.all(jnp.isfinite(dps))

    def test_wind_tendencies_zero(self, grid, sigma):
        state = _make_rest_state(grid, sigma)
        du, dv, _, _, _ = cgrid_latlon_hydrostatic_tendencies(state, grid, sigma)
        assert jnp.allclose(du, 0.0, atol=1e-8)
        assert jnp.allclose(dv, 0.0, atol=1e-8)

    def test_ps_tendency_zero(self, grid, sigma):
        state = _make_rest_state(grid, sigma)
        _, _, _, dps, _ = cgrid_latlon_hydrostatic_tendencies(state, grid, sigma)
        assert jnp.allclose(dps, 0.0, atol=1e-10)


# ==============================================================================
# Adiabatic correction: κ T v·∇(ln p_s)
# ==============================================================================

class TestAdiabaticCorrection:

    def test_uniform_T_sloping_ps_produces_dT(self, grid, sigma):
        """Uniform T + zonal wind + sloping p_s must produce nonzero dT/dt
        from the κ T v·∇(ln p_s) correction term.

        With flux-form continuity the ω/p and v·∇ln p_s terms are more
        consistent, so the residual is smaller (~1e-8) than with advective
        continuity.  The test verifies the correction is non-trivially
        above machine zero.
        """
        n_lat, n_lon = grid.n_lat, grid.n_lon
        nlev = sigma.n_levels
        # Uniform zonal wind at u-faces, zero v
        u0 = 20.0
        u = jnp.full((n_lat, n_lon + 1, nlev), u0)
        v = jnp.zeros((n_lat + 1, n_lon, nlev))
        # Uniform temperature
        T = jnp.full((n_lat, n_lon, nlev), 300.0)
        # Sloping surface pressure: east-west gradient
        p_s = 1.0e5 + 1000.0 * jnp.sin(grid.lon2d)
        phis = jnp.zeros((n_lat, n_lon))
        state = CGridLatLonHydrostaticState(u=u, v=v, T=T, p_s=p_s, phis=phis)

        _, _, dT, _, _ = cgrid_latlon_hydrostatic_tendencies(state, grid, sigma)

        # The κ T v·∇(ln p_s) term should be nonzero where u * d(ln p_s)/dx ≠ 0
        max_dT = float(jnp.max(jnp.abs(dT)))
        assert max_dT > 1e-12, (
            f"dT/dt should be nonzero from adiabatic correction but max |dT| = {max_dT:.2e}"
        )

    def test_uniform_ps_no_correction(self, grid, sigma):
        """Uniform p_s with zonal wind: the correction term should vanish
        (grad(ln p_s) = 0), so dT/dt comes only from advection+omega terms.
        """
        state = _make_rest_state(grid, sigma)
        _, _, dT_rest, _, _ = cgrid_latlon_hydrostatic_tendencies(state, grid, sigma)
        # Rest state with uniform T and p_s → dT/dt = 0
        assert jnp.allclose(dT_rest, 0.0, atol=1e-8)


# ==============================================================================
# Adapter round-trip tests
# ==============================================================================

class TestAdapter:

    def test_round_trip_preserves_scalars(self, grid, sigma):
        hs = held_suarez_init_latlon(grid, sigma)
        cgrid_state = hydrostatic_to_cgrid(hs, grid)
        hs_back = cgrid_to_hydrostatic(cgrid_state, grid)
        assert jnp.allclose(hs.T.data, hs_back.T.data)
        assert jnp.allclose(hs.p_s.data, hs_back.p_s.data)

    def test_cgrid_shapes(self, grid, sigma):
        hs = held_suarez_init_latlon(grid, sigma)
        cs = hydrostatic_to_cgrid(hs, grid)
        n_lat, n_lon = grid.n_lat, grid.n_lon
        nlev = sigma.n_levels
        assert cs.u.shape == (n_lat, n_lon + 1, nlev)
        assert cs.v.shape == (n_lat + 1, n_lon, nlev)


# ==============================================================================
# Single-step stability
# ==============================================================================

class TestSingleStep:

    def test_step_finite(self, model, grid, sigma):
        state = _make_rest_state(grid, sigma)
        state_new = model.step(state, dt=30.0)
        assert jnp.all(jnp.isfinite(state_new.u))
        assert jnp.all(jnp.isfinite(state_new.v))
        assert jnp.all(jnp.isfinite(state_new.T))
        assert jnp.all(jnp.isfinite(state_new.p_s))

    def test_step_from_held_suarez_ic(self, model, grid, sigma):
        hs = held_suarez_init_latlon(grid, sigma)
        state = hydrostatic_to_cgrid(hs, grid)
        state_new = model.step(state, dt=30.0)
        assert jnp.all(jnp.isfinite(state_new.T))
        assert jnp.all(jnp.isfinite(state_new.p_s))


# ==============================================================================
# Multi-step stability
# ==============================================================================

class TestMultiStepStability:

    def test_rest_state_50_steps(self, grid, sigma):
        config = CGridLatLonPrimitiveEquationConfig(fix_mass=True)
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        state = _make_rest_state(grid, sigma)
        target_mass = model.compute_mass(state)

        dt = 30.0
        for _ in range(50):
            state = model.step(state, dt, target_mass=target_mass)

        assert jnp.all(jnp.isfinite(state.T))
        assert float(jnp.max(jnp.abs(state.u))) < 1.0
        assert float(jnp.max(jnp.abs(state.v))) < 1.0

    def test_held_suarez_100_steps(self, grid, sigma):
        config = CGridLatLonPrimitiveEquationConfig(fix_mass=True, A_h=1e5)
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        hs = held_suarez_init_latlon(grid, sigma)
        state = hydrostatic_to_cgrid(hs, grid)
        target_mass = model.compute_mass(state)

        dt = 30.0
        for _ in range(100):
            state = model.step(state, dt, target_mass=target_mass)

        assert jnp.all(jnp.isfinite(state.T))
        assert float(jnp.min(state.T)) > 0.0
        assert float(jnp.min(state.p_s)) > 0.0


# ==============================================================================
# Mass conservation — uses explicit target_mass API
# ==============================================================================

class TestMassConservation:

    def test_mass_conservation_50_steps(self, grid, sigma):
        config = CGridLatLonPrimitiveEquationConfig(fix_mass=True)
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        hs = held_suarez_init_latlon(grid, sigma)
        state = hydrostatic_to_cgrid(hs, grid)
        target_mass = model.compute_mass(state)

        area64 = grid.area.astype(jnp.float64)
        mass_init = float(jnp.sum(state.p_s.astype(jnp.float64) * area64))

        dt = 30.0
        for _ in range(50):
            state = model.step(state, dt, target_mass=target_mass)

        mass_final = float(jnp.sum(state.p_s.astype(jnp.float64) * area64))
        # iter-164: centralized helper (NaN-aware).
        from legoesm.diagnostics import compute_relative_drift
        rel_err = compute_relative_drift([mass_init, mass_final])
        assert rel_err < 1e-5, f"Mass conservation error: {rel_err}"

    def test_pfloor_preserved_after_mass_correction(self, grid, sigma):
        """When one column sits at p_floor, the mass fixer's negative
        correction must not push it below p_floor.
        """
        config = CGridLatLonPrimitiveEquationConfig(
            fix_mass=True, p_floor=100.0)
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)

        n_lat, n_lon, nlev = grid.n_lat, grid.n_lon, sigma.n_levels
        # Normal p_s everywhere except one column at p_floor
        p_s = jnp.full((n_lat, n_lon), 1.0e5)
        p_s = p_s.at[0, 0].set(config.p_floor)
        state = CGridLatLonHydrostaticState(
            u=jnp.zeros((n_lat, n_lon + 1, nlev)),
            v=jnp.zeros((n_lat + 1, n_lon, nlev)),
            T=jnp.full((n_lat, n_lon, nlev), 300.0),
            p_s=p_s,
            phis=jnp.zeros((n_lat, n_lon)),
        )
        # Use a large target_mass that forces a negative correction
        # (target < actual → correction < 0 → pushes floor column down)
        target_mass = model.compute_mass(state) * 0.99
        state_new = model.step(state, dt=30.0, target_mass=target_mass)

        min_ps = float(jnp.min(state_new.p_s))
        assert min_ps >= config.p_floor, (
            f"p_s={min_ps:.1f} dropped below p_floor={config.p_floor} "
            f"after mass correction"
        )


# ==============================================================================
# Energy conservation (KE + enthalpy)
# ==============================================================================

class TestEnergyConservation:

    def _total_energy(self, state, grid, sigma):
        """Compute area-weighted column-integrated KE + enthalpy.

        E = ∫ (0.5*(u²+v²) + c_p*T) dp/g  summed over area.
        """
        u_c = 0.5 * (state.u[:, :-1, :] + state.u[:, 1:, :])
        v_c = 0.5 * (state.v[:-1, :, :] + state.v[1:, :, :])
        ke = 0.5 * (u_c**2 + v_c**2)
        enthalpy = constants.c_pd * state.T
        dsigma = sigma.dsigma  # (nlev,)
        # Column integral: sum over levels weighted by p_s * dsigma / g
        integrand = (ke + enthalpy) * state.p_s[:, :, None] * dsigma / constants.g
        column = jnp.sum(integrand, axis=-1)  # (n_lat, n_lon)
        return float(jnp.sum(column * grid.area))

    def test_energy_drift_bounded(self, grid, sigma):
        """Total energy drift should be < 1% over 50 steps on isothermal rest state."""
        config = CGridLatLonPrimitiveEquationConfig(fix_mass=True, A_h=0.0)
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        state = _make_rest_state(grid, sigma)
        target_mass = model.compute_mass(state)

        E0 = self._total_energy(state, grid, sigma)
        dt = 30.0
        for _ in range(50):
            state = model.step(state, dt, target_mass=target_mass)
        E1 = self._total_energy(state, grid, sigma)

        rel_drift = abs(E1 - E0) / abs(E0)
        assert rel_drift < 0.01, f"Energy drift = {rel_drift:.4e}"


# ==============================================================================
# Physics coupling (Held-Suarez)
# ==============================================================================

class TestPhysicsCoupling:

    def test_held_suarez_1arg_closure(self, grid, sigma):
        """1-arg closure wrapping held_suarez_forcing_latlon."""
        config = CGridLatLonPrimitiveEquationConfig(fix_mass=True)
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        hs = held_suarez_init_latlon(grid, sigma)
        state = hydrostatic_to_cgrid(hs, grid)

        def physics_fn(hs_state):
            return held_suarez_forcing_latlon(hs_state, grid, sigma)

        state_new = model.step_with_physics(state, dt=30.0, physics_fn=physics_fn)
        assert jnp.all(jnp.isfinite(state_new.T))
        assert jnp.all(jnp.isfinite(state_new.p_s))

    def test_held_suarez_3arg_direct(self, grid, sigma):
        """Pass held_suarez_forcing_latlon directly (3-arg legacy contract)."""
        config = CGridLatLonPrimitiveEquationConfig(fix_mass=True)
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        hs = held_suarez_init_latlon(grid, sigma)
        state = hydrostatic_to_cgrid(hs, grid)

        # Direct 3-arg function — must not require a closure wrapper
        state_new = model.step_with_physics(
            state, dt=30.0, physics_fn=held_suarez_forcing_latlon)
        assert jnp.all(jnp.isfinite(state_new.T))
        assert jnp.all(jnp.isfinite(state_new.p_s))

    def test_tuple_returning_physics(self, grid, sigma):
        """Physics that returns (tendencies, aux) tuple must be unwrapped."""
        from legoesm.core.state import HydrostaticTendencies
        from legoesm.core.field import Field

        config = CGridLatLonPrimitiveEquationConfig(fix_mass=True)
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        state = _make_rest_state(grid, sigma)
        n_lat, n_lon, nlev = grid.n_lat, grid.n_lon, sigma.n_levels
        dims_3d = ("lat", "lon", "level")
        dims_2d = ("lat", "lon")

        def tuple_physics(hs_state, grid_, sigma_):
            zeros_3d = jnp.zeros((n_lat, n_lon, nlev))
            zeros_2d = jnp.zeros((n_lat, n_lon))
            tend = HydrostaticTendencies(
                du_dt=Field(data=zeros_3d, name="du_dt", dims=dims_3d, units="m/s2"),
                dT_dt=Field(data=zeros_3d, name="dT_dt", dims=dims_3d, units="K/s"),
                dp_s_dt=Field(data=zeros_2d, name="dp_s_dt", dims=dims_2d, units="Pa/s"),
                dphis_dt=Field(data=zeros_2d, name="dphis_dt", dims=dims_2d, units="m2/s3"),
            )
            return (tend, None)  # (tendencies, aux) tuple

        state_new = model.step(state, dt=30.0, physics_fn=tuple_physics)
        assert jnp.all(jnp.isfinite(state_new.T))


# ==============================================================================
# Driver-compatible API (HydrostaticState in/out, physics_fn=None)
# ==============================================================================

class TestDriverAPI:
    """Reproduce the exact path used by ModelDriver: HydrostaticState in,
    step_with_physics(state, dt) with no physics_fn, HydrostaticState out.
    """

    def test_step_with_physics_no_physics(self, grid, sigma):
        """step_with_physics(hs, dt) must work with physics_fn=None."""
        config = CGridLatLonPrimitiveEquationConfig(fix_mass=True)
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        hs = held_suarez_init_latlon(grid, sigma)

        # This is exactly what the driver calls:
        hs_new = model.step_with_physics(hs, dt=30.0)

        # Must return HydrostaticState, not CGridLatLonHydrostaticState
        from legoesm.core.state import HydrostaticState
        assert isinstance(hs_new, HydrostaticState)
        assert jnp.all(jnp.isfinite(hs_new.T.data))
        assert jnp.all(jnp.isfinite(hs_new.p_s.data))

    def test_step_with_physics_with_closure(self, grid, sigma):
        """step_with_physics(hs, dt, physics_fn=closure) on HydrostaticState."""
        config = CGridLatLonPrimitiveEquationConfig(fix_mass=True)
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        hs = held_suarez_init_latlon(grid, sigma)

        def physics_fn(state):
            return held_suarez_forcing_latlon(state, grid, sigma)

        hs_new = model.step_with_physics(hs, dt=30.0, physics_fn=physics_fn)

        from legoesm.core.state import HydrostaticState
        assert isinstance(hs_new, HydrostaticState)
        assert jnp.all(jnp.isfinite(hs_new.T.data))

    def test_step_hydrostatic_state_multistep(self, grid, sigma):
        """Multi-step with HydrostaticState input matches driver loop pattern."""
        config = CGridLatLonPrimitiveEquationConfig(fix_mass=True, A_h=1e5)
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        state = held_suarez_init_latlon(grid, sigma)

        dt = 30.0
        for _ in range(20):
            state = model.step_with_physics(state, dt)

        assert jnp.all(jnp.isfinite(state.T.data))
        assert float(jnp.min(state.p_s.data)) > 0.0

    def test_step_is_pure_restart_branching(self, grid, sigma):
        """Stepping state1 then state2 on the same model must use state2.

        This is the restart/branching contract: replacing the driver's
        state must take effect on the next step.
        """
        config = CGridLatLonPrimitiveEquationConfig(fix_mass=True)
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)

        # state1: Held-Suarez IC at ~300 K
        hs1 = held_suarez_init_latlon(grid, sigma)
        out1 = model.step_with_physics(hs1, dt=30.0)

        # state2: same shape but T shifted to ~350 K
        from legoesm.core.field import Field
        T_hot = hs1.T.replace(data=hs1.T.data + 50.0)
        hs2 = hs1._replace(T=T_hot)
        out2 = model.step_with_physics(hs2, dt=30.0)

        # out2 must reflect the 350 K state, not the 300 K state
        mean_T1 = float(jnp.mean(out1.T.data))
        mean_T2 = float(jnp.mean(out2.T.data))
        assert mean_T2 > mean_T1 + 40.0, (
            f"step() ignored state2: mean_T1={mean_T1:.2f}, mean_T2={mean_T2:.2f}"
        )

    def test_physics_cannot_produce_negative_pressure(self, grid, sigma):
        """Physics dp_s_dt that would drop p_s below p_floor must be clamped."""
        from legoesm.core.state import HydrostaticTendencies
        from legoesm.core.field import Field

        config = CGridLatLonPrimitiveEquationConfig(
            fix_mass=True, p_floor=100.0)
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        state = _make_rest_state(grid, sigma)
        n_lat, n_lon, nlev = grid.n_lat, grid.n_lon, sigma.n_levels
        dims_3d = ("lat", "lon", "level")
        dims_2d = ("lat", "lon")

        def extreme_physics(hs, grid_, sigma_):
            """Return a huge negative dp_s/dt that would crash without rails."""
            zeros_3d = jnp.zeros((n_lat, n_lon, nlev))
            return HydrostaticTendencies(
                du_dt=Field(data=zeros_3d, name="du_dt", dims=dims_3d, units="m/s2"),
                dT_dt=Field(data=zeros_3d, name="dT_dt", dims=dims_3d, units="K/s"),
                dp_s_dt=Field(
                    data=jnp.full((n_lat, n_lon), -2e5),
                    name="dp_s_dt", dims=dims_2d, units="Pa/s"),
                dphis_dt=Field(data=jnp.zeros((n_lat, n_lon)),
                               name="dphis_dt", dims=dims_2d, units="m2/s3"),
            )

        out = model.step(state, dt=1.0, physics_fn=extreme_physics)
        min_ps = float(jnp.min(out.p_s))
        assert min_ps >= config.p_floor, (
            f"p_s={min_ps} dropped below p_floor={config.p_floor} after physics"
        )


# ==============================================================================
# Native C-grid vs HydrostaticState path equivalence
# ==============================================================================

class TestNativeCGridEquivalence:

    def test_single_step_T_ps_match(self, grid, sigma):
        """One step via native C-grid must produce same T/p_s as one step
        via the HydrostaticState path (which converts in and out once).
        """
        config = CGridLatLonPrimitiveEquationConfig(fix_mass=True, A_h=1e5)
        hs = held_suarez_init_latlon(grid, sigma)
        dt = 30.0

        # Native C-grid path
        model_a = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        cgrid_in = hydrostatic_to_cgrid(hs, grid)
        cgrid_out = model_a.step(cgrid_in, dt)
        native_hs = cgrid_to_hydrostatic(cgrid_out, grid)

        # HydrostaticState path (converts in and out once)
        model_b = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        hs_out = model_b.step(hs, dt)

        # T and p_s must be identical (same computation, just wrapped)
        assert jnp.allclose(native_hs.T.data, hs_out.T.data, atol=1e-10)
        assert jnp.allclose(native_hs.p_s.data, hs_out.p_s.data, atol=1e-10)

    def test_driver_multistep_matches_native_cgrid(self, grid, sigma):
        """Multi-step HydrostaticState (driver) path must match native
        C-grid evolution in T and p_s — the cached face winds prevent
        lossy re-projection between steps.
        """
        config = CGridLatLonPrimitiveEquationConfig(fix_mass=True, A_h=1e5)
        hs = held_suarez_init_latlon(grid, sigma)
        dt = 30.0
        n_steps = 10

        # Native C-grid loop
        model_nat = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        cgrid_state = hydrostatic_to_cgrid(hs, grid)
        target = model_nat.compute_mass(cgrid_state)
        for _ in range(n_steps):
            cgrid_state = model_nat.step(cgrid_state, dt, target_mass=target)
        native_hs = cgrid_to_hydrostatic(cgrid_state, grid)

        # Driver-style HydrostaticState loop (cache preserves face winds)
        model_drv = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        driver_state = hs
        for _ in range(n_steps):
            driver_state = model_drv.step_with_physics(driver_state, dt)

        # T and p_s must match closely (T is cell-centered in both paths)
        assert jnp.allclose(native_hs.T.data, driver_state.T.data, atol=1e-8), (
            f"T diverged: max diff = "
            f"{float(jnp.max(jnp.abs(native_hs.T.data - driver_state.T.data)))}"
        )
        assert jnp.allclose(native_hs.p_s.data, driver_state.p_s.data, atol=1e-4), (
            f"p_s diverged: max diff = "
            f"{float(jnp.max(jnp.abs(native_hs.p_s.data - driver_state.p_s.data)))}"
        )

    def test_cache_invalidated_on_restart(self, grid, sigma):
        """Replacing the driver state (restart/branching) must invalidate
        the face-wind cache so the new state's winds are used.
        """
        config = CGridLatLonPrimitiveEquationConfig(fix_mass=True)
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)

        hs1 = held_suarez_init_latlon(grid, sigma)
        out1 = model.step_with_physics(hs1, dt=30.0)  # populates cache

        # Create a completely different state (restart)
        T_hot = hs1.T.replace(data=hs1.T.data + 50.0)
        hs2 = hs1._replace(T=T_hot)
        out2 = model.step_with_physics(hs2, dt=30.0)  # must invalidate cache

        # out2 must reflect the hot state
        mean_T1 = float(jnp.mean(out1.T.data))
        mean_T2 = float(jnp.mean(out2.T.data))
        assert mean_T2 > mean_T1 + 40.0, (
            f"Cache not invalidated on restart: "
            f"mean_T1={mean_T1:.2f}, mean_T2={mean_T2:.2f}"
        )


# ==============================================================================
# Pole consistency
# ==============================================================================

class TestPoleConsistency:

    def test_v_zero_at_poles(self, model, grid, sigma):
        hs = held_suarez_init_latlon(grid, sigma)
        state = hydrostatic_to_cgrid(hs, grid)
        state_new = model.step(state, dt=30.0)
        assert jnp.allclose(state_new.v[0, :, :], 0.0)
        assert jnp.allclose(state_new.v[-1, :, :], 0.0)


# ==============================================================================
# Hybrid sigma-pressure coordinate
# ==============================================================================

class TestHybridCoordinate:

    def test_hybrid_rest_state(self, grid):
        hybrid = make_hybrid_levels(n_levels=10)
        n_lat, n_lon = grid.n_lat, grid.n_lon
        nlev = hybrid.n_levels
        state = CGridLatLonHydrostaticState(
            u=jnp.zeros((n_lat, n_lon + 1, nlev)),
            v=jnp.zeros((n_lat + 1, n_lon, nlev)),
            T=jnp.full((n_lat, n_lon, nlev), 300.0),
            p_s=jnp.full((n_lat, n_lon), 1.0e5),
            phis=jnp.zeros((n_lat, n_lon)),
        )
        du, dv, dT, dps, _ = cgrid_latlon_hydrostatic_tendencies(state, grid, hybrid)
        assert jnp.all(jnp.isfinite(du))
        assert jnp.allclose(du, 0.0, atol=1e-6)

    def test_hybrid_step_stable(self, grid):
        hybrid = make_hybrid_levels(n_levels=10)
        config = CGridLatLonPrimitiveEquationConfig(fix_mass=True)
        model = CGridLatLonPrimitiveEquationModel(grid, hybrid, config)
        hs = held_suarez_init_latlon(grid, hybrid)
        state = hydrostatic_to_cgrid(hs, grid)
        state_new = model.step(state, dt=30.0)
        assert jnp.all(jnp.isfinite(state_new.T))
        assert jnp.all(jnp.isfinite(state_new.p_s))

    def test_hybrid_sloping_ps_nonzero_dps(self, grid):
        """Hybrid coords with uniform wind + sloping p_s must produce
        nonzero dp_s/dt from the div(dp*v) closure (not dp*div(v)).
        """
        hybrid = make_hybrid_levels(n_levels=10)
        n_lat, n_lon = grid.n_lat, grid.n_lon
        nlev = hybrid.n_levels
        u0 = 20.0
        u = jnp.full((n_lat, n_lon + 1, nlev), u0)
        v = jnp.zeros((n_lat + 1, n_lon, nlev))
        T = jnp.full((n_lat, n_lon, nlev), 300.0)
        p_s = 1.0e5 + 1000.0 * jnp.sin(grid.lon2d)
        phis = jnp.zeros((n_lat, n_lon))
        state = CGridLatLonHydrostaticState(
            u=u, v=v, T=T, p_s=p_s, phis=phis)
        _, _, _, dps, _ = cgrid_latlon_hydrostatic_tendencies(
            state, grid, hybrid)
        max_dps = float(jnp.max(jnp.abs(dps)))
        assert max_dps > 1e-4, (
            f"dp_s/dt should be nonzero with sloping p_s in hybrid coords "
            f"but max |dp_s/dt| = {max_dps:.2e}"
        )


# ==============================================================================
# Polar filter for PE
# ==============================================================================

class TestPolarFilterPE:

    def test_polar_filter_stable(self, grid, sigma):
        dt = 60.0  # within pole-cell CFL for n_lat=32
        config = CGridLatLonPrimitiveEquationConfig(
            fix_mass=True, use_polar_filter=True,
            polar_filter_cutoff_deg=60.0, polar_filter_max_wave_speed=300.0,
        )
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config, dt=dt)
        hs = held_suarez_init_latlon(grid, sigma)
        state = hydrostatic_to_cgrid(hs, grid)
        target_mass = model.compute_mass(state)

        for _ in range(20):
            state = model.step(state, dt, target_mass=target_mass)

        assert jnp.all(jnp.isfinite(state.T))
        assert jnp.all(jnp.isfinite(state.p_s))


# ==============================================================================
# Time integrator flexibility
# ==============================================================================

class TestTimeIntegratorsPE:

    def test_rk4_pe(self, grid, sigma):
        config = CGridLatLonPrimitiveEquationConfig(fix_mass=True, time_integrator="rk4")
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        state = _make_rest_state(grid, sigma)
        state_new = model.step(state, dt=30.0)
        assert jnp.all(jnp.isfinite(state_new.T))

    def test_ssp_rk54_pe(self, grid, sigma):
        config = CGridLatLonPrimitiveEquationConfig(fix_mass=True, time_integrator="ssp_rk54")
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        state = _make_rest_state(grid, sigma)
        state_new = model.step(state, dt=30.0)
        assert jnp.all(jnp.isfinite(state_new.T))


# ==============================================================================
# PPM transport
# ==============================================================================

class TestPPMTransportPE:

    def test_ppm_transport_3d(self, grid):
        from legoesm.core.operators_fv_latlon_3d import cgrid_fv_flux_divergence_latlon_3d
        n_lat, n_lon = grid.n_lat, grid.n_lon
        nlev = 5
        q = jnp.ones((n_lat, n_lon, nlev))
        u = jnp.ones((n_lat, n_lon + 1, nlev)) * 10.0
        v = jnp.zeros((n_lat + 1, n_lon, nlev))
        tend = cgrid_fv_flux_divergence_latlon_3d(q, u, v, grid)
        assert jnp.max(jnp.abs(tend)) < 1e-10


# ==============================================================================
# Fallback transport (use_ppm_transport=False)
# ==============================================================================

class TestFallbackTransportPE:

    def test_no_ppm_rest_state_stable(self, grid, sigma):
        """Cell-centered gradient fallback should be stable for 50 steps."""
        config = CGridLatLonPrimitiveEquationConfig(
            fix_mass=True, use_ppm_transport=False,
        )
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        state = _make_rest_state(grid, sigma)
        target_mass = model.compute_mass(state)

        dt = 30.0
        for _ in range(50):
            state = model.step(state, dt, target_mass=target_mass)

        assert jnp.all(jnp.isfinite(state.T))
        assert jnp.all(jnp.isfinite(state.p_s))
        assert float(jnp.max(jnp.abs(state.u))) < 1.0

    def test_no_ppm_held_suarez_stable(self, grid, sigma):
        """Held-Suarez IC with fallback transport should be stable for 50 steps."""
        config = CGridLatLonPrimitiveEquationConfig(
            fix_mass=True, use_ppm_transport=False, A_h=1e5,
        )
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        hs = held_suarez_init_latlon(grid, sigma)
        state = hydrostatic_to_cgrid(hs, grid)
        target_mass = model.compute_mass(state)

        dt = 30.0
        for _ in range(50):
            state = model.step(state, dt, target_mass=target_mass)

        assert jnp.all(jnp.isfinite(state.T))
        assert float(jnp.min(state.T)) > 0.0
        assert float(jnp.min(state.p_s)) > 0.0


# ==============================================================================
# Tracer transport
# ==============================================================================

class TestTracerTransport:

    def test_tracer_advected(self, grid, sigma):
        """A tracer initialized with a bump should be transported."""
        config = CGridLatLonPrimitiveEquationConfig(fix_mass=True, A_h=1e5)
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        hs = held_suarez_init_latlon(grid, sigma)
        state = hydrostatic_to_cgrid(hs, grid)

        # Add a moisture-like tracer with a Gaussian bump
        n_lat, n_lon, nlev = grid.n_lat, grid.n_lon, sigma.n_levels
        q_v = 0.01 * jnp.exp(-((grid.lat2d - 0.3)**2 + grid.lon2d**2) / 0.1)
        q_v = jnp.broadcast_to(q_v[:, :, None], (n_lat, n_lon, nlev))
        state = state._replace(tracers={"q_v": q_v})

        target_mass = model.compute_mass(state)
        dt = 30.0
        for _ in range(20):
            state = model.step(state, dt, target_mass=target_mass)

        assert "q_v" in state.tracers
        assert jnp.all(jnp.isfinite(state.tracers["q_v"]))
        assert state.tracers["q_v"].shape == (n_lat, n_lon, nlev)

    def test_uniform_tracer_stays_uniform(self, grid, sigma):
        """A spatially uniform tracer should remain uniform under advection."""
        config = CGridLatLonPrimitiveEquationConfig(fix_mass=True)
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        state = _make_rest_state(grid, sigma)
        n_lat, n_lon, nlev = grid.n_lat, grid.n_lon, sigma.n_levels
        state = state._replace(tracers={"q_v": jnp.full((n_lat, n_lon, nlev), 0.01)})
        target_mass = model.compute_mass(state)

        dt = 30.0
        for _ in range(10):
            state = model.step(state, dt, target_mass=target_mass)

        assert jnp.allclose(state.tracers["q_v"], 0.01, atol=1e-8)

    def test_physics_tracer_tendency_applied(self, grid, sigma):
        """Physics dq_v/dt = +1 s^-1 for one step must change q_v by exactly dt."""
        from legoesm.core.state import HydrostaticState, HydrostaticTendencies
        from legoesm.core.field import Field

        config = CGridLatLonPrimitiveEquationConfig(fix_mass=True)
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        state = _make_rest_state(grid, sigma)
        n_lat, n_lon, nlev = grid.n_lat, grid.n_lon, sigma.n_levels
        state = state._replace(
            tracers={"q_v": jnp.zeros((n_lat, n_lon, nlev))})

        dims_3d = ("lat", "lon", "level")
        dims_2d = ("lat", "lon")

        def dummy_physics(hs):
            """Return +1 s^-1 tendency for q_v, zero for everything else."""
            zeros_3d = jnp.zeros((n_lat, n_lon, nlev))
            zeros_2d = jnp.zeros((n_lat, n_lon))
            return HydrostaticTendencies(
                du_dt=Field(data=zeros_3d, name="du_dt", dims=dims_3d, units="m/s2"),
                dT_dt=Field(data=zeros_3d, name="dT_dt", dims=dims_3d, units="K/s"),
                dp_s_dt=Field(data=zeros_2d, name="dp_s_dt", dims=dims_2d, units="Pa/s"),
                dphis_dt=Field(data=zeros_2d, name="dphis_dt", dims=dims_2d, units="m2/s3"),
                dv_dt=Field(data=zeros_3d, name="dv_dt", dims=dims_3d, units="m/s2"),
                tracer_tendencies={
                    "q_v": Field(data=jnp.ones((n_lat, n_lon, nlev)),
                                 name="dq_v_dt", dims=dims_3d, units="1/s"),
                },
            )

        dt = 1.0
        state_new = model.step(state, dt, physics_fn=dummy_physics)

        expected = dt * 1.0  # dq_v/dt = 1, applied for dt seconds
        assert "q_v" in state_new.tracers
        assert jnp.allclose(state_new.tracers["q_v"], expected, atol=1e-10), (
            f"q_v should be {expected}, got {float(jnp.mean(state_new.tracers['q_v']))}"
        )


# ==============================================================================
# Convergence test (C32 vs C64)
# ==============================================================================

class TestConvergencePE:

    def test_tendency_residual_decreases_with_resolution(self, grid, sigma):
        """Area-weighted L2 temperature tendency on the isothermal rest state
        (analytic tendency = 0) should decrease from C32 → C64.
        """
        fine_grid = create_latlon_grid(
            n_lat=64, radius=constants.R_earth, omega=constants.Omega,
        )

        state_32 = _make_rest_state(grid, sigma)
        _, _, dT_32, _, _ = cgrid_latlon_hydrostatic_tendencies(state_32, grid, sigma)
        l2_32 = float(jnp.sqrt(
            jnp.sum(dT_32**2 * grid.area[:, :, None])
            / (grid.total_area * sigma.n_levels)
        ))

        state_64 = _make_rest_state(fine_grid, sigma)
        _, _, dT_64, _, _ = cgrid_latlon_hydrostatic_tendencies(state_64, fine_grid, sigma)
        l2_64 = float(jnp.sqrt(
            jnp.sum(dT_64**2 * fine_grid.area[:, :, None])
            / (fine_grid.total_area * sigma.n_levels)
        ))

        # Both should be near zero; fine grid should be no worse
        assert l2_32 < 1e-6, f"dT/dt L2 at C32 not near zero: {l2_32:.4e}"
        assert l2_64 <= l2_32 * 1.1, (
            f"Fine grid L2 ({l2_64:.4e}) worse than coarse ({l2_32:.4e})"
        )


# ==============================================================================
# Pole-region accuracy
# ==============================================================================

class TestPoleRegionAccuracy:

    def test_polar_tendencies_bounded(self, grid, sigma):
        """Near-pole tendency magnitudes should not blow up relative to mid-latitudes."""
        hs = held_suarez_init_latlon(grid, sigma)
        state = hydrostatic_to_cgrid(hs, grid)
        du, dv, dT, dps, _ = cgrid_latlon_hydrostatic_tendencies(state, grid, sigma)

        # Compare max |dT| at poles (first/last 2 rows) vs mid-latitudes
        n_lat = grid.n_lat
        mid_slice = slice(n_lat // 4, 3 * n_lat // 4)
        pole_rows = jnp.concatenate([dT[:2], dT[-2:]], axis=0)
        mid_rows = dT[mid_slice]

        max_pole = float(jnp.max(jnp.abs(pole_rows)))
        max_mid = float(jnp.max(jnp.abs(mid_rows)))

        # Pole tendencies should not be more than 10x mid-latitude values
        assert max_pole < max(max_mid * 10.0, 1e-6), (
            f"Pole |dT| = {max_pole:.4e} vs mid |dT| = {max_mid:.4e}"
        )

    def test_pole_ps_tendency_symmetric(self, grid, sigma):
        """Surface pressure tendency should be zonally symmetric at poles."""
        hs = held_suarez_init_latlon(grid, sigma)
        state = hydrostatic_to_cgrid(hs, grid)
        _, _, _, dps, _ = cgrid_latlon_hydrostatic_tendencies(state, grid, sigma)

        # First and last latitude rows of dp_s/dt should be nearly uniform
        dps_south = dps[0, :]
        dps_north = dps[-1, :]
        assert float(jnp.std(dps_south)) < float(jnp.abs(jnp.mean(dps_south))) + 1e-10
        assert float(jnp.std(dps_north)) < float(jnp.abs(jnp.mean(dps_north))) + 1e-10


# ==============================================================================
# Baroclinic wave smoke test (DCMIP-style)
# ==============================================================================

class TestBaroclinicWave:

    def test_baroclinic_wave_10_steps(self, grid, sigma):
        """Baroclinic wave IC should be stable for 10 steps."""
        config = CGridLatLonPrimitiveEquationConfig(fix_mass=True, A_h=1e5)
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        hs = baroclinic_wave_init_latlon(grid, sigma, perturbed=True)
        state = hydrostatic_to_cgrid(hs, grid)
        target_mass = model.compute_mass(state)

        dt = 30.0
        for _ in range(10):
            state = model.step(state, dt, target_mass=target_mass)

        assert jnp.all(jnp.isfinite(state.T))
        assert jnp.all(jnp.isfinite(state.p_s))
        assert float(jnp.min(state.T)) > 100.0
        assert float(jnp.max(jnp.abs(state.u))) < 200.0


# ==============================================================================
# Sigma-coordinate flux-form continuity
# ==============================================================================

class TestSigmaContinuityFluxForm:
    """Verify that the sigma continuity uses div(p_s·v), not p_s·div(v)."""

    def test_sloping_ps_sigma_dp_s_dt(self, grid, sigma):
        """With uniform wind + sloping p_s, flux-form dp_s/dt must differ
        from the advective form (p_s * div(v)).  For uniform wind on a
        lat-lon grid, div(v) is nonzero (spherical geometry), so both
        forms give nonzero dp_s/dt.  The key assertion is that the result
        matches the flux-form formula.
        """
        n_lat, n_lon = grid.n_lat, grid.n_lon
        nlev = sigma.n_levels
        u0 = 20.0
        u = jnp.full((n_lat, n_lon + 1, nlev), u0)
        v = jnp.zeros((n_lat + 1, n_lon, nlev))
        T = jnp.full((n_lat, n_lon, nlev), 300.0)
        # Sloping p_s with large gradient to amplify the difference
        p_s = 1.0e5 + 5000.0 * jnp.sin(grid.lon2d)
        phis = jnp.zeros((n_lat, n_lon))
        state = CGridLatLonHydrostaticState(
            u=u, v=v, T=T, p_s=p_s, phis=phis)

        _, _, _, dps, _ = cgrid_latlon_hydrostatic_tendencies(
            state, grid, sigma)

        # dp_s/dt should be nonzero (nonzero wind + sloping p_s)
        max_dps = float(jnp.max(jnp.abs(dps)))
        assert max_dps > 1e-3, (
            f"dp_s/dt should be nonzero with sloping p_s + wind "
            f"but max |dp_s/dt| = {max_dps:.2e}"
        )
        assert jnp.all(jnp.isfinite(dps))

    def test_uniform_ps_sigma_zero_dp_s(self, grid, sigma):
        """Uniform p_s + uniform wind: dp_s/dt should be zero because
        div(p_s * v) = p_s * div(v) when p_s is constant.
        """
        state = _make_rest_state(grid, sigma)
        _, _, _, dps, _ = cgrid_latlon_hydrostatic_tendencies(
            state, grid, sigma)
        assert jnp.allclose(dps, 0.0, atol=1e-10)


# ==============================================================================
# Tracer mass conservation (mass-weighted)
# ==============================================================================

class TestTracerMassConservation:
    """Verify that ∫ q·dp·dA (tracer mass) is conserved by flux-form transport."""

    def _tracer_mass(self, q, p_s, sigma, grid):
        """Compute area-integrated mass-weighted tracer: ∫ q·(p_s·dσ)·dA / g."""
        dp = p_s[:, :, None] * sigma.dsigma  # (n_lat, n_lon, nlev)
        integrand = q * dp  # (n_lat, n_lon, nlev)
        column = jnp.sum(integrand, axis=-1)  # (n_lat, n_lon)
        return float(jnp.sum(column.astype(jnp.float64)
                             * grid.area.astype(jnp.float64))) / constants.g

    def test_tracer_mass_conserved_50_steps(self, grid, sigma):
        """A Gaussian tracer bump should conserve ∫ q·dp·dA over 50 steps."""
        config = CGridLatLonPrimitiveEquationConfig(
            fix_mass=True, A_h=1e5)
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
        hs = held_suarez_init_latlon(grid, sigma)
        state = hydrostatic_to_cgrid(hs, grid)

        # Add a Gaussian tracer bump
        n_lat, n_lon, nlev = grid.n_lat, grid.n_lon, sigma.n_levels
        q_v = 0.01 * jnp.exp(
            -((grid.lat2d - 0.3)**2 + grid.lon2d**2) / 0.2)
        q_v = jnp.broadcast_to(q_v[:, :, None], (n_lat, n_lon, nlev))
        state = state._replace(tracers={"q_v": q_v})
        target_mass = model.compute_mass(state)

        mass_init = self._tracer_mass(
            state.tracers["q_v"], state.p_s, sigma, grid)

        dt = 30.0
        for _ in range(50):
            state = model.step(state, dt, target_mass=target_mass)

        mass_final = self._tracer_mass(
            state.tracers["q_v"], state.p_s, sigma, grid)

        # iter-164: centralized helper.
        from legoesm.diagnostics import compute_relative_drift
        rel_err = compute_relative_drift([mass_init, mass_final])
        assert rel_err < 1e-3, (
            f"Tracer mass conservation error: {rel_err:.4e} "
            f"(init={mass_init:.6e}, final={mass_final:.6e})"
        )

    def test_tracer_mass_conserved_hybrid_20_steps(self, grid):
        """Tracer mass ∫ q·dp·dA must be conserved on hybrid coords too.

        For hybrid, dp = dA + dB·p_s, so the mass fixer's p_s adjustment
        must use the per-level dp ratio, not the p_s ratio.
        """
        from legoesm.grids.vertical import dp_from_hybrid
        hybrid = make_hybrid_levels(n_levels=10)
        config = CGridLatLonPrimitiveEquationConfig(
            fix_mass=True, A_h=1e5)
        model = CGridLatLonPrimitiveEquationModel(grid, hybrid, config)
        hs = held_suarez_init_latlon(grid, hybrid)
        state = hydrostatic_to_cgrid(hs, grid)

        n_lat, n_lon, nlev = grid.n_lat, grid.n_lon, hybrid.n_levels
        q_v = 0.01 * jnp.exp(
            -((grid.lat2d - 0.3)**2 + grid.lon2d**2) / 0.2)
        q_v = jnp.broadcast_to(q_v[:, :, None], (n_lat, n_lon, nlev))
        state = state._replace(tracers={"q_v": q_v})
        target_mass = model.compute_mass(state)

        dp_init = dp_from_hybrid(hybrid, state.p_s)
        mass_init = float(jnp.sum(
            (state.tracers["q_v"] * dp_init).astype(jnp.float64)
            * grid.area.astype(jnp.float64)[:, :, None]))

        dt = 30.0
        for _ in range(20):
            state = model.step(state, dt, target_mass=target_mass)

        dp_final = dp_from_hybrid(hybrid, state.p_s)
        mass_final = float(jnp.sum(
            (state.tracers["q_v"] * dp_final).astype(jnp.float64)
            * grid.area.astype(jnp.float64)[:, :, None]))

        # iter-164: centralized helper.
        from legoesm.diagnostics import compute_relative_drift
        rel_err = compute_relative_drift([mass_init, mass_final])
        assert rel_err < 1e-3, (
            f"Hybrid tracer mass conservation error: {rel_err:.4e} "
            f"(init={mass_init:.6e}, final={mass_final:.6e})"
        )


# ==============================================================================
# #836: top sponge (Rayleigh damping increasing toward the model lid)
# ==============================================================================

class TestTopSponge:
    """The hydrostatic lat-lon C-grid dycore gains a config-gated top sponge that
    damps horizontal momentum toward rest in the upper levels (absorbs
    gravity-wave energy that would else reflect off the rigid lid — #836)."""

    def _uniform_wind_state(self, grid, sigma, u0=20.0):
        n_lat, n_lon = grid.n_lat, grid.n_lon
        nlev = sigma.n_levels
        return CGridLatLonHydrostaticState(
            u=jnp.full((n_lat, n_lon + 1, nlev), u0),
            v=jnp.full((n_lat + 1, n_lon, nlev), 5.0),
            T=jnp.full((n_lat, n_lon, nlev), 300.0),
            p_s=jnp.full((n_lat, n_lon), 1.0e5),
            phis=jnp.zeros((n_lat, n_lon)),
        )

    def test_sponge_off_is_byte_identical(self, grid, sigma):
        """sponge_coeff=0 (default) is bit-for-bit the no-sponge tendency."""
        state = self._uniform_wind_state(grid, sigma)
        cfg_off = CGridLatLonPrimitiveEquationConfig(sponge_coeff=0.0)
        du0, dv0, *_ = cgrid_latlon_hydrostatic_tendencies(state, grid, sigma, cfg_off)
        du_def, dv_def, *_ = cgrid_latlon_hydrostatic_tendencies(state, grid, sigma)
        assert jnp.array_equal(du0, du_def)
        assert jnp.array_equal(dv0, dv_def)

    def test_sponge_damps_top_toward_rest_and_leaves_surface(self, grid, sigma):
        """The sponge contribution du_on - du_off = -k(z)*u is: negative for u>0
        (damps toward rest), NONZERO at the top level, ZERO at the surface (below
        the sponge base), and increasing toward the lid."""
        state = self._uniform_wind_state(grid, sigma, u0=20.0)
        cfg_off = CGridLatLonPrimitiveEquationConfig(sponge_coeff=0.0)
        cfg_on = CGridLatLonPrimitiveEquationConfig(
            sponge_coeff=1.0 / 86400.0, sponge_width_m=10000.0)
        du_off, dv_off, *_ = cgrid_latlon_hydrostatic_tendencies(
            state, grid, sigma, cfg_off)
        du_on, dv_on, *_ = cgrid_latlon_hydrostatic_tendencies(
            state, grid, sigma, cfg_on)

        d_du = du_on - du_off        # == -k(z) * u,  (n_lat, n_lon+1, nlev)
        # SIGN: u = +20 > 0 -> sponge tendency is <= 0 everywhere (toward rest).
        assert float(jnp.max(d_du)) <= 1e-12
        assert float(jnp.min(d_du)) < 0.0
        # TOP level (index 0 = smallest sigma) is damped; SURFACE (index -1) is not.
        per_lev = jnp.max(jnp.abs(d_du), axis=(0, 1))   # (nlev,)
        assert float(per_lev[0]) > 0.0, "top level not damped by the sponge"
        assert float(per_lev[-1]) == 0.0, "surface level should be below the sponge base"
        # Ramp increases toward the lid (top >= a mid level >= surface).
        assert float(per_lev[0]) >= float(per_lev[-1])
        # v is damped by the SAME profile (momentum, not just u).
        d_dv = dv_on - dv_off
        assert float(jnp.min(d_dv)) < 0.0     # v = +5 > 0 -> damped negative
        assert float(jnp.max(jnp.abs(d_dv[..., -1]))) == 0.0

    def test_sponge_is_differentiable(self, grid):
        """grad through the ISOLATED sponge (du_on - du_off == -k(z)*u) is finite
        AND actually driven by the sponge — not by an unrelated term. Loss on the
        on-minus-off contribution ONLY: its grad must be nonzero at the lid (sponge
        live) and EXACTLY zero at the surface (below the sponge base). Differencing
        cancels every term the sponge does not touch, so a deleted/no-op sponge
        gives an all-zero grad here (a bare grad(loss_on)!=0 would pass via
        Coriolis even with no sponge — codex R1). Exercises the hybrid-L40 matrix
        path, not the L10 sigma fixture."""
        from legoesm.grids.vertical import standard_hybrid_levels
        hyb = standard_hybrid_levels(40)
        state = self._uniform_wind_state(grid, hyb, u0=20.0)
        cfg_off = CGridLatLonPrimitiveEquationConfig(sponge_coeff=0.0)
        cfg_on = CGridLatLonPrimitiveEquationConfig(
            sponge_coeff=1.0 / 3600.0, sponge_width_m=10000.0)

        def sponge_only_loss(u):
            s = state._replace(u=u)
            du_on, *_ = cgrid_latlon_hydrostatic_tendencies(s, grid, hyb, cfg_on)
            du_off, *_ = cgrid_latlon_hydrostatic_tendencies(s, grid, hyb, cfg_off)
            return jnp.sum((du_on - du_off) ** 2)   # == sum((k(z)*u)^2), sponge-only

        g = jax.grad(sponge_only_loss)(state.u)
        assert jnp.all(jnp.isfinite(g))                 # no NaN-grad trap
        assert float(jnp.max(jnp.abs(g[..., 0]))) > 0.0   # lid: sponge drives grad
        assert float(jnp.max(jnp.abs(g[..., -1]))) == 0.0  # surface: sponge absent


# ==============================================================================
# Biharmonic (del-4) hyperdiffusion
# ==============================================================================

class TestHyperdiffusion:
    """``nu_del4`` damps grid-scale u, v, T with the shallow-water lane's
    per-row pole cap; 0.0 is OFF."""

    NU4 = 2.9e15
    DT = 200.0

    def _checkerboard_state(self, grid, sigma, amp_u=1.0, amp_T=1.0):
        st = _make_rest_state(grid, sigma)
        n_lat, n_lon = grid.n_lat, grid.n_lon
        sgn_c = jnp.where(jnp.arange(n_lon) % 2 == 0, 1.0, -1.0)
        sgn_u = jnp.where(jnp.arange(n_lon + 1) % 2 == 0, 1.0, -1.0)
        u = st.u + amp_u * sgn_u[None, :, None]
        T = st.T + amp_T * sgn_c[None, :, None]
        return st._replace(u=u, T=T)

    def _del4_part(self, state, grid, sigma, nu=None):
        on = CGridLatLonPrimitiveEquationConfig(nu_del4=nu or self.NU4)
        off = CGridLatLonPrimitiveEquationConfig()
        t_on = cgrid_latlon_hydrostatic_tendencies(
            state, grid, sigma, on, dt=self.DT)
        t_off = cgrid_latlon_hydrostatic_tendencies(state, grid, sigma, off)
        return [a - b for a, b in zip(t_on[:4], t_off[:4])]

    def test_off_is_byte_identical(self, grid, sigma):
        state = self._checkerboard_state(grid, sigma)
        a = cgrid_latlon_hydrostatic_tendencies(
            state, grid, sigma, CGridLatLonPrimitiveEquationConfig(nu_del4=0.0),
            dt=self.DT)
        b = cgrid_latlon_hydrostatic_tendencies(state, grid, sigma)
        for x, y in zip(a[:4], b[:4]):
            assert jnp.array_equal(x, y)

    def test_requires_dt(self, grid, sigma):
        state = _make_rest_state(grid, sigma)
        cfg = CGridLatLonPrimitiveEquationConfig(nu_del4=self.NU4)
        with pytest.raises(ValueError, match="requires"):
            cgrid_latlon_hydrostatic_tendencies(state, grid, sigma, cfg)

    def test_damps_zonal_checkerboard_at_the_del4_rate(self, grid, sigma):
        """A lon-alternating u and T decay at nu*(4/dx^2)^2 on an equatorial
        row, and the del-4 part opposes the perturbation everywhere."""
        state = self._checkerboard_state(grid, sigma)
        du, dv, dT, dps = self._del4_part(state, grid, sigma)
        j = grid.n_lat // 2
        dx = float(grid.radius * grid.dlon * grid.cos_lat[j])
        rate = self.NU4 * (4.0 / dx**2) ** 2
        pert_T = state.T - 300.0
        assert float(jnp.max(dT * pert_T)) <= 0.0
        assert float(jnp.max(du * state.u)) <= 0.0
        got = -float(jnp.mean(dT[j] / pert_T[j]))
        assert got == pytest.approx(rate, rel=0.05)
        assert float(jnp.max(jnp.abs(dps))) == 0.0

    def test_dissipates_random_noise(self, grid, sigma):
        """Sign check on an arbitrary state: the del-4 part removes variance
        from u, v and T (a sign flip would add it)."""
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
        st = _make_rest_state(grid, sigma)
        st = st._replace(
            u=jax.random.normal(k1, st.u.shape),
            v=jax.random.normal(k2, st.v.shape).at[0].set(0.0).at[-1].set(0.0),
            T=st.T + jax.random.normal(k3, st.T.shape))
        du, dv, dT, _ = self._del4_part(st, grid, sigma)
        assert float(jnp.sum(du * st.u)) < 0.0
        assert float(jnp.sum(dv * st.v)) < 0.0
        assert float(jnp.sum(dT * (st.T - 300.0))) < 0.0

    def test_pole_rows_are_capped(self, grid, sigma):
        """A coefficient far above every row's stability cap still yields a
        finite tendency bounded by cfl_frac/dt on the checkerboard."""
        state = self._checkerboard_state(grid, sigma)
        du, dv, dT, _ = self._del4_part(state, grid, sigma, nu=1.0e20)
        assert bool(jnp.all(jnp.isfinite(dT)))
        cfg = CGridLatLonPrimitiveEquationConfig()
        # nu*lam^2 <= cfl_frac/dt and |lap^2 checkerboard| <= lam^2 * amp
        assert float(jnp.max(jnp.abs(dT))) <= cfg.nu_del4_cfl_frac / self.DT * 1.0001

    def _terrain_state(self, grid, coord):
        k1, k2 = jax.random.split(jax.random.PRNGKey(1))
        nlev = coord.n_levels
        st = _make_rest_state(grid, coord)
        lat = grid.lat[:, None]
        p_s = 1.0e5 - 2.0e4 * jnp.exp(-((lat - 0.3) / 0.3) ** 2) * jnp.cos(
            3.0 * grid.lon[None, :]) ** 2
        return st._replace(
            T=250.0 + 5.0 * jax.random.normal(k1, (grid.n_lat, grid.n_lon, nlev)),
            u=jax.random.normal(k2, st.u.shape), p_s=p_s)

    @pytest.mark.parametrize("hybrid", [True, False])
    def test_T_term_conserves_layer_mass_weighted_heat(self, grid, sigma, hybrid):
        """sum(dp * dT_del4 * area) vanishes to roundoff over varying terrain,
        including the per-row pole cap (a huge nu engages it on every row)."""
        from legoesm.grids.vertical import standard_hybrid_levels, dp_from_hybrid
        coord = standard_hybrid_levels(10) if hybrid else sigma
        st = self._terrain_state(grid, coord)
        dp = (dp_from_hybrid(coord, st.p_s) if hybrid
              else st.p_s[..., None] * coord.dsigma)
        for nu in (self.NU4, 1.0e20):
            dT = self._del4_part(st, grid, coord, nu=nu)[2]
            heat = jnp.sum(dp * dT * grid.area[..., None])
            scale = jnp.sum(jnp.abs(dp * dT) * grid.area[..., None])
            assert float(jnp.abs(heat) / scale) < 1e-12
            var = jnp.sum(dp * (st.T - 250.0) * dT * grid.area[..., None])
            assert float(var) < 0.0
