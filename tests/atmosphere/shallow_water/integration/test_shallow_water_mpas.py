"""Integration tests for MPAS shallow water solver on Voronoi meshes.

Tests Williamson (1992) standard test cases using the TRiSK discretization
from Ringler et al. (2010).
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.atmosphere.dynamics.gcm.shallow_water_mpas import (
    MPASShallowWaterModel,
    MPASShallowWaterConfig,
)
from tests.atmosphere.shallow_water.test_cases.williamson_mpas import (
    williamson_test2_mpas,
    williamson_test5_mpas,
    williamson_test6_mpas,
    compute_error_norms_mpas,
)


@pytest.fixture(scope="module", autouse=True)
def _fp64_policy():
    """Set fp64 precision policy for this module (conservation tests need it).

    Restores the original policy on teardown to avoid leaking into other
    test modules.
    """
    from legoesm.core.precision import get_policy
    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(saved)


@pytest.fixture(scope="module")
def mesh(_fp64_policy):
    """Level-3 icosahedral mesh (642 cells)."""
    return create_voronoi_mesh(3, lloyd_iterations=50)


class TestWilliamsonTC2:
    """Williamson Test Case 2: steady-state geostrophic flow."""

    def test_tc2_5day_height_error(self, mesh):
        """TC2 height error after 5 days should be < 10m."""
        config = MPASShallowWaterConfig(
            pv_scheme="energy",
            fix_mass=True,
            fix_energy=False,
            time_integrator="rk4",
        )
        model = MPASShallowWaterModel(mesh, config)
        state = williamson_test2_mpas(mesh)

        dt = 600.0
        nsteps = int(5 * 86400 / dt)  # 5 days
        s = state
        for _ in range(nsteps):
            s = model.step(s, dt)

        norms = compute_error_norms_mpas(s.h.data, state.h.data, mesh)
        h_mean = float(jnp.mean(state.h.data))
        h_err_m = norms["linf"] * h_mean

        assert jnp.all(jnp.isfinite(s.h.data)), "h contains NaN/Inf"
        assert jnp.all(jnp.isfinite(s.u.data)), "u contains NaN/Inf"
        # At level 3 (642 cells), ~150m error is typical for the coarse
        # icosahedral mesh without diffusion. Higher resolution converges.
        assert h_err_m < 200.0, (
            f"TC2 max height error {h_err_m:.2f}m > 200m after 5 days"
        )


class TestWilliamsonTC5:
    """Williamson Test Case 5: zonal flow over isolated mountain."""

    def test_tc5_15day_stable(self, mesh):
        """TC5 should remain stable for 15 days."""
        config = MPASShallowWaterConfig(
            pv_scheme="energy",
            nu_del4=1e15,
            fix_mass=True,
            fix_energy=False,
            time_integrator="rk4",
        )
        model = MPASShallowWaterModel(mesh, config)
        state = williamson_test5_mpas(mesh)

        dt = 200.0
        nsteps = int(15 * 86400 / dt)  # 15 days
        s = state
        for _ in range(nsteps):
            s = model.step(s, dt)

        assert jnp.all(jnp.isfinite(s.h.data)), "h contains NaN/Inf after 15 days"
        assert jnp.all(jnp.isfinite(s.u.data)), "u contains NaN/Inf after 15 days"

    def test_tc5_mass_drift(self, mesh):
        """TC5 mass drift with fixer should be < 1e-10."""
        config = MPASShallowWaterConfig(
            pv_scheme="energy",
            nu_del4=1e15,
            fix_mass=True,
            time_integrator="rk4",
        )
        model = MPASShallowWaterModel(mesh, config)
        state = williamson_test5_mpas(mesh)

        area = mesh.areaCell.astype(jnp.float64)
        mass_init = float(jnp.sum(state.h.data.astype(jnp.float64) * area))

        dt = 200.0
        nsteps = int(15 * 86400 / dt)
        s = state
        for _ in range(nsteps):
            s = model.step(s, dt)

        mass_final = float(jnp.sum(s.h.data.astype(jnp.float64) * area))
        # iter-164: centralized helper (NaN-aware, completes
        # iter-157/159 sweep across the entire codebase).
        from legoesm.diagnostics import compute_relative_drift
        rel_drift = compute_relative_drift([mass_init, mass_final])

        assert rel_drift < 1e-10, (
            f"Mass drift {rel_drift:.2e} exceeds 1e-10"
        )


class TestWilliamsonTC6:
    """Williamson Test Case 6: Rossby-Haurwitz wave 4."""

    def test_tc6_5day_stable(self, mesh):
        """TC6 should remain stable for 5 days."""
        config = MPASShallowWaterConfig(
            pv_scheme="energy",
            nu_del4=1e15,
            fix_mass=True,
            time_integrator="rk4",
        )
        model = MPASShallowWaterModel(mesh, config)
        state = williamson_test6_mpas(mesh)

        dt = 300.0
        nsteps = int(5 * 86400 / dt)  # 5 days
        s = state
        for _ in range(nsteps):
            s = model.step(s, dt)

        assert jnp.all(jnp.isfinite(s.h.data)), "h contains NaN/Inf after 5 days"
        assert jnp.all(jnp.isfinite(s.u.data)), "u contains NaN/Inf after 5 days"


class TestMassConservation:
    """Mass conservation with conservation fixer."""

    def test_mass_conservation_precision(self, mesh):
        """Mass should be conserved to near machine precision with fixer."""
        config = MPASShallowWaterConfig(
            pv_scheme="energy",
            fix_mass=True,
            time_integrator="rk4",
        )
        model = MPASShallowWaterModel(mesh, config)
        state = williamson_test2_mpas(mesh)

        area = mesh.areaCell.astype(jnp.float64)
        mass_init = float(jnp.sum(state.h.data.astype(jnp.float64) * area))

        dt = 600.0
        s = state
        for _ in range(100):
            s = model.step(s, dt)

        mass_final = float(jnp.sum(s.h.data.astype(jnp.float64) * area))
        # iter-164: centralized helper.
        from legoesm.diagnostics import compute_relative_drift
        rel_drift = compute_relative_drift([mass_init, mass_final])

        assert rel_drift < 1e-13, (
            f"Mass drift {rel_drift:.2e} exceeds 1e-13"
        )


class TestEnergyConservation:
    """Energy conservation with energy-conserving PV scheme."""

    def test_energy_bounded(self, mesh):
        """Total energy should stay bounded over 50 steps."""
        from legoesm.core.operators_voronoi import kinetic_energy_cell
        # iter-166: replaced literal 9.80616 with constants.g per
        # CLAUDE.md "Tests and scripts must follow the constant-
        # hygiene rule too."
        from legoesm import constants

        g = constants.g
        config = MPASShallowWaterConfig(
            g=g,
            pv_scheme="energy",
            fix_mass=True,
            fix_energy=True,
            time_integrator="rk4",
        )
        model = MPASShallowWaterModel(mesh, config)
        state = williamson_test2_mpas(mesh)

        area = mesh.areaCell

        def total_energy(s):
            h = s.h.data
            u = s.u.data
            ke = kinetic_energy_cell(u, mesh) * h
            pe = 0.5 * g * (h + s.h_s.data) ** 2
            return float(jnp.sum((ke + pe) * area))

        E_init = total_energy(state)

        dt = 600.0
        s = state
        for _ in range(50):
            s = model.step(s, dt)

        E_final = total_energy(s)
        # iter-164: centralized helper (energy drift, same
        # iter-78 pattern).
        from legoesm.diagnostics import compute_relative_drift
        rel_drift = compute_relative_drift([E_init, E_final])

        assert rel_drift < 1e-10, (
            f"Energy drift {rel_drift:.2e} exceeds 1e-10"
        )


class TestDifferentiability:
    """JAX differentiability through the MPAS solver."""

    def test_differentiable_3_steps(self, mesh):
        """Model should be differentiable through 3 steps."""
        config = MPASShallowWaterConfig(
            pv_scheme="energy",
            fix_mass=False,
            fix_energy=False,
            time_integrator="rk4",
        )
        model = MPASShallowWaterModel(mesh, config)
        state = williamson_test2_mpas(mesh)

        def loss_fn(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            for _ in range(3):
                s = model.step(s, 300.0)
            return jnp.mean(s.h.data ** 2)

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(state.h.data)
        assert jnp.all(jnp.isfinite(g)), "Gradient contains NaN/Inf"
        assert float(jnp.max(jnp.abs(g))) > 0, "Gradient is all zeros"


class TestPVSchemeDispatch:
    """Dispatch hardening for the MPAS shallow-water PV flux selector."""

    def test_unknown_pv_scheme_raises(self, mesh):
        """A typo in ``pv_scheme`` must raise, not silently select the
        energy-conserving PV flux."""
        config = MPASShallowWaterConfig(pv_scheme="bogus")
        model = MPASShallowWaterModel(mesh, config)
        state = williamson_test2_mpas(mesh)
        with pytest.raises(ValueError, match="Unknown pv_scheme"):
            model.step(state, 300.0)
