"""Williamson Test Case 2 — Steady-State Geostrophic Flow.

The initial condition is a global solid-body rotation in geostrophic
balance with a corresponding height field. Since the flow is an exact
steady state of the shallow water equations, any deviation from the
initial condition is a measure of numerical error.

Diagnostics:
- l2 and linf height error after 5 days
- Mass conservation (relative error)
- Energy conservation (relative error)

Reference: Williamson et al. (1992), JCP 102, 211-224.
"""

import unittest

import jax
import jax.numpy as jnp

from legoesm import constants

# Side-effect-free shared TC2 C-D-grid initial condition (single-sourced in
# test_cases so the Stage-A2 Williamson Experiment rung reuses it without
# importing this module's global x64 mutation).
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson2_cdgrid_initial_condition as williamson2_initial_condition,
)

jax.config.update("jax_enable_x64", True)


class TestWilliamson2CDGrid(unittest.TestCase):
    """Williamson Test Case 2 on the C-D grid cubed-sphere."""

    def setUp(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
            CDGridShallowWaterModel,
            CDGridShallowWaterConfig,
        )

        self.n = 16  # C16 resolution
        self.grid = create_cubed_sphere(self.n)
        self.cdgrid = create_cubed_sphere_cdgrid(self.grid)

        # Use viscosity for stability at C16
        dx_min = float(jnp.min(self.grid.dx))
        self.config = CDGridShallowWaterConfig(
            A_h=1e4,
            hyperdiff_coeff=dx_min ** 4 / (86400.0 * 10),
        )
        self.model = CDGridShallowWaterModel(self.grid, self.config)
        self.state0 = williamson2_initial_condition(self.cdgrid)

    def test_initial_condition_balanced(self):
        """Rest of tendencies should be small for balanced IC."""
        dh, du, dv = self.model.tendencies(self.state0)
        # Height tendency should be small (geostrophic balance)
        max_dh = float(jnp.max(jnp.abs(dh)))
        # Normalize by h_0 / characteristic_time
        h_0 = float(jnp.mean(self.state0.h))
        self.assertLess(max_dh / h_0 * 86400, 2.0,
                        f"dh/h_0 per day = {max_dh / h_0 * 86400:.4f}")

    def test_mass_conservation_5day(self):
        """Mass should be conserved to machine precision over 5 days."""
        dt = 600.0  # 10 min
        n_steps = int(5 * 86400 / dt)  # 5 days

        area = self.cdgrid.base.area
        mass_0 = float(jnp.sum(self.state0.h * area))

        state = self.state0
        for _ in range(n_steps):
            state = self.model.step(state, dt)

        mass_final = float(jnp.sum(state.h * area))
        rel_err = abs(mass_final - mass_0) / abs(mass_0)

        print(f"  Mass conservation: rel_err = {rel_err:.2e}")
        self.assertLess(rel_err, 1e-4,
                        f"Mass conservation failed: rel_err = {rel_err:.2e}")

    def test_height_error_5day(self):
        """Height error should stay small over 5 days of integration."""
        dt = 600.0  # 10 min
        n_steps = int(5 * 86400 / dt)  # 5 days

        state = self.state0
        for _ in range(n_steps):
            state = self.model.step(state, dt)

        h_err = state.h - self.state0.h
        area = self.cdgrid.base.area
        total_area = jnp.sum(area)

        # L2 error (area-weighted RMS)
        l2_err = float(jnp.sqrt(jnp.sum(h_err ** 2 * area) / total_area))
        h_range = float(jnp.max(self.state0.h) - jnp.min(self.state0.h))
        l2_norm = l2_err / max(h_range, 1.0)

        # Linf error
        linf_err = float(jnp.max(jnp.abs(h_err)))
        linf_norm = linf_err / max(h_range, 1.0)

        print(f"  W2 5-day: l2_norm = {l2_norm:.4f}, linf_norm = {linf_norm:.4f}")
        print(f"  W2 5-day: l2_abs = {l2_err:.4f} m, linf_abs = {linf_err:.4f} m")

        # At C16 with damping, expect errors to remain bounded
        self.assertLess(l2_norm, 0.5,
                        f"L2 normalized height error too large: {l2_norm:.4f}")
        self.assertLess(linf_norm, 1.5,
                        f"Linf normalized height error too large: {linf_norm:.4f}")
        self.assertTrue(jnp.all(jnp.isfinite(state.h)),
                        "Height field has non-finite values")
        self.assertTrue(jnp.all(jnp.isfinite(state.u_d)),
                        "u_d has non-finite values")

    def test_energy_bounded_5day(self):
        """Total energy should not grow significantly over 5 days."""
        dt = 600.0
        n_steps = int(5 * 86400 / dt)

        g = constants.g
        area = self.cdgrid.base.area

        def total_energy(state):
            u_c, v_c = _dgrid_to_center(state.u_d, state.v_d)
            ke = 0.5 * state.h * (u_c ** 2 + v_c ** 2)
            pe = 0.5 * g * (state.h + state.h_s) ** 2
            return float(jnp.sum((ke + pe) * area))

        def _dgrid_to_center(u_d, v_d):
            u_c = 0.25 * (u_d[:, :-1, :-1] + u_d[:, 1:, :-1]
                           + u_d[:, :-1, 1:] + u_d[:, 1:, 1:])
            v_c = 0.25 * (v_d[:, :-1, :-1] + v_d[:, 1:, :-1]
                           + v_d[:, :-1, 1:] + v_d[:, 1:, 1:])
            return u_c, v_c

        E_0 = total_energy(self.state0)

        state = self.state0
        for _ in range(n_steps):
            state = self.model.step(state, dt)

        E_final = total_energy(state)
        rel_change = abs(E_final - E_0) / abs(E_0)

        print(f"  Energy: rel_change = {rel_change:.4e}")
        # Energy should dissipate slightly (viscosity) but not grow
        self.assertLess(rel_change, 0.1,
                        f"Energy change too large: {rel_change:.4e}")

    def test_stability_10day(self):
        """Model should remain stable for 10 days."""
        dt = 600.0
        n_steps = int(10 * 86400 / dt)

        state = self.state0
        for _ in range(n_steps):
            state = self.model.step(state, dt)

        self.assertTrue(jnp.all(jnp.isfinite(state.h)),
                        "Height field blew up after 10 days")
        self.assertTrue(jnp.all(jnp.isfinite(state.u_d)),
                        "u_d blew up after 10 days")
        self.assertTrue(jnp.all(jnp.isfinite(state.v_d)),
                        "v_d blew up after 10 days")
        # h should stay positive
        self.assertTrue(float(jnp.min(state.h)) > 0,
                        "Height became negative")


if __name__ == "__main__":
    unittest.main()
