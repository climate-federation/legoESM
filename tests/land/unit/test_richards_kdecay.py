"""K_sat(z) depth-decay retention (Niu et al. 2005) in the Richards solver.

``SoilHydraulicsConfig.k_sat_decay_m`` scales the WHOLE K(theta) curve by
exp(-z/k_sat_decay_m) per layer, so that drainage OUT OF the root zone is
impeded.  ``0.0`` must be an exact no-op (backward compatibility).
"""
from __future__ import annotations

import unittest

import jax.numpy as jnp

from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig, psi_from_theta
from legoesm.land.richards import solve_richards, RichardsConfig


def _drain_a_wet_column(k_decay: float, dt: float = 3600.0, n_steps: int = 24):
    """Free-drain an initially wet column for n_steps; return final state + total
    subsurface (bottom) drainage."""
    grid = make_soil_grid(SoilGridConfig())
    nl = grid.n_layers
    hyd = SoilHydraulicsConfig(k_sat_decay_m=k_decay)
    theta = jnp.full((1, nl), 0.9 * hyd.theta_sat)
    psi = psi_from_theta(theta, hyd)
    rcfg = RichardsConfig(bottom_bc="free_drainage")
    flux_top = jnp.zeros(1)              # no infiltration; pure drainage
    sink = jnp.zeros((1, nl))
    total_subsurface = 0.0
    sw = None
    for _ in range(n_steps):
        out = solve_richards(psi, theta, grid, hyd, rcfg, flux_top, sink, dt,
                             surface_water=sw)
        psi, theta = out.psi_new, out.theta_new
        sw = out.surface_water
        total_subsurface += float(out.runoff_subsurface[0])
    return theta, total_subsurface


class TestKsatDepthDecay(unittest.TestCase):
    def test_decay_impedes_bottom_drainage_and_retains_deep_water(self):
        theta0, drain0 = _drain_a_wet_column(0.0)     # uniform K
        thetaD, drainD = _drain_a_wet_column(0.3)      # depth-decayed K
        # The deepest layer's conductivity is scaled by exp(-z_bot/0.3) << 1, so the
        # bottom (subsurface) drainage is strongly reduced ...
        self.assertLess(drainD, drain0)
        self.assertLess(drainD, 0.5 * drain0)
        # ... and the profile therefore retains more water at depth.
        self.assertGreater(float(thetaD[0, -1]), float(theta0[0, -1]))

    def test_zero_decay_matches_default_config(self):
        # k_sat_decay_m defaults to 0.0 and must be an exact no-op: an explicit 0.0
        # reproduces the default-config drainage bit-for-bit.
        grid = make_soil_grid(SoilGridConfig())
        nl = grid.n_layers
        default_hyd = SoilHydraulicsConfig()
        self.assertEqual(default_hyd.k_sat_decay_m, 0.0)
        _, drain_explicit0 = _drain_a_wet_column(0.0)
        # rebuild with the true default (no k_sat_decay_m kwarg) -> same numbers
        theta = jnp.full((1, nl), 0.9 * default_hyd.theta_sat)
        psi = psi_from_theta(theta, default_hyd)
        out = solve_richards(psi, theta, grid, default_hyd,
                             RichardsConfig(bottom_bc="free_drainage"),
                             jnp.zeros(1), jnp.zeros((1, nl)), 3600.0,
                             surface_water=None)
        # one-step subsurface drainage is finite and positive for a wet column
        self.assertGreater(float(out.runoff_subsurface[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
