"""Explicit-flux CFL cap in the Richards solver (stability at high K_sat).

The gravity-drainage / surface-infiltration / bottom-drainage fluxes are applied
EXPLICITLY, so they must respect the Courant bound ``K*dt/dz <= 1``.  A sandy
soil (``K_sat ~ 4e-5 m/s``) at a 30-min step violates this and — because the
mixed-form solver reconstructs ``theta = theta_sat + S_s*theta_sat*psi`` with an
UNBOUNDED specific-storage term — ``psi``/``theta`` run away (observed
``theta ~ 5e5`` at a real sandy EC site).  ``solve_richards`` caps all three
explicit conductivity paths (interface ``K_half``, surface ``_Ksat``, bottom
``K_bot``) at ``_CFL_SAFETY * dz / dt``.

These tests lock in:
  1. a high-K_sat column under sustained saturating infiltration stays BOUNDED
     (``theta <= theta_sat`` to a tiny ponding-storage margin), no blow-up / NaN;
  2. the cap is a NO-OP below the CFL limit (loam drainage == uncapped K_sat) and
     ENGAGES above it (reported bottom drainage == the capped rate, not K_sat);
  3. water is still CONSERVED with the cap active (budget residual ~ 0) — the
     capped-but-not-infiltrated water leaves as runoff, it is not destroyed.
"""
from __future__ import annotations

import unittest

import jax.numpy as jnp

from legoesm import constants
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig, psi_from_theta
from legoesm.land.richards import solve_richards, RichardsConfig, _CFL_SAFETY

_SANDY_KSAT = 4.05e-5   # loamy-sand K_sat [m/s] (USDA Carsel-Parrish) — CFL-violating
_LOAM_KSAT = 2.0e-6     # loam-ish K_sat [m/s] — safely below dz0/dt at 30 min
_DT = 1800.0            # half-hourly EC-site step [s]


def _column(k_sat, theta_frac=0.6):
    grid = make_soil_grid(SoilGridConfig())
    nl = grid.n_layers
    hyd = SoilHydraulicsConfig(K_sat=k_sat, k_sat_decay_m=0.0)
    theta = jnp.full((1, nl), theta_frac * hyd.theta_sat)
    psi = psi_from_theta(theta, hyd)
    rcfg = RichardsConfig(bottom_bc="free_drainage")
    return grid, hyd, rcfg, psi, theta, nl


class TestRichardsCflCap(unittest.TestCase):
    def test_high_ksat_infiltration_stays_bounded(self):
        """Sustained saturating infiltration on a sandy column: theta must stay
        physically bounded (no specific-storage runaway) over many steps."""
        grid, hyd, rcfg, psi, theta, nl = _column(_SANDY_KSAT)
        # A large downward surface flux — far more than the column can drain per
        # step — is exactly the infiltration-excess case that drove the blow-up.
        flux_top = jnp.full(1, 5.0e-4)          # [m/s] downward (positive = into soil)
        sink = jnp.zeros((1, nl))
        sw = None
        theta_sat = float(hyd.theta_sat)
        for _ in range(200):
            out = solve_richards(psi, theta, grid, hyd, rcfg, flux_top, sink, _DT,
                                 surface_water=sw)
            psi, theta, sw = out.psi_new, out.theta_new, out.surface_water
            self.assertTrue(bool(jnp.all(jnp.isfinite(theta))))
            # theta_sat plus at most a tiny elastic ponding-storage margin.
            self.assertLessEqual(float(jnp.max(theta)), theta_sat + 1e-2)

    def test_cap_engages_only_above_cfl_limit(self):
        """Reported bottom drainage is clamped to ``_CFL_SAFETY*dz_bot/dt`` when
        (and only when) the physical conductivity exceeds it.  Uses a thin uniform
        grid so the bottom cap is testable (the production grid's ~3 m bottom
        layer makes ``dz_bot/dt`` far larger than any soil K, so it never binds
        there — by design; the surface/interface caps do the stabilising work)."""
        from legoesm.land.soil_grid import make_soil_grid_custom
        grid = make_soil_grid_custom([0.05] * 10)       # uniform 5 cm layers
        nl = grid.n_layers
        cfl_bot = _CFL_SAFETY * float(grid.dz[-1]) / _DT   # = 0.9*0.05/1800
        rcfg = RichardsConfig(bottom_bc="free_drainage")
        flux_sat = jnp.full(1, 1.0e-2)                   # keep the column saturated

        # Above the limit: K_sat >> cfl_bot -> drainage clamped to cfl_bot, not K_sat.
        hyd_hi = SoilHydraulicsConfig(K_sat=1.0e-3, k_sat_decay_m=0.0)
        theta = jnp.full((1, nl), float(hyd_hi.theta_sat))
        psi = psi_from_theta(theta, hyd_hi)
        self.assertGreater(1.0e-3, cfl_bot)
        out = solve_richards(psi, theta, grid, hyd_hi, rcfg, flux_sat,
                             jnp.zeros((1, nl)), _DT, surface_water=jnp.zeros(1))
        drain_hi = float(out.runoff_subsurface[0]) / float(constants.rho_water)
        self.assertAlmostEqual(drain_hi, cfl_bot, places=9)   # capped
        self.assertLess(drain_hi, 1.0e-3)

        # Below the limit: K_sat < cfl_bot -> cap inactive, drainage set by physics.
        hyd_lo = SoilHydraulicsConfig(K_sat=1.0e-6, k_sat_decay_m=0.0)
        theta = jnp.full((1, nl), float(hyd_lo.theta_sat))
        psi = psi_from_theta(theta, hyd_lo)
        self.assertLess(1.0e-6, cfl_bot)
        out = solve_richards(psi, theta, grid, hyd_lo, rcfg, flux_sat,
                             jnp.zeros((1, nl)), _DT, surface_water=jnp.zeros(1))
        drain_lo = float(out.runoff_subsurface[0]) / float(constants.rho_water)
        self.assertLess(drain_lo, cfl_bot)                    # cap inactive
        self.assertAlmostEqual(drain_lo, 1.0e-6, places=9)    # == K_sat (saturated)

    def test_mass_conserved_with_cap_active(self):
        """With the cap engaged, the one-step water budget still closes:
        flux_in*dt = d(soil water) + d(pond) + (drainage + runoff)*dt + sink*dt."""
        grid, hyd, rcfg, psi, theta, nl = _column(_SANDY_KSAT, theta_frac=0.7)
        dz = grid.dz
        flux_top = jnp.full(1, 3.0e-4)          # [m/s] strong infiltration
        sink = jnp.zeros((1, nl))
        sw0 = jnp.zeros(1)
        soil0 = float(jnp.sum(theta[0] * dz))
        out = solve_richards(psi, theta, grid, hyd, rcfg, flux_top, sink, _DT,
                             surface_water=sw0)
        soil1 = float(jnp.sum(out.theta_new[0] * dz))
        pond1 = float(out.surface_water[0])
        rho = float(constants.rho_water)
        drain = float(out.runoff_subsurface[0]) / rho      # m/s
        runoff = float(out.runoff_surface[0]) / rho        # m/s
        d_storage = (soil1 - soil0) + (pond1 - float(sw0[0]))
        residual = float(flux_top[0]) * _DT - drain * _DT - runoff * _DT - d_storage
        # Budget closes to solver tolerance (fixed-iteration Picard slack).
        self.assertLess(abs(residual), 1e-4 * float(flux_top[0]) * _DT + 1e-9)


if __name__ == "__main__":
    unittest.main()
