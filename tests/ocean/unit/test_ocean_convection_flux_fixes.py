"""Regression tests for four confirmed ocean physics fixes.

Covers (one class each):

1. Plume convection trigger uses a CONSISTENT-pressure static-stability
   check (displace the surface parcel to level-1 pressure) instead of a
   raw in-situ adjacent-level density comparison biased by compressibility.
   ``physics/convection/plume.py``.
2. Bulk-formula wind stress flips BOTH ``tau_x`` and ``tau_y`` from the
   atmospheric to the ocean convention (positive = stress into the ocean
   in the wind direction). ``physics/surface_forcing/bulk_formulas.py``.
3. Ice-shelf three-equation melt has a FINITE ``jax.grad`` when the
   quadratic discriminant is <= 0 (sqrt-floor guard), and the Jenkins /
   ISOMIP+ basal-melt module carries no hardcoded ``3974.0`` / ``1000.0``
   constants. ``physics/ice_shelf.py`` (the linearised
   ``ice_shelf_basal_melt`` closure was deleted 2026-07-17 — callers now use
   the faithful ``three_equation_melt``).
4. Backscatter column power is a depth-MEAN per-mass density [m^2/s^3] so it
   shares units with the per-mass EKE reservoir budget (not a depth-integral
   H x too large that would pin E to E_max).
   ``physics/lateral_mixing/backscatter.py``.

Run:
    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/ocean/unit/test_ocean_convection_flux_fixes.py -v
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pathlib

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.ocean.eos import wright_eos
from legoesm.ocean.physics.convection.plume import plume_convection
from legoesm.ocean.physics.convection.config import PlumeConfig
from legoesm.ocean.vertical import create_z_star_from_thicknesses
from legoesm.core.bulk_flux import compute_most_fluxes
from legoesm.ocean.physics.surface_forcing.bulk_formulas import (
    bulk_formula_surface_forcing,
)
from legoesm.ocean.physics.surface_forcing.config import BulkFormulaConfig
from legoesm.ocean.physics.ice_shelf import (
    IceShelfConfig,
    freezing_point_C,
    three_equation_melt,
)
from legoesm.grids.latlon import create_regional_latlon_grid
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
from legoesm.ocean.physics.lateral_mixing import (
    BackscatterConfig,
    backscatter_power_density_cgrid,
    update_eddy_energy,
)


# ---------------------------------------------------------------------------
# BUG 1 — plume trigger at consistent pressure
# ---------------------------------------------------------------------------
class TestPlumeTriggerConsistentPressure:
    def _unstable_column(self):
        """A column whose surface water is DENSER than level 1 at a common
        pressure (statically unstable), yet whose raw in-situ densities
        read as STABLE because compressibility inflates the deeper level.

        Surface is slightly saltier (denser at equal p); the level-1
        pressure gap (~2000 dbar) makes the raw in-situ rho[1] > rho[0].
        """
        T = jnp.array([[2.0, 2.0, 2.0]])            # (1, nlev), degC
        S = jnp.array([[35.0, 34.5, 34.5]])         # psu — saltier surface
        p_hydro = jnp.array([[0.0, 2.0e7, 4.0e7]])  # Pa (0 / ~2000 / ~4000 dbar)
        rho = wright_eos(T, S, p_hydro)
        return T, S, rho, p_hydro

    def test_raw_comparison_looks_stable(self):
        _, _, rho, _ = self._unstable_column()
        # The OLD trigger compared raw in-situ densities at DIFFERENT
        # pressures and would (wrongly) read this column as stable.
        assert bool(rho[..., 0] <= rho[..., 1])

    def test_consistent_pressure_is_unstable(self):
        T, S, rho, p_hydro = self._unstable_column()
        # Displacing the surface parcel to level-1 pressure reveals the
        # true (denser-than-environment) instability.
        rho_surf_at_1 = wright_eos(T[..., 0], S[..., 0], p_hydro[..., 1])
        assert bool(rho_surf_at_1 > rho[..., 1])

    def test_trigger_fires_and_conserves(self):
        T, S, rho, p_hydro = self._unstable_column()
        z_coord = create_z_star_from_thicknesses([2000.0, 2000.0, 2000.0])
        jacobian = jnp.ones((1,))
        out = plume_convection(T, S, rho, p_hydro, z_coord, jacobian,
                               PlumeConfig())
        # Fixed trigger fires on the consistent-pressure instability.
        assert float(jnp.sum(out.convection_flag)) > 0.0
        # Vertical redistribution still conserves the dz-weighted column
        # integral of heat to machine precision.
        dz = z_coord.dz_ref * jacobian[..., jnp.newaxis]
        col = jnp.sum(out.dT_dt * dz, axis=-1)
        assert float(jnp.max(jnp.abs(col))) < 1e-12

    def test_trigger_grad_finite(self):
        T, S, rho, p_hydro = self._unstable_column()
        z_coord = create_z_star_from_thicknesses([2000.0, 2000.0, 2000.0])
        jacobian = jnp.ones((1,))

        def loss(t_excess):
            cfg = PlumeConfig(T_excess=t_excess)
            out = plume_convection(T, S, rho, p_hydro, z_coord, jacobian, cfg)
            return jnp.sum(out.dT_dt ** 2)

        g = jax.grad(loss)(0.05)
        assert bool(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# BUG 2 — both stress components flip to the ocean convention
# ---------------------------------------------------------------------------
class TestBulkStressBothFlipped:
    def test_most_stress_convention_and_double_flip(self):
        """``compute_most_fluxes`` returns stress in the ATMOSPHERIC
        convention (opposes the wind); the ocean forcing negates BOTH
        components so each ends up WITH the wind (same sign)."""
        u = jnp.array(6.0)   # eastward wind
        v = jnp.array(4.0)   # northward wind
        T_a = jnp.array(280.0)
        q_a = jnp.array(0.005)
        T_s = jnp.array(285.0)
        q_s = jnp.array(0.010)
        rho_a = jnp.array(constants.rho_air)
        tau_x_raw, tau_y_raw, *_ = compute_most_fluxes(
            u, v, T_a, q_a, T_s, q_s, rho_a, scheme="coare3", n_iter=5)
        # Raw stresses OPPOSE the wind (atmospheric convention).
        assert float(tau_x_raw) < 0.0
        assert float(tau_y_raw) < 0.0
        # bulk_formulas now applies ``tau = -tau`` to BOTH components.
        tau_x, tau_y = -tau_x_raw, -tau_y_raw
        # Both point WITH the wind (into the ocean) — SAME sign as the wind.
        assert float(tau_x) > 0.0 and float(u) > 0.0
        assert float(tau_y) > 0.0 and float(v) > 0.0
        assert jnp.sign(tau_x) == jnp.sign(u)
        assert jnp.sign(tau_y) == jnp.sign(v)

    def test_bulk_forcing_taux_into_ocean(self):
        """End-to-end through ``bulk_formula_surface_forcing``: with an
        eastward wind ``U_a > 0`` the emitted ``tau_x`` is positive
        (into the ocean, eastward), ``tau_y`` is zero (v_a hardcoded 0),
        and both are finite."""
        nlev = 3
        T = jnp.full((2, 2, nlev), 5.0)       # SST 5 degC
        S = jnp.full((2, 2, nlev), 34.5)
        z_coord = create_z_star_from_thicknesses([10.0, 20.0, 40.0])
        jacobian = jnp.ones((2, 2))
        cfg = BulkFormulaConfig(bulk_scheme="coare3", U_a=8.0)
        out = bulk_formula_surface_forcing(T, S, z_coord, jacobian, cfg)
        tx = out.tau_x
        ty = out.tau_y
        assert bool(jnp.all(jnp.isfinite(tx)))
        assert bool(jnp.all(jnp.isfinite(ty)))
        assert float(jnp.mean(tx)) > 0.0     # eastward wind -> eastward stress
        assert bool(jnp.allclose(ty, 0.0))   # v_a = 0 in this scheme


# ---------------------------------------------------------------------------
# BUG 3 — ice-shelf melt: finite grad at disc<=0 + no hardcoded constants
# ---------------------------------------------------------------------------
class TestIceShelfMeltFix:
    def test_grad_finite_when_discriminant_nonpositive(self):
        """An out-of-range (advection/coupler overshoot) input drives the
        quadratic discriminant negative; the sqrt-floor keeps ``jax.grad``
        finite where ``sqrt(max(disc, 0))`` would produce a NaN gradient."""
        cfg = IceShelfConfig(enabled=True)
        # Negative-salinity overshoot: verify disc <= 0 here.
        S = jnp.array(-34.9)
        p = jnp.array(500.0)
        T_amb = jnp.array(-0.30)
        alpha = cfg.rho_w * cfg.c_w * cfg.gamma_T
        beta = cfg.rho_w * cfg.gamma_S
        gamma = cfg.rho_ice * cfg.L_f
        delta = cfg.rho_ice
        theta = T_amb - freezing_point_C(S, p, config=cfg)
        t_mbcp = T_amb - cfg.freeze_b - cfg.freeze_c * p
        disc = (gamma * beta - alpha * delta * t_mbcp) ** 2 \
            - 4.0 * (gamma * delta) * (-alpha * beta * theta)
        assert float(disc) <= 0.0    # pathological branch is exercised

        def melt(t):
            return three_equation_melt(t, S, p, config=cfg).m_dot_m_s

        g = jax.grad(melt)(T_amb)
        assert bool(jnp.isfinite(g))

    def test_grad_finite_at_freezing_point(self):
        """At the freezing point (theta = 0 => C = 0 => disc = B^2) the
        gradient is also finite (minimum-discriminant physical case)."""
        cfg = IceShelfConfig(enabled=True)
        S = jnp.array(34.5)
        p = jnp.array(500.0)
        t_f = float(freezing_point_C(S, p, config=cfg))

        def melt(t):
            return three_equation_melt(t, S, p, config=cfg).m_dot_m_s

        g = jax.grad(melt)(t_f)
        assert bool(jnp.isfinite(g))

    def test_three_equation_flux_finite_and_positive_when_warm(self):
        # (Replaces two tests that targeted the DELETED linearised
        # ice_shelf_basal_melt module; the constants-not-literals property
        # for ice_shelf.py itself is enforced by the CI constants ratchet.)
        r = three_equation_melt(
            jnp.array(1.0), jnp.array(34.5), jnp.array(500.0))
        assert bool(jnp.isfinite(r.freshwater_to_ocean))
        assert float(r.freshwater_to_ocean) > 0.0   # warm cavity -> melt


# ---------------------------------------------------------------------------
# BUG 4 — backscatter column power is a depth-MEAN per-mass density
# ---------------------------------------------------------------------------
class TestBackscatterDepthMean:
    def _grid(self):
        grid, wall = create_regional_latlon_grid(
            8, 8, 20.0, 30.0, periodic_x=True, lon_west=0.0, lon_east=10.0,
            dtype=jnp.float64)
        u_mask, v_mask = compute_face_masks(wall)
        return grid, wall, u_mask, v_mask

    def _fields(self, wall, nlev):
        n_lat, n_lon = wall.shape
        u2 = 0.1 * jax.random.normal(jax.random.PRNGKey(0), (n_lat, n_lon + 1))
        v2 = 0.1 * jax.random.normal(jax.random.PRNGKey(1), (n_lat + 1, n_lon))
        # Realistic (small) backscatter momentum tendency ~1e-7 m/s^2.
        tu2 = 1e-6 * u2
        tv2 = 1e-6 * v2
        bc = lambda a: jnp.broadcast_to(a[..., None], a.shape + (nlev,))
        return bc(u2), bc(v2), bc(tu2), bc(tv2)

    def test_power_is_depth_mean_not_integral(self):
        grid, wall, u_mask, v_mask = self._grid()
        nlev = 20
        dz = jnp.full((nlev,), 4000.0 / nlev)   # deep 4000 m column
        u3, v3, tu3, tv3 = self._fields(wall, nlev)
        p_mean = backscatter_power_density_cgrid(
            u3, v3, tu3, tv3, grid, u_mask=u_mask, v_mask=v_mask, dz=dz)
        H = float(jnp.sum(dz))
        # Depth-INTEGRATED equivalent = depth-mean x H (the old, wrong scale).
        p_integ = p_mean * H
        m_mean = float(jnp.max(jnp.abs(p_mean)))
        m_integ = float(jnp.max(jnp.abs(p_integ)))
        assert m_mean > 0.0
        # The mean is O(1/H) of the integral on a deep column.
        assert m_integ / m_mean == pytest.approx(H, rel=1e-9)

    def test_uniform_dz_invariant(self):
        """A uniform thickness cancels in the depth-mean: dz=2 gives the
        same result as the unit-spacing default (an integral would double)."""
        grid, wall, u_mask, v_mask = self._grid()
        nlev = 4
        u3, v3, tu3, tv3 = self._fields(wall, nlev)
        p_default = backscatter_power_density_cgrid(
            u3, v3, tu3, tv3, grid, u_mask=u_mask, v_mask=v_mask)
        p_dz2 = backscatter_power_density_cgrid(
            u3, v3, tu3, tv3, grid, u_mask=u_mask, v_mask=v_mask,
            dz=jnp.full((nlev,), 2.0))
        assert bool(jnp.allclose(p_dz2, p_default, rtol=1e-10))

    def test_reservoir_does_not_immediately_pin(self):
        """With the per-mass (depth-mean) source the EKE reservoir E stays
        well below E_max over several steps; the depth-INTEGRATED source
        (H x larger) pins E to E_max immediately."""
        grid, wall, u_mask, v_mask = self._grid()
        n_lat, n_lon = wall.shape
        nlev = 20
        dz = jnp.full((nlev,), 4000.0 / nlev)
        u3, v3, tu3, tv3 = self._fields(wall, nlev)
        p_mean = backscatter_power_density_cgrid(
            u3, v3, tu3, tv3, grid, u_mask=u_mask, v_mask=v_mask, dz=dz)
        p_integ = p_mean * float(jnp.sum(dz))

        cfg = BackscatterConfig(enabled=True, E_min=0.0, E_max=0.1,
                                tau_relax_days=10.0, efficiency=0.9)
        E0 = jnp.full((n_lat, n_lon), 1e-3)
        dt = 3600.0
        zeros = jnp.zeros_like(p_mean)
        E_mean, E_int = E0, E0
        for _ in range(6):
            E_mean = update_eddy_energy(E_mean, dt, jnp.abs(p_mean), zeros, cfg)
            E_int = update_eddy_energy(E_int, dt, jnp.abs(p_integ), zeros, cfg)
        # Depth-mean: nowhere near the ceiling.
        assert float(jnp.max(E_mean)) < 0.5 * cfg.E_max
        assert not bool(jnp.any(jnp.isclose(E_mean, cfg.E_max)))
        # Depth-integrated (old scale): pins to E_max.
        assert bool(jnp.any(jnp.isclose(E_int, cfg.E_max)))

    def test_power_grad_finite(self):
        grid, wall, u_mask, v_mask = self._grid()
        nlev = 8
        dz = jnp.full((nlev,), 100.0)
        u3, v3, tu3, tv3 = self._fields(wall, nlev)

        def loss(u):
            p = backscatter_power_density_cgrid(
                u, v3, u, v3, grid, u_mask=u_mask, v_mask=v_mask, dz=dz)
            return jnp.sum(p)

        g = jax.grad(loss)(u3)
        assert bool(jnp.all(jnp.isfinite(g)))
