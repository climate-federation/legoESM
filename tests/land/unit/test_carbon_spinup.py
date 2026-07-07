"""Tests for the semi-analytic soil-carbon spin-up (legoesm.land.carbon.spinup)."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from legoesm.land.carbon.config import CarbonDiagnostics, CarbonState
from legoesm.land.carbon.spinup import (
    SlowPoolFluxes,
    analytic_slow_pool_equilibrium,
    integrate_annual_pools,
    run_semi_analytic_spinup,
)


def _state(**kw):
    d = dict(C_lab=100.0, C_fol=200.0, C_root=300.0,
             C_wood=1000.0, C_lit=400.0, C_som=5000.0)
    d.update(kw)
    return CarbonState(**{k: jnp.array([v]) for k, v in d.items()})


class TestAnalyticSlowPoolEquilibrium(unittest.TestCase):

    def test_wood_equilibrium_is_input_over_rate(self):
        """C_wood_eq = C_wood * a_wood / wood_litter."""
        st = _state(C_wood=1000.0)
        fx = SlowPoolFluxes(
            a_wood=jnp.array([10.0]), wood_litter=jnp.array([5.0]),
            lit_to_som=jnp.array([20.0]), r_het_som=jnp.array([10.0]))
        out = analytic_slow_pool_equilibrium(st, fx, cwd_humification_eff=0.3)
        # 1000 * 10/5 = 2000
        npt.assert_allclose(out.C_wood, 2000.0, rtol=1e-9)

    def test_som_equilibrium_includes_humified_cwd(self):
        """C_som_eq uses lit_to_som + cwd_humification_eff*a_wood as input."""
        st = _state(C_som=5000.0)
        fx = SlowPoolFluxes(
            a_wood=jnp.array([10.0]), wood_litter=jnp.array([5.0]),
            lit_to_som=jnp.array([20.0]), r_het_som=jnp.array([10.0]))
        out = analytic_slow_pool_equilibrium(st, fx, cwd_humification_eff=0.3)
        # som_in = 20 + 0.3*10 = 23; C_som_eq = 5000 * 23/10 = 11500
        npt.assert_allclose(out.C_som, 11500.0, rtol=1e-9)

    def test_reset_pool_is_a_fixed_point(self):
        """At the analytic equilibrium, input == loss (steady state)."""
        st = _state(C_som=5000.0, C_wood=1000.0)
        fx = SlowPoolFluxes(
            a_wood=jnp.array([12.0]), wood_litter=jnp.array([7.0]),
            lit_to_som=jnp.array([15.0]), r_het_som=jnp.array([9.0]))
        cwd = 0.3
        out = analytic_slow_pool_equilibrium(st, fx, cwd_humification_eff=cwd)
        # Wood: loss rate k = wood_litter/C_wood; at eq, k*C_wood_eq == a_wood.
        k_wood = float(fx.wood_litter[0]) / float(st.C_wood[0])
        npt.assert_allclose(k_wood * float(out.C_wood[0]), float(fx.a_wood[0]),
                            rtol=1e-9)
        # SOM: k*C_som_eq == som_in_eq (input).
        k_som = float(fx.r_het_som[0]) / float(st.C_som[0])
        som_in_eq = float(fx.lit_to_som[0]) + cwd * float(fx.a_wood[0])
        npt.assert_allclose(k_som * float(out.C_som[0]), som_in_eq, rtol=1e-9)

    def test_fast_pools_unchanged(self):
        st = _state()
        fx = SlowPoolFluxes(
            a_wood=jnp.array([10.0]), wood_litter=jnp.array([5.0]),
            lit_to_som=jnp.array([20.0]), r_het_som=jnp.array([10.0]))
        out = analytic_slow_pool_equilibrium(st, fx, cwd_humification_eff=0.3)
        for f in ("C_lab", "C_fol", "C_root", "C_lit"):
            npt.assert_allclose(getattr(out, f), getattr(st, f), rtol=0, atol=0)

    def test_zero_loss_leaves_pool_unchanged(self):
        """Zero loss flux (no inferable turnover) => pool left as-is, not an
        artefact (0 or C/eps)."""
        st = _state(C_wood=1234.0, C_som=6789.0)
        # Zero loss but NON-zero input: no finite equilibrium -> leave unchanged.
        fx = SlowPoolFluxes(
            a_wood=jnp.array([5.0]), wood_litter=jnp.array([0.0]),
            lit_to_som=jnp.array([7.0]), r_het_som=jnp.array([0.0]))
        out = analytic_slow_pool_equilibrium(st, fx, cwd_humification_eff=0.3)
        self.assertTrue(jnp.all(jnp.isfinite(out.C_wood)))
        self.assertTrue(jnp.all(jnp.isfinite(out.C_som)))
        npt.assert_allclose(out.C_wood, 1234.0, rtol=0, atol=0)
        npt.assert_allclose(out.C_som, 6789.0, rtol=0, atol=0)

    def test_batched(self):
        st = CarbonState(
            C_lab=jnp.full((3,), 100.0), C_fol=jnp.full((3,), 200.0),
            C_root=jnp.full((3,), 300.0), C_wood=jnp.full((3,), 1000.0),
            C_lit=jnp.full((3,), 400.0), C_som=jnp.full((3,), 5000.0))
        fx = SlowPoolFluxes(
            a_wood=jnp.full((3,), 10.0), wood_litter=jnp.full((3,), 5.0),
            lit_to_som=jnp.full((3,), 20.0), r_het_som=jnp.full((3,), 10.0))
        out = analytic_slow_pool_equilibrium(st, fx, cwd_humification_eff=0.3)
        self.assertEqual(out.C_som.shape, (3,))
        npt.assert_allclose(out.C_wood, 2000.0, rtol=1e-9)


def _const_diag(ncol, **over):
    """A stationary CarbonDiagnostics (per-day rates) with closed allocation.

    Defaults: a_wood=10, wood_litter=5, lit_to_som=20, r_het_som=10 so the
    analytic slow-pool reset is analytically checkable; allocation closes
    (a_fol+a_lab+a_root+a_wood == max(npp,0)) with npp=10, r_auto=5.
    """
    z = lambda v: jnp.full((ncol,), float(v))
    d = dict(gpp=15.0, npp=10.0, r_maint=3.0, r_growth=2.0, r_auto=5.0,
             r_het_lit=0.0, r_het_som=10.0, r_het_cwd=0.0, r_het=10.0, nee=0.0,
             unmet_npp_deficit=0.0, a_fol=0.0, a_lab=0.0, a_root=0.0, a_wood=10.0,
             lab_release=0.0, leaf_litter=0.0, root_litter=0.0, wood_litter=5.0,
             wood_to_som=1.5, lit_to_som=20.0, lai=4.0)
    d.update(over)
    return CarbonDiagnostics(**{k: z(v) for k, v in d.items()})


class TestRunSemiAnalyticSpinup(unittest.TestCase):
    """Plumbing test for the shared 3-phase driver with a frozen-carbon toy
    'model' (constant stationary fluxes), so the analytic reset + annual
    accumulation + mass-balance NEE are exactly checkable without the land
    model."""

    def _run(self, ncol):
        # A clean model year: steps_per_year * dt == 365 days.
        steps_per_year = 4
        dt = 365.0 * 86400.0 / steps_per_year
        carbon0 = CarbonState(
            C_lab=jnp.full((ncol,), 100.0), C_fol=jnp.full((ncol,), 200.0),
            C_root=jnp.full((ncol,), 300.0), C_wood=jnp.full((ncol,), 1000.0),
            C_lit=jnp.full((ncol,), 400.0), C_som=jnp.full((ncol,), 5000.0))
        state0 = jnp.zeros((ncol,))  # opaque dummy state

        def forcing_fn(doy, hour):
            return None  # the toy step ignores forcing

        def step_fn(state, carbon, forcing, doy):
            # Frozen carbon (returned unchanged) + stationary diagnostics.
            return state, carbon, _const_diag(ncol)

        return run_semi_analytic_spinup(
            step_fn, state0, carbon0, forcing_fn,
            n_spinup=3, n_verify=2, steps_per_year=steps_per_year, dt=dt,
            cwd_humification_eff=0.3)

    def test_analytic_reset_and_shapes(self):
        _fs, fc, annual = self._run(ncol=1)
        # Wood: C_wood_eq = 1000 * a_wood/wood_litter = 1000 * 10/5 = 2000.
        npt.assert_allclose(np.asarray(fc.C_wood), 2000.0, rtol=1e-6)
        # SOM: som_in = lit_to_som + cwd*a_wood = 20 + 0.3*10 = 23; loss 10.
        #      C_som_eq = 5000 * 23/10 = 11500.
        npt.assert_allclose(np.asarray(fc.C_som), 11500.0, rtol=1e-6)
        # Fast pools untouched by the reset and frozen by the toy step.
        for f, v in (("C_lab", 100.0), ("C_fol", 200.0),
                     ("C_root", 300.0), ("C_lit", 400.0)):
            npt.assert_allclose(np.asarray(getattr(fc, f)), v, rtol=1e-6)
        # Per-verify-year annual dict, shape (n_verify, ncol).
        for key in ("gpp", "npp", "nee_model", "alloc_resid", "lai_sum",
                    "lai_max", "nsteps", "C_som"):
            self.assertIn(key, annual)
            self.assertEqual(np.asarray(annual[key]).shape, (2, 1), key)
        # Allocation closes -> residual ~0; LAI reductions; steps counted.
        npt.assert_allclose(np.asarray(annual["alloc_resid"]), 0.0, atol=1e-9)
        npt.assert_allclose(np.asarray(annual["lai_max"]), 4.0, rtol=1e-6)
        npt.assert_allclose(np.asarray(annual["nsteps"]), 4.0, rtol=0)
        # Frozen carbon -> zero mass-balance NEE over each verify year.
        npt.assert_allclose(np.asarray(annual["nee_model"]), 0.0, atol=1e-6)
        # Annual GPP total = per-day rate * 365 (one clean model year).
        npt.assert_allclose(np.asarray(annual["gpp"]), 15.0 * 365.0, rtol=1e-6)

    def test_batched_over_columns(self):
        _fs, fc, annual = self._run(ncol=3)
        self.assertEqual(fc.C_som.shape, (3,))
        npt.assert_allclose(np.asarray(fc.C_wood), 2000.0, rtol=1e-6)
        self.assertEqual(np.asarray(annual["gpp"]).shape, (2, 3))


class TestIntegrateAnnualPools(unittest.TestCase):
    """Raw forward integrator (NO analytic reset) with the frozen-carbon toy:
    per-year pools are recorded with the IC as row 0, so a drift metric over the
    trajectory reads ~0 when the carbon is frozen."""

    def test_frozen_carbon_flat_series_with_ic_row(self):
        ncol = 2
        steps_per_year = 4
        dt = 365.0 * 86400.0 / steps_per_year
        carbon0 = CarbonState(
            C_lab=jnp.full((ncol,), 100.0), C_fol=jnp.full((ncol,), 200.0),
            C_root=jnp.full((ncol,), 300.0), C_wood=jnp.full((ncol,), 1000.0),
            C_lit=jnp.full((ncol,), 400.0), C_som=jnp.full((ncol,), 5000.0))
        state0 = jnp.zeros((ncol,))  # opaque dummy state

        def forcing_fn(doy, hour):
            return None

        def step_fn(state, carbon, forcing, doy):
            return state, carbon, _const_diag(ncol)  # carbon frozen

        n_years = 3
        out = integrate_annual_pools(
            step_fn, state0, carbon0, forcing_fn,
            n_years=n_years, steps_per_year=steps_per_year, dt=dt)
        for p in ("C_lab", "C_fol", "C_root", "C_wood", "C_lit", "C_som"):
            arr = np.asarray(out[p])
            # One (n_years+1, ncol) array per pool; row 0 == the IC.
            self.assertEqual(arr.shape, (n_years + 1, ncol), p)
            npt.assert_allclose(arr[0], np.asarray(getattr(carbon0, p)),
                                rtol=1e-6)
            # Frozen carbon -> every recorded year identical -> zero drift
            # (broadcast the IC row explicitly; assert_allclose won't).
            npt.assert_allclose(arr, np.broadcast_to(arr[0], arr.shape),
                                rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
