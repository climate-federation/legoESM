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


class TestPoolFieldsGuard(unittest.TestCase):
    """The hardcoded ``spinup._POOL_FIELDS`` copy MUST track
    ``CarbonState._fields``.  The closed-column mass balance
    (``_total_carbon`` / annual-pool dict) is keyed off it, so a drift after
    the 6->8 SOM split (or any future pool change) would silently break
    conservation.  Guards Risk #1/#4 of the multi-pool-SOM plan."""

    def test_pool_fields_matches_carbon_state(self):
        from legoesm.land.carbon.spinup import _POOL_FIELDS
        self.assertEqual(tuple(_POOL_FIELDS), CarbonState._fields)


def _state(**kw):
    # A2: all three SOM pools are live and the analytic solve resets each of
    # them, so they are seeded non-zero (a zero loss flux would trip the
    # degenerate-column guard and leave that pool unchanged).
    d = dict(C_lab=100.0, C_fol=200.0, C_root=300.0,
             C_wood=1000.0, C_lit=400.0,
             C_som_active=5000.0, C_som_slow=2000.0, C_som_passive=4000.0)
    d.update(kw)
    return CarbonState(**{k: jnp.array([v]) for k, v in d.items()})


_CWD, _F_AS, _F_SP = 0.3, 0.3, 0.3


def _fx(ncol=1, **over):
    """Stationary mean-annual SlowPoolFluxes for the cascade solve.

    Defaults (per column): a_wood=10, wood_litter=5, lit_to_som=20 and per-pool
    SOM losses som_active_loss=10, som_slow_loss=5, som_passive_loss=3.
    """
    z = lambda v: jnp.full((ncol,), float(v))
    d = dict(a_wood=10.0, wood_litter=5.0, lit_to_som=20.0,
             som_active_loss=10.0, som_slow_loss=5.0, som_passive_loss=3.0)
    d.update(over)
    return SlowPoolFluxes(**{k: z(v) for k, v in d.items()})


def _solve(st, fx, cwd=_CWD, f_as=_F_AS, f_sp=_F_SP):
    return analytic_slow_pool_equilibrium(
        st, fx, cwd_humification_eff=cwd,
        f_active_to_slow=f_as, f_slow_to_passive=f_sp)


class TestAnalyticSlowPoolEquilibrium(unittest.TestCase):

    def test_wood_equilibrium_is_input_over_rate(self):
        """C_wood_eq = C_wood * a_wood / wood_litter."""
        out = _solve(_state(C_wood=1000.0), _fx())
        npt.assert_allclose(out.C_wood, 2000.0, rtol=1e-9)  # 1000 * 10/5

    def test_active_equilibrium_includes_humified_cwd(self):
        """C_active_eq uses lit_to_som + cwd*a_wood as input, som_active_loss out."""
        out = _solve(_state(C_som_active=5000.0), _fx())
        # i_active = 20 + 0.3*10 = 23; C_active_eq = 5000 * 23/10 = 11500.
        npt.assert_allclose(out.C_som_active, 11500.0, rtol=1e-9)

    def test_slow_and_passive_forward_substitution(self):
        """Slow/passive equilibria cascade from the pool above:
        C_slow_eq = C_slow * (f_as*i_active) / som_slow_loss;
        C_passive_eq = C_passive * (f_sp*i_slow) / som_passive_loss."""
        out = _solve(_state(C_som_slow=2000.0, C_som_passive=4000.0), _fx())
        # i_active=23; i_slow=0.3*23=6.9; C_slow_eq=2000*6.9/5=2760.
        npt.assert_allclose(out.C_som_slow, 2760.0, rtol=1e-9)
        # i_passive=0.3*6.9=2.07; C_passive_eq=4000*2.07/3=2760.
        npt.assert_allclose(out.C_som_passive, 2760.0, rtol=1e-9)

    def test_every_pool_is_a_fixed_point(self):
        """At the reset, each pool's inferred loss k_X*C_X_eq == its input I_X
        (the forward-substitution fixed-point condition, all three SOM pools)."""
        st = _state(C_wood=1000.0, C_som_active=5000.0,
                    C_som_slow=2000.0, C_som_passive=4000.0)
        fx = _fx(a_wood=12.0, wood_litter=7.0, lit_to_som=15.0,
                 som_active_loss=9.0, som_slow_loss=4.0, som_passive_loss=2.5)
        out = _solve(st, fx)
        # Wood.
        k_wood = float(fx.wood_litter[0]) / float(st.C_wood[0])
        npt.assert_allclose(k_wood * float(out.C_wood[0]), float(fx.a_wood[0]),
                            rtol=1e-9)
        # Active: input = lit_to_som + cwd*a_wood.
        i_active = float(fx.lit_to_som[0]) + _CWD * float(fx.a_wood[0])
        k_a = float(fx.som_active_loss[0]) / float(st.C_som_active[0])
        npt.assert_allclose(k_a * float(out.C_som_active[0]), i_active, rtol=1e-9)
        # Slow: input = f_as * i_active.
        i_slow = _F_AS * i_active
        k_s = float(fx.som_slow_loss[0]) / float(st.C_som_slow[0])
        npt.assert_allclose(k_s * float(out.C_som_slow[0]), i_slow, rtol=1e-9)
        # Passive: input = f_sp * i_slow.
        i_passive = _F_SP * i_slow
        k_p = float(fx.som_passive_loss[0]) / float(st.C_som_passive[0])
        npt.assert_allclose(k_p * float(out.C_som_passive[0]), i_passive,
                            rtol=1e-9)

    def test_fast_pools_unchanged(self):
        out = _solve(_state(), _fx())
        for f in ("C_lab", "C_fol", "C_root", "C_lit"):
            npt.assert_allclose(getattr(out, f), getattr(_state(), f),
                                rtol=0, atol=0)

    def test_zero_loss_leaves_each_pool_unchanged(self):
        """A per-pool zero loss flux (no inferable turnover) leaves THAT pool as
        is, not an artefact (0 or C/eps); other pools still solve."""
        st = _state(C_wood=1234.0, C_som_active=6789.0,
                    C_som_slow=1111.0, C_som_passive=2222.0)
        # Wood + slow have zero loss; active + passive solve normally.
        fx = _fx(wood_litter=0.0, som_slow_loss=0.0)
        out = _solve(st, fx)
        for f in ("C_wood", "C_som_slow"):
            self.assertTrue(jnp.all(jnp.isfinite(getattr(out, f))))
        npt.assert_allclose(out.C_wood, 1234.0, rtol=0, atol=0)
        npt.assert_allclose(out.C_som_slow, 1111.0, rtol=0, atol=0)
        # Active still solved: 6789 * 23/10 = 15614.7.
        npt.assert_allclose(out.C_som_active, 6789.0 * 23.0 / 10.0, rtol=1e-9)

    def test_degenerate_active_feeds_zero_transfer_downstream(self):
        """FIX #4 (codex A2): a dead/collapsed active pool (loss <= eps,
        itself left unchanged) must NOT hand its raw litter/CWD forcing
        input downstream unconditionally.  The transfer INTO slow (and
        cascading into passive) is the upstream pool's ACTUAL (realised)
        equilibrium loss, which is 0 when the upstream is degenerate -- so
        with slow/passive's OWN loss still > eps (i.e. themselves alive, NOT
        protected by their own zero-loss guard), their correct zero-input
        linear equilibrium is ``C * 0 / loss == 0``, NOT the spurious
        ``C * f * i_active / loss`` the un-gated (pre-fix) formula
        manufactured from a transfer the dead active pool cannot actually
        produce."""
        st = _state(C_som_active=6789.0, C_som_slow=1111.0, C_som_passive=2222.0)
        # Active dead (loss <= eps); slow/passive keep the _fx() defaults
        # (som_slow_loss=5, som_passive_loss=3), i.e. themselves alive, so
        # their OWN per-pool zero-loss guard does NOT already protect them --
        # only the upstream input-gating fix does.
        fx = _fx(som_active_loss=0.0)
        out = _solve(st, fx)
        # Active itself: unaffected by this fix, still left at its spun-up
        # value (som_active_loss <= eps triggers the pre-existing guard).
        npt.assert_allclose(out.C_som_active, 6789.0, rtol=0, atol=0)
        self.assertTrue(jnp.all(jnp.isfinite(out.C_som_slow)))
        self.assertTrue(jnp.all(jnp.isfinite(out.C_som_passive)))
        # Correct zero-input equilibrium -- NOT the pre-fix spurious values
        # (i_active=23, i_slow=0.3*23=6.9: C_slow_spurious=1111*6.9/5=1533.18;
        # i_passive=0.3*6.9=2.07: C_passive_spurious=2222*2.07/3=1533.18).
        npt.assert_allclose(out.C_som_slow, 0.0, rtol=0, atol=1e-9)
        npt.assert_allclose(out.C_som_passive, 0.0, rtol=0, atol=1e-9)

    def test_batched(self):
        st = CarbonState(
            C_lab=jnp.full((3,), 100.0), C_fol=jnp.full((3,), 200.0),
            C_root=jnp.full((3,), 300.0), C_wood=jnp.full((3,), 1000.0),
            C_lit=jnp.full((3,), 400.0),
            C_som_active=jnp.full((3,), 5000.0),
            C_som_slow=jnp.full((3,), 2000.0),
            C_som_passive=jnp.full((3,), 4000.0))
        out = _solve(st, _fx(ncol=3))
        self.assertEqual(out.C_som_passive.shape, (3,))
        npt.assert_allclose(out.C_wood, 2000.0, rtol=1e-9)
        npt.assert_allclose(out.C_som_passive, 2760.0, rtol=1e-9)

    def test_cold_column_equilibrates_higher_than_warm(self):
        """REALISM via the production solver: two columns with identical SOM
        INPUTS (same a_wood/lit_to_som) but the cold column's smaller
        decomposition (freeze + temperature suppression -> smaller per-pool loss
        flux) equilibrates to MORE total SOC.  Losses scaled by the modifier
        ratio m_cold/m_warm reproduce the cold column's slower turnover."""
        from legoesm.land.carbon.carbon_cycle import _som_decomp_modifier
        from legoesm.land.carbon.config import CarbonConfig, som_total
        cfg = CarbonConfig(scheme="differland")
        precip = jnp.array([cfg.precip_ref])
        m_cold = float(_som_decomp_modifier(jnp.array([270.0]), precip, cfg)[0])
        m_warm = float(_som_decomp_modifier(jnp.array([298.0]), precip, cfg)[0])
        # Same stocks + same SOM inputs; per-pool loss flux scales with m
        # (loss = C * m * k * yr).  Warm -> larger loss -> smaller equilibrium.
        st = _state(C_som_active=5000.0, C_som_slow=2000.0, C_som_passive=4000.0)
        base = dict(a_wood=10.0, wood_litter=5.0, lit_to_som=20.0)
        warm = _solve(st, _fx(som_active_loss=10.0 * m_warm,
                              som_slow_loss=5.0 * m_warm,
                              som_passive_loss=3.0 * m_warm, **base))
        cold = _solve(st, _fx(som_active_loss=10.0 * m_cold,
                              som_slow_loss=5.0 * m_cold,
                              som_passive_loss=3.0 * m_cold, **base))
        self.assertGreater(float(som_total(cold)[0]), float(som_total(warm)[0]))
        self.assertGreater(float(som_total(cold)[0]) / float(som_total(warm)[0]),
                           10.0)

    def test_out_of_range_transfer_fraction_raises(self):
        """codex A2 follow-up: analytic_slow_pool_equilibrium is a public
        entry point called DIRECTLY by run_lmip.py (bypassing
        step_carbon_differland's guard), so it must independently validate
        f_active_to_slow/f_slow_to_passive to [0, 1]."""
        with self.assertRaises(ValueError):
            _solve(_state(), _fx(), f_as=1.5)
        with self.assertRaises(ValueError):
            _solve(_state(), _fx(), f_as=-0.1)
        with self.assertRaises(ValueError):
            _solve(_state(), _fx(), f_sp=1.5)
        with self.assertRaises(ValueError):
            _solve(_state(), _fx(), f_sp=-0.1)


def _const_diag(ncol, **over):
    """A stationary CarbonDiagnostics (per-day rates) with closed allocation.

    Defaults: a_wood=10, wood_litter=5, lit_to_som=20 with per-pool SOM losses
    som_active_loss=10, som_slow_loss=5, som_passive_loss=3 so the analytic
    forward-substitution cascade reset is analytically checkable; allocation
    closes (a_fol+a_lab+a_root+a_wood == max(npp,0)) with npp=10, r_auto=5.
    """
    z = lambda v: jnp.full((ncol,), float(v))
    d = dict(gpp=15.0, npp=10.0, r_maint=3.0, r_growth=2.0, r_auto=5.0,
             r_het_lit=0.0, r_het_som=10.0, r_het_cwd=0.0, r_het=10.0, nee=0.0,
             unmet_npp_deficit=0.0, a_fol=0.0, a_lab=0.0, a_root=0.0, a_wood=10.0,
             lab_release=0.0, leaf_litter=0.0, root_litter=0.0, wood_litter=5.0,
             wood_to_som=1.5, lit_to_som=20.0,
             som_active_loss=10.0, som_slow_loss=5.0, som_passive_loss=3.0,
             lai=4.0)
    d.update(over)
    return CarbonDiagnostics(**{k: z(v) for k, v in d.items()})


class TestRunSemiAnalyticSpinup(unittest.TestCase):
    """Plumbing test for the shared 3-phase driver with a frozen-carbon toy
    'model' (constant stationary fluxes), so the analytic reset + annual
    accumulation + mass-balance NEE are exactly checkable without the land
    model."""

    def _run(self, ncol, remat=False):
        # A clean model year: steps_per_year * dt == 365 days.
        steps_per_year = 4
        dt = 365.0 * 86400.0 / steps_per_year
        carbon0 = CarbonState(
            C_lab=jnp.full((ncol,), 100.0), C_fol=jnp.full((ncol,), 200.0),
            C_root=jnp.full((ncol,), 300.0), C_wood=jnp.full((ncol,), 1000.0),
            C_lit=jnp.full((ncol,), 400.0),
            C_som_active=jnp.full((ncol,), 5000.0),
            C_som_slow=jnp.full((ncol,), 2000.0),
            C_som_passive=jnp.full((ncol,), 4000.0))
        state0 = jnp.zeros((ncol,))  # opaque dummy state

        def forcing_fn(doy, hour):
            return None  # the toy step ignores forcing

        def step_fn(state, carbon, forcing, doy):
            # Frozen carbon (returned unchanged) + stationary diagnostics.
            return state, carbon, _const_diag(ncol)

        return run_semi_analytic_spinup(
            step_fn, state0, carbon0, forcing_fn,
            n_spinup=3, n_verify=2, steps_per_year=steps_per_year, dt=dt,
            cwd_humification_eff=0.3, f_active_to_slow=0.3, f_slow_to_passive=0.3,
            remat=remat)

    def test_analytic_reset_and_shapes(self):
        _fs, fc, annual, reset_fluxes = self._run(ncol=1)
        # Wood: C_wood_eq = 1000 * a_wood/wood_litter = 1000 * 10/5 = 2000.
        npt.assert_allclose(np.asarray(fc.C_wood), 2000.0, rtol=1e-6)
        # Active: i_active = lit_to_som + cwd*a_wood = 20 + 0.3*10 = 23; loss 10.
        #         C_som_active_eq = 5000 * 23/10 = 11500.
        npt.assert_allclose(np.asarray(fc.C_som_active), 11500.0, rtol=1e-6)
        # Slow: i_slow = f_as*i_active = 0.3*23 = 6.9; loss 5.
        #       C_som_slow_eq = 2000 * 6.9/5 = 2760.
        npt.assert_allclose(np.asarray(fc.C_som_slow), 2760.0, rtol=1e-6)
        # Passive: i_passive = f_sp*i_slow = 0.3*6.9 = 2.07; loss 3.
        #          C_som_passive_eq = 4000 * 2.07/3 = 2760.
        npt.assert_allclose(np.asarray(fc.C_som_passive), 2760.0, rtol=1e-6)
        # Fast pools untouched by the reset and frozen by the toy step.
        for f, v in (("C_lab", 100.0), ("C_fol", 200.0),
                     ("C_root", 300.0), ("C_lit", 400.0)):
            npt.assert_allclose(np.asarray(getattr(fc, f)), v, rtol=1e-6)
        # 4th return: the LAST-TRANSIENT-year SlowPoolFluxes the reset consumed
        # (constant stationary toy diagnostics) -- exposed so the fast-analytic SOC
        # precompute records the reset-consistent lit_to_som/a_wood (not a shifted
        # post-verify phase of the wood pool).  These are ANNUAL totals
        # ``sum_t(rate*dt_days)`` = per-day toy rate * 365 (one clean model year), so
        # a_wood=10*365, wood_litter=5*365, lit_to_som=20*365, and the SOM losses
        # 10/5/3 * 365 -- the SAME fluxes ``analytic_slow_pool_equilibrium`` consumed.
        npt.assert_allclose(np.asarray(reset_fluxes.a_wood), 10.0 * 365.0, rtol=1e-6)
        npt.assert_allclose(np.asarray(reset_fluxes.wood_litter), 5.0 * 365.0, rtol=1e-6)
        npt.assert_allclose(np.asarray(reset_fluxes.lit_to_som), 20.0 * 365.0, rtol=1e-6)
        npt.assert_allclose(np.asarray(reset_fluxes.som_active_loss), 10.0 * 365.0, rtol=1e-6)
        npt.assert_allclose(np.asarray(reset_fluxes.som_slow_loss), 5.0 * 365.0, rtol=1e-6)
        npt.assert_allclose(np.asarray(reset_fluxes.som_passive_loss), 3.0 * 365.0, rtol=1e-6)
        # Per-verify-year annual dict, shape (n_verify, ncol).
        for key in ("gpp", "npp", "nee_model", "alloc_resid", "lai_sum",
                    "lai_max", "nsteps", "C_som_active"):
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
        _fs, fc, annual, _reset = self._run(ncol=3)
        self.assertEqual(fc.C_som_active.shape, (3,))
        npt.assert_allclose(np.asarray(fc.C_wood), 2000.0, rtol=1e-6)
        self.assertEqual(np.asarray(annual["gpp"]).shape, (2, 3))

    def test_remat_is_numerically_identical(self):
        # The opt-in `remat` kwarg (jax.checkpoint on the per-year body) only
        # changes the store-vs-recompute schedule for reverse-mode AD, never the
        # forward values -- so remat=True must reproduce remat=False exactly.
        _fs0, fc0, annual0, _r0 = self._run(ncol=2, remat=False)
        _fs1, fc1, annual1, _r1 = self._run(ncol=2, remat=True)
        for field in CarbonState._fields:
            npt.assert_allclose(
                np.asarray(getattr(fc1, field)),
                np.asarray(getattr(fc0, field)), rtol=0, atol=0, err_msg=field)
        for key in ("gpp", "npp", "nee_model", "C_som_active"):
            npt.assert_allclose(
                np.asarray(annual1[key]), np.asarray(annual0[key]),
                rtol=0, atol=0, err_msg=key)


class TestRunSemiAnalyticSpinupGuards(unittest.TestCase):
    """F7: fail early on degenerate run controls (raises at entry, before any
    scan / JIT), matching the repo's dispatch-hardening discipline."""

    def _kwargs(self, **over):
        ncol = 1
        steps_per_year = 4
        dt = 365.0 * 86400.0 / steps_per_year
        carbon0 = CarbonState(**{f: jnp.full((ncol,), 1.0)
                                 for f in CarbonState._fields})
        kw = dict(
            step_fn=lambda s, c, f, d: (s, c, _const_diag(ncol)),
            state0=jnp.zeros((ncol,)), carbon0=carbon0,
            forcing_fn=lambda doy, hour: None,
            n_spinup=3, n_verify=2, steps_per_year=steps_per_year, dt=dt,
            cwd_humification_eff=0.3, f_active_to_slow=0.3, f_slow_to_passive=0.3)
        kw.update(over)
        return kw

    def test_zero_spinup_raises(self):
        with self.assertRaises(ValueError):
            run_semi_analytic_spinup(**self._kwargs(n_spinup=0))

    def test_one_verify_year_raises(self):
        with self.assertRaises(ValueError):
            run_semi_analytic_spinup(**self._kwargs(n_verify=1))

    def test_zero_steps_per_year_raises(self):
        with self.assertRaises(ValueError):
            run_semi_analytic_spinup(**self._kwargs(steps_per_year=0))

    def test_out_of_range_transfer_fraction_raises(self):
        """codex A2 follow-up: this guard must fire BEFORE the expensive
        Phase-1 transient scan, not only inside
        analytic_slow_pool_equilibrium after the scan has already run."""
        with self.assertRaises(ValueError):
            run_semi_analytic_spinup(**self._kwargs(f_active_to_slow=1.5))
        with self.assertRaises(ValueError):
            run_semi_analytic_spinup(**self._kwargs(f_slow_to_passive=-0.1))


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
            C_lit=jnp.full((ncol,), 400.0),
            C_som_active=jnp.full((ncol,), 5000.0),
            C_som_slow=jnp.zeros((ncol,)), C_som_passive=jnp.zeros((ncol,)))
        state0 = jnp.zeros((ncol,))  # opaque dummy state

        def forcing_fn(doy, hour):
            return None

        def step_fn(state, carbon, forcing, doy):
            return state, carbon, _const_diag(ncol)  # carbon frozen

        n_years = 3
        out = integrate_annual_pools(
            step_fn, state0, carbon0, forcing_fn,
            n_years=n_years, steps_per_year=steps_per_year, dt=dt)
        for p in CarbonState._fields:
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
