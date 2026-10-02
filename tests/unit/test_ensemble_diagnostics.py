"""Tests for ensemble diagnostics: CRPS, rank histogram, spread tracking.

Validates:
- ensemble_crps produces correct scores for known distributions
- ensemble_rank_histogram is flat for well-calibrated ensembles
- EnsembleDiagnosticCollector tracks spread timeseries
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.parallel.ensemble import (
    ensemble_crps,
    ensemble_rank_histogram,
    ensemble_spread,
    ensemble_mean,
    ensemble_std,
)


class TestEnsembleCRPS:
    """CRPS computation for ensemble forecasts."""

    def test_perfect_forecast_crps_zero(self):
        """Identical ensemble members matching obs → CRPS ≈ 0."""
        obs = jnp.ones((4, 4))
        ens = jnp.ones((10, 4, 4))  # 10 members, all = 1
        crps = ensemble_crps(ens, obs)
        np.testing.assert_allclose(float(crps), 0.0, atol=1e-6)

    def test_biased_forecast_positive_crps(self):
        """Ensemble consistently above obs → positive CRPS."""
        obs = jnp.zeros((4, 4))
        ens = jnp.ones((10, 4, 4))  # all members = 1, obs = 0
        crps = ensemble_crps(ens, obs)
        assert float(crps) > 0.0

    def test_spread_reduces_crps(self):
        """A spread ensemble around the obs should have lower CRPS than biased."""
        obs = jnp.full((4, 4), 5.0)
        # Biased: all at 6.0
        ens_biased = jnp.full((10, 4, 4), 6.0)
        # Spread: centered on 5.0
        key = jax.random.PRNGKey(0)
        ens_spread = 5.0 + 0.5 * jax.random.normal(key, (10, 4, 4))
        crps_biased = float(ensemble_crps(ens_biased, obs))
        crps_spread = float(ensemble_crps(ens_spread, obs))
        assert crps_spread < crps_biased

    def test_crps_scalar_output(self):
        """CRPS should return a scalar (spatially averaged)."""
        obs = jnp.zeros((6, 8, 8))
        ens = jnp.ones((5, 6, 8, 8))
        crps = ensemble_crps(ens, obs)
        assert crps.shape == ()


class TestRankHistogram:
    """Rank histogram (Talagrand diagram)."""

    def test_correct_bin_count(self):
        """N members → N+1 bins."""
        n = 10
        ens = jax.random.normal(jax.random.PRNGKey(0), (n, 100))
        obs = jnp.zeros(100)
        hist = ensemble_rank_histogram(ens, obs)
        assert hist.shape == (n + 1,)
        assert int(jnp.sum(hist)) == 100  # total = n_gridpoints

    def test_well_calibrated_is_flat(self):
        """If obs is drawn from the same distribution as members, histogram is ~flat."""
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        n_members = 20
        n_points = 10000
        ens = jax.random.normal(k1, (n_members, n_points))
        obs = jax.random.normal(k2, (n_points,))

        hist = ensemble_rank_histogram(ens, obs)
        expected_per_bin = n_points / (n_members + 1)
        # Allow 30% deviation from uniform
        assert jnp.all(hist > expected_per_bin * 0.5)
        assert jnp.all(hist < expected_per_bin * 1.5)

    def test_biased_ensemble_not_flat(self):
        """If ensemble is biased high, obs ranks should cluster at 0."""
        n_members = 10
        n_points = 1000
        ens = jnp.ones((n_members, n_points)) * 10.0  # biased high
        obs = jnp.zeros(n_points)  # obs = 0
        hist = ensemble_rank_histogram(ens, obs)
        # All obs should be rank 0 (below all members)
        assert int(hist[0]) == n_points


class TestEnsembleDiagnosticCollector:
    """EnsembleDiagnosticCollector tracks spread and produces checkpoints."""

    def test_collect_spread(self):
        """Spread timeseries accumulates correctly."""
        from legoesm.driver.diagnostics import DiagnosticCollector, EnsembleDiagnosticCollector
        from legoesm.driver.compiled_segments import SegmentCarry

        nlev = 3
        base = DiagnosticCollector(
            nlev=nlev,
            sigma_full=np.linspace(0.1, 1.0, nlev),
            dsigma=np.full(nlev, 1.0/nlev),
        )
        ens_diag = EnsembleDiagnosticCollector(base, n_members=3)

        # Create a batched carry (3 members, small grid)
        s3 = (3, 6, 4, 4, nlev)
        s2 = (3, 6, 4, 4)
        carry = SegmentCarry(
            u=jnp.ones(s3) * jnp.array([1.0, 2.0, 3.0]).reshape(3,1,1,1,1),
            v=jnp.zeros(s3),
            T=jnp.full(s3, 280.0) + jnp.array([0, 1, 2]).reshape(3,1,1,1,1),
            p_s=jnp.full(s2, 101325.0),
            phis=jnp.zeros(s2),
            q_v=jnp.ones(s3) * 0.01,
            q_c=jnp.zeros(s3),
            q_r=jnp.zeros(s3),
            conv_prog=jnp.zeros((3, 6 * 4 * 4)),
            held_dT_rad=jnp.zeros(s3),
            held_sw_net_sfc=jnp.zeros(s2),
            held_lw_net_sfc=jnp.zeros(s2),
            held_sw_up_toa=jnp.zeros(s2),
            held_lw_up_toa=jnp.zeros(s2),
            held_sw_up_toa_clr=jnp.zeros(s2),
            held_lw_up_toa_clr=jnp.zeros(s2),
            held_sw_down_toa=jnp.zeros(s2),
            step_index=jnp.zeros(3, dtype=jnp.int32),  # batched scalar
            target_moisture=jnp.zeros(3),
            target_mass=jnp.zeros(3),
            max_cfl=jnp.zeros(3),
            precip_accum=jnp.zeros(s2),
            shflx_accum=jnp.zeros(s2),
            lhflx_accum=jnp.zeros(s2),
            evap_accum=jnp.zeros(s2),
            sw_up_toa_accum=jnp.zeros(s2),
            lw_up_toa_accum=jnp.zeros(s2),
            sw_up_toa_clr_accum=jnp.zeros(s2),
            lw_up_toa_clr_accum=jnp.zeros(s2),
            sw_down_toa_accum=jnp.zeros(s2),
            sw_net_sfc_accum=jnp.zeros(s2),
            lw_net_sfc_accum=jnp.zeros(s2),
            t_low_accum=jnp.zeros(s2),
            T_land=jnp.zeros(s2),
        )

        mean_carry = ensemble_mean(carry)

        info = ens_diag.collect_ensemble(5.0, carry, mean_carry)
        assert info['spread_T'] > 0.0  # members differ in T
        assert info['spread_u'] > 0.0  # members differ in u
        assert len(ens_diag.spread_times) == 1

        # Second collection
        ens_diag.collect_ensemble(10.0, carry, mean_carry)
        assert len(ens_diag.spread_times) == 2

    def test_save_spread(self, tmp_path):
        """Spread timeseries saves to npz."""
        from legoesm.driver.diagnostics import DiagnosticCollector, EnsembleDiagnosticCollector

        nlev = 3
        base = DiagnosticCollector(
            nlev=nlev,
            sigma_full=np.linspace(0.1, 1.0, nlev),
            dsigma=np.full(nlev, 1.0/nlev),
        )
        ens_diag = EnsembleDiagnosticCollector(base, n_members=2)

        # Add fake spread data
        ens_diag.spread_times = [5.0, 10.0]
        ens_diag.spread_T = [0.5, 0.6]
        ens_diag.spread_u = [1.0, 1.1]
        ens_diag.spread_ps = [10.0, 12.0]

        ens_diag.save(tmp_path)

        # Verify file
        data = np.load(tmp_path / "ensemble_spread.npz")
        assert len(data['days']) == 2
        np.testing.assert_allclose(data['spread_T'], [0.5, 0.6])
