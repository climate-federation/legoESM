"""Tests for the moisture budget tracker.

Validates:
- Column water computation is physically correct
- The CLOSURE residual E − P − dW/dt is ~0 for a conserving system
- Precipitation is correctly converted to mm/day
- Evaporation (from lhflx) is correctly converted to mm/day
- A synthetic vapor sink is DETECTED (positive residual) — the
  non-vacuous tripwire self-test; the pre-2026-07-22 residual
  (dW/dt + P, no E) provably could not see it
- Flush/clear works for long-run memory management
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.diagnostics.energy_budget import MoistureBudgetTracker


def _lhflx_for_evap_mm_day(shape, evap_mm_day):
    """lhflx [W/m²] whose implied E = evap_mm_day (E = lhflx/L_v)."""
    return jnp.full(shape, evap_mm_day / 86400.0 * constants.L_v)


class TestMoistureBudgetTracker:

    def _make_tracker_with_data(self):
        """Create tracker and feed it 3 time steps of synthetic data.

        Drying step balanced by precip with E = 0 — a closed sequence.
        """
        tracker = MoistureBudgetTracker()
        nlev = 5
        dsigma = jnp.full(nlev, 0.2)
        no_evap = jnp.zeros((6, 4, 4))

        # Step 1: t=0s
        q_v_1 = jnp.full((6, 4, 4, nlev), 0.01)   # 10 g/kg
        p_s_1 = jnp.full((6, 4, 4), 100000.0)       # 1000 hPa
        precip_1 = jnp.zeros((6, 4, 4))              # no precip
        tracker.update(q_v_1, p_s_1, dsigma, precip_1, no_evap,
                       elapsed_seconds=0.0)

        # Step 2: t=86400s (1 day), slightly drier
        q_v_2 = jnp.full((6, 4, 4, nlev), 0.009)   # 9 g/kg
        p_s_2 = jnp.full((6, 4, 4), 100000.0)
        # Precip rate that accounts for the moisture loss
        # dW = (0.009 - 0.01) * 100000 * 1.0 / 9.81 = -101.94 kg/m²
        # over 86400s → loss rate = -1.18e-3 kg/m²/s
        precip_2 = jnp.full((6, 4, 4), 1.18e-3)     # matching precip
        tracker.update(q_v_2, p_s_2, dsigma, precip_2, no_evap,
                       elapsed_seconds=86400.0)

        # Step 3: t=172800s (2 days)
        q_v_3 = jnp.full((6, 4, 4, nlev), 0.009)
        precip_3 = jnp.zeros((6, 4, 4))
        tracker.update(q_v_3, p_s_2, dsigma, precip_3, no_evap,
                       elapsed_seconds=172800.0)

        return tracker

    def test_column_water_physical(self):
        """Column water = (1/g) * sum(q * p_s * dsigma)."""
        tracker = MoistureBudgetTracker()
        nlev = 5
        dsigma = jnp.full(nlev, 0.2)
        q_v = jnp.full((6, 4, 4, nlev), 0.01)
        p_s = jnp.full((6, 4, 4), 100000.0)
        precip = jnp.zeros((6, 4, 4))

        budget = tracker.update(q_v, p_s, dsigma, precip,
                                jnp.zeros((6, 4, 4)), 0.0)

        # Expected: W = q * p_s * sum(dsigma) / g = 0.01 * 100000 * 1.0 / g
        expected_W = 0.01 * 100000.0 / constants.g
        np.testing.assert_allclose(budget.column_water, expected_W, rtol=1e-4)

    def test_precip_in_mm_per_day(self):
        """Precipitation rate is correctly converted to mm/day."""
        tracker = MoistureBudgetTracker()
        nlev = 3
        dsigma = jnp.full(nlev, 1.0/nlev)
        q_v = jnp.full((2, 2, nlev), 0.01)
        p_s = jnp.full((2, 2), 100000.0)

        # 1 mm/day = 1 kg/m²/day = 1/86400 kg/m²/s ≈ 1.157e-5
        precip = jnp.full((2, 2), 1.0 / 86400.0)  # 1 mm/day

        budget = tracker.update(q_v, p_s, dsigma, precip,
                                jnp.zeros((2, 2)), 0.0)
        np.testing.assert_allclose(budget.precip_rate, 1.0, rtol=1e-4)

    def test_evap_in_mm_per_day(self):
        """lhflx [W/m²] converts to evap mm/day via L_v."""
        tracker = MoistureBudgetTracker()
        nlev = 3
        dsigma = jnp.full(nlev, 1.0/nlev)
        q_v = jnp.full((2, 2, nlev), 0.01)
        p_s = jnp.full((2, 2), 100000.0)
        precip = jnp.zeros((2, 2))

        budget = tracker.update(q_v, p_s, dsigma, precip,
                                _lhflx_for_evap_mm_day((2, 2), 2.5), 0.0)
        np.testing.assert_allclose(budget.evap_rate, 2.5, rtol=1e-6)

    def test_tendency_computed(self):
        """dW/dt is computed from successive column water values."""
        tracker = self._make_tracker_with_data()

        # After 3 updates, we should have 3 entries
        assert len(tracker.times) == 3
        assert len(tracker.dW_dt) == 3
        # First dW/dt is 0 (no previous), second should be negative (drying)
        assert tracker.dW_dt[0] == 0.0
        assert tracker.dW_dt[1] < 0.0  # q decreased

    def test_summary_format(self):
        """Summary produces readable output."""
        tracker = self._make_tracker_with_data()
        s = tracker.summary()
        assert "Moisture Budget Summary" in s
        assert "CWV" in s
        assert "Precip" in s
        assert "Evap" in s
        assert "PASS" in s or "CHECK" in s

    def test_flush_clears_lists(self):
        """flush_to_lists returns data (incl. evap_rate) and clears state."""
        tracker = self._make_tracker_with_data()
        assert len(tracker.times) == 3

        data = tracker.flush_to_lists()
        assert len(data["times"]) == 3
        assert len(data["evap_rate"]) == 3
        assert len(tracker.times) == 0  # cleared

    def test_conserving_system_small_residual(self):
        """A closed system (E = P = dW/dt = 0) has ~0 residual."""
        tracker = MoistureBudgetTracker()
        nlev = 5
        dsigma = jnp.full(nlev, 0.2)
        p_s = jnp.full((4, 4), 100000.0)

        q_v = jnp.full((4, 4, nlev), 0.01)
        precip = jnp.zeros((4, 4))
        no_evap = jnp.zeros((4, 4))

        for i in range(5):
            tracker.update(q_v, p_s, dsigma, precip, no_evap,
                           float(i * 86400))

        res = np.array(tracker.residual[1:])
        np.testing.assert_allclose(res, 0.0, atol=1e-6)

    def test_evap_balances_precip_zero_residual(self):
        """Steady state with E = P = 2 mm/day closes: residual ~0."""
        tracker = MoistureBudgetTracker()
        nlev = 5
        dsigma = jnp.full(nlev, 0.2)
        p_s = jnp.full((4, 4), 100000.0)
        q_v = jnp.full((4, 4, nlev), 0.01)          # constant → dW/dt = 0
        precip = jnp.full((4, 4), 2.0 / 86400.0)    # 2 mm/day sink
        lhflx = _lhflx_for_evap_mm_day((4, 4), 2.0)  # 2 mm/day source

        for i in range(4):
            tracker.update(q_v, p_s, dsigma, precip, lhflx, float(i * 86400))

        res = np.array(tracker.residual[1:])
        np.testing.assert_allclose(res, 0.0, atol=1e-4)

    def test_synthetic_vapor_sink_detected(self):
        """Non-vacuous tripwire: a 1.4 mm/day hidden vapor sink shows as
        residual ≈ +1.4 (E enters, storage flat, no precip — the 2-yr AMIP
        pilot signature).  The pre-fix residual (dW/dt + P) is 0 on this
        exact sequence, i.e. provably blind to it."""
        tracker = MoistureBudgetTracker()
        nlev = 5
        dsigma = jnp.full(nlev, 0.2)
        p_s = jnp.full((4, 4), 100000.0)
        q_v = jnp.full((4, 4, nlev), 0.01)          # storage FLAT
        precip = jnp.zeros((4, 4))                   # no precip
        lhflx = _lhflx_for_evap_mm_day((4, 4), 1.4)  # E = 1.4 mm/day in

        for i in range(4):
            tracker.update(q_v, p_s, dsigma, precip, lhflx, float(i * 86400))

        res = np.array(tracker.residual[1:])
        np.testing.assert_allclose(res, 1.4, rtol=1e-4)
        # The OLD definition (dW/dt + P) evaluates to exactly 0 here —
        # assert that fact so the reason for the redefinition stays
        # machine-documented.
        old_style = np.array(tracker.dW_dt[1:]) * 86400.0 + np.array(
            tracker.precip_rate[1:])
        np.testing.assert_allclose(old_style, 0.0, atol=1e-6)
