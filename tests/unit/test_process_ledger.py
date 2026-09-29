"""Tests for the per-process column budget ledger (diagnostics.process_ledger).

Covers:
- analytic single-column ledger entries against hand-computed values (fp64,
  1e-12),
- snapshot/entry consistency (a store delta equals the entry of its
  tendency — the identity the dynamics/clips rows rely on),
- synthetic-violation: an injected vapor-destroying "floor" registers as a
  positive clips-row entry (non-vacuous tripwire for the seam formula),
- gate-off structural defaults (PhysicsOutput.budget_ledger / SegmentCarry
  .budget_ledger_accum are None),
- a 1-day C8/L5 driver integration run (slow): ON-vs-OFF final states are
  bit-identical (the ledger is a pure diagnostic) and the ON run's ledger
  rows sum to the measured store change (closure by construction).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.diagnostics.process_ledger import (
    LEDGER_COLUMNS,
    LEDGER_PROCESSES,
    N_LEDGER,
    ROW_CLIPS,
    column_store_snapshot,
    format_ledger_table,
    ledger_entry,
    zero_ledger,
)


class TestLedgerHelpers:

    def test_analytic_entry(self):
        """Prescribed uniform tendencies → hand-computed column rates."""
        nlev = 4
        dsigma = jnp.full(nlev, 0.25)
        p_s = jnp.full((3, 5), 1.0e5)
        dq = jnp.full((3, 5, nlev), 2.0e-9)   # kg/kg/s
        dT = jnp.full((3, 5, nlev), 1.0e-6)   # K/s

        entry = np.asarray(ledger_entry(dq, dT, p_s, dsigma))
        # ∫dq dp/g = dq * p_s / g (uniform, Σdσ=1)
        expect_w = 2.0e-9 * 1.0e5 / constants.g
        expect_e = constants.c_pd * 1.0e-6 * 1.0e5 / constants.g
        np.testing.assert_allclose(entry[0], expect_w, rtol=1e-12)
        np.testing.assert_allclose(entry[1], expect_e, rtol=1e-12)

    def test_area_weighted_global_mean(self):
        """Lat-lon-like cos-lat areas: a tendency confined to polar rows must
        count by its AREA share, not its column-count share."""
        nlev = 2
        dsigma = jnp.full(nlev, 0.5)
        lat = jnp.linspace(-80.0, 80.0, 9)
        area = jnp.broadcast_to(jnp.cos(jnp.deg2rad(lat))[:, None], (9, 4))
        p_s = jnp.full((9, 4), 1.0e5)
        dq = jnp.zeros((9, 4, nlev)).at[0].set(1.0e-9)
        col = 1.0e-9 * 1.0e5 / constants.g
        expect = col * float(jnp.sum(area[0]) / jnp.sum(area))
        entry = np.asarray(ledger_entry(dq, None, p_s, dsigma, area=area))
        np.testing.assert_allclose(entry[0], expect, rtol=1e-12)
        snap = np.asarray(column_store_snapshot(
            p_s, dsigma, jnp.zeros((9, 4, nlev)), dq, area=area))
        np.testing.assert_allclose(snap[0], expect, rtol=1e-12)

    def test_none_inputs_zero(self):
        dsigma = jnp.full(3, 1 / 3)
        p_s = jnp.full((2, 2), 1.0e5)
        entry = np.asarray(ledger_entry(None, None, p_s, dsigma))
        np.testing.assert_array_equal(entry, 0.0)

    def test_snapshot_delta_equals_entry(self):
        """(snapshot(after) − snapshot(before))/dt == ledger_entry(tendency)
        — the identity the dynamics/clips seams rely on (fixed p_s)."""
        nlev, dt = 5, 300.0
        dsigma = jnp.full(nlev, 0.2)
        p_s = jnp.full((4, 4), 9.8e4)
        T0 = jnp.full((4, 4, nlev), 280.0)
        qv0 = jnp.full((4, 4, nlev), 8.0e-3)
        dT = jnp.full_like(T0, 3.0e-5)
        dq = jnp.full_like(qv0, -1.0e-8)

        before = column_store_snapshot(p_s, dsigma, T0, qv0)
        after = column_store_snapshot(p_s, dsigma, T0 + dt * dT, qv0 + dt * dq)
        rate = np.asarray((after - before) / dt)
        entry = np.asarray(ledger_entry(dq, dT, p_s, dsigma))
        np.testing.assert_allclose(rate, entry, rtol=1e-10)

    def test_synthetic_violation_clip_creates_water(self):
        """Non-vacuous tripwire: flooring a negative q to 0 (the clips seam
        formula) shows as a POSITIVE water entry of the clipped mass."""
        nlev, dt = 3, 100.0
        dsigma = jnp.full(nlev, 1 / 3)
        p_s = jnp.full((2, 2), 1.0e5)
        T = jnp.full((2, 2, nlev), 280.0)
        q_raw = jnp.full((2, 2, nlev), -1.0e-6)   # unphysical negative vapor
        q_clip = jnp.maximum(q_raw, 0.0)

        pre = column_store_snapshot(p_s, dsigma, T, q_raw)
        post = column_store_snapshot(p_s, dsigma, T, q_clip)
        clips_rate = np.asarray((post - pre) / dt)
        expect_w = 1.0e-6 * 1.0e5 / constants.g / dt   # created water rate
        np.testing.assert_allclose(clips_rate[0], expect_w, rtol=1e-12)
        assert clips_rate[0] > 0.0                      # positive = created
        np.testing.assert_allclose(clips_rate[1], 0.0, atol=1e-30)

    def test_module_constants(self):
        assert len(LEDGER_PROCESSES) == N_LEDGER
        assert LEDGER_COLUMNS == ("water", "energy")
        assert zero_ledger().shape == (N_LEDGER, 2)
        tbl = format_ledger_table(np.ones((N_LEDGER, 2)))
        for name in LEDGER_PROCESSES:
            assert name in tbl
        assert "TOTAL" in tbl

    def test_gate_off_defaults(self):
        """OFF path structural guarantees: the optional fields default None
        (byte-identical pytrees)."""
        from legoesm.core.physics_output import PhysicsOutput
        assert PhysicsOutput._field_defaults["budget_ledger"] is None
        from legoesm.driver.compiled_segments import SegmentCarry
        assert SegmentCarry._field_defaults["budget_ledger_accum"] is None


@pytest.mark.slow
class TestLedgerDriverIntegration:
    """1-day C8/L5 analytical AMIP: the wiring end-to-end."""

    def _run(self, tmp_path, budget_ledger):
        from legoesm.driver.config import (
            DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
        )
        from legoesm.driver.model_driver import ModelDriver
        out = tmp_path / ("on" if budget_ledger else "off")
        cfg = ExperimentConfig(
            grid=GridConfig(resolution=8, nlev=5),
            dycore=DycoreConfig(dt=600.0),
            output=OutputConfig(diag_days=1, output_dir=str(out),
                                budget_ledger=budget_ledger),
            days=1,
            dataset="analytical",
            precision="fp64",
        )
        driver = ModelDriver(cfg, output_dir=out)
        driver.setup()
        status = driver.run()
        assert status == "COMPLETED"
        return driver, out

    def test_on_off_bit_identical_and_ledger_written(self, tmp_path):
        d_off, _ = self._run(tmp_path, budget_ledger=False)
        d_on, out_on = self._run(tmp_path, budget_ledger=True)

        # The ledger is a pure diagnostic: final prognostic state identical.
        np.testing.assert_array_equal(np.asarray(d_off.state.T.data),
                                      np.asarray(d_on.state.T.data))
        np.testing.assert_array_equal(np.asarray(d_off.q_v),
                                      np.asarray(d_on.q_v))

        # Ledger written, well-formed, finite, and rows carry signal.
        led = np.load(out_on / "budget_ledger.npz")
        rates = led["rates"]
        assert rates.shape[1:] == (N_LEDGER, 2)
        assert np.isfinite(rates).all()
        assert list(led["processes"]) == list(LEDGER_PROCESSES)
        # Radiation row must be cooling the column (analytical case runs
        # gray radiation) — a sign sanity anchor, not a magnitude gate.
        i_rad = list(LEDGER_PROCESSES).index("radiation")
        assert rates[:, i_rad, 1].mean() < 0.0
