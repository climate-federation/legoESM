"""Tests for the canopy leaf-level stomatal primitives.

Covers ``legoesm.land.stomata`` — the explicit-parameter
``ball_berry_gs`` / ``medlyn_gs`` used by the two-leaf canopy energy
balance, which pass per-leaf-class slope/intercept ``(m, b0)`` rather
than a scalar ``StomataConfig``.

The config-based stomatal roster (``jarvis_gs``,
``coupled_farquhar_stomata``, ``compute_stomatal_beta``, and the
config-signature ``ball_berry_gs`` / ``medlyn_gs``) lives in
``legoesm.land.stomata`` and is tested in ``test_stomata.py``.
``StomataConfig`` is imported here only to source representative default
slope/intercept values, which are then passed explicitly.
"""

import unittest

import jax.numpy as jnp

from legoesm.land.stomata import ball_berry_gs, medlyn_gs
from legoesm.land.stomata import StomataConfig


class TestBallBerry(unittest.TestCase):
    """Ball-Berry stomatal conductance — explicit m / b0 signature."""

    def setUp(self):
        self.cfg = StomataConfig()
        # Convenience aliases — both passed explicitly, not via config.
        self.m = self.cfg.g1_bb
        self.b0 = self.cfg.g0

    def test_minimum_conductance(self):
        """gs >= b0 always."""
        gs = ball_berry_gs(
            jnp.array(-5.0), jnp.array(0.8), jnp.array(400.0),
            self.m, self.b0)
        self.assertAlmostEqual(float(gs), self.b0, places=5)

    def test_increases_with_A(self):
        gs_low = ball_berry_gs(
            jnp.array(5.0), jnp.array(0.8), jnp.array(400.0),
            self.m, self.b0)
        gs_high = ball_berry_gs(
            jnp.array(20.0), jnp.array(0.8), jnp.array(400.0),
            self.m, self.b0)
        self.assertGreater(float(gs_high), float(gs_low))

    def test_increases_with_RH(self):
        gs_dry = ball_berry_gs(
            jnp.array(10.0), jnp.array(0.3), jnp.array(400.0),
            self.m, self.b0)
        gs_wet = ball_berry_gs(
            jnp.array(10.0), jnp.array(0.9), jnp.array(400.0),
            self.m, self.b0)
        self.assertGreater(float(gs_wet), float(gs_dry))

    def test_decreases_with_Cs(self):
        gs_low_co2 = ball_berry_gs(
            jnp.array(10.0), jnp.array(0.8), jnp.array(200.0),
            self.m, self.b0)
        gs_high_co2 = ball_berry_gs(
            jnp.array(10.0), jnp.array(0.8), jnp.array(800.0),
            self.m, self.b0)
        self.assertGreater(float(gs_low_co2), float(gs_high_co2))

    def test_per_column_m_b0(self):
        """Ball-Berry accepts per-column m / b0 arrays (needed by canopy)."""
        An = jnp.array([5.0, 10.0, 15.0])
        RH = jnp.full(3, 0.8)
        Cs = jnp.full(3, 400.0)
        m = jnp.array([9.0, 7.0, 4.0])
        b0 = jnp.array([0.01, 0.02, 0.04])
        gs = ball_berry_gs(An, RH, Cs, m, b0)
        self.assertEqual(gs.shape, (3,))
        self.assertTrue(jnp.all(gs >= b0 - 1e-12))


class TestMedlyn(unittest.TestCase):
    """Medlyn optimal stomatal conductance — explicit g1 / g0 signature."""

    def setUp(self):
        self.cfg = StomataConfig()
        self.g1 = self.cfg.g1_med
        self.g0 = self.cfg.g0

    def test_minimum_conductance(self):
        gs = medlyn_gs(
            jnp.array(-5.0), jnp.array(1.0), jnp.array(400.0),
            self.g1, self.g0)
        self.assertAlmostEqual(float(gs), self.g0, places=5)

    def test_increases_with_A(self):
        gs_low = medlyn_gs(
            jnp.array(5.0), jnp.array(1.0), jnp.array(400.0),
            self.g1, self.g0)
        gs_high = medlyn_gs(
            jnp.array(20.0), jnp.array(1.0), jnp.array(400.0),
            self.g1, self.g0)
        self.assertGreater(float(gs_high), float(gs_low))

    def test_decreases_with_VPD(self):
        gs_wet = medlyn_gs(
            jnp.array(10.0), jnp.array(0.5), jnp.array(400.0),
            self.g1, self.g0)
        gs_dry = medlyn_gs(
            jnp.array(10.0), jnp.array(3.0), jnp.array(400.0),
            self.g1, self.g0)
        self.assertGreater(float(gs_wet), float(gs_dry))

    def test_reasonable_magnitude(self):
        gs = medlyn_gs(
            jnp.array(15.0), jnp.array(1.0), jnp.array(400.0),
            self.g1, self.g0)
        self.assertGreater(float(gs), self.g0)
        self.assertLess(float(gs), 1.0)


class TestTwoLeafSelectorDispatch(unittest.TestCase):
    """The two-leaf ``_compute_gs_and_ci`` selector must raise on an unknown
    ``stomatal_model`` (dispatch-hardening), not silently fall back to
    Ball-Berry — mirroring the big-leaf ``solve_coupled_farquhar_ci`` guard."""

    def _args(self):
        return dict(
            An=jnp.array(10.0), RH_c=jnp.array(0.7), VPD_c=jnp.array(1000.0),
            Ca=jnp.array(400.0), Tf=jnp.array(300.0), Ps=jnp.array(101325.0),
            m=jnp.array(9.0), b0=jnp.array(0.01))

    def test_valid_models_run(self):
        from legoesm.land.canopy.energy_balance import _compute_gs_and_ci
        for model in ("ball_berry", "medlyn"):
            rs, gs, Ci = _compute_gs_and_ci(**self._args(), stomatal_model=model)
            self.assertTrue(jnp.isfinite(gs))

    def test_unknown_model_raises(self):
        from legoesm.land.canopy.energy_balance import _compute_gs_and_ci
        with self.assertRaises(ValueError):
            _compute_gs_and_ci(**self._args(), stomatal_model="bal_berry")

    def test_config_validate_rejects_typo_early(self):
        # Fail-early backstop: CanopyConfig.validate() rejects a typo'd
        # stomatal_model at setup (before any jitted canopy solve).
        from legoesm.land.canopy.config import CanopyConfig
        CanopyConfig(stomatal_model="ball_berry").validate()
        CanopyConfig(stomatal_model="medlyn").validate()
        with self.assertRaises(ValueError):
            CanopyConfig(stomatal_model="bal_berry").validate()


if __name__ == "__main__":
    unittest.main()
