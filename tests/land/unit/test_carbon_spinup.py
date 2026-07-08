"""Tests for the semi-analytic soil-carbon spin-up (legoesm.land.carbon.spinup)."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy.testing as npt

from legoesm.land.carbon.config import CarbonState
from legoesm.land.carbon.spinup import (
    SlowPoolFluxes,
    analytic_slow_pool_equilibrium,
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


if __name__ == "__main__":
    unittest.main()
