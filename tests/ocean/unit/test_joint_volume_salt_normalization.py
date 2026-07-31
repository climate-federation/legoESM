"""Joint volume+salt normalization for the LOCAL-salinity freshwater closure.

The fixed-``S_ref`` virtual-salt closure removes salt at the S_ref rate however
little is present, so at a river mouth (S_local -> 0) it drives salinity
negative -- the eORCA1 d90 state carries 111 such cells, min -22.31 PSU, worst
at the Amazon mouth.  NEMO's own ``tra_sbc`` convention multiplies by the LOCAL
surface salinity instead, which decays toward zero and cannot cross it, but that
convention could not be combined with the global freshwater normalization
because a zero-mean FRESHWATER flux is not a zero-mean SALT flux once the
multiplier varies in space.

``joint_volume_salt_virtual_salt_flux`` closes both budgets.  These tests pin
the two conservation statements analytically (not by smoke test), plus the
positivity property that motivated the change.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.freshwater import (
    FreshwaterForcing,
    joint_volume_salt_virtual_salt_flux,
    normalize_freshwater_net,
    virtual_salt_flux_from_net,
)

RHO_0 = 1026.0


def _setup(n=64, seed=0):
    """Spatially varying S, freshwater and cell area on an all-wet patch."""
    rng = np.random.default_rng(seed)
    area = jnp.asarray(rng.uniform(0.5, 2.0, n) * 1.0e10)
    mask = jnp.ones(n)
    h_top = jnp.asarray(rng.uniform(1.0, 10.0, n))
    # salinity spanning fresh river mouths to open ocean
    S = jnp.asarray(rng.uniform(0.0, 35.0, n))
    fw = FreshwaterForcing(
        precip=jnp.asarray(rng.uniform(0.0, 1e-4, n)),
        evap=jnp.asarray(rng.uniform(0.0, 1e-4, n)),
        runoff=jnp.asarray(rng.uniform(0.0, 5e-4, n)),
        ice_fw=jnp.asarray(rng.uniform(-1e-4, 1e-4, n)),
    )
    return fw, S, h_top, area, mask


def _salt_integral(dS_dt, h_top, area):
    """d/dt of total salt content [PSU kg/s]: rho_0 * dz * dS/dt * area."""
    return float(jnp.sum(RHO_0 * h_top * dS_dt * area))


class TestJointNormalization:
    def test_global_salt_tendency_is_zero(self):
        """The whole point: local-S closure that still conserves global salt."""
        fw, S, h_top, area, mask = _setup()
        dS = joint_volume_salt_virtual_salt_flux(fw, S, h_top, RHO_0, area, mask)
        total = _salt_integral(dS, h_top, area)
        # scale by a representative term so the tolerance is relative
        scale = float(jnp.sum(jnp.abs(RHO_0 * h_top * dS * area)))
        assert abs(total) <= 1e-10 * max(scale, 1.0), (total, scale)

    def test_uncorrected_local_S_closure_does_NOT_conserve(self):
        """Control: the UNCORRECTED local-S closure leaves a real residual.

        (Named precisely: this is local S WITHOUT the lambda correction,
        not the fixed-S_ref closure -- codex round-1 #8.)

        Without this, the test above could pass for a trivial reason (e.g. a
        degenerate input) and would not prove the correction does anything.
        """
        fw, S, h_top, area, mask = _setup()
        F = fw.precip - fw.evap + fw.runoff + fw.ice_fw
        F = normalize_freshwater_net(F, area, mask)          # volume-normalized
        dS_old = virtual_salt_flux_from_net(F, S, h_top, RHO_0)   # local S, no salt fix
        total_old = _salt_integral(dS_old, h_top, area)
        scale = float(jnp.sum(jnp.abs(RHO_0 * h_top * dS_old * area)))
        assert abs(total_old) > 1e-4 * scale, (total_old, scale)

    def test_volume_normalization_unchanged(self):
        """The freshwater (volume) budget must still integrate to zero."""
        fw, _S, h_top, area, mask = _setup()
        F = fw.precip - fw.evap + fw.runoff + fw.ice_fw
        wet = mask * (h_top > 1.0e-3).astype(mask.dtype)
        Fp = normalize_freshwater_net(F, area, wet)
        assert abs(float(jnp.sum(Fp * area))) <= 1e-9 * float(
            jnp.sum(jnp.abs(F * area)))

    def test_tendency_is_EXACTLY_zero_at_zero_salinity(self):
        """S_local = 0 => dS/dt == 0 exactly, so salinity cannot cross zero.

        This is the property the fixed-S_ref closure lacks and that produced the
        -22.31 PSU Amazon cells.  It is also why the correction is distributed
        in proportion to S rather than uniformly: a UNIFORM offset would leave
        ``-<S F'>`` at a zero-salinity cell and could still drive it negative.
        Exact equality (not a tolerance) is the whole point.
        """
        fw, S, h_top, area, mask = _setup()
        zero_cell = S.at[0].set(0.0)
        dS = joint_volume_salt_virtual_salt_flux(fw, zero_cell, h_top, RHO_0,
                                                 area, mask)
        assert float(dS[0]) == 0.0, float(dS[0])

    def test_uniform_offset_alternative_would_FAIL_that_property(self):
        """Non-vacuity for the choice above: show the rejected design breaks it.

        Reproduce the uniform-offset variant on the same inputs and assert it
        does NOT vanish at S=0 -- otherwise the test above would pass for both
        designs and prove nothing about the one we shipped.
        """
        fw, S, h_top, area, mask = _setup()
        zero_cell = S.at[0].set(0.0)
        F = fw.precip - fw.evap + fw.runoff + fw.ice_fw
        wet = mask * (h_top > 1.0e-3).astype(mask.dtype)
        Fp = normalize_freshwater_net(F, area, wet)
        G_uniform = normalize_freshwater_net(zero_cell * Fp, area, wet)
        assert float(G_uniform[0]) != 0.0

    def test_scales_linearly_with_local_salinity(self):
        """Doubling S_local doubles the salt flux (before the mean removal).

        A perturbation check that the local multiplier is actually applied --
        a closure that ignored S_local would be flat here.
        """
        fw, S, h_top, area, mask = _setup()
        d1 = joint_volume_salt_virtual_salt_flux(fw, S, h_top, RHO_0, area, mask)
        d2 = joint_volume_salt_virtual_salt_flux(fw, 2.0 * S, h_top, RHO_0,
                                                 area, mask)
        # lambda(2S) = 2*lambda(S) and G is homogeneous of degree 1 in S, so
        # this is EXACT up to float round-off -- assert elementwise, not a 5%
        # band (codex round-1 #9: the loose band could not distinguish the
        # corrected closure from the uncorrected one).
        assert jnp.allclose(d2, 2.0 * d1, rtol=1e-9, atol=0.0)

    def test_restoring_is_not_globally_redistributed(self):
        """``restoring`` is a LOCAL relaxation: excluded from both means."""
        fw, S, h_top, area, mask = _setup()
        rest = jnp.asarray(np.random.default_rng(3).uniform(-1e-5, 1e-5, S.size))
        with_rest = joint_volume_salt_virtual_salt_flux(
            fw._replace(restoring=rest), S, h_top, RHO_0, area, mask)
        without = joint_volume_salt_virtual_salt_flux(fw, S, h_top, RHO_0,
                                                      area, mask)
        delta = with_rest - without
        expected = jnp.where(h_top > 1.0e-3,
                             -(S * rest) / (RHO_0 * jnp.maximum(h_top, 1.0e-3)),
                             0.0)
        assert jnp.allclose(delta, expected, rtol=1e-10, atol=0.0)

    def test_thin_cell_guard(self):
        """h_top below 1 mm yields exactly zero tendency, not a huge one."""
        fw, S, h_top, area, mask = _setup()
        h = h_top.at[0].set(1.0e-6)
        dS = joint_volume_salt_virtual_salt_flux(fw, S, h, RHO_0, area, mask)
        assert float(dS[0]) == 0.0

    def test_dry_cell_with_thick_layer_gets_no_restoring(self):
        """A masked (dry) cell must get zero tendency even with h_top > 1 mm.

        The output gate tests THICKNESS, not the land mask, so before the fix a
        dry cell carrying a thick layer received the un-masked ``restoring``
        term.  The earlier restoring test ran on an all-wet patch and could not
        see this (codex round-1 #3 / #10).
        """
        fw, S, h_top, area, mask = _setup()
        mask = mask.at[0].set(0.0)                 # dry, but h_top[0] >> 1 mm
        rest = jnp.full(S.size, 1.0e-5)
        dS = joint_volume_salt_virtual_salt_flux(
            fw._replace(restoring=rest), S, h_top, RHO_0, area, mask)
        assert float(dS[0]) == 0.0, float(dS[0])

    def test_all_fresh_domain_is_finite_and_inert(self):
        """S == 0 everywhere: lambda has no basis to divide by -> 0, not NaN."""
        fw, S, h_top, area, mask = _setup()
        dS = joint_volume_salt_virtual_salt_flux(
            fw, jnp.zeros_like(S), h_top, RHO_0, area, mask)
        assert bool(jnp.all(jnp.isfinite(dS)))
        assert float(jnp.max(jnp.abs(dS))) == 0.0

    def test_negative_salinity_cells_do_not_break_conservation(self):
        """Signed S must not cancel the denominator.

        Using S itself as the redistribution basis lets ``∮S dA`` pass through
        zero while ``∮S F' dA`` does not, silently dropping the correction --
        and negative cells are exactly the state this closure exists to repair.
        The nonnegative basis ``max(S,0)`` removes that failure mode.
        """
        fw, S, h_top, area, mask = _setup()
        S_signed = S.at[0].set(-20.0).at[1].set(-15.0)
        dS = joint_volume_salt_virtual_salt_flux(fw, S_signed, h_top, RHO_0,
                                                 area, mask)
        total = _salt_integral(dS, h_top, area)
        scale = float(jnp.sum(jnp.abs(RHO_0 * h_top * dS * area)))
        assert abs(total) <= 1e-10 * max(scale, 1.0), (total, scale)

    def test_jit_and_grad_are_finite(self):
        """Differentiability: the lambda guard must not leak a 0/0 NaN VJP."""
        import jax

        fw, S, h_top, area, mask = _setup()

        def loss(s):
            return jnp.sum(joint_volume_salt_virtual_salt_flux(
                fw, s, h_top, RHO_0, area, mask) ** 2)

        g = jax.jit(jax.grad(loss))(S)
        assert bool(jnp.all(jnp.isfinite(g))), g
        # and at the degenerate all-fresh point, where den -> 0
        g0 = jax.jit(jax.grad(loss))(jnp.zeros_like(S))
        assert bool(jnp.all(jnp.isfinite(g0))), g0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
