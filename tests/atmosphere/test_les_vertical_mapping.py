"""Unit tests for :mod:`legoesm.atmosphere.dynamics.les.les_vertical_mapping`.

Stage-5 vertical mapping: GCM-column→LES-grid interpolation + Newtonian
top-relaxation.  Analytic interp checks (node recovery, linear midpoint,
endpoint clamp), the relaxation tendency sign/zero, the reused sponge-shape
rate profile (0 below the layer, inv_tau at top), and AD/jit safety.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.les.les_vertical_mapping import (
    build_top_relaxation,
    interpolate_column_to_les,
    relaxation_tendency,
    top_relaxation_rate,
)

jax.config.update("jax_enable_x64", True)


def test_interp_node_recovery_and_midpoint():
    src_z = jnp.array([0.0, 100.0, 200.0])
    src_f = jnp.array([10.0, 30.0, 20.0])
    les_z = jnp.array([0.0, 50.0, 100.0, 150.0, 200.0])
    out = interpolate_column_to_les(src_z, src_f, les_z)
    # nodes recovered, midpoints linear
    np.testing.assert_allclose(np.asarray(out), [10.0, 20.0, 30.0, 25.0, 20.0], rtol=1e-12)


def test_interp_clamps_outside_range():
    src_z = jnp.array([100.0, 200.0])
    src_f = jnp.array([5.0, 9.0])
    les_z = jnp.array([0.0, 300.0])  # both outside [100,200]
    out = interpolate_column_to_les(src_z, src_f, les_z)
    assert float(out[0]) == pytest.approx(5.0)  # below -> first value
    assert float(out[1]) == pytest.approx(9.0)  # above -> last value


def test_interp_handles_descending_source():
    """Top-down (descending-z) source must interpolate correctly (sorted)."""
    src_z = jnp.array([200.0, 100.0, 0.0])      # descending (dycore order)
    src_f = jnp.array([20.0, 30.0, 10.0])       # matching values
    les_z = jnp.array([0.0, 50.0, 100.0, 200.0])
    out = interpolate_column_to_les(src_z, src_f, les_z)
    np.testing.assert_allclose(np.asarray(out), [10.0, 20.0, 30.0, 20.0], rtol=1e-12)


def test_sam_rational_top_value_is_canonical():
    """sam_rational reaches 100/101·inv_tau at the very top (documented)."""
    inv_tau = 1.0 / 600.0
    r_top = float(top_relaxation_rate(
        jnp.array([4000.0]), 4000.0, 1000.0, inv_tau, shape="sam_rational")[0])
    assert r_top == pytest.approx(inv_tau * 100.0 / 101.0, rel=1e-10)
    # sin2 hits it exactly.
    r_top_sin2 = float(top_relaxation_rate(
        jnp.array([4000.0]), 4000.0, 1000.0, inv_tau, shape="sin2")[0])
    assert r_top_sin2 == pytest.approx(inv_tau, rel=1e-10)


def test_build_raises_when_les_top_exceeds_gcm_top():
    les_z = jnp.linspace(0.0, 5000.0, 9)
    gcm_z = jnp.array([0.0, 2000.0, 4000.0])  # column tops at 4 km < LES 5 km
    with pytest.raises(ValueError, match="exceeds the GCM column top"):
        build_top_relaxation(les_z, 5000.0, gcm_z, jnp.array([300.0, 305.0, 320.0]),
                             relax_width_m=1000.0, inv_tau=1e-3)


def test_top_relaxation_rate_zero_below_inv_tau_at_top():
    H, width, inv_tau = 4000.0, 1000.0, 1.0 / 600.0
    les_z = jnp.array([0.0, 2500.0, 3000.0, 3500.0, 4000.0])
    rate = top_relaxation_rate(les_z, H, width, inv_tau)
    r = np.asarray(rate)
    # Below the sponge base (H-width=3000): zero.
    assert r[0] == pytest.approx(0.0)
    assert r[1] == pytest.approx(0.0)  # at 2500 < 3000
    assert r[2] == pytest.approx(0.0)  # at exactly the base
    # At the top: inv_tau (sin² of π/2 = 1).
    assert r[4] == pytest.approx(inv_tau, rel=1e-10)
    # Monotone non-decreasing toward the top.
    assert bool(np.all(np.diff(r) >= -1e-15))


def test_top_relaxation_rate_unknown_shape_raises():
    with pytest.raises(ValueError, match="Unknown sponge profile shape"):
        top_relaxation_rate(jnp.array([0.0, 4000.0]), 4000.0, 1000.0, 1e-3,
                            shape="bogus")


def test_relaxation_tendency_sign_and_zero():
    field = jnp.array([2.0, 5.0])
    target = jnp.array([1.0, 5.0])
    rate = jnp.array([0.01, 0.01])
    tend = relaxation_tendency(field, target, rate)
    # -rate*(field-target): pulls toward target (negative where field>target).
    assert float(tend[0]) == pytest.approx(-0.01 * 1.0)
    assert float(tend[1]) == 0.0  # field == target


def test_relaxation_tendency_zero_where_rate_zero():
    field = jnp.array([9.0, 9.0])
    target = jnp.array([1.0, 1.0])
    rate = jnp.array([0.0, 0.02])
    tend = relaxation_tendency(field, target, rate)
    assert float(tend[0]) == 0.0  # rate 0 -> no relaxation below the layer
    assert float(tend[1]) < 0.0


def test_build_top_relaxation_composes():
    les_z = jnp.linspace(0.0, 4000.0, 9)
    gcm_z = jnp.array([0.0, 2000.0, 4000.0])
    gcm_field = jnp.array([300.0, 305.0, 320.0])
    target, rate = build_top_relaxation(
        les_z, 4000.0, gcm_z, gcm_field, relax_width_m=1000.0, inv_tau=1e-3
    )
    assert target.shape == les_z.shape
    assert rate.shape == les_z.shape
    # target equals the standalone interpolation
    np.testing.assert_allclose(
        np.asarray(target),
        np.asarray(interpolate_column_to_les(gcm_z, gcm_field, les_z)))
    # rate zero at the surface, positive at the top
    assert float(rate[0]) == pytest.approx(0.0)
    assert float(rate[-1]) > 0.0


def test_jit_and_grad():
    les_z = jnp.linspace(0.0, 4000.0, 9)
    gcm_z = jnp.array([0.0, 2000.0, 4000.0])

    def loss(gcm_field):
        target, rate = build_top_relaxation(
            les_z, 4000.0, gcm_z, gcm_field, relax_width_m=1000.0, inv_tau=1e-3)
        les_field = jnp.full_like(target, 305.0)
        return jnp.sum(relaxation_tendency(les_field, target, rate) ** 2)

    out = jax.jit(loss)(jnp.array([300.0, 305.0, 320.0]))
    assert jnp.isfinite(out)
    g = jax.grad(loss)(jnp.array([300.0, 305.0, 320.0]))
    assert bool(jnp.all(jnp.isfinite(g)))
