"""ppm_edge_values_axis0 must stay formula-identical to the default-path
ppm_edge_values — the axis-0 twin exists only to kill the moveaxis
round-trip in the lat-lon latitude PPM (input_transpose_fusion,
370 us/step on the LL2048@64 trace). Change one formula and this test
forces the other."""

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.operators_fv import ppm_edge_values, ppm_edge_values_axis0


def test_axis0_matches_moveaxis_reference():
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal((12, 7, 5)))  # (swept, lon, lev)
    ref = jnp.moveaxis(ppm_edge_values(jnp.moveaxis(q, 0, 1)), 1, 0)
    got = ppm_edge_values_axis0(q)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


def test_axis0_is_transpose_free():
    jaxpr = str(jax.make_jaxpr(ppm_edge_values_axis0)(
        jnp.zeros((12, 7, 5))))
    assert "transpose" not in jaxpr, (
        "the axis-0 twin re-grew a transpose — its whole point is gone")
