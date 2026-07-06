"""Unit tests for the RRTMGP gas-optics table interpolation primitives.

Focus: the CPU/GPU gather path (``combine_linearly_gather``) must be a
forward- AND gradient-equivalent drop-in for the one-hot+einsum path
(``combine_linearly`` on the TPU branch).  The gather path exists to shrink the
XLA graph and turn the dense one-hot-einsum reverse-mode VJP into a sparse
scatter-add corner sum; these tests pin that it does so WITHOUT changing the
answer (to floating-point re-association) or the gradient.

Run under x64 for the tight tolerance:
    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/atmosphere/hydrostatic/unit/test_optics_utils_interp.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.radiation.rrtmgp.optics import optics_utils as ou


def _einsum_reference(vals, weight_idx_list):
    """One-hot + einsum reference, forced regardless of the host backend.

    Re-uses the production TPU branch of ``combine_linearly`` by temporarily
    making ``jax.default_backend`` report ``"tpu"`` (the einsum path runs
    correctly on any backend; it is just slower).  This guarantees the
    reference is the EXACT shipped one-hot formula, not a re-derivation.
    """
    orig = jax.default_backend
    jax.default_backend = lambda *a, **k: "tpu"
    try:
        return ou.combine_linearly(vals, weight_idx_list)
    finally:
        jax.default_backend = orig


def _make_axis(key, x_ref, *, kind, offset_ref=None):
    """Build one interpolation axis from a random continuous input.

    kind="interp": a 2-endpoint ``Interpolant`` (the dependent/independent
        reference-axis case, e.g. temperature/pressure/eta).
    kind="iw": a single ``IndexAndWeight`` (the already-resolved-endpoint case
        the einsum path encodes as a single weighted one-hot).
    Returns (entry, x) so the test can differentiate the output w.r.t. x.
    """
    lo = float(x_ref[0])
    hi = float(x_ref[-1])
    x = jax.random.uniform(key, (4, 3), minval=lo + 1e-3, maxval=hi - 1e-3)
    interp = ou.create_linear_interpolant(x, x_ref)
    if kind == "interp":
        return interp, x
    # Collapse to a single IndexAndWeight (use the low endpoint).
    return interp.interp_low, x


@pytest.mark.parametrize("ranks", [(7,), (5, 4), (6, 5, 4), (4, 3, 5, 4)])
def test_gather_matches_einsum_forward(ranks):
    """Gather corner-sum == one-hot einsum, to float re-association (x64)."""
    if not jax.config.jax_enable_x64:
        pytest.skip("needs JAX_ENABLE_X64=1 for the tight tolerance")
    key = jax.random.PRNGKey(0)
    k_vals, *axis_keys = jax.random.split(key, len(ranks) + 1)
    vals = jax.random.normal(k_vals, ranks)
    # Alternate Interpolant / IndexAndWeight axes to exercise both branches.
    weight_idx_list = []
    for i, (ak, n) in enumerate(zip(axis_keys, ranks)):
        x_ref = jnp.linspace(0.0, 10.0, n)
        kind = "interp" if i % 2 == 0 else "iw"
        entry, _ = _make_axis(ak, x_ref, kind=kind)
        weight_idx_list.append(entry)

    gather = ou.combine_linearly_gather(vals, weight_idx_list)
    einsum = _einsum_reference(vals, weight_idx_list)
    assert gather.shape == einsum.shape
    np.testing.assert_allclose(gather, einsum, rtol=1e-12, atol=1e-12)


def test_gather_matches_einsum_gradient():
    """d/dx of the interpolated value matches between gather and einsum.

    Differentiates through the FULL path (continuous input -> weights ->
    output): this is the property that matters for end-to-end ``jax.grad`` of
    radiation.  The integer floor index is non-differentiable in both paths, so
    the gradients must agree.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("needs JAX_ENABLE_X64=1 for the tight tolerance")
    ranks = (6, 5, 4)
    key = jax.random.PRNGKey(7)
    k_vals, *axis_keys = jax.random.split(key, len(ranks) + 1)
    vals = jax.random.normal(k_vals, ranks)
    x_refs = [jnp.linspace(0.0, 10.0, n) for n in ranks]
    kinds = ["interp", "iw", "interp"]
    # Differentiate w.r.t. the continuous axis inputs that drive the weights.
    xs0 = []
    for ak, xr, kind in zip(axis_keys, x_refs, kinds):
        _, x = _make_axis(ak, xr, kind=kind)
        xs0.append(x)

    def _build(xs):
        wl = []
        for xr, kind, x in zip(x_refs, kinds, xs):
            interp = ou.create_linear_interpolant(x, xr)
            wl.append(interp if kind == "interp" else interp.interp_low)
        return wl

    def f_gather(xs):
        return jnp.sum(ou.combine_linearly_gather(vals, _build(xs)))

    def f_einsum(xs):
        return jnp.sum(_einsum_reference(vals, _build(xs)))

    g_gather = jax.grad(f_gather)(xs0)
    g_einsum = jax.grad(f_einsum)(xs0)
    for gg, ge in zip(g_gather, g_einsum):
        np.testing.assert_allclose(gg, ge, rtol=1e-10, atol=1e-10)


def test_gather_zero_on_out_of_bounds_matches_onehot():
    """An offset index past the table edge contributes ZERO (one-hot parity).

    ``jax.nn.one_hot`` zeros an OOB index; advanced-index gather clamps to the
    edge.  ``combine_linearly_gather`` masks OOB corners to zero so the two
    paths agree even when a ``create_linear_interpolant`` ``offset`` pushes an
    index past ``vals.shape[axis]``.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("needs JAX_ENABLE_X64=1 for the tight tolerance")
    n = 5
    vals = jax.random.normal(jax.random.PRNGKey(11), (n, n))
    x_ref = jnp.linspace(0.0, 10.0, n)
    # Force the top index to the edge, then add an offset that overruns it.
    x = jnp.array([[9.99, 9.99, 9.99]])
    base = ou.create_linear_interpolant(x, x_ref)
    over = ou.create_linear_interpolant(x, x_ref, offset=jnp.full(x.shape, 3))
    wl = [base, over]
    gather = ou.combine_linearly_gather(vals, wl)
    einsum = _einsum_reference(vals, wl)
    # The one-hot reference also zeros the overrun endpoints; assert parity.
    np.testing.assert_allclose(gather, einsum, rtol=1e-12, atol=1e-12)
    assert jnp.all(jnp.isfinite(gather))


def test_gather_grad_finite_no_nan():
    """Reverse-mode gradient is finite (no 0*inf from the gather/stop_grad)."""
    ranks = (5, 4)
    key = jax.random.PRNGKey(3)
    k_vals, k0, k1 = jax.random.split(key, 3)
    vals = jax.random.normal(k_vals, ranks)
    x_refs = [jnp.linspace(0.0, 10.0, n) for n in ranks]

    def f(xs):
        wl = [ou.create_linear_interpolant(x, xr)
              for x, xr in zip(xs, x_refs)]
        return jnp.sum(ou.combine_linearly_gather(vals, wl))

    x0 = [jax.random.uniform(k0, (4, 3), minval=0.1, maxval=9.9),
          jax.random.uniform(k1, (4, 3), minval=0.1, maxval=9.9)]
    grads = jax.grad(f)(x0)
    for g in grads:
        assert jnp.all(jnp.isfinite(g))
