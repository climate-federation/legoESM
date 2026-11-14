"""convtran's interface log-mean in float32: tiny hydrometeor amounts must give
finite tendencies AND finite, correct gradients.  The unscaled form floored at
maxc*1e-12 underflowed (log(0)*0 = NaN forward) and its gradient overflowed
for adjacent amounts ~1e-30 (NaN gradient)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.atmosphere.physics.convection._zm_cam6 import convtran


def _column(dtype, c5, c4=0.0):
    ncol, nlev = 2, 12
    c = np.zeros((ncol, nlev)); c[:, 4] = c4; c[:, 5] = c5; c[:, 7] = 1e-6
    mu = np.zeros((ncol, nlev)); mu[:, 3:10] = 0.05
    md = np.zeros((ncol, nlev)); md[:, 4:9] = -0.02
    du = np.full((ncol, nlev), 1e-6); eu = np.full((ncol, nlev), 2e-6)
    ed = np.full((ncol, nlev), -1e-6); dp = np.full((ncol, nlev), 50.0)
    a = lambda x: jnp.asarray(x, dtype)
    jt = jnp.full((ncol,), 2, jnp.int32); mx = jnp.full((ncol,), 10, jnp.int32)
    return a(c), (a(mu), a(md), a(du), a(eu), a(ed), a(dp), jt, mx, 0)


def _loss(c, rest):
    # Non-uniform weights: a plain sum lets the interface-mean derivatives
    # cancel between the two layers sharing an interface.
    w = jnp.arange(1.0, 1.0 + c.shape[1], dtype=c.dtype) ** 2
    return jnp.sum(w * convtran(c, *rest))


@pytest.mark.parametrize("c5", [1e-30, 1e-36])
def test_convtran_float32_tiny_tracer_is_finite(c5):
    c, rest = _column(jnp.float32, c5)
    out = jax.jit(convtran)(c, *rest)
    assert out.dtype == jnp.float32
    assert np.all(np.isfinite(np.asarray(out)))
    assert np.all(np.isfinite(np.asarray(jax.grad(_loss)(c, rest))))


def test_convtran_float32_adjacent_tiny_gradient_is_scale_free():
    # convtran is homogeneous of degree 1 in the tracer, so its gradient is
    # scale-free: the float32 gradient at amounts ~1e-25 must equal the
    # float32 gradient of the same column scaled by 1e25 (same dtype, so the
    # netflux round-off zeroing acts alike on both).  At ~1e-30 the floored
    # log-mean next to a zero layer (~3e-11 * 1e-30) is itself below the
    # float32 normal range and flushes to 0, tying min(chat, 0); that is
    # float32 range, not this path.
    c, rest = _column(jnp.float32, 1e-25, c4=2e-25)
    g_tiny = np.asarray(jax.grad(_loss)(c, rest))
    assert np.all(np.isfinite(g_tiny))
    big, _ = _column(jnp.float32, 1.0, c4=2.0)
    big = big.at[:, 7].set(1e19)  # 1e-6 * 1e25
    g_big = np.asarray(jax.grad(_loss)(big, rest))
    np.testing.assert_allclose(g_tiny, g_big, rtol=1e-4, atol=1e-6)


def test_convtran_check_grads_float64_default():
    # O(1) all-positive distinct values (convtran is degree-1 homogeneous):
    # the min()/log-mean switches sit far from their kinks for the FD step.
    _, rest = _column(jnp.float64, 1e-5)
    c = jnp.asarray((1.0 + 0.3 * np.arange(12.0))[None, :].repeat(2, 0))
    check_grads(lambda x: convtran(x, *rest), (c,), order=2, modes=("fwd", "rev"))
