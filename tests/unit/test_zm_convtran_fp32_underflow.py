"""convtran's interface log-mean must not turn tiny hydrometeor amounts into
NaN in float32 (the 1e-12 floor underflows there).  Float64 bitwise parity
with the pre-fix code was checked by an A/B recorded in the commit."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.convection._zm_cam6 import convtran


def _column(dtype, tiny):
    ncol, nlev = 2, 12
    c = np.zeros((ncol, nlev)); c[:, 5] = tiny; c[:, 7] = 1e-6
    mu = np.zeros((ncol, nlev)); mu[:, 3:10] = 0.05
    md = np.zeros((ncol, nlev)); md[:, 4:9] = -0.02
    du = np.full((ncol, nlev), 1e-6); eu = np.full((ncol, nlev), 2e-6)
    ed = np.full((ncol, nlev), -1e-6); dp = np.full((ncol, nlev), 50.0)  # hPa-like layer
    a = lambda x: jnp.asarray(x, dtype)
    jt = jnp.full((ncol,), 2, jnp.int32); mx = jnp.full((ncol,), 10, jnp.int32)
    return a(c), a(mu), a(md), a(du), a(eu), a(ed), a(dp), jt, mx


@pytest.mark.parametrize("tiny", [1e-30, 1e-36])
def test_convtran_float32_tiny_tracer_is_finite(tiny):
    c, mu, md, du, eu, ed, dp, jt, mx = _column(jnp.float32, tiny)
    out = jax.jit(convtran)(c, mu, md, du, eu, ed, dp, jt, mx, 0)
    assert out.dtype == jnp.float32
    assert np.all(np.isfinite(np.asarray(out)))
    g = jax.grad(lambda x: jnp.sum(convtran(x, mu, md, du, eu, ed, dp, jt, mx, 0)))(c)
    assert np.all(np.isfinite(np.asarray(g)))
