"""check_grads (order 2, forward and reverse) on the elementwise custom JVPs
that had no derivative test (review F8): the two SI3 square roots and the
NEMO binary64 rounding barrier.  Interior points only (x > 0): at x == 0 the
rules deliberately return a finite zero subgradient instead of the infinite
derivative, which no finite-difference check can confirm.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from legoesm.ice import dynamics as _dyn  # noqa: E402
from legoesm.ice import ridging as _rdg  # noqa: E402
from legoesm.ocean.physics.vertical_mixing import tke as _tke  # noqa: E402

FNS = {
    "si3_sqrt": (_dyn._si3_sqrt, lambda x: 0.5 / jnp.sqrt(x)),
    "si3_jpl1_sqrt": (_rdg._si3_jpl1_sqrt, lambda x: 0.5 / jnp.sqrt(x)),
    "nemo_binary64_round": (_tke._nemo_binary64_round, jnp.ones_like),
}
X = jnp.asarray([0.37, 2.5, 41.0])   # FD-resolvable (eps=1e-4 at x~1e-3 is not)


@pytest.mark.parametrize("name", sorted(FNS))
def test_check_grads_order2(name):
    fn, _ = FNS[name]
    check_grads(fn, (X,), order=2, modes=["fwd", "rev"])


@pytest.mark.parametrize("name", sorted(FNS))
def test_derivative_value(name):
    fn, dfdx = FNS[name]
    g = jax.vmap(jax.grad(fn))(X)
    np.testing.assert_allclose(g, dfdx(X), rtol=1e-14)


@pytest.mark.parametrize("name", ["si3_sqrt", "si3_jpl1_sqrt"])
def test_zero_subgradient_is_finite(name):
    fn, _ = FNS[name]
    assert float(jax.grad(fn)(jnp.asarray(0.0))) == 0.0
