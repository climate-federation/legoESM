"""IfsSmooth (shared CLOUDSC / CUMASTRN switch forms): dtype of the exact
comparison weights."""
import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.physics._ifs_smooth import IfsSmooth

_T = 273.16


@pytest.mark.parametrize("hard", [True, False])
@pytest.mark.parametrize("x64", [True, False])
def test_comparison_weight_keeps_the_array_dtype(hard, x64):
    """A python-scalar operand (a constant threshold, or the constant on the
    left as in ``ge(RTT, T)``) must not set the weight's dtype: under x64 the
    exact form used to return float64 for float32 temperatures, which broke
    the float32 scan carries of the IFS ascent."""
    sm = IfsSmooth(hard)
    t = jnp.asarray([250.0, 273.16, 290.0], jnp.float32)
    with jax.enable_x64(x64):
        for m in (sm.gt(_T, t, 1.0), sm.ge(_T, t, 1.0), sm.lt(_T, t, 1.0),
                  sm.le(_T, t, 1.0), sm.gt(t, _T, 1.0), sm.lt(t, _T, 1.0),
                  sm.on(t, _T, 1.0), sm.off(t, _T, 1.0)):
            assert m.dtype == jnp.float32
