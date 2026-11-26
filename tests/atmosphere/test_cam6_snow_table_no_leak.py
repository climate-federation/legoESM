"""The cached Mitchell snow table must not hold tracers.

``_snow_optics_table`` is an lru_cache; first called inside a jit trace, a table
built with jnp would cache a tracer and the next trace would fail with
UnexpectedTracerError (seen in a level 4 run on 30 October 2026).
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.radiation import integration as R

PATH = os.environ.get("LEGOESM_ICEOPTICS", "")


@pytest.mark.skipif(not os.path.exists(PATH), reason="CAM iceoptics file not staged")
def test_table_is_numpy_and_survives_two_traces():
    R._snow_optics_table.cache_clear()

    @jax.jit
    def f(x):
        t = R._snow_optics_table(PATH)
        return x * jnp.sum(t.sw_ext[:, 0])

    a = f(jnp.ones(3))
    b = f(jnp.ones(4))          # a second trace reuses the cached table
    assert np.isfinite(np.asarray(a)).all() and np.isfinite(np.asarray(b)).all()
    t = R._snow_optics_table(PATH)
    assert all(isinstance(x, np.ndarray) for x in t)
