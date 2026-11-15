"""Unit tests for the near-neutral N2 probe's derivative floor and interface counter."""
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not jax.config.jax_enable_x64, reason="needs JAX_ENABLE_X64=1")

_P = (Path(__file__).resolve().parents[3]
      / "scripts/validate/ocean_fidelity/orca2_l4/orca2_near_neutral_n2.py")
_spec = importlib.util.spec_from_file_location("near_neutral", _P)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def test_floor_changes_only_the_derivative():
    from legoesm.core.profile_metrics import safe_sqrt_nan

    x = jnp.array([-1.0, 0.0, 1e-16, 1e-6, 4.0])
    f = probe.floored_sqrt(1e-10)
    np.testing.assert_array_equal(f(x), safe_sqrt_nan(x))
    g = jax.vmap(jax.grad(f))(x)
    g0 = jax.vmap(jax.grad(safe_sqrt_nan))(x)
    np.testing.assert_allclose(g[:2], 0.0)
    np.testing.assert_allclose(g[2], 0.5 / np.sqrt(1e-10))       # floored
    assert g0[2] > 100 * g[2]                                     # unfloored is far larger
    np.testing.assert_allclose(g[3:], g0[3:])                     # above the floor: unchanged


def test_counter_bins_wet_interfaces():
    probe._COUNTS.clear()
    probe._RECORD["on"] = True
    seen = {}

    def fake(forcing, depth_cell, dz_w, H, N2, **kw):
        seen["called"] = True
        return N2, N2

    wrapped = probe.counting(fake)
    n2 = jnp.array([-1.0, 0.0, 5e-13, 5e-11, 5e-9, 1e-5, 1e-13])
    dz = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])           # last interface dry
    jax.block_until_ready(jax.jit(lambda a, b: wrapped(None, None, b, None, a))(n2, dz))
    probe._RECORD["on"] = False
    wet, nonpos, b12, b10, b8 = probe._COUNTS[0]
    assert (wet, nonpos, b12, b10, b8) == (6, 2, 1, 2, 3) and seen["called"]
