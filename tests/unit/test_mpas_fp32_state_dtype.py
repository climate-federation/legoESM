"""An fp32 MPAS run under JAX_ENABLE_X64 keeps its atmosphere state float32
(the dry-mass fixer used to hand back a float64 p_s that promoted it)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.unit.test_mpas_multilayer_land_port import (
    _build_driver, _patch_land_loaders, TWO_STEPS_DAYS)


def test_fp32_mpas_state_stays_float32_with_x64(monkeypatch, tmp_path):
    if not jax.config.jax_enable_x64:
        pytest.skip("needs JAX_ENABLE_X64 (the promotion only exists with x64)")
    _patch_land_loaders(monkeypatch)
    d = _build_driver(str(tmp_path), TWO_STEPS_DAYS, precision="fp32",
                      use_multilayer=False, microphysics="kessler")
    assert d.run() == "COMPLETED"
    leaves = jax.tree_util.tree_leaves_with_path(d.state)
    f64 = [jax.tree_util.keystr(p) for p, x in leaves
           if hasattr(x, "dtype") and x.dtype == jnp.float64]
    assert not f64, f64
    assert np.all(np.isfinite(np.asarray(d.state.p_s.data)))

