"""The ERA5 initial state is built in float64; an fp32 MPAS run must still
start with a float32 state (the IC is cast once to the seed-state dtypes)."""
from __future__ import annotations

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver

_ERA5 = os.environ.get("ERA5_IC", "")
_ETOPO = os.environ.get("ETOPO", "")


def _build(precision: str, microphysics: str = "none") -> ModelDriver:
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=3, nlev=20,
                        vertical_coord="hybrid"),
        dycore=DycoreConfig(discretization="mpas", dt=300.0),
        output=OutputConfig(output_dir="", diag_days=0),
        days=1, dataset="analytical", radiation="gray", convection="none",
        turbulence="none", precision=precision, ic="era5", ic_path=_ERA5,
        microphysics=microphysics,
        topography=_ETOPO, topo_smoothing=4,
    )
    d = ModelDriver(cfg, output_dir=tempfile.mkdtemp())
    d.setup()
    return d


@pytest.mark.skipif(not (os.path.exists(_ERA5) and os.path.exists(_ETOPO)),
                    reason="ERA5_IC / ETOPO input not available")
@pytest.mark.parametrize("precision, micro, ps_dtype", [
    ("fp32", "none", jnp.float32),
    ("fp32", "kessler", jnp.float32),     # moist: dycore tracers cast too
    ("mixed", "kessler", jnp.float64),    # 'mixed' keeps a float64 p_s (#1675)
])
def test_era5_ic_starts_in_the_run_precision(precision, micro, ps_dtype):
    if not jax.config.jax_enable_x64:
        pytest.skip("the float64 IC only exists with JAX_ENABLE_X64")
    d = _build(precision, micro)
    leaves = jax.tree_util.tree_leaves_with_path(d.state)
    f64 = sorted(jax.tree_util.keystr(p) for p, x in leaves
                 if getattr(x, "dtype", None) == jnp.float64)
    want = ([] if ps_dtype == jnp.float32
            else [k for k in f64 if k.startswith(".p_s")])
    assert f64 == want, f64
    assert d.state.p_s.data.dtype == ps_dtype
    assert d.tracers["q_v"].dtype == jnp.float32
    assert np.all(np.isfinite(np.asarray(d.state.T.data)))
