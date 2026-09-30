"""fv3_duo coarse-precision FOUNDATION guards (2026-08-28).

The dtype phase-gates were relaxed from strict-float64 to
UNIFORMITY, so the anti-silent-downcast protection the strict gates gave
incidentally (an f32 IC in an fp64-intended run) moved to the MODEL
BOUNDARY: ``FV3DuoDynamicsModel.step`` refuses an incoming bundle whose
inexact leaves are not the configured ``storage_dtype``.  These pin that
boundary (the regression the triple review flagged) plus the
construct-time fp32-not-yet-wired refusal.

Cheap by construction: the boundary guard fires BEFORE the jitted step,
so none of these compile ``fv_dynamics`` (no 15-minute trace).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (  # noqa: E402
    FV3DuoConfig, FV3DuoDynamicsModel)
from legoesm.grids.factory import create_fv3_duo_grid  # noqa: E402


@pytest.fixture(scope="module")
def grid():
    return create_fv3_duo_grid(12)


def _f32_bundle(bundle):
    """Downcast every inexact leaf of the IC bundle to float32."""
    return jax.tree_util.tree_map(
        lambda a: (a.astype(jnp.float32)
                   if jnp.issubdtype(getattr(a, "dtype", np.int64),
                                     jnp.inexact) else a),
        bundle)


def test_fp64_default_constructs_and_ic_is_f64(grid):
    m = FV3DuoDynamicsModel(grid, FV3DuoConfig(km=5))
    assert m._storage_dtype == np.float64
    ic = m.dcmip16_initial_state(do_pert=True)
    assert ic["state"]["delp"].dtype == jnp.float64


def test_step_rejects_a_silent_f32_bundle_under_fp64(grid):
    """THE REGRESSION the triple review caught: with the phase gates now
    dtype-UNIFORMITY (not strict f64), a uniformly-f32 bundle would sail
    through them and run silently in f32. step()'s boundary guard must
    reject it -- and BEFORE the jitted step compiles."""
    m = FV3DuoDynamicsModel(grid, FV3DuoConfig(km=5))
    ic = m.dcmip16_initial_state(do_pert=True)
    bad = _f32_bundle(ic)
    with pytest.raises(TypeError, match="storage_dtype"):
        m.step(bad, 120.0)


def test_fp32_constructs_and_steps(grid):
    """Increment 2: fp32 now RUNS end-to-end. Construct an fp32 model,
    build the (f32) IC, step once, and assert the carry stays float32 and
    finite. (Compiles fv_dynamics once -- the heaviest test here.)"""
    m = FV3DuoDynamicsModel(grid, FV3DuoConfig(km=5, storage_dtype="float32"))
    assert m._storage_dtype == np.float32
    ic = m.dcmip16_initial_state(do_pert=True)
    assert ic["state"]["delp"].dtype == jnp.float32
    out = m.step(ic, 120.0)
    for k, v in out["state"].items():
        assert v.dtype == jnp.float32, (k, v.dtype)
        assert bool(jnp.isfinite(v).all()), (k, "non-finite")


def test_mixed_storage_still_refused(grid):
    """True per-op MIXED (fp64 pressure column / energy fixer, fp32
    elsewhere) is a later increment -- np.dtype('mixed') is not a real
    dtype, so it errors; only 'float32'/'float64' are wired."""
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        FV3DuoDynamicsModel(grid, FV3DuoConfig(km=5, storage_dtype="mixed"))


def test_bad_dtype_string_is_a_value_error(grid):
    """np.dtype normalisation gives a loud error on a garbage dtype
    string rather than a silent mismatch later."""
    with pytest.raises((ValueError, TypeError)):
        FV3DuoDynamicsModel(grid, FV3DuoConfig(km=5, storage_dtype="flt64"))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
