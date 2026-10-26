"""#1675 site 2: ``step_carbon`` preserves the pool STORAGE dtype.

In mixed-precision mode (x64 enabled, fp32 storage) the f64 forcing/lat/doy and
the traced f64 ``carbon_phi`` promote the fp32 carbon pools to f64.  A one-time
init cast does NOT hold — the carry re-promotes every step.  ``step_carbon``
casts the returned pools back to the input dtype (the single chokepoint), so a
rollout keeps fp32 storage.  No-op for strict-fp32 (x64 off) and fp64.

Runs with x64 ON — that IS the mixed condition (f64 arithmetic, fp32 storage).
"""
import jax
import jax.numpy as jnp
import pytest

# Needs x64 ON (the mixed condition: f64 arithmetic, fp32 storage).  SKIP rather
# than force-set x64 at import — forcing it leaks a global into later tests
# (collection-order precision bleed).  The sci-test run sets JAX_ENABLE_X64=1.
if not jax.config.read("jax_enable_x64"):
    pytest.skip(
        "needs x64 (run under JAX_ENABLE_X64=1)", allow_module_level=True)

from legoesm.land.carbon.carbon_cycle import (
    step_carbon, step_carbon_differland, init_carbon_state)
from legoesm.land.carbon.config import CarbonConfig

_CFG = CarbonConfig(scheme="differland")
_NCOL = 8


def _forcing(ncol):
    # All f64 under x64 — these are the promoting inputs.
    return dict(
        sw_down=jnp.full(ncol, 300.0),
        T=jnp.full(ncol, 288.0),
        co2_ppmv=jnp.full(ncol, 400.0),
        beta=jnp.full(ncol, 0.8),
        lat=jnp.linspace(-1.0, 1.0, ncol),
        precip=jnp.full(ncol, 3.0e-5),
    )


def test_step_carbon_preserves_fp32_storage_under_f64_forcing():
    state = jax.tree.map(
        lambda a: a.astype(jnp.float32),
        init_carbon_state((_NCOL,), _CFG))
    assert all(getattr(state, f).dtype == jnp.float32 for f in state._fields)

    f = _forcing(_NCOL)
    assert f["sw_down"].dtype == jnp.float64, "forcing must be f64 to promote"

    for step in range(4):
        state, _flux = step_carbon(
            state, f["sw_down"], f["T"], f["co2_ppmv"], f["beta"],
            f["lat"], float(step), f["precip"], _CFG, dt=86400.0)
        for name in state._fields:
            assert getattr(state, name).dtype == jnp.float32, (
                f"pool {name} promoted to {getattr(state, name).dtype} at "
                f"step {step}: step_carbon must cast the carry to storage dtype")


def test_step_carbon_fp64_is_byte_identical():
    # fp64 in: both casts (entry upcast + exit cast) are astype(f64->f64) =
    # identity, so the wrapped step must be BYTE-IDENTICAL to the un-wrapped
    # differland step — not merely the same dtype.
    state = init_carbon_state((_NCOL,), _CFG)  # f64 under x64
    assert all(getattr(state, x).dtype == jnp.float64 for x in state._fields)
    f = _forcing(_NCOL)
    args = (state, f["sw_down"], f["T"], f["co2_ppmv"], f["beta"],
            f["lat"], 1.0, f["precip"], _CFG)
    wrapped, flux_w = step_carbon(*args, dt=86400.0)
    direct, flux_d = step_carbon_differland(*args, dt=86400.0)
    assert jnp.array_equal(flux_w, flux_d)
    for name in state._fields:
        assert getattr(wrapped, name).dtype == jnp.float64
        assert jnp.array_equal(
            getattr(wrapped, name), getattr(direct, name)), name
