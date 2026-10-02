"""Direct unit tests for the shared ocean physics tendency helpers.

Covers the leaf module ``legoesm.ocean.physics.tendencies`` that the per-scheme
integration factories and the combiner share (zero / wrap / no-op), introduced
when these were de-duplicated out of ``combined`` to break the
combined<->integration import cycle.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.ocean.state import OceanState
from legoesm.ocean.physics.tendencies import (
    make_none_physics_fn,
    wrap_ocean_tendencies,
    zero_ocean_tendencies,
)

DIMS_3D = ("face", "x", "y", "level")
DIMS_2D = ("face", "x", "y")


def _mini_state(nlev=3):
    shape_3d = (1, 2, 2, nlev)
    shape_2d = (1, 2, 2)

    def f3(name):
        return Field(data=jnp.ones(shape_3d), name=name, dims=DIMS_3D, units="x")

    def f2(name):
        return Field(data=jnp.ones(shape_2d), name=name, dims=DIMS_2D, units="x")

    return OceanState(
        u=f3("u"), v=f3("v"), T=f3("T"), S=f3("S"),
        eta=f2("eta"), H_bathy=f2("H_bathy"), land_mask=f2("land_mask"),
    )


def test_zero_ocean_tendencies_all_zero_correct_shapes_dims():
    state = _mini_state()
    t = zero_ocean_tendencies(state)
    for name in ("du_dt", "dv_dt", "dT_dt", "dS_dt"):
        fld = getattr(t, name)
        assert fld.data.shape == state.u.data.shape
        assert fld.dims == DIMS_3D
        assert jnp.all(fld.data == 0.0)
    for name in ("deta_dt", "dH_bathy_dt", "dland_mask_dt"):
        fld = getattr(t, name)
        assert fld.data.shape == state.eta.data.shape
        assert fld.dims == DIMS_2D
        assert jnp.all(fld.data == 0.0)


def test_wrap_ocean_tendencies_passes_through_and_zeroes_none():
    state = _mini_state()
    du = jnp.full(state.u.data.shape, 2.0)
    dT = jnp.full(state.u.data.shape, -1.5)
    # dv, dS omitted -> must become zero; 2-D fields always zero.
    t = wrap_ocean_tendencies(du, None, dT, None, state)
    assert jnp.allclose(t.du_dt.data, 2.0)
    assert jnp.all(t.dv_dt.data == 0.0)
    assert jnp.allclose(t.dT_dt.data, -1.5)
    assert jnp.all(t.dS_dt.data == 0.0)
    assert jnp.all(t.deta_dt.data == 0.0)
    assert t.du_dt.dims == DIMS_3D and t.deta_dt.dims == DIMS_2D


def test_make_none_physics_fn_returns_zero_tendencies():
    state = _mini_state()
    fn = make_none_physics_fn()
    # grid / z_coord are ignored by the no-op; surface_forcing defaults None.
    out = fn(state, None, None)
    ref = zero_ocean_tendencies(state)
    for name in ("du_dt", "dv_dt", "dT_dt", "dS_dt", "deta_dt",
                 "dH_bathy_dt", "dland_mask_dt"):
        assert jnp.array_equal(getattr(out, name).data, getattr(ref, name).data)


def test_none_physics_fn_accepts_dt_like_enabled_schemes():
    state = _mini_state()
    out = make_none_physics_fn()(state, None, None, dt=900.0)
    assert jnp.array_equal(out.dT_dt.data, zero_ocean_tendencies(state).dT_dt.data)


def test_helpers_are_differentiable():
    """wrap/zero must carry gradients (ocean physics is jax.grad-trained)."""
    state = _mini_state()

    def loss(scale):
        du = scale * jnp.ones(state.u.data.shape)
        t = wrap_ocean_tendencies(du, None, None, None, state)
        return jnp.sum(t.du_dt.data ** 2)

    g = jax.grad(loss)(3.0)
    assert jnp.isfinite(g)
    # d/dscale sum((scale*1)^2) = 2*scale*N
    assert jnp.allclose(g, 2.0 * 3.0 * state.u.data.size)
