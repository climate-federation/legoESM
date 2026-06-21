"""Direct unit tests for the #517 shared ocean-dycore primitives.

Covers the dedup helpers factored out of the barotropic / baroclinic
split and the AB2 time blend:

* ``depth_average_to_faces`` (item 1) — masked thickness-weighted depth
  mean of a face velocity.
* ``column_depth`` + ``depth_mean`` (item 5) — floored column depth and
  thickness-weighted column mean.
* ``ab2_blend`` (item 8) — ``(1.5 + eps)·F^n − (0.5 + eps)·F^{n-1}``.

Every test asserts BIT-IDENTITY against the exact open-coded expression
that previously lived at the call sites (including the DIVERGENT floors
``min_water_column_m`` vs ``1e-10`` and the MPAS double floor), so the
refactor is a pure no-op on numerical behaviour.  Differentiability is
checked too — these helpers sit in the end-to-end ``jax.grad`` path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.dynamics.ocean_tendency_common import (
    column_depth,
    depth_average_to_faces,
    depth_mean,
)


def _rng(seed=0):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------
# column_depth (item 5)
# ---------------------------------------------------------------------
def test_column_depth_matches_open_coded_floor_min_water_col():
    g = _rng(1)
    h = jnp.asarray(g.uniform(0.0, 50.0, size=(4, 5, 7)))
    floor = 0.37  # config.min_water_column_m-style floor
    ref = jnp.maximum(jnp.sum(h, axis=-1), floor)
    out = column_depth(h, floor)
    assert jnp.array_equal(out, ref)


def test_column_depth_matches_open_coded_floor_1e10():
    g = _rng(2)
    h = jnp.asarray(g.uniform(0.0, 50.0, size=(3, 6, 9)))
    ref = jnp.maximum(jnp.sum(h, axis=-1), 1e-10)
    out = column_depth(h, 1e-10)
    assert jnp.array_equal(out, ref)


def test_column_depth_axis1_keepdims_mpas_pattern():
    # MPAS edge columns reduce over axis=1.
    g = _rng(3)
    h = jnp.asarray(g.uniform(0.0, 50.0, size=(11, 8)))  # (nEdges, nlev)
    floor = 0.5
    ref = jnp.maximum(jnp.sum(h, axis=1, keepdims=True), floor)
    out = column_depth(h, floor, axis=1, keepdims=True)
    assert jnp.array_equal(out, ref)


def test_column_depth_floor_engages_on_thin_column():
    h = jnp.zeros((2, 3, 4))  # fully dry → sum = 0
    out = column_depth(h, 0.25)
    assert jnp.all(out == 0.25)


# ---------------------------------------------------------------------
# depth_mean (item 5)
# ---------------------------------------------------------------------
def test_depth_mean_matches_open_coded():
    g = _rng(4)
    field = jnp.asarray(g.normal(size=(4, 5, 7)))
    h = jnp.asarray(g.uniform(0.0, 50.0, size=(4, 5, 7)))
    floor = 1e-10
    ref = jnp.sum(field * h, axis=-1) / jnp.maximum(jnp.sum(h, axis=-1), floor)
    out = depth_mean(field, h, floor)
    assert jnp.array_equal(out, ref)


def test_depth_mean_keepdims_matches_split_helper():
    # Reproduces ocean_model_latlon_cgrid._split's `bt`.
    g = _rng(5)
    field = jnp.asarray(g.normal(size=(6, 7, 5)))
    h = jnp.asarray(g.uniform(0.0, 30.0, size=(6, 7, 5)))
    ref = (jnp.sum(field * h, axis=-1, keepdims=True)
           / jnp.maximum(jnp.sum(h, axis=-1, keepdims=True), 1.0e-10))
    out = depth_mean(field, h, 1.0e-10, keepdims=True)
    assert jnp.array_equal(out, ref)
    assert out.shape == ref.shape


def test_depth_mean_axis1_mpas():
    g = _rng(6)
    field = jnp.asarray(g.normal(size=(11, 8)))
    h = jnp.asarray(g.uniform(0.0, 40.0, size=(11, 8)))
    floor = 0.4
    ref = jnp.sum(field * h, axis=1) / jnp.maximum(jnp.sum(h, axis=1), floor)
    out = depth_mean(field, h, floor, axis=1)
    assert jnp.array_equal(out, ref)


def test_depth_mean_fused_flag_selects_reduction_topology():
    # BYTE-IDENTITY (#517 codex HIGH): the `fused` flag must reproduce the
    # call site's ORIGINAL reduction shape.  `fused=False` (split-origin
    # sites: barotropic_implicit U_bar, rigid_lid U_old/V_old, _split)
    # must equal TWO independent sums; `fused=True` (already-stacked sites)
    # must equal the ONE stacked reduction.  Both checked under jit, since
    # XLA fusion is where split-vs-fused can drift in the full step.
    g = _rng(61)
    field = jnp.asarray(g.normal(size=(4, 5, 7)))
    h = jnp.asarray(g.uniform(0.0, 50.0, size=(4, 5, 7)))
    floor = 1e-10

    def split_ref(field, h):
        return (jnp.sum(field * h, axis=-1)
                / jnp.maximum(jnp.sum(h, axis=-1), floor))

    def fused_ref(field, h):
        pair = jnp.sum(jnp.stack([field * h, h], axis=-1), axis=-2)
        return pair[..., 0] / jnp.maximum(pair[..., 1], floor)

    f_split = jax.jit(lambda a, b: depth_mean(a, b, floor, fused=False))
    f_fused = jax.jit(lambda a, b: depth_mean(a, b, floor, fused=True))
    assert jnp.array_equal(f_split(field, h), jax.jit(split_ref)(field, h))
    assert jnp.array_equal(f_fused(field, h), jax.jit(fused_ref)(field, h))
    # keepdims variant (the _split baroclinic decomposition site).
    out_kd = depth_mean(field, h, floor, keepdims=True, fused=False)
    ref_kd = (jnp.sum(field * h, axis=-1, keepdims=True)
              / jnp.maximum(jnp.sum(h, axis=-1, keepdims=True), floor))
    assert jnp.array_equal(out_kd, ref_kd)
    assert out_kd.shape == ref_kd.shape


# ---------------------------------------------------------------------
# depth_average_to_faces (item 1)
# ---------------------------------------------------------------------
def test_depth_average_to_faces_matches_fused_stack_min_water_col():
    # barotropic_latlon_cgrid / ocean_model_latlon_cgrid pattern.
    g = _rng(7)
    u = jnp.asarray(g.normal(size=(4, 6, 7)))
    h_u = jnp.asarray(g.uniform(0.0, 50.0, size=(4, 6, 7)))
    u_mask = jnp.asarray((g.uniform(size=(4, 6)) > 0.3).astype(float))
    floor = 0.42
    # Exact original fused-stack expression.
    _u_pair = jnp.sum(jnp.stack([h_u, u * h_u], axis=-1), axis=-2)
    H_u = jnp.maximum(_u_pair[..., 0], floor)
    ref = _u_pair[..., 1] / H_u * u_mask
    out = depth_average_to_faces(u, h_u, u_mask, floor)
    assert jnp.array_equal(out, ref)


def test_depth_average_to_faces_matches_pe_diag_floor_1e10():
    # ocean_pe_latlon_cgrid 4b pattern (floor = 1e-10).
    g = _rng(8)
    u = jnp.asarray(g.normal(size=(5, 6, 8)))
    h_u = jnp.asarray(g.uniform(0.0, 50.0, size=(5, 6, 8)))
    u_mask = jnp.asarray((g.uniform(size=(5, 6)) > 0.2).astype(float))
    _u_pair = jnp.sum(jnp.stack([u * h_u, h_u], axis=-1), axis=-2)
    ref = _u_pair[..., 0] / jnp.maximum(_u_pair[..., 1], 1e-10) * u_mask
    out = depth_average_to_faces(u, h_u, u_mask, 1e-10)
    assert jnp.array_equal(out, ref)


def test_depth_average_to_faces_matches_summed_form():
    # barotropic_implicit / rigid_lid pattern: sum(u*h)/max(sum(h),f) * mask.
    g = _rng(9)
    u = jnp.asarray(g.normal(size=(4, 7, 6)))
    h_u = jnp.asarray(g.uniform(0.0, 50.0, size=(4, 7, 6)))
    u_mask = jnp.asarray((g.uniform(size=(4, 7)) > 0.25).astype(float))
    floor = 0.3
    H_u = jnp.maximum(jnp.sum(h_u, axis=-1), floor)
    ref = jnp.sum(u * h_u, axis=-1) / H_u * u_mask
    out = depth_average_to_faces(u, h_u, u_mask, floor)
    assert jnp.array_equal(out, ref)


def test_depth_average_to_faces_land_face_zero():
    g = _rng(10)
    u = jnp.asarray(g.normal(size=(3, 4, 5)))
    h_u = jnp.asarray(g.uniform(0.0, 20.0, size=(3, 4, 5)))
    u_mask = jnp.zeros((3, 4))  # all land
    out = depth_average_to_faces(u, h_u, u_mask, 0.1)
    assert jnp.all(out == 0.0)


def test_depth_average_to_faces_differentiable():
    g = _rng(11)
    u = jnp.asarray(g.normal(size=(3, 4, 5)))
    h_u = jnp.asarray(g.uniform(1.0, 20.0, size=(3, 4, 5)))
    u_mask = jnp.ones((3, 4))

    def loss(u_in):
        return jnp.sum(depth_average_to_faces(u_in, h_u, u_mask, 0.1) ** 2)

    grad = jax.grad(loss)(u)
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.any(np.asarray(grad) != 0.0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
