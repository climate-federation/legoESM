"""NEMO-order QCO metric transports and continuity divergence."""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    _nemo_external_mode_boundary_association,
    _nemo_ssh_avg_apply,
    nemo_literal_continuity_divergence,
    nemo_literal_metric_transports,
)
from legoesm.core.source_rounding import nemo_source_round
from legoesm.grids.tripole import (
    create_synthetic_tripole,
    create_synthetic_tripole_pivot,
)


def _case():
    # Deliberately non-power-of-two values make association/order observable.
    h_u = jnp.array([[3.1, 4.7, 5.9, 3.1], [7.3, 8.9, 9.7, 7.3]], dtype=jnp.float64)
    h_v = jnp.array([[0.0, 0.0, 0.0], [2.3, 3.7, 4.1], [0.0, 0.0, 0.0]], dtype=jnp.float64)
    u = jnp.array([[0.017, -0.023, 0.031, 0.017], [-0.043, 0.053, -0.061, -0.043]], dtype=jnp.float64)
    v = jnp.array([[0.0, 0.0, 0.0], [0.071, -0.083, 0.097], [0.0, 0.0, 0.0]], dtype=jnp.float64)
    um = jnp.ones_like(h_u)
    vm = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=jnp.float64)
    grid = SimpleNamespace(
        dy=jnp.array([2.0 * 39711.3, 2.0 * 51227.9], dtype=jnp.float64),
        dx_v=jnp.array([[0.0, 0.0, 0.0], [43319.7, 43319.7, 43319.7],
                        [0.0, 0.0, 0.0]], dtype=jnp.float64),
        area=jnp.array([[1.31e9, 1.31e9, 1.31e9],
                        [2.07e9, 2.07e9, 2.07e9]], dtype=jnp.float64),
    )
    return h_u, h_v, u, v, um, vm, grid


def test_matches_nemo_literal_source_order_bitwise():
    h_u, h_v, u, v, um, vm, grid = _case()
    got_u, got_v = nemo_literal_metric_transports(h_u, h_v, u, v, um, vm, grid)
    got = nemo_literal_continuity_divergence(h_u, h_v, u, v, um, vm, grid)
    zh_u = ((grid.dy[:, None] * 0.5 * u) * h_u) * um
    zh_v = ((grid.dx_v * v) * h_v) * vm
    expected = ((zh_u[:, 1:] - zh_u[:, :-1])
                + (zh_v[1:] - zh_v[:-1])) * (1.0 / grid.area)
    np.testing.assert_array_equal(np.asarray(got_u), np.asarray(zh_u))
    np.testing.assert_array_equal(np.asarray(got_v), np.asarray(zh_v))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_closed_periodic_domain_telescopes_area_weighted_divergence():
    h_u, h_v, u, v, um, vm, grid = _case()
    div = nemo_literal_continuity_divergence(h_u, h_v, u, v, um, vm, grid)
    assert abs(float(jnp.sum(div * grid.area))) < 1.0e-10


def test_private_unmasked_v_transport_omits_only_final_v_mask():
    h_u, h_v, u, v, um, vm, grid = _case()
    h_v = h_v.at[0, 1].set(3.25)
    v = v.at[0, 1].set(-0.125)
    grid.dx_v = grid.dx_v.at[0, 1].set(17.0)
    masked_u, masked_v = nemo_literal_metric_transports(
        h_u, h_v, u, v, um, vm, grid)
    unmasked_u, unmasked_v = nemo_literal_metric_transports(
        h_u, h_v, u, v, um, vm, grid, mask_v=False)

    np.testing.assert_array_equal(np.asarray(unmasked_u), np.asarray(masked_u))
    np.testing.assert_array_equal(
        np.asarray(unmasked_v), np.asarray((grid.dx_v * v) * h_v))
    changed = np.argwhere(np.asarray(masked_v) != np.asarray(unmasked_v))
    np.testing.assert_array_equal(changed, np.array([[0, 1]]))


def test_gradient_is_finite():
    h_u, h_v, u, v, um, vm, grid = _case()

    def loss(uu):
        div = nemo_literal_continuity_divergence(h_u, h_v, uu, v, um, vm, grid)
        return jnp.sum(div * div)

    grad = jax.grad(loss)(u)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_rich_geometry_uses_full_u_face_metric():
    h_u, h_v, u, v, um, vm, grid = _case()
    grid.dy_u = jnp.array([[39711.3, 39712.3, 39713.3, 39714.3],
                           [51227.9, 51228.9, 51229.9, 51230.9]],
                          dtype=jnp.float64)
    got_u, _ = nemo_literal_metric_transports(h_u, h_v, u, v, um, vm, grid)
    expected = ((grid.dy_u * u) * h_u) * um
    np.testing.assert_array_equal(np.asarray(got_u), np.asarray(expected))


def test_external_mode_boundary_association_is_one_seven_field_map():
    grid = create_synthetic_tripole(6, 8)
    u = jnp.arange(6 * 9, dtype=jnp.float64).reshape(6, 9)
    v = jnp.arange(7 * 8, dtype=jnp.float64).reshape(7, 8)
    depth_u = u + 100.0
    depth_v = v + 200.0
    inverse_u = u + 300.0
    inverse_v = v + 400.0
    eta = jnp.arange(6 * 8, dtype=jnp.float64).reshape(6, 8)
    got = _nemo_external_mode_boundary_association(
        u, v, depth_u, depth_v, inverse_u, inverse_v, eta, grid)

    for value, source in ((got[0], u), (got[2], depth_u),
                          (got[4], inverse_u)):
        np.testing.assert_array_equal(np.asarray(value[:, 0]),
                                      np.asarray(source[:, -1]))
        np.testing.assert_array_equal(np.asarray(value[:, 1:]),
                                      np.asarray(source[:, 1:]))
    for value, source, sign in ((got[1], v, -1.0),
                                (got[3], depth_v, 1.0),
                                (got[5], inverse_v, 1.0)):
        np.testing.assert_array_equal(np.asarray(value[0]),
                                      np.zeros(8, dtype=np.float64))
        np.testing.assert_array_equal(np.asarray(value[1:-1]),
                                      np.asarray(source[1:-1]))
        expected_north = sign * source[-2, grid.fold.perm_T]
        np.testing.assert_array_equal(np.asarray(value[-1]),
                                      np.asarray(expected_north))
    np.testing.assert_array_equal(np.asarray(got[6]), np.asarray(eta))


def test_external_mode_u_fold_preserves_signed_zero_semantics():
    grid = create_synthetic_tripole_pivot(4, 8)
    u = jnp.zeros((4, 9), dtype=jnp.float64).at[-1, 1:5].set(
        jnp.array([0.0, -0.0, 0.0, -0.0], dtype=jnp.float64))
    scalar_u = jnp.ones_like(u)
    v = jnp.ones((5, 8), dtype=jnp.float64)
    eta = jnp.ones((4, 8), dtype=jnp.float64)
    got = _nemo_external_mode_boundary_association(
        u, v, scalar_u, v, scalar_u, v, eta, grid)[0]
    expected = -np.asarray(u[-1, 4:0:-1])
    np.testing.assert_array_equal(
        np.asarray(got[-1, 5:]).view(np.uint64), expected.view(np.uint64))


def test_external_mode_u_fold_is_compiled_half_row_map():
    grid = create_synthetic_tripole_pivot(4, 8)
    u = jnp.arange(4 * 9, dtype=jnp.float64).reshape(4, 9)
    scalar_u = jnp.ones_like(u)
    v = jnp.ones((5, 8), dtype=jnp.float64)
    eta = jnp.ones((4, 8), dtype=jnp.float64)
    got = _nemo_external_mode_boundary_association(
        u, v, scalar_u, v, scalar_u, v, eta, grid)[0]
    np.testing.assert_array_equal(np.asarray(got[-1, 1:5]),
                                  np.asarray(u[-1, 1:5]))
    np.testing.assert_array_equal(np.asarray(got[-1, 5:]),
                                  np.asarray(-u[-1, 4:0:-1]))


def test_t_pivot_v_depth_uses_associated_row_below_as_north_neighbor():
    grid = create_synthetic_tripole_pivot(4, 8, dtype=jnp.float64)
    u_mask = jnp.ones((4, 9), dtype=jnp.float64)
    v_mask = jnp.ones((5, 8), dtype=jnp.float64)
    eta = jnp.arange(32, dtype=jnp.float64).reshape(4, 8) * 0.013
    prep = (
        jnp.full((4, 9), 1000.0, dtype=jnp.float64),
        jnp.full((5, 8), 1000.0, dtype=jnp.float64),
        jnp.full((4, 9), 1.0e-9, dtype=jnp.float64),
        jnp.full((5, 8), 1.0e-9, dtype=jnp.float64),
        None,
    )

    def evaluate(use_row_below):
        return _nemo_ssh_avg_apply(
            eta, u_mask, v_mask, grid, grid.area, prep,
            nemo_t_pivot_north_neighbor=use_row_below)[1]

    old = evaluate(False)
    got = evaluate(True)
    got_jit = jax.jit(evaluate, static_argnums=0)(True)

    fold = grid.fold
    b = nemo_source_round
    local = b(grid.area[-1] * eta[-1])
    north = b(
        grid.area[-2, fold.perm_T] * eta[-2, fold.perm_T])
    scale = b(0.5 * prep[3][-1])
    ssh_avg = b(b(scale * b(local + north)) * v_mask[-1])
    expected = b(prep[1][-1] + ssh_avg)

    np.testing.assert_array_equal(np.asarray(got[-1]), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(got_jit), np.asarray(got))
    assert not np.array_equal(np.asarray(old[-1]), np.asarray(expected))
