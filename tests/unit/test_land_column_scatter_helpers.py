"""Single-process contract tests for the flattened-column scatter/gather
helpers that ModelDriver uses to distribute the multilayer-land soil columns
under cube-face MPI (``model_driver._scatter_flat_columns`` etc.).

The MPI parity (gathered N-rank soil == serial) is gated by
``tests/distributed/test_multilayer_land_scatter_mpi.py``; here we pin the
reshape/ordering + pytree-leaf-selection logic without a launcher (a
single-rank layout makes scatter/gather the identity).
"""

from __future__ import annotations

import collections

import jax.numpy as jnp
import numpy as np

from legoesm.driver.model_driver import (
    _gather_flat_columns,
    _map_flat_column_leaves,
    _scatter_flat_columns,
)
from legoesm.parallel.layout import make_layout

_N = 4
_NCOL = 6 * _N * _N


def test_single_rank_scatter_is_identity_and_preserves_column_order():
    """With one rank the scatter is the identity — and the (6*n*n,...) ->
    (6,n,n,...) -> (6*n*n,...) round-trip must preserve the exact column
    order the ColumnAdapter uses (face-major, row-major within a face)."""
    layout = make_layout(0, 1, _N)
    for trailing in ((), (8,), (3, 2)):
        arr = jnp.asarray(
            np.arange(int(np.prod((_NCOL,) + trailing)), dtype=np.float64)
            .reshape((_NCOL,) + trailing))
        out = _scatter_flat_columns(arr, layout, _N)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(arr))
        back = _gather_flat_columns(out, layout, _N)
        np.testing.assert_array_equal(np.asarray(back), np.asarray(arr))


def test_reshape_matches_face_raster():
    """A flattened array built as ``(6,n,n).reshape(-1)`` must round-trip
    through the (6,n,n) reshape unchanged — i.e. the helper's implicit column
    ordering IS the face-major row-major raster the physics adapter flattens,
    so a soil column lands on its own (face,i,j)."""
    faces = jnp.asarray(
        np.arange(6 * _N * _N, dtype=np.float64).reshape(6, _N, _N))
    flat = faces.reshape(-1)
    layout = make_layout(0, 1, _N)
    recon = _scatter_flat_columns(flat, layout, _N).reshape(6, _N, _N)
    np.testing.assert_array_equal(np.asarray(recon), np.asarray(faces))


def test_map_selects_only_column_leaves():
    """_map_flat_column_leaves touches ONLY leaves whose leading axis equals
    the column count; scalars, config ints, and differently-shaped params
    pass through untouched (so scattering the params pytree never mangles a
    per-PFT table or a scalar hyperparameter)."""
    Tree = collections.namedtuple("Tree", "col_1d col_2d n_pft scalar")
    tree = Tree(
        col_1d=jnp.arange(_NCOL, dtype=jnp.float64),
        col_2d=jnp.ones((_NCOL, 5)),
        n_pft=jnp.ones((17, 3)),          # per-PFT table — NOT a column leaf
        scalar=jnp.asarray(2.5),          # scalar — NOT a column leaf
    )
    seen = []

    def fn(x, n):
        seen.append(x.shape)
        return x * 2.0

    out = _map_flat_column_leaves(tree, _N, _NCOL, fn)
    # only the two column leaves were transformed
    assert sorted(seen) == sorted([(_NCOL,), (_NCOL, 5)])
    np.testing.assert_array_equal(np.asarray(out.n_pft), np.asarray(tree.n_pft))
    np.testing.assert_array_equal(
        np.asarray(out.scalar), np.asarray(tree.scalar))
    np.testing.assert_allclose(
        np.asarray(out.col_1d), 2.0 * np.asarray(tree.col_1d))
