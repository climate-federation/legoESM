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
    _scatter_1based_columns,
    _scatter_flat_columns,
    _slice_grid_info_to_rank,
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


# ---------------------------------------------------------------------------
# CLM-ML MPI enablement: 1-based canopy scatter + per-rank grid_info slice
# (model_driver._scatter_1based_columns / _slice_grid_info_to_rank).  scatter()
# is pure index-selection (no MPI comm), so a 2-rank DistributedLayout built
# single-process exercises REAL multi-rank column distribution on rank 0.
# ---------------------------------------------------------------------------

import collections as _collections  # noqa: E402

_GridInfo = _collections.namedtuple("_GridInfo", "p ncan ntop nbot")


def _owned_global_columns(layout, n_tile):
    """The global column indices rank 0 owns, in the SAME local order the
    scatter produces (scatter a global-index array)."""
    gidx = jnp.arange(6 * n_tile * n_tile, dtype=jnp.int32)
    return np.asarray(_scatter_flat_columns(gidx, layout, n_tile)).astype(int)


def test_scatter_1based_single_rank_is_identity():
    """One rank: the 1-based scatter is the identity (index 0 preserved, body
    round-trips) for scalar-, vector- and matrix-per-column canopy fields."""
    layout = make_layout(0, 1, _N)
    for trailing in ((), (11, 3), (11, 3, 2)):  # (ncol+1,) / profile / leaf shapes
        arr = jnp.asarray(np.arange(
            int(np.prod((_NCOL + 1,) + trailing)), dtype=np.float64)
            .reshape((_NCOL + 1,) + trailing))
        out = _scatter_1based_columns(arr, layout, _N)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(arr))


def test_scatter_1based_equals_zerobased_body_plus_sentinel():
    """The 1-based scatter == the (MPI-validated) 0-based scatter on the body,
    with index 0 preserved — proven on a REAL 2-rank layout (rank 0's columns)."""
    layout = make_layout(0, 2, _N)          # rank 0 of a 2-rank run
    arr = jnp.asarray(np.arange(
        (_NCOL + 1) * 3, dtype=np.float64).reshape((_NCOL + 1, 3)))
    out = _scatter_1based_columns(arr, layout, _N)
    body_ref = _scatter_flat_columns(arr[1:], layout, _N)   # 0-based on the body
    # index 0 sentinel preserved; body == the trusted 0-based scatter
    np.testing.assert_array_equal(np.asarray(out[0]), np.asarray(arr[0]))
    np.testing.assert_array_equal(np.asarray(out[1:]), np.asarray(body_ref))
    # rank-local body carries exactly rank 0's owned global columns (col c -> row c+1)
    owned = _owned_global_columns(layout, _N)
    np.testing.assert_array_equal(
        np.asarray(out[1:]), np.asarray(arr[1:][owned]))


def test_slice_grid_info_to_rank_picks_owned_and_renumbers_local():
    """grid_info slice: rank 0 gets ITS owned global columns' structure, with
    ``.p`` renumbered to local 1-based 1..n_local (what the interface requires)."""
    layout = make_layout(0, 2, _N)
    # global grid_info: entry c is global patch c+1 with a distinctive ncan.
    gi = tuple(_GridInfo(p=c + 1, ncan=100 + c, ntop=c, nbot=1) for c in range(_NCOL))
    local = _slice_grid_info_to_rank(gi, layout, _N, _NCOL)
    owned = _owned_global_columns(layout, _N)
    assert len(local) == len(owned)
    # local patch numbering is 1..n_local, in scattered order
    assert [g.p for g in local] == list(range(1, len(owned) + 1))
    # each local entry carries the OWNED global column's structure (ncan tags it)
    assert [g.ncan for g in local] == [100 + int(c) for c in owned]


def test_slice_grid_info_single_rank_is_identity():
    """One rank owns all columns: the slice is the identity (p already 1..ncol)."""
    layout = make_layout(0, 1, _N)
    gi = tuple(_GridInfo(p=c + 1, ncan=100 + c, ntop=c, nbot=1) for c in range(_NCOL))
    local = _slice_grid_info_to_rank(gi, layout, _N, _NCOL)
    assert local == gi
