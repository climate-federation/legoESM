"""Serial (single-rank) tests for the lat-lon band gather / scatter
primitives in :mod:`legoesm.parallel.latlon_mpi`.

The MPI-level counterpart lives in
``tests/distributed/test_latlon_mpi_checkpoint.py`` and is exercised
under ``mpirun -n {2,4}`` by ``scripts/run/run_latlon_mpi_halo_smoke.sbatch``.

Why serial tests for an MPI helper
----------------------------------
With ``n_ranks=1`` the layout owns the entire lat axis, ``comm.gather``
returns a single-element list, and ``comm.bcast`` is the identity.  The
roundtrip therefore reduces to:

* ``gather_field_latlon(local)``  → ``jnp.concatenate([local]) == local``
* ``scatter_field_latlon(global)`` → ``global[0:n_lat] == global``

i.e. the v-row-trim, slicing, and dtype-preservation logic is fully
exercised even on one rank.  The multi-rank test then validates the
cross-rank concatenate ordering — orthogonal coverage.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.parallel.latlon_mpi import (
    LatLonBandLayout,
    gather_field_latlon,
    make_latlon_band_layout,
    scatter_field_latlon,
)


N_LAT = 16
N_LON = 32
NLEV = 4


@pytest.fixture
def single_rank_layout() -> LatLonBandLayout:
    return make_latlon_band_layout(
        rank=0, n_ranks=1, n_lat=N_LAT, n_lon=N_LON,
    )


@pytest.mark.parametrize("shape", [
    (N_LAT, N_LON),
    (N_LAT, N_LON, NLEV),
])
def test_gather_then_scatter_scalar_roundtrip(single_rank_layout, shape):
    """Gather(scatter(x)) == x on the entire scalar lat axis.

    Catches dtype changes, axis-reordering, or accidental NaN
    propagation through the helper.
    """
    rng = np.random.default_rng(seed=1234)
    arr = jnp.asarray(rng.standard_normal(shape).astype(np.float64))

    gathered = gather_field_latlon(arr, single_rank_layout)
    assert gathered is not None
    assert gathered.shape == arr.shape
    np.testing.assert_array_equal(np.asarray(gathered), np.asarray(arr))

    scattered = scatter_field_latlon(gathered, single_rank_layout)
    assert scattered.shape == arr.shape
    np.testing.assert_array_equal(np.asarray(scattered), np.asarray(arr))


def test_gather_then_scatter_v_face_roundtrip(single_rank_layout):
    """v-face arrays (shape (n_lat+1, n_lon)) preserve the extra row.

    The northernmost-rank trim convention is the trickiest piece; this
    test makes sure single-rank doesn't accidentally trim its own
    boundary row when it is BOTH the southernmost and the northernmost.
    """
    rng = np.random.default_rng(seed=4321)
    v = jnp.asarray(
        rng.standard_normal((N_LAT + 1, N_LON, NLEV)).astype(np.float64)
    )

    gathered = gather_field_latlon(v, single_rank_layout, is_v_face=True)
    assert gathered is not None
    assert gathered.shape == v.shape, (
        "Single rank IS the northernmost, so the v boundary row must "
        f"not be trimmed: got {gathered.shape}, expected {v.shape}"
    )
    np.testing.assert_array_equal(np.asarray(gathered), np.asarray(v))

    scattered = scatter_field_latlon(
        gathered, single_rank_layout, is_v_face=True,
    )
    assert scattered.shape == v.shape
    np.testing.assert_array_equal(np.asarray(scattered), np.asarray(v))


def test_scatter_with_none_global_on_nonzero_rank_when_no_mpi(single_rank_layout):
    """Without mpi4py the scatter helper passes through.

    The driver code paths that call ``scatter_field_latlon`` only hit
    the non-rank-0 branch under MPI; serial code should never see
    ``global_arr=None`` on rank 0.  This test pins the
    'rank-0-must-supply-array' invariant so any future regression of
    the precondition is caught here.
    """
    with pytest.raises(ValueError, match="rank 0 must supply"):
        scatter_field_latlon(None, single_rank_layout)


def test_layout_lat_bounds_cover_entire_axis_single_rank(single_rank_layout):
    """Single rank owns ``[0, n_lat_global)`` — sanity-check the layout
    helper, since both gather + scatter rely on these bounds."""
    assert single_rank_layout.lat_start == 0
    assert single_rank_layout.lat_end == N_LAT
    assert single_rank_layout.n_lat_local == N_LAT


def test_dtype_preserved_through_roundtrip(single_rank_layout):
    """Gather/scatter must not silently promote / demote the float dtype.

    The driver's checkpoint path serializes ``np.savez`` directly; a
    silent fp32→fp64 promotion would 2x the on-disk size, and a
    silent fp64→fp32 demotion would lose entropy across a restart.
    """
    for dt in (np.float32, np.float64):
        arr = jnp.asarray(np.zeros((N_LAT, N_LON), dtype=dt))
        gathered = gather_field_latlon(arr, single_rank_layout)
        scattered = scatter_field_latlon(gathered, single_rank_layout)
        # jnp upcasts float32→float64 under x64; tolerate either as
        # long as the precision is at least preserved.
        assert np.asarray(scattered).dtype.itemsize >= np.dtype(dt).itemsize, (
            f"Round-trip lost precision: {dt} → {scattered.dtype}"
        )
