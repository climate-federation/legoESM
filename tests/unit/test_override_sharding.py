"""Localize a GLOBAL per-column turbulence override to a rank's MPI tile.

Unit-covers ``legoesm.atmosphere.physics.turbulence.override_sharding`` — the
driver-path (lenient) slicer the dycore build calls so a deployed GLOBAL
per-column ``clubb_lite`` override reaches each rank's LOCAL columns.  The real
2-rank distributed validation lives in
``tests/distributed/test_latlon_mpi_override.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.turbulence.config import (
    CLUBBLiteConfig,
    TurbulenceConfig,
)
from legoesm.atmosphere.physics.turbulence.override_sharding import (
    localize_turbulence_override,
)
from legoesm.parallel.latlon_mpi import (
    make_latlon_2d_layout,
    make_latlon_band_layout,
)

N_LAT, N_LON = 4, 4
NCOL = N_LAT * N_LON


def _global_override(ncol=NCOL):
    return TurbulenceConfig(scheme="clubb_lite", clubb_lite=CLUBBLiteConfig(
        C_K=jnp.arange(ncol, dtype=jnp.float64) * 0.01 + 0.3))


def _band(rank, n_ranks):
    return make_latlon_band_layout(rank=rank, n_ranks=n_ranks, n_lat=N_LAT, n_lon=N_LON)


def test_band_single_rank_identity():
    ov = _global_override()
    local = localize_turbulence_override(ov, _band(0, 1))
    np.testing.assert_allclose(
        np.asarray(local.clubb_lite.C_K), np.asarray(ov.clubb_lite.C_K))


def test_band_multirank_reassembles():
    ov = _global_override()
    parts = [np.asarray(localize_turbulence_override(ov, _band(r, 2)).clubb_lite.C_K)
             for r in range(2)]
    np.testing.assert_allclose(np.concatenate(parts), np.asarray(ov.clubb_lite.C_K))


def test_2d_pencil_slices():
    ov = _global_override()
    lay = make_latlon_2d_layout(3, 2, 2, N_LAT, N_LON)  # rank 3 = SE block
    local = np.asarray(localize_turbulence_override(ov, lay).clubb_lite.C_K)
    gidx = (np.arange(NCOL).reshape(N_LAT, N_LON)
            [lay.lat_start:lay.lat_end, lay.lon_start:lay.lon_end].reshape(-1))
    np.testing.assert_allclose(local, np.asarray(ov.clubb_lite.C_K)[gidx])


# --------------------------------------------------------------------------- #
# Lenient pass-through: anything that does NOT need slicing is returned verbatim.
# --------------------------------------------------------------------------- #
def test_scalar_override_passthrough():
    ov = TurbulenceConfig(scheme="clubb_lite", clubb_lite=CLUBBLiteConfig(C_K=0.7))
    assert localize_turbulence_override(ov, _band(0, 2)) is ov


def test_non_clubb_override_passthrough():
    ov = TurbulenceConfig(scheme="mynn25")
    assert localize_turbulence_override(ov, _band(0, 2)) is ov


def test_already_local_override_passthrough():
    # A per-column field NOT at the global ncol (e.g. already a rank's 8 columns)
    # is left as-is — the localizer is idempotent / safe to call repeatedly.
    ov = _global_override(ncol=8)            # 8 != 16 (the layout's global ncol)
    assert localize_turbulence_override(ov, _band(0, 2)) is ov


def test_none_passthrough():
    assert localize_turbulence_override(None, _band(0, 2)) is None


# --------------------------------------------------------------------------- #
# Cubed-sphere: reuse the model's own `scatter` partition (face-only + tiled).
# --------------------------------------------------------------------------- #
def _cubed_global_override(n):
    ncol = 6 * n * n
    return TurbulenceConfig(scheme="clubb_lite", clubb_lite=CLUBBLiteConfig(
        C_K=jnp.arange(ncol, dtype=jnp.float64) * 0.001 + 0.3))


@pytest.mark.parametrize("n_ranks", [1, 2, 3, 6])
def test_cubed_sphere_face_only_reassembles(n_ranks):
    from legoesm.parallel.layout import make_layout, scatter

    n = 4
    ov = _cubed_global_override(n)
    global_ck = np.asarray(ov.clubb_lite.C_K)
    rebuilt = np.full(6 * n * n, np.nan)
    for rank in range(n_ranks):
        layout = make_layout(rank, n_ranks, n)
        local = np.asarray(
            localize_turbulence_override(ov, layout).clubb_lite.C_K)
        # The local slice must equal the model's own scatter of the global field.
        expected = np.asarray(scatter(global_ck.reshape(6, n, n), layout)).reshape(-1)
        np.testing.assert_allclose(local, expected)
        # Map back to global positions to confirm a complete, non-overlapping cover.
        gidx = np.asarray(
            scatter(np.arange(6 * n * n).reshape(6, n, n), layout)).reshape(-1)
        rebuilt[gidx] = local
    np.testing.assert_allclose(rebuilt, global_ck)


def test_cubed_sphere_tiled_slices():
    # >6 ranks → sub-face tiling: localize must still match the model's scatter.
    from legoesm.parallel.layout import make_layout, scatter

    n, n_ranks = 4, 24            # 4 tiles/face
    ov = _cubed_global_override(n)
    global_ck = np.asarray(ov.clubb_lite.C_K)
    layout = make_layout(7, n_ranks, n)
    local = np.asarray(localize_turbulence_override(ov, layout).clubb_lite.C_K)
    expected = np.asarray(scatter(global_ck.reshape(6, n, n), layout)).reshape(-1)
    np.testing.assert_allclose(local, expected)


# --------------------------------------------------------------------------- #
# Cubed-sphere via the halo CommTopology → resolve to the active DistributedLayout.
# --------------------------------------------------------------------------- #
def _comm_topology(rank=0, n_processes=2):
    from legoesm.parallel.comm import CommTopology
    return CommTopology(
        rank=rank, n_processes=n_processes, local_face_ids=(0, 1, 2),
        neighbor_ranks={}, neighbor_info={}, tiling=(1, 1), tile_index=(0, 0),
        tile_neighbors={})


def test_comm_topology_resolves_to_active_layout():
    # get_mpi_topology() returns a CommTopology under cubed MPI; localize must
    # resolve it to the active DistributedLayout (carrying global_n) and slice.
    from legoesm.parallel.distributed import set_active_layout
    from legoesm.parallel.layout import make_layout, scatter

    n = 4
    layout = make_layout(rank=0, n_ranks=2, global_n=n)
    set_active_layout(layout)
    try:
        ov = _cubed_global_override(n)
        local = np.asarray(
            localize_turbulence_override(ov, _comm_topology()).clubb_lite.C_K)
        expected = np.asarray(
            scatter(np.asarray(ov.clubb_lite.C_K).reshape(6, n, n), layout)
        ).reshape(-1)
        np.testing.assert_allclose(local, expected)
    finally:
        set_active_layout(None)


def test_comm_topology_without_active_layout_raises():
    from legoesm.parallel.distributed import set_active_layout

    set_active_layout(None)
    with pytest.raises(RuntimeError, match="no DistributedLayout is registered"):
        localize_turbulence_override(_cubed_global_override(4), _comm_topology())


# --------------------------------------------------------------------------- #
# Unrecognized decomposition → pass-through (the escape hatch).
# --------------------------------------------------------------------------- #
class _UnknownLayout:
    """A layout type the localizer does not recognize."""


def test_unknown_layout_passes_through():
    ov = _global_override()
    assert localize_turbulence_override(ov, _UnknownLayout()) is ov


def test_unknown_layout_scalar_passthrough():
    ov = TurbulenceConfig(scheme="clubb_lite", clubb_lite=CLUBBLiteConfig(C_K=0.7))
    assert localize_turbulence_override(ov, _UnknownLayout()) is ov


def test_inconsistent_lengths_raise():
    ov = TurbulenceConfig(scheme="clubb_lite", clubb_lite=CLUBBLiteConfig(
        C_K=jnp.ones(NCOL), Pr_t=jnp.ones(8)))   # one global, one not
    with pytest.raises(ValueError, match="inconsistent column counts"):
        localize_turbulence_override(ov, _band(0, 2))
