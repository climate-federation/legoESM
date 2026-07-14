"""Single-process unit tests for cubed-sphere MPI face-scatter slicing.

These cover the rank-local grid/metric slicing in isolation (no MPI): correct
owned-face shapes, the FULL-keep of the global-face-indexed halo tables, and
the dispatch guard.  The replicated-vs-scattered numerical equivalence (which
needs ≥2 ranks) lives in ``tests/distributed/test_cube_face_scatter_mpi.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.parallel.cube_face_scatter import (
    _GLOBAL_FACE_INDEXED_FIELDS,
    make_rank_local_cube_model,
    slice_cubed_sphere_cdgrid,
    slice_cubed_sphere_grid,
)

_N = 8
_OWNED = (2, 3)  # the np=3 rank-1 partition


@pytest.fixture(scope="module")
def grid():
    return create_cubed_sphere(_N)


@pytest.fixture(scope="module")
def cdgrid(grid):
    return create_cubed_sphere_cdgrid(grid)


def test_grid_geometric_metrics_sliced_to_owned_faces(grid):
    sg = slice_cubed_sphere_grid(grid, _OWNED)
    # Representative geometric metrics drop to len(owned) on the leading axis.
    for name in ("lat", "lon", "area", "dx", "dy"):
        full = getattr(grid, name)
        sl = getattr(sg, name)
        assert sl.shape == (len(_OWNED),) + full.shape[1:], name
        # Values are exactly faces [2, 3] of the full array.
        assert jnp.array_equal(sl, full[jnp.asarray(_OWNED)]), name


def test_grid_keeps_interp_offsets_and_duogrid_full(grid):
    """interp_offsets / duogrid are GLOBAL-face-indexed by the halo machinery —
    they must NOT be sliced even though their leading dim is 6."""
    sg = slice_cubed_sphere_grid(grid, _OWNED)
    for name in _GLOBAL_FACE_INDEXED_FIELDS:
        full = getattr(grid, name, None)
        if full is None:
            continue
        sl = getattr(sg, name)
        if isinstance(full, (jax.Array, jnp.ndarray)):
            assert sl.shape == full.shape, f"{name} must stay full {full.shape}"
            assert jnp.array_equal(sl, full), name
        else:
            # duogrid (DuoGridData or None) kept by identity.
            assert sl is full, name


def test_cdgrid_metrics_sliced_and_base_recursed(grid, cdgrid):
    sc = slice_cubed_sphere_cdgrid(cdgrid, _OWNED)
    # A C/D metric slices on the leading face axis.
    assert sc.cosa_u.shape == (len(_OWNED),) + cdgrid.cosa_u.shape[1:]
    assert jnp.array_equal(sc.cosa_u, cdgrid.cosa_u[jnp.asarray(_OWNED)])
    # The nested base grid is recursed: geometric sliced, interp tables full.
    assert sc.base.area.shape[0] == len(_OWNED)
    assert sc.base.halo_interp_offsets.shape[0] == 6


def test_owned_face_index_validation(grid):
    with pytest.raises(ValueError):
        slice_cubed_sphere_grid(grid, ())  # empty
    with pytest.raises(ValueError):
        slice_cubed_sphere_grid(grid, (5, 6))  # out of range


def test_make_rank_local_model_rejects_zero_mean_ps_tendency(grid):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationConfig,
        CDGridPrimitiveEquationModel,
    )
    from legoesm.grids.vertical import create_sigma_coordinate

    sigma = create_sigma_coordinate(4)
    cfg = CDGridPrimitiveEquationConfig(zero_mean_ps_tendency=True)
    model = CDGridPrimitiveEquationModel(grid, sigma, cfg)
    with pytest.raises(NotImplementedError, match="zero_mean_ps_tendency"):
        make_rank_local_cube_model(model, _OWNED)


def test_make_rank_local_model_slices_and_clears_target(grid):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationConfig,
        CDGridPrimitiveEquationModel,
    )
    from legoesm.grids.vertical import create_sigma_coordinate

    sigma = create_sigma_coordinate(4)
    cfg = CDGridPrimitiveEquationConfig(
        fix_mass=True, anchor_mass_to_initial=True, zero_mean_ps_tendency=False
    )
    model = CDGridPrimitiveEquationModel(grid, sigma, cfg)
    model._target_mass = 12345.0  # pretend a stale full-grid target was cached
    make_rank_local_cube_model(model, _OWNED)
    assert model.grid.area.shape[0] == len(_OWNED)
    assert model.cdgrid.cosa_u.shape[0] == len(_OWNED)
    assert model.grid.halo_interp_offsets.shape[0] == 6  # kept full
    assert model._target_mass is None  # recomputed from scattered IC
