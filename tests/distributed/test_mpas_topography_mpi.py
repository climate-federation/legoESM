"""MPI correctness scaffold for MPAS partial-cell realistic bathymetry.

P7 of ``docs/ocean_experiments/realistic_geometry_mpas_plan.md``.

**Status**: SCAFFOLD ONLY — auto-skipped today.  The MPAS *atmosphere*
has a full MPI step path (``legoesm.parallel.voronoi_mpi``,
exercised by ``test_voronoi_mpi.py``), but the MPAS *ocean* model
(``ocean_model_mpas.MPASOceanModel``) does not yet have a distributed
counterpart.  This file pins down the test contract so it runs
cleanly the moment MPASOceanModel-MPI lands.

Run with::

    mpirun -np 2 python -m pytest tests/distributed/test_mpas_topography_mpi.py -v

When the MPAS-ocean MPI step exists, the planned tests are:

1. **State scatter round-trip** — scatter MPASOceanState (including
   H_bathy, land_mask) to per-rank tiles; gather back; assert equal.
2. **Partial-cell coord scatter** — scatter
   OceanPartialCellCoordinate (h_partial, bottom_level, is_active);
   confirm per-rank construction matches gather + global construct.
3. **Halo exchange of H_bathy + bottom_level + is_active** — verify
   halo-cell values agree across ranks.
4. **Single-step equivalence** — N-rank step matches single-rank
   reference to PCG tolerance (the harder check; depends on the
   implicit-CN PCG converging consistently across decompositions).
5. **Mass conservation** — global ``Sum areaCell * eta`` drift
   matches single-rank baseline at PCG-tolerance level.
6. **AD through scatter** — ``jax.grad(H_bathy → eta)`` at one
   rank produces the same gradient as the single-rank version.

Open follow-up issue: implement ``MPASOceanModel`` MPI step
mirroring the atmosphere's ``make_voronoi_mpi_step`` pattern.
"""

import os
import pytest

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _skip_if_no_mpi_or_ocean_mpi():
    """Skip until the MPAS ocean MPI step lands."""
    try:
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_size() < 2:
            pytest.skip("Need at least 2 MPI ranks")
    except ImportError:
        pytest.skip("mpi4py not available")
    # The function we'd need to drive a distributed MPAS ocean step.
    # When it lands, drop this skip.
    try:
        from legoesm.parallel.voronoi_mpi import make_voronoi_mpi_ocean_step  # noqa: F401
    except ImportError:
        pytest.skip(
            "MPAS ocean MPI step not yet implemented — see P7 of "
            "docs/ocean_experiments/realistic_geometry_mpas_plan.md"
        )


def test_partial_cell_state_scatter_roundtrip_placeholder():
    """Placeholder — see module docstring for the planned contract."""
    _skip_if_no_mpi_or_ocean_mpi()


def test_partial_cell_step_matches_single_rank_placeholder():
    _skip_if_no_mpi_or_ocean_mpi()


def test_partial_cell_mass_conservation_under_mpi_placeholder():
    _skip_if_no_mpi_or_ocean_mpi()


def test_partial_cell_grad_through_mpi_placeholder():
    _skip_if_no_mpi_or_ocean_mpi()
