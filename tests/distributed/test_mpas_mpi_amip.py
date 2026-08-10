"""End-to-end MPAS (Voronoi/TRiSK) AMIP under MPI: serial-vs-MPI parity.

Run under MPI::

    mpirun -n 2 python -m pytest tests/distributed/test_mpas_mpi_amip.py
    mpirun -n 4 python -m pytest tests/distributed/test_mpas_mpi_amip.py

Companion to ``tests/distributed/test_latlon_mpi_amip.py`` (the lat-lon
band analogue) and ``tests/distributed/test_voronoi_mpi.py`` (which
validates the bare dynamical-core step, not the AMIP physics column).

What this guards
----------------
The MPAS AMIP-under-MPI path: ``_create_grid`` partitions the global
Voronoi mesh into this rank's owned+halo cells; the step
(``make_voronoi_mpi_step(..., return_phys_state=True)``) does the full
operator-split — dynamics RK with per-stage cell/edge/tracer halo
exchange, physics with threaded SST forcing + ``phys_state`` carry, then
the owned-cell global mass fixer.

Strategy (deadlock-free, no gather): every rank *independently* runs the
serial global model (``distributed=False`` ⇒ no MPI collectives) to build
the reference, then runs the 2-/4-rank MPI model and compares its
owned-cell state against the serial global state at the SAME global cell
indices (``partition.local_cells[:n_owned]``).  The serial blocks carry no
collectives, and the MPI collectives are matched across ranks, so the two
phases interleave without deadlock.

Tolerance: the residual is the mass-fixer global-reduction order
(allreduce vs a single sequential sum) seeding a small, *damping*
perturbation that the dynamics spreads — the dynamics halo itself is
bit-exact (see ``test_voronoi_mpi``).  We assert a loose absolute bound
that the validated run clears with margin (T ~3.7e-2 K, p_s ~1.1 Pa at
day 1 on the level-3 mesh) and, crucially, that the MPI run is finite and
physical.
"""
from __future__ import annotations

import numpy as np
import pytest

mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

from legoesm.driver.config import (  # noqa: E402
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
    OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver  # noqa: E402

RES, NLEV, NDAYS, DT = 3, 20, 1, 300.0  # level 3 = 642 cells


def _build(distributed: bool):
    import tempfile
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=RES, nlev=NLEV,
                        vertical_coord="hybrid"),
        dycore=DycoreConfig(discretization="mpas", dt=DT),
        output=OutputConfig(output_dir="", diag_days=0),
        days=NDAYS, dataset="analytical", radiation="gray",
        convection="none", turbulence="none", precision="fp64",
        distributed=distributed,
    )
    d = ModelDriver(cfg, output_dir=tempfile.mkdtemp())
    d.setup()
    return d


def test_mpas_mpi_amip_matches_serial_owned_cells():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # 1. Serial reference (independent per rank, no MPI collectives).
    ref = _build(False)
    ref.run()
    T_ref = np.asarray(ref.state.T.data)
    ps_ref = np.asarray(ref.state.p_s.data)
    u_ref = np.asarray(ref.state.u.data)
    n_global = T_ref.shape[0]

    # 2. MPI run (collectives matched across ranks).
    d = _build(True)
    # Guard: the MPAS distributed path must activate (partition built).
    assert d._voronoi_layout is not None, (
        "MPAS distributed run did not build a Voronoi partition layout"
    )
    d.run()

    part = d._voronoi_layout.partition
    n_owned = part.n_owned_cells
    owned_cell_gids = np.asarray(part.local_cells[:n_owned])
    owned_edge_gids = np.asarray(part.local_edges[:part.n_owned_edges])

    T_mpi = np.asarray(d.state.T.data[:n_owned])
    ps_mpi = np.asarray(d.state.p_s.data[:n_owned])
    u_mpi = np.asarray(d.state.u.data[:part.n_owned_edges])

    # Regression guard: state lives on the rank-local (owned+halo) mesh,
    # NOT the global mesh.
    assert d.state.T.data.shape[0] == part.n_local_cells
    assert n_owned < n_global, "partition gave a rank all global cells"

    # Physical sanity on the MPI owned-cell state.
    assert np.all(np.isfinite(T_mpi)), "non-finite T in MPI run"
    assert np.all(np.isfinite(ps_mpi)), "non-finite p_s in MPI run"
    assert 150.0 < float(T_mpi.mean()) < 350.0, (
        f"MPI mean T {float(T_mpi.mean()):.1f} K unphysical"
    )

    # Serial-vs-MPI parity at matching global indices.
    dT = np.max(np.abs(T_mpi - T_ref[owned_cell_gids]))
    dps = np.max(np.abs(ps_mpi - ps_ref[owned_cell_gids]))
    du = np.max(np.abs(u_mpi - u_ref[owned_edge_gids]))

    # Loose bounds: the validated run clears these with margin; they are
    # tight enough that a real halo / advection / mass-fix bug (which would
    # be O(1), not O(0.1)) trips them.
    assert dT < 1.0, f"T owned-cell |serial-mpi| max = {dT:.3e} K (too large)"
    assert dps < 50.0, f"p_s owned-cell |serial-mpi| max = {dps:.3e} Pa"
    assert du < 1.0, f"u owned-edge |serial-mpi| max = {du:.3e} m/s"

    if rank == 0:
        print(f"\nMPAS MPI vs serial (owned cells): "
              f"dT={dT:.3e}K dps={dps:.3e}Pa du={du:.3e}m/s")


def test_multirank_cmor_request_refuses_loudly():
    """#1517: a multi-rank MPAS run with CMOR output explicitly requested
    must REFUSE AT STARTUP (RuntimeError on every rank) — historically it
    completed "successfully" with an empty ``cmor/`` directory and an
    empty accumulator sidecar, artifacts indistinguishable from a healthy
    run (the cell-partition feed is unimplemented)."""
    import tempfile

    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("needs mpirun -n >= 2 (1-rank layouts feed fine)")

    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=RES, nlev=NLEV,
                        vertical_coord="hybrid"),
        dycore=DycoreConfig(discretization="mpas", dt=DT),
        output=OutputConfig(output_dir="", diag_days=1, cmip_output=True),
        days=NDAYS, dataset="analytical", radiation="gray",
        convection="none", turbulence="none", precision="fp64",
        distributed=True,
    )
    d = ModelDriver(cfg, output_dir=tempfile.mkdtemp())
    d.setup()
    with pytest.raises(RuntimeError, match="UNSUPPORTED under cell-partition"):
        d.run()


if __name__ == "__main__":
    test_mpas_mpi_amip_matches_serial_owned_cells()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("OK")
