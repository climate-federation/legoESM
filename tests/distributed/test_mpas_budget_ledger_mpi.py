"""--budget-ledger on the MULTI-rank MPAS lane (#2171).

The ledger used to be refused under MPI because the cell-partition step
bypassed the serial model's per-step side channel.  The MPI step now
publishes the same per-column rows; the writer gathers the owned rows to the
global mesh.  Guards: the file is written, it is GLOBAL (every cell once, with
global area weights), finite, and the dynamics row actually carries a signal.

Run under MPI::

    mpirun -n 2 python -m pytest tests/distributed/test_mpas_budget_ledger_mpi.py
"""
from __future__ import annotations

import os
import tempfile

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
from legoesm.driver import model_driver as md  # noqa: E402


def _shared_tmpdir(prefix):
    comm = MPI.COMM_WORLD
    d = None
    if comm.Get_rank() == 0:
        d = tempfile.mkdtemp(prefix=prefix, dir=os.environ.get(
            "LEGOESM_TEST_SHARED_TMP", "/work/bd1083/b309178/diffESM"))
    return comm.bcast(d, root=0)


def test_ledger_runs_and_writes_global_rows_under_mpi():
    comm = MPI.COMM_WORLD
    out = _shared_tmpdir("ledger_mpi_")
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=3, nlev=20,
                        vertical_coord="hybrid"),
        dycore=DycoreConfig(discretization="mpas", dt=300.0),
        output=OutputConfig(output_dir="", diag_days=1, checkpoint_days=0,
                            budget_ledger=True),
        days=1, dataset="analytical", radiation="gray", convection="none",
        turbulence="none", precision="fp64", distributed=True,
    )
    d = md.ModelDriver(cfg, output_dir=out)
    d.setup()
    assert md._is_mpas_cell_partitioned(d), "not a multi-rank cell partition"
    d.run()
    comm.Barrier()

    verdict = None
    if comm.Get_rank() == 0:
        path = os.path.join(out, "budget_ledger_columns.npz")
        n_global = d._voronoi_layout.partition.nCells_global
        if not os.path.exists(path):
            verdict = "no budget_ledger_columns.npz was written under MPI"
        else:
            from legoesm.diagnostics.process_ledger import ROW_DYNAMICS
            with np.load(path) as z:
                rates = np.asarray(z["ledger_rates"])
                area = np.asarray(z["area_cell"]) if "area_cell" in z else None
                part = bool(z["cell_partitioned"])
                n = int(z["n_steps"])
            if rates.shape[0] != n_global:
                verdict = (f"ledger has {rates.shape[0]} rows, global mesh "
                           f"{n_global}: rank-local rows were written")
            elif area is None or area.size != n_global:
                verdict = "area weights are missing or not global"
            elif part:
                verdict = "artifact stamped rank-local after the gather"
            elif n != 288:
                verdict = f"accumulated {n} steps, expected 288"
            elif not np.all(np.isfinite(rates)):
                verdict = "non-finite ledger rates"
            elif float(np.max(np.abs(rates[:, ROW_DYNAMICS, 1]))) == 0.0:
                verdict = "dynamics row is identically zero (vacuous ledger)"
            elif np.any(np.all(rates == 0.0, axis=(1, 2))):
                verdict = "some global cells got no rows (gather hole)"
    verdict = comm.bcast(verdict, root=0)
    assert verdict is None, verdict
