"""Partial CMOR flux interval across a MULTI-rank MPAS restart (#2171).

The per-step flux sums behind the CMOR interval means (#1353) are per-cell and
rank-local under a cell partition.  They used to be dropped at every
multi-rank checkpoint, so each chain link lost one diagnostic interval.  They
are now gathered to the global checkpoint like the other per-cell payloads and
scattered back on load, so the next link resumes the interval.

Guards:
* the checkpoint holds GLOBAL sums at the right global slots (every rank
  checks its own owned cells against the file);
* the load cuts them back to this rank, value-exact on owned cells;
* the next run actually RESUMES the interval (restore called, slots restored)
  instead of dropping it.

Run under MPI::

    mpirun -n 2 python -m pytest tests/distributed/test_mpas_cmor_flux_restart_mpi.py
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

DT = 300.0
N_PRIOR = 10          # samples already in the interval at the checkpoint
SLOTS = (0, 6)        # two slots, so a slot mix-up cannot pass


def _shared_tmpdir(prefix):
    comm = MPI.COMM_WORLD
    d = None
    if comm.Get_rank() == 0:
        d = tempfile.mkdtemp(prefix=prefix, dir=os.environ.get(
            "LEGOESM_TEST_SHARED_TMP", "/work/bd1083/b309178/diffESM"))
    return comm.bcast(d, root=0)


def _driver(out):
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=3, nlev=20,
                        vertical_coord="hybrid"),
        dycore=DycoreConfig(discretization="mpas", dt=DT),
        output=OutputConfig(output_dir="", diag_days=1, cmip_output=True,
                            monthly_means=True, cmip_resolution_deg=10.0),
        days=1, dataset="analytical", radiation="gray", convection="none",
        turbulence="none", precision="fp64", distributed=True,
    )
    d = md.ModelDriver(cfg, output_dir=out)
    d.setup()
    assert md._is_mpas_cell_partitioned(d), "not a multi-rank cell partition"
    return d


def _planted(gids, slot):
    """Rank-invariant per-cell value: same global cell, same number."""
    return 1000.0 * slot + np.asarray(gids, dtype=np.float64)


def test_partial_flux_interval_survives_a_multirank_restart(monkeypatch):
    comm = MPI.COMM_WORLD
    out = _shared_tmpdir("cmorflux_ckpt_")
    a = _driver(out)
    a.run()
    part = a._voronoi_layout.partition
    gids = np.asarray(part.local_cells)
    n_owned = part.n_owned_cells

    # A partial interval: N_PRIOR samples into a 1-day window ending at day 2.
    acc = md._MPASSfcFluxAccum(expected_steps=int(86400.0 / DT),
                               window_start_day=1.0 - N_PRIOR * DT / 86400.0,
                               dt_s=DT)
    for s in SLOTS:
        acc._sum[s] = _planted(gids, s)
        acc._n[s] = N_PRIOR
    acc._steps = N_PRIOR
    a._mpas_sfc_accum = acc
    a.save_checkpoint(int(round(86400.0 / DT)), 1.0)
    comm.Barrier()
    path = os.path.join(out, "checkpoint_day_0001.npz")

    with np.load(path) as z:
        for s in SLOTS:
            key = f"cmor_fluxsum_{s}"
            assert key in z.files, (
                f"{key} missing: the partial interval was dropped at the "
                "multi-rank checkpoint")
            glob = np.asarray(z[key])
            assert glob.shape == (part.nCells_global,), glob.shape
            np.testing.assert_array_equal(
                glob[gids[:n_owned]], _planted(gids[:n_owned], s),
                err_msg="this rank's owned sums landed at wrong global slots")

    restored = []
    real_restore = md._MPASSfcFluxAccum.restore

    def _spy(self, staged, **kw):
        n = real_restore(self, staged, **kw)
        restored.append((n, self._steps,
                         {s: np.asarray(self._sum[s]) for s in self._sum}))
        return n
    monkeypatch.setattr(md._MPASSfcFluxAccum, "restore", _spy)

    b = _driver(out)
    step0, day0 = b.load_checkpoint(path)
    for s in SLOTS:
        staged = np.asarray(b._carry_aux[f"cmor_fluxsum_{s}"])
        assert staged.shape == (part.n_local_cells,), staged.shape
        np.testing.assert_array_equal(
            staged[:n_owned], _planted(gids[:n_owned], s),
            err_msg="load did not cut the global sums back to this rank")
    b.run(start_step=step0, start_day=day0)

    assert restored, "the run never offered the staged interval to restore"
    n, steps, sums = restored[0]
    assert n == len(SLOTS), (
        f"restored {n} slots, expected {len(SLOTS)}: the multi-rank link "
        "dropped the partial interval")
    assert steps == N_PRIOR
    for s in SLOTS:
        np.testing.assert_array_equal(sums[s][:n_owned],
                                      _planted(gids[:n_owned], s))
