"""MPAS cell-partition MPI: real topography and the ERA5 initial condition
must be identical to a serial build on every local (owned + halo) cell.

Both are smoothed with a neighbour stencil; built on the rank-local mesh, the
outer halo has no off-rank neighbours, so the surface came out different near
every partition boundary (up to 35 m at 16 ranks on the production res6 mesh).
The production ERA5 IC is checked; the analytic default IC adds a random
lowest-level perturbation drawn per rank, so it is decomposition-dependent by
construction and not compared here.

    mpirun -n 2 python -m pytest tests/distributed/test_mpas_mpi_topography_ic.py
"""
from __future__ import annotations

import os

import numpy as np
import pytest

mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

from legoesm.driver.config import (  # noqa: E402
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver  # noqa: E402

_INPUTS = os.environ.get("AMIP_INPUTS", "/glade/work/pg2328/amip_inputs")
_ETOPO = os.environ.get("ETOPO", os.path.join(_INPUTS, "etopo_1deg_nodup.nc"))
_ERA5 = os.environ.get("ERA5_IC", os.path.join(_INPUTS, "era5_ic_1979-01-01.zarr"))


def _build(distributed: bool, era5: bool):
    import tempfile
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=3, nlev=20,
                        vertical_coord="hybrid"),
        dycore=DycoreConfig(discretization="mpas", dt=300.0),
        output=OutputConfig(output_dir="", diag_days=0),
        days=1, dataset="analytical", radiation="gray", convection="none",
        turbulence="none", precision="fp64", distributed=distributed,
        topography=_ETOPO, topo_smoothing=4,
        ic="era5" if era5 else "default", ic_path=_ERA5 if era5 else "",
    )
    d = ModelDriver(cfg, output_dir=tempfile.mkdtemp())
    d.setup()
    return d


def test_topography_and_era5_ic_match_serial_on_every_local_cell():
    era5 = True
    if MPI.COMM_WORLD.Get_size() < 2:
        pytest.skip("needs >= 2 MPI ranks")
    if not os.path.exists(_ETOPO) or (era5 and not os.path.exists(_ERA5)):
        pytest.skip("topography / ERA5 IC input not available")
    ref = _build(False, era5)          # serial, no collectives
    d = _build(True, era5)
    part = d._voronoi_layout.partition
    lc = np.asarray(part.local_cells)
    le = np.asarray(part.local_edges)
    assert len(lc) < np.asarray(ref.state.T.data).shape[0]
    # inputs: the ETOPO-derived surface and land fraction
    np.testing.assert_array_equal(np.asarray(d._phis_data), np.asarray(ref._phis_data)[lc])
    np.testing.assert_array_equal(np.asarray(d._f_land), np.asarray(ref._f_land)[lc])
    # the initial state the dynamics starts from
    for name, idx in (("phis", lc), ("p_s", lc), ("T", lc), ("u", le)):
        np.testing.assert_array_equal(
            np.asarray(getattr(d.state, name).data), np.asarray(getattr(ref.state, name).data)[idx],
            err_msg=f"{name} differs from the serial build")
    np.testing.assert_array_equal(
        np.asarray(d.tracers["q_v"]), np.asarray(ref.tracers["q_v"])[lc],
        err_msg="driver q_v differs from the serial build")
    dyn = getattr(ref.state, "tracers", None) or {}
    if "q_v" in dyn:
        np.testing.assert_array_equal(
            np.asarray(d.state.tracers["q_v"].data), np.asarray(dyn["q_v"].data)[lc],
            err_msg="dycore q_v differs from the serial build")
