"""Serial MPAS energy-tracker sample is unchanged by the multi-rank port.

``ModelDriver._mpas_energy_sample`` on a single-process run must give
exactly what a direct ``EnergyBudgetTracker.update`` gives on the same
state, fluxes and area weights (the pre-port inline code), and a missing
input must give NaN without the flux-timing stamp.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "distributed"))
import test_mpas_energy_tracker_mpi as mpi_case  # noqa: E402

from legoesm.diagnostics.energy_budget import (  # noqa: E402
    EnergyBudgetTracker,
    area_weighted_mean,
)
from legoesm.driver.model_driver import _cell_winds  # noqa: E402


@pytest.fixture(scope="module")
def drv():
    d = mpi_case._build(False)
    gid = np.arange(d.state.T.data.shape[0])
    mpi_case._install_fluxes(d, gid)
    mpi_case._install_water(d, gid)
    mpi_case._install_T_and_u(d, np.asarray(d.state.T.data), gid,
                              np.arange(d.state.u.data.shape[0]))
    return d


def _direct(d, tracker, elapsed_day):
    sd, tr, dg = d.model._sfc_diag, d.state.tracers, d.diagnostics
    p_s = d.state.p_s.data
    uc, vc = _cell_winds(d.state, d.grid)
    return tracker.update(
        d.state.T.data, tr["q_v"].data, uc, vc, d.state.phis.data, p_s,
        dg.dsigma, dg.sigma_full,
        sd[5].data, sd[4].data, sd[3].data, sd[0].data, sd[1].data,
        elapsed_seconds=elapsed_day * 86400.0, area_weights=dg._area_w,
        dp=dg._dp(p_s), p_full=dg._p_full(p_s), q_frozen=tr["q_i"].data)


def test_serial_sample_equals_direct_tracker_update(drv):
    drv.diagnostics.energy_tracker = EnergyBudgetTracker()
    ref = EnergyBudgetTracker()
    for day, dT in ((0.0, 0.0), (1.0, 1.0)):
        mpi_case._shift_T(drv, dT)
        got = drv._mpas_energy_sample(drv.state.p_s.data, day)
        want = _direct(drv, ref, day)
    assert want.dE_dt != 0.0
    assert got["energy_toa_net"] == float(want.toa_net)
    assert got["energy_dE_dt"] == float(want.dE_dt)
    assert got["energy_residual"] == float(want.residual)
    assert got["sw_net_sfc"] == float(want.sfc_sw_net)
    assert got["lw_net_sfc"] == float(want.sfc_lw_net)
    assert got["energy_flux_interval_mean"] == 0.0
    awt = drv.diagnostics._area_w
    sd = drv.model._sfc_diag
    assert got["hfss"] == float(area_weighted_mean(sd[6].data, awt))
    assert got["hfls"] == float(area_weighted_mean(sd[7].data, awt))
    assert got["evspsbl"] == float(
        area_weighted_mean(sd[mpi_case.N_SLOTS - 1].data, awt))


def test_serial_sample_without_toa_flux_is_nan_and_unstamped(drv):
    sd = list(drv.model._sfc_diag)
    saved, sd[5] = sd[5], None
    drv.model._sfc_diag = tuple(sd)
    try:
        out = drv._mpas_energy_sample(drv.state.p_s.data, 2.0)
    finally:
        sd[5] = saved
        drv.model._sfc_diag = tuple(sd)
    assert set(out) == set(drv._ENERGY_SERIES_KEYS)
    assert all(np.isnan(v) for v in out.values())


def test_serial_mpas_run_writes_its_timeseries():
    """A single-process run (rank None) is its own writer: the end-of-run
    timeseries.npz and results.txt must exist and carry the energy series."""
    d = mpi_case._build(False)
    d.run()
    out = Path(d._output_dir)
    assert (out / "results.txt").exists()
    ts = np.load(out / "timeseries.npz")
    assert "energy_toa_net" in ts.files
