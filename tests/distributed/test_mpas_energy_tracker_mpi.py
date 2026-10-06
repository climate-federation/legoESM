"""MPAS energy tracker under the multi-rank cell partition (#1354).

Run under MPI::

    mpirun -n 2 python -m pytest tests/distributed/test_mpas_energy_tracker_mpi.py
    mpirun -n 4 python -m pytest tests/distributed/test_mpas_energy_tracker_mpi.py

On the partition every per-cell array is this rank's owned+halo slice, so the
tracker's global means must come from OWNED cells and an allreduce
(``ModelDriver._mpas_energy_partitioned``).  The reference is the serial
tracker on the SAME global initial state with the SAME surface/TOA fluxes,
built independently on every rank (no collectives), as in
``test_mpas_mpi_amip``.  Nothing is time-stepped: the two drivers start from
the identical analytical state, so any difference is the reduction.

Each case is chosen to fail for one specific defect: a dropped allreduce
(rank-local means), a dropped owned mask (halo cells counted twice), a
dropped edge-halo refresh (cell winds of owned boundary cells read stale
halo edges), and a non-unanimous sample (ranks mixing flux timings, or one
rank failing locally while the others record).
"""
from __future__ import annotations

import types

import numpy as np
import pytest

mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

import jax.numpy as jnp  # noqa: E402

from legoesm.driver.config import (  # noqa: E402
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
    OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver, _evap_sfc_slot  # noqa: E402

RES, NLEV, DT = 3, 20, 300.0  # level 3 = 642 cells
N_SLOTS = _evap_sfc_slot() + 1

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() < 2, reason="needs mpirun -n >= 2")


def _build(distributed: bool):
    import tempfile
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=RES, nlev=NLEV,
                        vertical_coord="hybrid"),
        dycore=DycoreConfig(discretization="mpas", dt=DT),
        output=OutputConfig(output_dir="", diag_days=0),
        days=1, dataset="analytical", radiation="gray",
        convection="none", turbulence="none", precision="fp64",
        distributed=distributed,
    )
    d = ModelDriver(cfg, output_dir=tempfile.mkdtemp())
    d.setup()
    return d


def _flux(slot: int, gid: np.ndarray) -> np.ndarray:
    """A deterministic, cell-varying flux for ``slot`` at global cells
    ``gid`` (positive, O(100) W/m^2, different per slot)."""
    return 100.0 + 10.0 * slot + 50.0 * np.sin(0.37 * gid + slot)


def _install_fluxes(d, gid: np.ndarray) -> None:
    d.model._sfc_diag = tuple(
        types.SimpleNamespace(data=jnp.asarray(_flux(i, gid)))
        for i in range(N_SLOTS))
    d._mpas_sfc_accum = None   # snapshot timing on both drivers


def _install_water(d, gid: np.ndarray) -> None:
    """Vapour and cloud ice as functions of the global cell id: the analytical
    state is dry, and the latent and frozen terms must be exercised too."""
    nlev = d.state.T.data.shape[1]
    phase = 0.21 * gid[:, None] + np.arange(nlev)[None, :]
    q_v = 5.0e-3 * (1.0 + 0.5 * np.sin(phase))
    q_i = 1.0e-5 * (1.0 + np.cos(phase))
    tr = dict(d.state.tracers or {})
    tr["q_v"] = d.state.T.replace(data=jnp.asarray(q_v), name="q_v",
                                  units="kg/kg")
    tr["q_i"] = d.state.T.replace(data=jnp.asarray(q_i), name="q_i",
                                  units="kg/kg")
    d.state = d.state._replace(tracers=tr)


def _install_T_and_u(d, T_global: np.ndarray, cell_gid: np.ndarray,
                     edge_gid: np.ndarray) -> None:
    """The idealized seed draws its lowest-level noise per LOCAL cell count,
    so each rank starts from a different T than the serial driver; give both
    the serial T, and a nonzero wind so the kinetic-energy term is tested."""
    nlev = T_global.shape[1]
    u = 20.0 * np.cos(0.11 * edge_gid[:, None] + 0.3 * np.arange(nlev)[None, :])
    d.state = d.state._replace(
        T=d.state.T.replace(data=jnp.asarray(T_global[cell_gid])),
        u=d.state.u.replace(data=jnp.asarray(u)))


def _shift_T(d, dT: float) -> None:
    d.state = d.state._replace(T=d.state.T.replace(data=d.state.T.data + dT))


def _serial_and_mpi():
    ref = _build(False)
    gid_ref = np.arange(ref.state.T.data.shape[0])
    _install_fluxes(ref, gid_ref)
    _install_water(ref, gid_ref)
    T_global = np.asarray(ref.state.T.data)
    _install_T_and_u(ref, T_global, gid_ref,
                     np.arange(ref.state.u.data.shape[0]))
    d = _build(True)
    assert d._voronoi_layout is not None
    part = d._voronoi_layout.partition
    _install_fluxes(d, np.asarray(part.local_cells))
    _install_water(d, np.asarray(part.local_cells))
    _install_T_and_u(d, T_global, np.asarray(part.local_cells),
                     np.asarray(part.local_edges))
    return ref, d


KEYS = ("energy_toa_net", "sw_net_sfc", "lw_net_sfc", "hfss", "hfls",
        "evspsbl")


# Area weights are float32 (grid_area), and the serial mean sums them in a
# different order than owned-cell partial sums + allreduce: ~1e-7 relative.
# A dropped allreduce or owned mask is off by O(1e-2) or more.
RTOL = 1e-6


def _assert_close(got: dict, want: dict, keys, *, rtol=RTOL):
    for k in keys:
        assert np.isfinite(got[k]), f"{k} is not finite on the partition"
        assert got[k] == pytest.approx(want[k], rel=rtol, abs=1e-6), (
            f"{k}: partition {got[k]!r} vs serial {want[k]!r}")


def test_partitioned_tracker_matches_serial_including_dE_dt():
    ref, d = _serial_and_mpi()
    want0 = ref._mpas_energy_sample(ref.state.p_s.data, 0.0)
    got0 = d._mpas_energy_sample(d.state.p_s.data, 0.0)
    _assert_close(got0, want0, KEYS)
    assert got0["energy_flux_interval_mean"] == 0.0
    # Column energy itself (the dE/dt numerator), not just the fluxes.
    E_ser = ref.diagnostics.energy_tracker.column_energy[-1]
    E_mpi = d.diagnostics.energy_tracker.column_energy[-1]
    assert E_mpi == pytest.approx(E_ser, rel=RTOL)

    # Second sample one day later with T warmed by 1 K everywhere: dE/dt and
    # the residual are now nonzero and must match too.
    for drv in (ref, d):
        _shift_T(drv, 1.0)
    want1 = ref._mpas_energy_sample(ref.state.p_s.data, 1.0)
    got1 = d._mpas_energy_sample(d.state.p_s.data, 1.0)
    assert abs(want1["energy_dE_dt"]) > 1.0, "control: dE/dt must be nonzero"
    _assert_close(got1, want1, KEYS + ("energy_dE_dt", "energy_residual"))


def test_stale_halo_edges_are_refreshed_before_the_ke_term():
    """Corrupt this rank's HALO edges (indices past the owned edges): the
    partition path must refresh them from their owners, so the column energy
    still matches serial.  Without the refresh, owned boundary cells
    reconstruct their winds from garbage edges."""
    ref, d = _serial_and_mpi()
    part = d._voronoi_layout.partition
    u = np.array(d.state.u.data)
    assert u.shape[0] > part.n_owned_edges, "no halo edges to corrupt"
    u[part.n_owned_edges:] = 400.0
    d.state = d.state._replace(u=d.state.u.replace(data=jnp.asarray(u)))
    ref._mpas_energy_sample(ref.state.p_s.data, 0.0)
    d._mpas_energy_sample(d.state.p_s.data, 0.0)
    E_ser = ref.diagnostics.energy_tracker.column_energy[-1]
    E_mpi = d.diagnostics.energy_tracker.column_energy[-1]
    assert E_mpi == pytest.approx(E_ser, rel=RTOL)


class _ReadyAccum:
    """An accumulator whose window is complete, returning the given means."""
    ENERGY_SLOTS = (0, 1, 3, 4, 5, 6, 7)

    def __init__(self, gid):
        self._gid = gid

    def window_ready(self, slots):
        return True

    def is_complete(self):
        return True

    def mean(self, i):
        return _flux(i, self._gid)


def test_mixed_flux_timing_across_ranks_gives_nan_on_every_rank():
    """Rank 0 has a complete interval-mean window, the others sample
    snapshots: a mixture would be a plausible number, so every rank must
    record NaN (and none may hang)."""
    _, d = _serial_and_mpi()
    if MPI.COMM_WORLD.Get_rank() == 0:
        d._mpas_sfc_accum = _ReadyAccum(
            np.asarray(d._voronoi_layout.partition.local_cells))
    out = d._mpas_energy_sample(d.state.p_s.data, 0.0)
    assert np.isnan(out["energy_toa_net"])
    assert "energy_flux_interval_mean" not in out
    assert d.diagnostics.energy_tracker.column_energy == []


def test_a_local_failure_on_one_rank_gives_nan_on_every_rank():
    """One rank's flux has the wrong length: that rank cannot form its sums,
    and every rank must agree to record NaN instead of the others publishing
    a partial-domain mean."""
    _, d = _serial_and_mpi()
    if MPI.COMM_WORLD.Get_rank() == MPI.COMM_WORLD.Get_size() - 1:
        bad = list(d.model._sfc_diag)
        bad[5] = types.SimpleNamespace(data=jnp.ones(3))
        d.model._sfc_diag = tuple(bad)
    out = d._mpas_energy_sample(d.state.p_s.data, 0.0)
    assert np.isnan(out["energy_toa_net"])
    assert d.diagnostics.energy_tracker.column_energy == []


def test_optional_flux_missing_on_one_rank_is_nan_not_partial():
    """hfss absent on one rank only: hfss is NaN everywhere, while the
    radiation budget (present on every rank) is still recorded."""
    ref, d = _serial_and_mpi()
    if MPI.COMM_WORLD.Get_rank() == 0:
        sd = list(d.model._sfc_diag)
        sd[6] = None
        d.model._sfc_diag = tuple(sd)
    want = ref._mpas_energy_sample(ref.state.p_s.data, 0.0)
    out = d._mpas_energy_sample(d.state.p_s.data, 0.0)
    assert np.isnan(out["hfss"])
    _assert_close(out, want, ("energy_toa_net", "hfls", "evspsbl"))
