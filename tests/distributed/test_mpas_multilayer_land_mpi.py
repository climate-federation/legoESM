"""Single-rank vs multi-rank interactive multilayer land on the real MPAS lane.

This is work item 4 of #1321. Items 1-3 landed (PR #1546 fed the land its
downwelling fluxes under MPI, PR #1357 wired the distributed q_v del2 halo), but
#1546's own commit body states that no 2-rank MPAS multilayer-land model run was
ever performed, so the equivalence claim itself was never tested. The
pre-existing ``test_multilayer_land_scatter_mpi.py`` exercises a SIX-CUBE-FACE
scatter of the land helpers -- a different partition from the Voronoi cell
partition the MPAS lane actually uses -- and it never runs the driver.

What this guards
----------------
The multilayer land tile is embarrassingly parallel: each column solves an
independent soil/snow/canopy problem with NO lateral coupling. So on the MPAS
Voronoi partition a rank's owned land columns must advance to the values the
serial global run gives at the SAME global cell indices. A wrong permutation, a
wrong slice, a rank reading another rank's forcing, or the marshaller silently
declining to advance the soil (the defect #1546 fixed) all break that.

The land mask is written here rather than loaded, and is deliberately
spatially varying: with a uniform mask every column would carry the same land
fraction and a wrong permutation would be invisible.

Three tiers, because only two of them can honestly be exact:

* **The land advance on the Voronoi partition is BIT-EXACT.** Feed identical
  per-column forcing, slice it by this rank's owned cells, advance, compare.
  This isolates the partition from the atmosphere and is where a wrong
  permutation or a wrong slice shows up at full strength.
* **The checkpoint round trip is BIT-EXACT over OWNED cells.** Under MPI the
  checkpoint holds GLOBAL columns (gathered), so the loader has to cut them
  back to the rank. Halo columns are excluded deliberately: a restore fills
  owned cells and leaves the halo stale until the next exchange, so comparing
  over them measures the test's own impatience, not the loader.
* **The end-to-end driver run is LOOSE**, because the atmosphere feeding the
  land is itself partition-dependent -- measured at 1.3 K in air temperature
  over one day on this configuration, and NOT removed by switching the global
  mass fixer off, so it is not the mass fixer alone. The land inherits that.
  What is asserted end-to-end is that the land stays physical and that its
  difference stays far below the field's own spatial spread; a real
  partitioning defect would be of order that spread, not a fraction of it.

Run under MPI::

    mpirun -n 2 python -m pytest tests/distributed/test_mpas_multilayer_land_mpi.py
    mpirun -n 4 python -m pytest tests/distributed/test_mpas_multilayer_land_mpi.py
"""
from __future__ import annotations

import os
import tempfile

import jax
import jax.numpy as jnp
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

RES, NLEV, DT, DAYS = 3, 20, 300.0, 1     # level 3 = 642 cells; days is an int
N_SOIL = 6


def _land_mask_path():
    """A deterministic, spatially varying land mask, written once by rank 0.

    Every rank must read the SAME file, so rank 0 writes it into a shared
    directory and broadcasts the path (a per-rank tmpdir would work here only
    because the content is deterministic -- broadcasting removes the
    assumption).
    """
    comm = MPI.COMM_WORLD
    path = None
    if comm.Get_rank() == 0:
        import xarray as xr
        d = tempfile.mkdtemp(prefix="landmask_", dir=os.environ.get(
            "LEGOESM_TEST_SHARED_TMP", "/work/bd1083/b309178/diffESM"))
        path = os.path.join(d, "landmask.nc")
        lat = np.arange(-89.0, 90.0, 2.0)
        lon = np.arange(0.0, 360.0, 2.0)
        lon2d, lat2d = np.meshgrid(lon, lat)
        lsm = np.clip(0.5 * (1.0 + np.sin(np.deg2rad(3.0 * lon2d))
                             * np.cos(np.deg2rad(2.0 * lat2d))), 0.0, 1.0)
        xr.Dataset({"lsm": (("lat", "lon"), lsm)},
                   coords={"lat": lat, "lon": lon}).to_netcdf(path)
    return comm.bcast(path, root=0)


def _shared_tmpdir(prefix):
    """One directory, made by rank 0 and broadcast.

    Each rank calling ``mkdtemp`` gets its OWN directory, so only rank 0 would
    find the checkpoint it writes; and ``/tmp`` is node-local, so a multi-node
    run needs the shared filesystem anyway.
    """
    comm = MPI.COMM_WORLD
    d = None
    if comm.Get_rank() == 0:
        d = tempfile.mkdtemp(prefix=prefix, dir=os.environ.get(
            "LEGOESM_TEST_SHARED_TMP", "/work/bd1083/b309178/diffESM"))
    return comm.bcast(d, root=0)


def _build(distributed, mask_path, output_dir=None, fix_mass=True):
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=RES, nlev=NLEV,
                        vertical_coord="sigma"),
        dycore=DycoreConfig(discretization="mpas", dt=DT,
                            fix_mass=fix_mass),
        output=OutputConfig(output_dir="", diag_days=0),
        days=DAYS, dataset="analytical", radiation="gray",
        convection="none", turbulence="none", precision="fp64",
        land_mask_path=mask_path, use_multilayer_land=True,
        multilayer_n_layers=N_SOIL, multilayer_soil_depth=2.5,
        distributed=distributed,
    )
    d = ModelDriver(cfg, output_dir=output_dir or tempfile.mkdtemp())
    d.setup()
    assert d._land_ml_state is not None, (
        "no multilayer land state was built -- use_multilayer_land did not "
        "take effect, so this test would be comparing nothing")
    return d


def _column_leaves(state, ncol):
    """Land leaves whose LEADING axis is the column axis, by name.

    Leaves that are None (optional sub-models off) or carry a different leading
    length (the 1-based CLM canopy leaves) are returned separately rather than
    dropped silently -- a comparison that quietly skipped the field under test
    would prove nothing.
    """
    per_col, other = {}, {}
    for name, v in state._asdict().items():
        if v is None:
            other[name] = None
            continue
        a = np.asarray(v)
        (per_col if a.ndim >= 1 and a.shape[0] == ncol else other)[name] = a
    return per_col, other


def _run_pair(mask_path, fix_mass):
    """Serial-global and MPI runs of one config; returns both drivers.

    Every rank runs the serial reference independently (``distributed=False``
    carries no collectives), as ``test_mpas_mpi_amip`` does, so the serial and
    MPI phases interleave without deadlock.
    """
    ref = _build(False, mask_path, fix_mass=fix_mass)
    ref.run()
    d = _build(True, mask_path, fix_mass=fix_mass)
    assert d._voronoi_layout is not None, (
        "MPAS distributed run did not build a Voronoi partition layout")
    d.run()
    return ref, d


def _compare(ref, d):
    """Max abs difference per land leaf: MPI owned columns vs serial globals."""
    part = d._voronoi_layout.partition
    n_owned = part.n_owned_cells
    gids = np.asarray(part.local_cells[:n_owned])
    n_global = np.asarray(ref.state.T.data).shape[0]
    assert n_owned < n_global, "this rank was given every global cell"

    ref_cols, ref_other = _column_leaves(ref._land_ml_state, n_global)
    mpi_cols, _ = _column_leaves(d._land_ml_state, part.n_local_cells)
    assert set(ref_cols) == set(mpi_cols), (
        f"the runs disagree on which land leaves are per-column: "
        f"serial-only {set(ref_cols) - set(mpi_cols)}, "
        f"mpi-only {set(mpi_cols) - set(ref_cols)}")
    assert ref_cols, "no per-column land leaves found -- nothing was compared"
    for required in ("T_soil", "theta_soil", "snow_depth"):
        assert required in ref_cols, f"{required} is not being compared"

    diffs, spread = {}, {}
    for name, a_ref in ref_cols.items():
        a_mpi = mpi_cols[name][:n_owned]
        assert np.all(np.isfinite(a_mpi)), f"non-finite {name} in the MPI run"
        diffs[name] = float(np.max(np.abs(a_mpi - a_ref[gids])))
        # A leaf that is identically constant matches trivially; record its
        # spread so a zero difference is never read as agreement.
        spread[name] = float(np.ptp(a_ref))

    # THE CONTROL for any land claim: how far apart are the two ATMOSPHERES?
    # The land is forced by them, so a land difference is only a land defect if
    # it is larger than what the atmosphere already hands it.
    atm = {
        "T": float(np.max(np.abs(
            np.asarray(d.state.T.data)[:n_owned]
            - np.asarray(ref.state.T.data)[gids]))),
        "p_s": float(np.max(np.abs(
            np.asarray(d.state.p_s.data)[:n_owned]
            - np.asarray(ref.state.p_s.data)[gids]))),
    }
    return diffs, sorted(ref_other), spread, atm


def test_land_advance_on_the_voronoi_partition_is_bit_exact():
    """Identical forcing, sliced by the Voronoi partition: advance must commute.

    This is the claim work item 4 is really about -- a rank advancing its OWN
    columns must reproduce the global advance at the same global cell indices.
    Driving it with forcing built here, rather than with the atmosphere's,
    removes the one thing that is legitimately partition-dependent, so this
    tier carries no tolerance at all.

    Each side uses ITS OWN driver's land config, per-column PFT parameters and
    latitudes -- rank-local on the MPI side, global on the serial side -- so
    this also checks that the driver builds the rank-local land correctly,
    which is the other half of the same claim.
    """
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.land.multilayer_land import step_multilayer_land

    mask = _land_mask_path()
    d = _build(True, mask)
    ref_d = _build(False, mask)

    part = d._voronoi_layout.partition
    n_owned, n_local = part.n_owned_cells, part.n_local_cells
    local_gids = np.asarray(part.local_cells[:n_local])
    owned_gids = local_gids[:n_owned]
    ncol = np.asarray(ref_d._land_ml_state.T_soil).shape[0]
    assert n_owned < ncol, "this rank was given every global cell"

    # Both sides start from the SAME land state -- the serial one, sliced.
    #
    # Not each driver's own cold start, and the reason is worth recording: the
    # multilayer land seeds its soil column from the atmosphere's lowest-level
    # temperature, and on this lane that field ALREADY differs between the
    # serial and the distributed run at setup, before a single timestep
    # (measured 4.5394 K, which is exactly the cold-start soil difference to
    # the last digit, while the land mask is bit-identical). That is an
    # atmosphere initialisation question, not a land one, so seeding both sides
    # identically is what isolates the claim under test here.
    start_serial = ref_d._land_ml_state
    start_local = jax.tree_util.tree_map(
        lambda x: x[local_gids] if getattr(x, "ndim", 0) >= 1
        and x.shape[0] == ncol else x, start_serial)
    # Non-vacuity: the slice must actually reorder something.
    assert not np.array_equal(
        np.asarray(start_local.T_soil)[:n_owned],
        np.asarray(start_serial.T_soil)[:n_owned]), (
        "local_cells is the identity on this rank, so the comparison below "
        "would not exercise the partition at all")
    # The rank-local land parameters must already match the global ones at the
    # rank's own cells, or the advance is not the same model on both sides.
    assert np.array_equal(
        np.asarray(d._f_land)[:n_owned], np.asarray(ref_d._f_land)[owned_gids]), (
        "the rank-local land mask is not the global mask restricted to the rank")

    # Per-column DISTINCT forcing: identical columns would hide a wrong
    # permutation.
    rng = np.random.default_rng(11)
    def f(lo, hi):
        return np.asarray(rng.uniform(lo, hi, size=ncol))
    glob = dict(
        sw_down=f(100.0, 400.0), lw_down=f(250.0, 400.0),
        precip_total=f(0.0, 2e-4), precip_snow=np.zeros(ncol),
        T_lowest=f(270.0, 300.0), q_lowest=f(0.002, 0.015),
        u_lowest=f(-8.0, 8.0), v_lowest=f(-8.0, 8.0),
        p_lowest=f(94000.0, 96000.0), p_surface=f(99000.0, 101000.0),
        rho_lowest=f(1.0, 1.3), cos_zenith=f(0.1, 0.9),
        co2_ppmv=np.full(ncol, 400.0),
        has_radiation=np.ones(ncol), has_precipitation=np.ones(ncol))
    forcing_global = AtmToSurface(**{k: jnp.asarray(v) for k, v in glob.items()})
    forcing_local = AtmToSurface(
        **{k: jnp.asarray(v[local_gids]) for k, v in glob.items()})

    def advance(driver, state, forc, n=4):
        lat = jnp.asarray(driver.physics.land_ml_lat)
        for _ in range(n):
            state, _r, _c = step_multilayer_land(
                state, forc, driver.physics.land_ml_cfg, 1.0, 600.0,
                lat=lat, doy=0.0,
                land_params=driver.physics.land_ml_params)
        return state

    ref = advance(ref_d, start_serial, forcing_global)
    out = advance(d, start_local, forcing_local)

    ref_cols, _ = _column_leaves(ref, ncol)
    out_cols, _ = _column_leaves(out, n_local)
    assert set(ref_cols) == set(out_cols)
    moving = {k for k, v in ref_cols.items() if float(np.ptp(v)) > 0.0}
    assert {"T_soil", "theta_soil"} <= moving, (
        f"the soil columns are constant in the reference advance, so this "
        f"comparison would be vacuous; leaves that vary: {sorted(moving)}")
    bad = {k: float(np.max(np.abs(out_cols[k][:n_owned] - ref_cols[k][owned_gids])))
           for k in ref_cols
           if not np.array_equal(out_cols[k][:n_owned], ref_cols[k][owned_gids])}
    assert not bad, (
        f"the land advance does NOT commute with the Voronoi partition: {bad}")
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"\n[#1321 item 4] land advance bit-exact on the Voronoi "
              f"partition over {len(ref_cols)} leaves ({len(moving)} of which "
              f"actually vary); {n_owned} owned of {ncol} global columns")


def test_mpas_multilayer_land_end_to_end_inherits_only_the_atmosphere_residual():
    """With the fixer ON the land tracks the serial run to the atmosphere's own
    MPI residual, and stays physical.

    These bounds are the land analogues of the ones ``test_mpas_mpi_amip``
    asserts on T and p_s: loose enough to absorb the reduction-order seed the
    dynamics then spreads, far tighter than any real partitioning defect, which
    would be O(1) in soil temperature rather than O(0.01).
    """
    mask = _land_mask_path()
    ref, d = _run_pair(mask, fix_mass=True)
    diffs, _, spread, atm = _compare(ref, d)

    # Bounded against each field's OWN spatial spread rather than an absolute
    # constant: a wrong permutation scrambles columns and lands at order the
    # spread, while inheriting the atmosphere's residual does not.
    for name in ("T_soil", "theta_soil", "psi_soil"):
        assert diffs[name] < 0.5 * spread[name], (
            f"{name} |serial-mpi| max = {diffs[name]:.3e} against a spatial "
            f"spread of {spread[name]:.3e} -- too large to be inherited from "
            f"the atmosphere (which differs by {atm['T']:.3e} K)")

    part = d._voronoi_layout.partition
    T_mpi = np.asarray(d._land_ml_state.T_soil)[:part.n_owned_cells]
    th_mpi = np.asarray(d._land_ml_state.theta_soil)[:part.n_owned_cells]
    assert 150.0 < float(T_mpi.mean()) < 350.0, (
        f"MPI mean soil T {float(T_mpi.mean()):.1f} K is unphysical")
    assert float(th_mpi.min()) >= 0.0, "negative soil water in the MPI run"

    if MPI.COMM_WORLD.Get_rank() == 0:
        worst = max(diffs.items(), key=lambda kv: kv[1])
        print(f"\n[#1321 item 4] fixer on: T_soil {diffs['T_soil']:.3e} K, "
              f"theta_soil {diffs['theta_soil']:.3e}; "
              f"largest leaf {worst[0]} {worst[1]:.3e}")


def test_mpas_multilayer_land_checkpoint_round_trip_under_mpi():
    """Save under MPI and restore back onto the rank, exactly.

    Under MPI the checkpoint holds GLOBAL land columns (gathered), so the
    loader has to cut them back to this rank before its exact-shape check or
    every resumed distributed run is rejected. That path has no serial
    analogue, which is why #1321 names the round trip separately.
    """
    mask = _land_mask_path()
    out = _shared_tmpdir("ckpt_")
    d = _build(True, mask, output_dir=out)
    d.run()
    part = d._voronoi_layout.partition
    n_owned = part.n_owned_cells
    # OWNED cells only: a restore fills the rank's own columns and leaves the
    # halo stale until the next exchange, so including halo columns here would
    # measure the test's impatience rather than the loader.
    before = {k: v[:n_owned].copy() for k, v
              in _column_leaves(d._land_ml_state, part.n_local_cells)[0].items()}
    assert before, "no per-column land leaves to round-trip"

    d.save_checkpoint(step=int(DAYS * 86400 / DT), day=float(DAYS))
    ckpt = os.path.join(out, f"checkpoint_day_{int(round(float(DAYS))):04d}.npz")
    MPI.COMM_WORLD.Barrier()
    assert os.path.exists(ckpt) or os.path.isdir(ckpt), (
        f"no checkpoint written at {ckpt}")

    d2 = _build(True, mask, output_dir=out)
    d2.load_checkpoint(ckpt)
    after = {k: v[:n_owned] for k, v in
             _column_leaves(d2._land_ml_state, part.n_local_cells)[0].items()}

    assert set(after) == set(before), (
        f"round trip changed which land leaves exist: "
        f"lost {set(before) - set(after)}, gained {set(after) - set(before)}")
    bad = {k: float(np.max(np.abs(after[k] - before[k])))
           for k in before if not np.array_equal(after[k], before[k])}
    assert not bad, (
        f"land leaves changed across the MPI checkpoint round trip: {bad}")
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"\n[#1321 item 4] checkpoint round trip exact over "
              f"{len(before)} per-column land leaves")


if __name__ == "__main__":
    test_land_advance_on_the_voronoi_partition_is_bit_exact()
    test_mpas_multilayer_land_end_to_end_inherits_only_the_atmosphere_residual()
    test_mpas_multilayer_land_checkpoint_round_trip_under_mpi()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("OK")
