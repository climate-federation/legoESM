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
import pathlib
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

# The land surface scheme defaulted to two-leaf after this file was written,
# and a canopy run refuses to start without the harmonized surfdata its
# per-PFT parameters come from -- so every test here had been failing on that
# refusal, unrelated to what they assert. Point them at the staged file.
#
# The path is resolved against THIS checkout's data directory, not one
# developer's absolute path, with an environment override for a shared copy.
SURFDATA = os.environ.get("LEGOESM_TEST_SURFDATA") or str(
    pathlib.Path(__file__).resolve().parents[2]
    / "data" / "legoesm_surfdata_c260716.nc")
# A module-level skip would turn these three tests from LOUDLY FAILING into
# SILENTLY GREEN wherever the file is absent, which is how a gate rots (GLM
# raised exactly this). So the absence is a FAILURE with an actionable message,
# and only an explicit opt-out skips -- someone who knowingly has no surfdata
# says so once, in the environment, instead of every machine quietly passing.
if not os.path.exists(SURFDATA):
    pytestmark = (
        pytest.mark.skip(reason=(
            f"LEGOESM_TEST_SKIP_NO_SURFDATA set and no surfdata at {SURFDATA}"))
        if os.environ.get("LEGOESM_TEST_SKIP_NO_SURFDATA")
        else pytest.mark.usefixtures("_surfdata_required"))


@pytest.fixture
def _surfdata_required():
    raise AssertionError(
        f"the multilayer land tests need the harmonized surfdata and none is "
        f"at {SURFDATA}. The land surface scheme defaults to two-leaf, and a "
        f"canopy run refuses to start without the per-PFT parameters that "
        f"file carries. Stage it, point LEGOESM_TEST_SURFDATA at a copy, or "
        f"set LEGOESM_TEST_SKIP_NO_SURFDATA=1 to skip deliberately.")


def _shared_tmp_root():
    """Directory both ranks can see, resolved rather than hard-coded.

    ``LEGOESM_TEST_SHARED_TMP`` wins when it exists.  The previous fallback was
    one machine's absolute scratch path, so on any other cluster every test in
    this module died in ``mkdtemp`` with ``FileNotFoundError`` before it
    reached a single assertion -- a distributed gate that could only ever run
    on the machine it was written on.  ``/tmp`` is the last resort and is
    node-local, which is correct for a single-node ``srun`` and is why the
    single-node case is the one this module documents.
    """
    env = os.environ.get("LEGOESM_TEST_SHARED_TMP")
    for cand in (env, os.environ.get("SCRATCH"), os.getcwd(), tempfile.gettempdir()):
        if cand and os.path.isdir(cand) and os.access(cand, os.W_OK):
            return cand
    raise RuntimeError("no writable shared temp directory found")


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
        d = tempfile.mkdtemp(prefix="landmask_", dir=_shared_tmp_root())
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
        d = tempfile.mkdtemp(prefix=prefix, dir=_shared_tmp_root())
    return comm.bcast(d, root=0)


def _surfdata_or_none():
    """Path to a harmonized surfdata file, or None.

    The canopy schemes REFUSE to run without it (they would otherwise run on
    generic constants with every tuned per-PFT value inert), so the canopy
    coverage below is opt-in on the file being present.

    It resolves the SAME path the module-level policy and ``_build`` use, which
    is the repository's staged copy with an environment override.  Consulting
    only the override was a merge artifact: with the file staged at the default
    path and no variable set, this returned None and the canopy rows SKIPPED
    while _build was handing the solver a perfectly good file (codex, merge
    review 2026-09-24).
    """
    return SURFDATA if SURFDATA and os.path.isfile(SURFDATA) else None


def _build(distributed, mask_path, output_dir=None, fix_mass=True,
           land_surface_scheme="simple_seb", land_ic=""):
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=RES, nlev=NLEV,
                        vertical_coord="sigma"),
        dycore=DycoreConfig(discretization="mpas", dt=DT,
                            fix_mass=fix_mass),
        output=OutputConfig(output_dir="", diag_days=0),
        days=DAYS, dataset="analytical", radiation="gray",
        convection="none", turbulence="none", precision="fp64",
        land_mask_path=mask_path, use_multilayer_land=True,
        surfdata_path=SURFDATA,
        multilayer_n_layers=N_SOIL, multilayer_soil_depth=2.5,
        # NAME the land surface scheme rather than inherit the driver default.
        # This test asks one question -- are the soil columns partitioned
        # correctly -- and the answer does not depend on which canopy sits on
        # top of them, so it takes the scheme with no external data
        # dependency. Leaving it unnamed was a hidden choice and it broke: the
        # default moved to the two-leaf canopy, which REFUSES to run without a
        # harmonized surfdata file, so every test in this module died in setup
        # at np=1 as well as under MPI -- a distributed gate that could not run
        # at all, on any rank count, and nothing noticed because nothing runs
        # it in CI.
        land_surface_scheme=land_surface_scheme,
        # Named for the same reason: the per-step surfdata refresh exists only
        # for the two-leaf canopy (production keeps it on there) and the
        # driver refuses it for any other scheme.
        mpas_land_params_refresh=(land_surface_scheme == "two_leaf"),
        land_ic_path=land_ic,
        distributed=distributed,
    )
    d = ModelDriver(cfg, output_dir=output_dir or tempfile.mkdtemp())
    d.setup()
    assert d._land_ml_state is not None, (
        "no multilayer land state was built -- use_multilayer_land did not "
        "take effect, so this test would be comparing nothing")
    return d


def _bitwise_equal(a, b) -> bool:
    """Strict equality for the BIT-EXACT tiers.

    ``np.array_equal`` is not bit equality: it accepts +0.0 == -0.0 and it
    compares across dtypes after promotion, so a float32 rank result would
    "equal" a float64 serial one (review finding). Comparing dtype, shape and
    then the raw bytes is what the word bit-exact in this module's docstring
    actually claims.
    """
    a, b = np.asarray(a), np.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


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


def _stamp_columns(state, ncol):
    """Write each column's GLOBAL index into every per-column leaf.

    Values stay in a physically harmless range (the state is only ever used to
    check index bookkeeping, never stepped), but every column is unique, so any
    mis-indexed scatter shows up as an exact mismatch rather than surviving
    because two columns happened to hold the same profile.
    """
    def _leaf(v):
        if v is None or not hasattr(v, "shape") or v.ndim < 1:
            return v
        if int(v.shape[0]) != ncol:
            return v
        ids = np.arange(ncol, dtype=np.float64).reshape(
            (ncol,) + (1,) * (v.ndim - 1))
        return jnp.asarray(np.asarray(v) + ids * 1e-3)
    return jax.tree_util.tree_map(_leaf, state)


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
    # At ONE rank the partition hands that rank every cell, so the comparison
    # below is a state against itself and proves nothing. That is a property
    # of the rank count, not a defect, so it SKIPS -- it used to fail, which
    # made the whole np=1 rung of the ladder red for a structural reason and
    # buried the real failures. At np>1 a rank that still owns everything IS a
    # defect and keeps failing.
    if MPI.COMM_WORLD.Get_size() == 1:
        pytest.skip("np=1: one rank owns the whole mesh, so this comparison "
                    "is vacuous by construction; run under mpirun -n >= 2")
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


@pytest.mark.parametrize("scheme", ["simple_seb", "two_leaf"])
def test_land_advance_on_the_voronoi_partition_is_bit_exact(scheme):
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
    if scheme != "simple_seb" and _surfdata_or_none() is None:
        pytest.skip(
            f"land_surface_scheme={scheme!r} needs a harmonized surfdata file; "
            "set LEGOESM_TEST_SURFDATA to cover the canopy leaves too "
            "(review finding: simple_seb omits the two-leaf per-column cache, "
            "so the simple_seb row alone does not cover every per-column leaf)")
    d = _build(True, mask, land_surface_scheme=scheme)
    ref_d = _build(False, mask, land_surface_scheme=scheme)

    part = d._voronoi_layout.partition
    n_owned, n_local = part.n_owned_cells, part.n_local_cells
    local_gids = np.asarray(part.local_cells[:n_local])
    owned_gids = local_gids[:n_owned]
    ncol = np.asarray(ref_d._land_ml_state.T_soil).shape[0]
    # At ONE rank the partition hands that rank every cell, so the comparison
    # below is a state against itself and proves nothing. That is a property
    # of the rank count, not a defect, so it SKIPS -- it used to fail, which
    # made the whole np=1 rung of the ladder red for a structural reason and
    # buried the real failures. At np>1 a rank that still owns everything IS a
    # defect and keeps failing.
    if MPI.COMM_WORLD.Get_size() == 1:
        pytest.skip("np=1: one rank owns the whole mesh, so this comparison "
                    "is vacuous by construction; run under mpirun -n >= 2")
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
        # Compiled, as on the production lane: run eagerly, the two-leaf
        # canopy compiles one executable per op and exhausts the process's
        # memory-mapping budget ("LLVM compilation error: Cannot allocate
        # memory").  Both sides take the same path.
        step = jax.jit(lambda s, f_: step_multilayer_land(
            s, f_, driver.physics.land_ml_cfg, 1.0, 600.0,
            lat=lat, doy=0.0, land_params=driver.physics.land_ml_params)[0])
        for _ in range(n):
            state = step(state, forc)
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
           if not _bitwise_equal(out_cols[k][:n_owned], ref_cols[k][owned_gids])}
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
           for k in before if not _bitwise_equal(after[k], before[k])}
    assert not bad, (
        f"land leaves changed across the MPI checkpoint round trip: {bad}")
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"\n[#1321 item 4] checkpoint round trip exact over "
              f"{len(before)} per-column land leaves")


def test_spun_up_land_ic_scatters_to_the_same_columns_the_serial_run_gets():
    """A global spun-up land state must load on every rank, at the right columns.

    The second thing blocking a multi-device production arm, and the mirror of
    the sea-ice one: that file is written once for the whole globe, but the
    loader was handed THIS RANK's column count and refused the file with a
    different number on every rank. So any arm started from a spun-up land
    state was forced back to a single device.

    A test that only checked it no longer crashes would pass on a scatter that
    hands every rank the wrong slice, so this compares values: each rank's
    owned columns against the SERIAL run's columns at the same global indices,
    and exactly -- loading and cutting a file is pure data movement with no
    atmosphere in it, so there is no tolerance to hide behind.
    """
    mask = _land_mask_path()
    shared = _shared_tmpdir("land_ic_")
    ic_path = os.path.join(shared, "spinup_land.npz")

    # The spun-up file is the serial cold-start state, which the aridity seed
    # already makes spatially varying -- checked below, because a uniform state
    # would match under ANY permutation.
    seed = _build(False, mask)
    # Stamp every per-column leaf with its own GLOBAL column index before
    # writing the file, so a wrong index is observable by construction.
    # Codex raised this against the first version, which only required the
    # state to be "spatially varying": the cold-start soil profile varies with
    # DEPTH, so a field could satisfy that while every column was identical,
    # and a scatter that handed a rank the wrong columns would still compare
    # equal. A per-column fingerprint cannot be fooled that way.
    seed._land_ml_state = _stamp_columns(
        seed._land_ml_state, np.asarray(seed.state.T.data).shape[0])
    if MPI.COMM_WORLD.Get_rank() == 0:
        from legoesm.land.clm_surface_map import download_clm_surfdata
        from legoesm.land.restart import (
            HYDRAULICS_SOURCE_CLM_MAP, save_land_restart,
            soil_hydraulics_stamp)
        from legoesm.land.soil_grid import make_soil_grid
        # Stamped with the run's own hydraulics, so the file is loaded verbatim
        # and the comparison below stays bit-exact.
        save_land_restart(
            ic_path, seed._land_ml_state, land_mode="multilayer",
            t_end_s=0.0, n_steps_completed=0,
            soil_dz=np.asarray(
                make_soil_grid(seed.physics.land_ml_cfg.soil_grid).dz),
            soil_hydraulics=soil_hydraulics_stamp(
                seed.physics.land_ml_cfg.hydraulics.retention_curve,
                HYDRAULICS_SOURCE_CLM_MAP, download_clm_surfdata()),
            hydraulics=seed.physics.land_ml_cfg.hydraulics)
    MPI.COMM_WORLD.Barrier()
    assert os.path.exists(ic_path), "the spun-up land state was never written"

    ref = _build(False, mask, land_ic=ic_path)
    d = _build(True, mask, land_ic=ic_path)
    assert d._voronoi_layout is not None, (
        "the distributed run built no Voronoi partition, so nothing was "
        "scattered and this test is not exercising the lane")

    part = d._voronoi_layout.partition
    n_owned = part.n_owned_cells
    gids = np.asarray(part.local_cells[:n_owned])
    n_global = np.asarray(ref.state.T.data).shape[0]
    assert n_owned < n_global, "this rank was given every global cell"

    ref_cols, _ = _column_leaves(ref._land_ml_state, n_global)
    mpi_cols, _ = _column_leaves(d._land_ml_state, part.n_local_cells)
    assert set(ref_cols) == set(mpi_cols), (
        f"the runs disagree on which land leaves are per-column: "
        f"serial-only {set(ref_cols) - set(mpi_cols)}, "
        f"mpi-only {set(mpi_cols) - set(ref_cols)}")
    for required in ("T_soil", "theta_soil"):
        assert required in ref_cols, f"{required} is not being compared"

    # Variation ACROSS COLUMNS, not across all axes: a profile that varies only
    # with depth is uniform in the dimension this test is about.
    varying = [k for k in ref_cols
               if float(np.ptp(np.asarray(ref_cols[k]).reshape(n_global, -1),
                               axis=0).max()) > 0.0]
    # Require the fingerprint on the fields the RESTART carries. Leaves the
    # restart does not restore (a canopy solver seed, say) come from the
    # cold-start template on both sides and are legitimately uniform; they are
    # still compared, they just cannot carry the proof.
    from legoesm.land.restart import _MULTILAYER_FIELDS
    restored = [k for k in ref_cols if k in _MULTILAYER_FIELDS]
    assert restored, "none of the restart's own fields are being compared"
    assert set(restored) <= set(varying), (
        f"these RESTORED land leaves are identical in every column, so a wrong "
        f"slice would be invisible in them: "
        f"{sorted(set(restored) - set(varying))}")

    # Compare the WHOLE local band, halo included, not just the owned prefix:
    # the scatter fills halo columns too and the land tile integrates them, so
    # a scatter that got the halo wrong would pass an owned-only check.
    all_gids = np.asarray(part.local_cells[:part.n_local_cells])
    for name, a_ref in ref_cols.items():
        np.testing.assert_array_equal(
            mpi_cols[name], a_ref[all_gids],
            err_msg=f"land leaf {name!r} landed on the wrong columns after "
                    f"the spun-up state was cut to this rank")

    # A per-column leaf whose column axis is NOT leading would be skipped by
    # both the scatter and this comparison, so refuse to leave one uninspected.
    _, ref_other = _column_leaves(ref._land_ml_state, n_global)
    for name, v in ref_other.items():
        if v is None:
            continue
        _shape = tuple(np.asarray(v).shape)
        assert n_global not in _shape and (n_global + 1) not in _shape, (
            f"land leaf {name!r} carries a global-length axis that is not "
            f"leading, so nothing cut it to this rank and nothing here "
            f"compared it")

    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"\n[land IC] {len(ref_cols)} per-column leaves scattered "
              f"exactly, {len(varying)} of them spatially varying")


if __name__ == "__main__":
    test_land_advance_on_the_voronoi_partition_is_bit_exact()
    test_mpas_multilayer_land_end_to_end_inherits_only_the_atmosphere_residual()
    test_mpas_multilayer_land_checkpoint_round_trip_under_mpi()
    test_spun_up_land_ic_scatters_to_the_same_columns_the_serial_run_gets()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("OK")
