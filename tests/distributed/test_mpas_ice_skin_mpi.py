"""Prognostic sea-ice skin on the distributed MPAS (Voronoi) lane.

Until now the feature refused this lane outright: the driver raised
"mpas_ice_skin_prognostic is not wired for the distributed Voronoi (MPI) lane
yet". Because the production AMIP deck turns the skin ON, that refusal is what
stopped every production arm from running on more than one GPU -- it surfaced
the first time a four-rank smoke run was attempted against the production deck.

What the wiring had to do, and therefore what this guards
--------------------------------------------------------
The skin update is a Semtner zero-layer balance evaluated per cell with no
neighbour stencil, and on this lane the prescribed SST/SIC forcing is already
built on each rank's LOCAL mesh. So the advance itself needs nothing but to be
reached -- it had been sitting inside the serial branch of the step loop, and
the cell-partition MPI step publishes the same surface-flux side channel. What
genuinely needed code is the checkpoint: the file is global, the field is not.

* **The checkpoint round trip is BIT-EXACT over OWNED cells.** The save gathers
  the rank's owned cells into the global field; the load cuts the global field
  back to the rank. A wrong permutation, an ungathered rank-local fragment
  written as if it were global, or a missing scatter all show up here at full
  strength. Halo cells are excluded deliberately: a restore fills owned cells
  and leaves the halo stale until the next exchange, so comparing over them
  would measure the test's impatience rather than the loader.
* **The skin advances, stays physical, and varies in space.** A skin frozen at
  its seed value would pass a permutation check trivially, so the seed is
  checked against the advanced field.
* **Serial and MPI agree to within what the ATMOSPHERE already differs by.**
  The skin is driven by the surface energy flux the atmosphere exports, and
  that flux is partition-dependent on this lane (the same effect the
  multilayer-land MPI test measures at ~1.3 K in air temperature over a day).
  An exact match is therefore not available and claiming one would be false;
  what is asserted is that the skin difference stays far below the skin's own
  spatial spread, which a real partitioning defect would exceed.

Run under MPI::

    mpirun -n 2 python -m pytest tests/distributed/test_mpas_ice_skin_mpi.py
    mpirun -n 4 python -m pytest tests/distributed/test_mpas_ice_skin_mpi.py
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

import jax.numpy as jnp  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.driver.config import (  # noqa: E402
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
    OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver  # noqa: E402

RES, NLEV, DT, DAYS = 3, 20, 300.0, 1     # level 3 = 642 cells


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


def _build(distributed, output_dir=None):
    # Gray radiation plus the analytical SST/SIC forcing is the cheapest
    # configuration the skin will accept: it needs a prescribed ice fraction to
    # carry a skin on, and refuses radiation='none' because there would be no
    # surface energy budget to integrate.
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=RES, nlev=NLEV,
                        vertical_coord="sigma"),
        dycore=DycoreConfig(discretization="mpas", dt=DT),
        output=OutputConfig(output_dir="", diag_days=0),
        days=DAYS, dataset="analytical", radiation="gray",
        convection="none", turbulence="none", precision="fp64",
        mpas_ice_skin_prognostic=True,
        distributed=distributed,
    )
    d = ModelDriver(cfg, output_dir=output_dir or tempfile.mkdtemp())
    d.setup()
    _install_ice_cap(d)
    return d


def _install_ice_cap(driver):
    """Give the run a polar ice cap, because the analytical forcing has none.

    Measured: ``analytical_sst_sic`` returns an ice fraction that is identically
    zero on this mesh, and the skin is only advanced where there is ice — so
    without this the skin would sit at its seed and every comparison below would
    pass by comparing two constants.

    The cap is a function of LATITUDE, a physical coordinate, so the same global
    cell gets the same ice fraction whatever rank owns it; and it is built from
    the driver's OWN cell latitudes, which are already this rank's local band on
    the distributed lane. It varies in space on purpose: a uniform cap would
    make a wrong permutation invisible.
    """
    lat = np.abs(np.degrees(np.asarray(driver.grid.latCell)).reshape(-1))
    sic = np.clip((lat - 55.0) / 20.0, 0.0, 1.0)
    assert float(np.max(sic)) > 0.5, (
        "the test mesh has no polar cells, so no ice cap could be installed")
    _fn = driver.get_sst_sic

    def _with_ice(day, _fn=_fn, _sic=jnp.asarray(sic)):
        sst, _ = _fn(day)
        return sst, _sic

    driver.get_sst_sic = _with_ice


def _owned(driver):
    part = driver._voronoi_layout.partition
    n_owned = part.n_owned_cells
    return n_owned, np.asarray(part.local_cells[:n_owned])


def test_distributed_run_advances_a_finite_varying_skin():
    d = _build(True)
    assert d._voronoi_layout is not None, (
        "the MPAS distributed run built no Voronoi partition layout, so this "
        "test is not exercising the partitioned lane at all")
    d.run()

    skin = np.asarray(d._ice_T_skin)
    assert skin.shape == (np.asarray(d.state.T.data).shape[0],), (
        f"the skin is {skin.shape} but this rank holds "
        f"{np.asarray(d.state.T.data).shape[0]} cells -- it was built on the "
        f"wrong mesh")
    assert np.all(np.isfinite(skin)), "the distributed run produced a non-finite skin"

    n_owned, _ = _owned(d)
    owned = skin[:n_owned]
    # Physical envelope: the zero-layer skin is conductively tied to the
    # seawater freezing point at the ice base and cannot melt past the fresh
    # freezing point, so a value outside this band is a defect, not weather.
    assert float(np.min(owned)) > 150.0, "the skin cooled through an unphysical floor"
    assert float(np.max(owned)) < constants.T_freeze + 5.0, (
        "the skin warmed well past the melting surface")
    # A skin still sitting exactly at its seed never advanced, and would pass
    # every permutation check below for free.
    assert float(np.max(np.abs(owned - constants.T_freeze_ocean))) > 1e-6, (
        "the skin is still at its seed value -- the advance never ran on this "
        "lane, which is the defect this test exists to catch")


def test_checkpoint_round_trip_is_exact_over_owned_cells():
    out = _shared_tmpdir("iceskin_ckpt_")
    d = _build(True, output_dir=out)
    d.run()
    before = np.asarray(d._ice_T_skin)
    n_owned, gids = _owned(d)

    d.save_checkpoint(int(round(DAYS * 86400.0 / DT)), float(DAYS))
    MPI.COMM_WORLD.Barrier()
    path = os.path.join(out, f"checkpoint_day_{DAYS:04d}.npz")
    assert os.path.exists(path), f"no checkpoint was written at {path}"

    # The file must be GLOBAL: a rank-local fragment written as if it were
    # global is the exact failure the save-side gather prevents, and it would
    # otherwise only surface as a wrong restart months later.
    #
    # Only rank 0 can read the file, but the VERDICT is broadcast and asserted
    # by every rank. A bare rank-0 assert would abort that rank inside a
    # collective region and leave the others waiting at the next barrier
    # forever -- the failure would present as a hang rather than a failure,
    # which is the worst way for a gate to report.
    verdict = None
    if MPI.COMM_WORLD.Get_rank() == 0:
        with np.load(path) as z:
            has = "ice_T_skin" in z.files
            n_global = int(z["ice_T_skin"].shape[0]) if has else -1
        if not has:
            verdict = ("the checkpoint carries no skin, so the chain would "
                       "silently re-run the multi-week spin-up every restart")
        elif n_global != d._voronoi_layout.partition.nCells_global:
            verdict = (
                f"the checkpoint holds {n_global} skin values but the global "
                f"mesh has {d._voronoi_layout.partition.nCells_global} cells "
                f"-- a rank-local fragment was written as a global field")
    verdict = MPI.COMM_WORLD.bcast(verdict, root=0)
    assert verdict is None, verdict

    fresh = _build(True, output_dir=out)
    fresh.load_checkpoint(path)
    staged = np.asarray(fresh._carry_aux["ice_T_skin"])
    assert staged.shape[0] == np.asarray(fresh.state.T.data).shape[0], (
        "the staged skin was not cut back to this rank's cells, so the seed "
        "overlay would reject it on a shape mismatch")
    np.testing.assert_array_equal(
        staged[:n_owned], before[:n_owned],
        err_msg="the skin did not survive the gather/scatter round trip over "
                "this rank's owned cells")
    # The same values must sit at the same GLOBAL cells, not merely somewhere
    # in the array: a permuted gather preserves the multiset and would pass a
    # sorted comparison.  Every rank checks ITS OWN cells against the file, so
    # a gather that scrambled one rank's band cannot hide behind rank 0's.
    with np.load(path) as z:
        glob = np.asarray(z["ice_T_skin"])
    np.testing.assert_array_equal(
        glob[gids], before[:n_owned],
        err_msg="this rank's owned cells landed at the wrong global slots")


def test_serial_and_distributed_skins_agree_within_the_atmospheric_spread():
    # Every rank runs the serial reference independently (distributed=False
    # carries no collectives), so the serial and MPI phases interleave without
    # deadlock -- the same arrangement the multilayer-land MPI test uses.
    ref = _build(False)
    ref.run()
    d = _build(True)
    d.run()

    n_owned, gids = _owned(d)
    a_ref = np.asarray(ref._ice_T_skin)
    a_mpi = np.asarray(d._ice_T_skin)[:n_owned]
    assert n_owned < a_ref.shape[0], "this rank was given every global cell"

    diff = float(np.max(np.abs(a_mpi - a_ref[gids])))
    spread = float(np.ptp(a_ref))
    assert spread > 1.0, (
        "the serial skin is nearly uniform, so agreement here would mean "
        "nothing -- the comparison has no signal to detect a permutation with")

    # THE CONTROL: the skin is forced by the atmosphere, and the atmosphere is
    # itself partition-dependent on this lane. A skin difference is only a skin
    # defect if it exceeds what its own forcing already differs by.
    atm = float(np.max(np.abs(
        np.asarray(d.state.T.data)[:n_owned] - np.asarray(ref.state.T.data)[gids])))
    # The multiplier is set from what was measured on this configuration, not
    # chosen for comfort: a ceiling of ten times the atmospheric difference
    # would admit a permutation that swaps a handful of cells, which was GLM's
    # objection to the first version of this bound.
    assert diff < max(3.0 * atm, 0.5), (
        f"the distributed skin differs from the serial skin by {diff:.3g} K, "
        f"far beyond the {atm:.3g} K the two atmospheres differ by -- that is "
        f"a partitioning defect, not inherited weather")
    assert diff < 0.25 * spread, (
        f"the skin difference {diff:.3g} K is a large fraction of the field's "
        f"own spatial spread {spread:.3g} K")


def test_save_straight_after_a_load_is_refused_not_silently_skinless():
    """A chain link that loads and checkpoints without stepping is REFUSED.

    Raised by GLM in review: under MPI the save overwrites its rank-local field
    with the global gather, and right after a load the live skin is cleared --
    the loader stages it and only the run adopts it. The worry was that such a
    link would write a checkpoint with no skin and the next one would restart
    from the seed, silently re-running a multi-week spin-up.

    Measured: it cannot. The driver already refuses to checkpoint a loaded-but-
    unadopted carry at all, because writing one would launder a mismatched
    restart into a plausible-looking file (#405/#413). The refusal is what
    protects the skin here, so the refusal is what this pins -- if it were ever
    relaxed, the skin path would need the staged-value handling on its own.
    """
    out = _shared_tmpdir("iceskin_relay_")
    d = _build(True, output_dir=out)
    d.run()
    d.save_checkpoint(int(round(DAYS * 86400.0 / DT)), float(DAYS))
    MPI.COMM_WORLD.Barrier()
    first = os.path.join(out, f"checkpoint_day_{DAYS:04d}.npz")

    relay = _build(True, output_dir=_shared_tmpdir("iceskin_relay2_"))
    relay.load_checkpoint(first)
    assert relay._ice_T_skin is None, (
        "the loader adopted the skin directly, so this test no longer "
        "exercises the staged-but-not-yet-adopted path it was written for")
    assert np.asarray(relay._carry_aux["ice_T_skin"]).shape[0] == (
        np.asarray(relay.state.T.data).shape[0]), (
        "the staged skin is not this rank's length, so the scatter on load "
        "did not happen")
    with pytest.raises(ValueError, match="NOT adopted into the save channel"):
        relay.save_checkpoint(int(round(DAYS * 86400.0 / DT)), float(DAYS))
