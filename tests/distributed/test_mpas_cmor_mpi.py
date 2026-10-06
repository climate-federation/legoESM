"""MPAS (Voronoi) CMOR accumulator feed under cell-partition MPI.

Run under MPI::

    mpirun -n 2 python -m pytest tests/distributed/test_mpas_cmor_mpi.py
    mpirun -n 4 python -m pytest tests/distributed/test_mpas_cmor_mpi.py

What this guards
----------------
``ModelDriver._feed_mpas_cmip_multirank`` — the owned-cell -> global
GATHER-to-root that lets ``--cmip-output`` produce a TRUE global CMOR ``Amon``
/ ``day`` file on N ranks.  Before it, ``_mpas_cmip_feed_enabled`` refused every
multi-rank layout (rank-local owned+halo cells binned through rank-local
regrid weights would have written a rank-local "global" mean), which forced
every scoreable MPAS AMIP run onto a single GPU.

Decisiveness
------------
The equivalence test does NOT run the dynamics: both arms are fed the SAME
deterministic analytic field, so the only thing that can move the CMOR output
is the gather + regrid path under test.  With identical inputs the gather is
pure data movement (``gather`` + slot assignment, no arithmetic) and the
regrid is the unchanged serial code, so the assertion is EXACT equality
(``assert_array_equal``), not a tolerance.  The analytic payload here is
float64; the dtype-promotion path of
:func:`gather_owned_cells_to_root` (widest contributed dtype, NumPy end to
end, no ``jnp.array`` round-trip) is covered separately by
``test_gather_promotes_to_the_widest_contributed_dtype``.  The file
deliberately does NOT gate itself on ``JAX_ENABLE_X64``, which would make it
silently no-op in default CI.

Non-vacuity is demonstrated in-file, not claimed: every rank's HALO entries
are poisoned with NaN, so a gather that shipped owned+halo (or that wrote the
wrong global slots) cannot accidentally agree with the serial reference.
"""
from __future__ import annotations

import functools
import types

import jax.numpy as jnp
import numpy as np
import pytest

mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

from legoesm.driver.diagnostics import DiagnosticCollector  # noqa: E402
from legoesm.driver.model_driver import ModelDriver  # noqa: E402
from legoesm.grids.factory import create_grid  # noqa: E402
from legoesm.parallel.voronoi_mpi import initialize_voronoi_mpi  # noqa: E402

RES, NLEV, DAY = 3, 8, 15.0   # level 3 = 642 cells
CMIP_DEG = 10.0
SIGMA_FULL = np.linspace(0.05, 0.98, NLEV)

# Fields whose leading axis is the cell axis (everything the driver feeds
# except the scalar ``flux_interval_days``).
CELL_FIELDS = ("T", "p_s", "lat_deg", "q_v", "u_east", "v_north",
               "precip", "phis", "tas", "rlut", "rsut", "rsdt",
               "hfss", "hfls")


@pytest.fixture(scope="module")
def mesh():
    return create_grid("mpas", RES, lloyd_iterations=10)


@pytest.fixture(scope="module")
def layout(mesh):
    _rank, _n, lay = initialize_voronoi_mpi(mesh)
    return lay


def _collector():
    """Collector configured exactly as ``_create_diagnostics`` does — GLOBAL
    mesh weights, which is what the multi-rank feed relies on."""
    dc = DiagnosticCollector(
        nlev=NLEV, sigma_full=SIGMA_FULL, dsigma=np.full(NLEV, 1.0 / NLEV),
        experiment_id="amip", monthly_means=True, cmip_output=True,
        n_days=30, cmip_resolution_deg=CMIP_DEG, start_year=1979,
    )
    return dc


def _global_fields(mesh):
    """Deterministic analytic fields on the GLOBAL cell axis.

    Every field varies with latitude, longitude AND the global cell index, so
    an index permutation in the gather cannot coincidentally reproduce the
    serial answer.
    """
    lat = np.asarray(mesh.latCell, dtype=np.float64)
    lon = np.asarray(mesh.lonCell, dtype=np.float64)
    n = lat.shape[0]
    idx = np.arange(n, dtype=np.float64)
    # A per-cell fingerprint: smooth in space plus a per-index ripple.
    fp = np.cos(lat) + 0.3 * np.sin(2.0 * lon) + 1e-3 * np.sin(idx)
    prof = (1.0 + 0.5 * SIGMA_FULL)[None, :]
    return {
        "T": 250.0 + 30.0 * fp[:, None] * prof,
        "p_s": 1.0e5 + 2.0e3 * fp,
        "lat_deg": np.degrees(lat),
        "q_v": 1e-3 * (1.0 + 0.4 * fp)[:, None] * (1.0 - SIGMA_FULL)[None, :],
        "u_east": 8.0 * fp[:, None] * prof,
        "v_north": 2.0 * np.sin(lon)[:, None] * prof,
        "precip": 2.0e-5 * (1.0 + 0.5 * fp),
        "phis": 500.0 * (1.0 + fp),
        "tas": 288.0 + 20.0 * fp,
        "rlut": 240.0 + 30.0 * fp,
        "rsut": 100.0 + 40.0 * fp,
        "rsdt": 340.0 + 50.0 * fp,
        "hfss": 20.0 + 10.0 * fp,
        "hfls": 80.0 + 30.0 * fp,
        "flux_interval_days": None,
    }


def _local_with_poisoned_halo(gf, part):
    """Slice the global fields to this rank's LOCAL (owned+halo) cell axis and
    fill the HALO rows with NaN.

    The poison is what makes the owned-vs-halo assertion non-vacuous: in a real
    run the halo values equal their owner's (post-exchange), so a gather that
    wrongly included them could still produce the right answer by accident.
    """
    loc = {}
    for k, v in gf.items():
        if k not in CELL_FIELDS:
            loc[k] = v
            continue
        a = np.array(np.asarray(v)[part.local_cells], dtype=np.float64)
        a[part.n_owned_cells:] = np.nan
        loc[k] = a
    return loc


def _edge_owner_values(lay):
    """The value every edge SHOULD hold once halos are exchanged: a function of
    the GLOBAL edge id, so it is rank-independent and an unexchanged halo can
    never match it by accident."""
    return np.asarray(lay.partition.local_edges)[:, None].astype(
        np.float64) * 10.0 + np.arange(NLEV, dtype=np.float64)[None, :]


def _poisoned_edges(lay):
    """Owner values on OWNED edges, rank-dependent poison on the HALO ones —
    the state ``state.u`` can be in at the CMOR feed point: under column-local
    physics (``physics_fn._column_local``) there is neither a pre- nor a
    post-physics exchange, so halo edges keep this rank's own guess."""
    part = lay.partition
    a = _edge_owner_values(lay).copy()
    a[int(part.n_owned_edges):] = -1000.0 - lay.rank
    return a


def _fake_driver(dc, lay, local, *, raise_on_rank=None, drop_on_rank=None):
    """Minimal driver stand-in exposing exactly what the feed reads.

    ``drop_on_rank`` is ``(field_name, rank)``: that rank omits the field, to
    exercise the field-set intersection.
    """
    seen_u_override = []

    def _kwargs(day, diag, u_override=None):
        # Record what phase 0 handed us: the feed MUST pass the exchanged
        # edge field, not None, or the Perot reconstruction in the real
        # helper would read stale halo edges around boundary-owned cells.
        seen_u_override.append(u_override)
        if raise_on_rank is not None and lay.rank == raise_on_rank:
            raise RuntimeError("synthetic per-rank field-build failure")
        if drop_on_rank is not None and lay.rank == drop_on_rank[1]:
            return {k: v for k, v in local.items() if k != drop_on_rank[0]}
        return dict(local)

    ns = types.SimpleNamespace(
        diagnostics=dc,
        _voronoi_layout=lay,
        _mpi_rank=lay.rank,
        _mpi_world_size=lay.n_ranks,
        _mpas_sfc_accum=None,
        # Phase 0 exchanges this rank's edge halos before building the fields.
        # A REAL edge array on the local edge axis, so the production exchange
        # runs unmodified — and its HALO rows are POISONED with a
        # rank-dependent value. That poison is what lets the hand-off test
        # below tell an exchanged field from an unexchanged one.
        state=types.SimpleNamespace(
            u=types.SimpleNamespace(data=jnp.asarray(_poisoned_edges(lay))),
        ),
        # Injected: this is the ONLY substitution — it lets a test vary the
        # per-rank field set / force a per-rank failure.  Everything the
        # change actually adds (the gather, the intersection, the rank-0
        # commit) runs as production code.
        _mpas_cmip_native_kwargs=_kwargs,
    )
    ns.seen_u_override = seen_u_override
    # Spy on the rank-0 moisture-budget hand-off (#1321): record what each rank
    # passes, without needing a tracker or a flux accumulator here.
    ns.moisture_calls = []
    ns._feed_mpas_moisture_budget = (
        lambda day, diag, kw, *, global_fields=False:
        ns.moisture_calls.append((lay.rank, dict(kw), global_fields)))
    # SimpleNamespace cannot inherit ModelDriver methods; bind the one the
    # feed dispatches to.
    ns._feed_mpas_cmip_multirank = functools.partial(
        ModelDriver._feed_mpas_cmip_multirank, ns)
    return ns


def _finalized(dc):
    return dc._spatial_monthly.finalize(min_sample_fraction=0)


def _serial_reference(mesh, gf):
    """Serial CMOR output for the SAME fields — no collectives, so every rank
    can build it independently (mirrors ``test_mpas_mpi_amip``'s pattern)."""
    dc = _collector()
    dc.set_cmip_grid_info(grid_type="mpas", grid=mesh, start_year=1979)
    dc.feed_cmip_accumulators_native(DAY, **gf)
    return _finalized(dc)


# ---------------------------------------------------------------------------
# 1. The decisive equivalence test
# ---------------------------------------------------------------------------

def test_multirank_cmor_matches_serial_exactly(mesh, layout):
    """N-rank CMOR ``Amon`` output == serial output, EXACTLY, for identical
    input fields.  Halo rows are NaN-poisoned, so this also proves the gather
    ships owned cells only and writes them to the right global slots."""
    if layout.n_ranks < 2:
        pytest.skip("single rank takes the SERIAL branch — this test would "
                    "pass without executing the multi-rank gather at all")
    comm = MPI.COMM_WORLD
    gf = _global_fields(mesh)
    ref = _serial_reference(mesh, gf)

    dc = _collector()
    dc.set_cmip_grid_info(grid_type="mpas", grid=mesh, start_year=1979)
    local = _local_with_poisoned_halo(gf, layout.partition)
    fake = _fake_driver(dc, layout, local)
    ModelDriver._feed_mpas_cmip_accumulators(fake, day=DAY)

    if layout.rank == 0:
        got = _finalized(dc)
        assert set(got) == set(ref), (set(got) ^ set(ref))
        assert got, "no CMOR fields produced under MPI"
        for k in sorted(ref):
            assert np.isfinite(np.asarray(got[k])).any(), (
                f"{k} is entirely non-finite -> halo NaN leaked into the gather")
            np.testing.assert_array_equal(
                np.asarray(got[k]), np.asarray(ref[k]),
                err_msg=f"multi-rank CMOR field {k} != serial")
    comm.Barrier()


def test_moisture_budget_gets_the_gathered_global_fields_on_rank0_only(
        mesh, layout):
    """#1321: the water-budget closure is fed ONCE, on rank 0, with the
    GATHERED global q_v / p_s / precip -- equal to the serial fields, halo
    poison absent -- and never by another rank."""
    if layout.n_ranks < 2:
        pytest.skip("single rank takes the SERIAL branch")
    gf = _global_fields(mesh)
    dc = _collector()
    dc.set_cmip_grid_info(grid_type="mpas", grid=mesh, start_year=1979)
    fake = _fake_driver(dc, layout, _local_with_poisoned_halo(gf, layout.partition))
    ModelDriver._feed_mpas_cmip_accumulators(fake, day=DAY)

    # Judged collectively: a one-rank assertion would strand the others.
    calls = fake.moisture_calls
    if layout.rank != 0:
        err = None if calls == [] else f"rank {layout.rank} fed the closure"
    elif len(calls) != 1 or calls[0][2] is not True:
        err = f"rank 0 calls: {[(r, g) for r, _, g in calls]}"
    else:
        bad = [k for k in ("q_v", "p_s", "precip")
               if not np.array_equal(np.asarray(calls[0][1][k]), gf[k])]
        err = f"gathered fields differ from global: {bad}" if bad else None
    errors = [e for e in MPI.COMM_WORLD.allgather(err) if e is not None]
    assert not errors, errors


def test_multirank_zonal_matches_serial_exactly(mesh, layout):
    """Same, for the zonal (``MonthlyAccumulator``) side of the feed."""
    if layout.n_ranks < 2:
        pytest.skip("single rank takes the SERIAL branch — this test would "
                    "pass without executing the multi-rank gather at all")
    comm = MPI.COMM_WORLD
    gf = _global_fields(mesh)
    dc_ref = _collector()
    dc_ref.set_cmip_grid_info(grid_type="mpas", grid=mesh, start_year=1979)
    dc_ref.feed_cmip_accumulators_native(DAY, **gf)
    ref = dc_ref.monthly_accum.finalize(min_sample_fraction=0)

    dc = _collector()
    dc.set_cmip_grid_info(grid_type="mpas", grid=mesh, start_year=1979)
    fake = _fake_driver(dc, layout,
                        _local_with_poisoned_halo(gf, layout.partition))
    ModelDriver._feed_mpas_cmip_accumulators(fake, day=DAY)

    if layout.rank == 0:
        got = dc.monthly_accum.finalize(min_sample_fraction=0)
        assert got and set(got) == set(ref)
        for k in sorted(ref):
            np.testing.assert_array_equal(
                np.asarray(got[k]), np.asarray(ref[k]),
                err_msg=f"multi-rank zonal field {k} != serial")
    comm.Barrier()


# ---------------------------------------------------------------------------
# 2. Non-vacuity: deliberately broken gathers must be CAUGHT
# ---------------------------------------------------------------------------

def _feed(mesh, lay, gf, poison_part=None):
    """Run the multi-rank feed with *lay* and return (rank-0 output, serial).

    *poison_part* is the partition used to decide which rows are HALO; it must
    be the TRUE partition even when *lay* carries a doctored one, or the poison
    lands nowhere and the injection is silently a no-op.
    """
    ref = _serial_reference(mesh, gf)
    dc = _collector()
    dc.set_cmip_grid_info(grid_type="mpas", grid=mesh, start_year=1979)
    fake = _fake_driver(dc, lay, _local_with_poisoned_halo(
        gf, poison_part if poison_part is not None else lay.partition))
    ModelDriver._feed_mpas_cmip_accumulators(fake, day=DAY)
    MPI.COMM_WORLD.Barrier()
    return (_finalized(dc), ref) if lay.rank == 0 else (None, None)


def _all_equal(got, ref):
    return all(np.array_equal(np.asarray(got[k]), np.asarray(ref[k]))
               for k in ref)


def test_nonvacuous_halo_cells_would_be_caught(mesh, layout):
    """Shipping owned+HALO instead of owned-only must change the answer.

    Injected at the data level, not by monkeypatching: a partition that
    CLAIMS every local cell is owned makes the feed's ``val[:n_owned]`` slice
    and its ``local_cells[:n_owned]`` index map both cover the halo — exactly
    the defect the old ``_mpas_cmip_feed_enabled`` docstring warned about.

    The halo rows must be POISONED for this to bite: when a halo row happens
    to equal its owner's value (a fresh scalar halo), re-writing the same
    global slot is idempotent and shipping it is harmless.  The rows that are
    NOT harmless are the ones a local submesh cannot reconstruct correctly for
    a halo cell (``u_east``/``v_north`` from the Perot reconstruction need
    edges the local mesh does not hold) — NaN stands in for that class here.
    """
    if layout.n_ranks < 2:
        pytest.skip("needs >= 2 ranks to have a halo")
    part = layout.partition
    broken = layout._replace(
        partition=part._replace(n_owned_cells=part.n_local_cells))
    # Poison from the TRUE partition: the doctored one claims no halo exists.
    got, ref = _feed(mesh, broken, _global_fields(mesh), poison_part=part)
    if layout.rank == 0:
        assert not _all_equal(got, ref), (
            "including halo cells did NOT change the CMOR output — the "
            "equivalence test would be vacuous")


def test_nonvacuous_wrong_index_map_would_be_caught(mesh, layout, monkeypatch):
    """Owned cells written to RANK-LOCAL slots instead of their global ids —
    the silent mis-binning that motivated this whole exercise."""
    if layout.n_ranks < 2:
        pytest.skip("needs >= 2 ranks for the index map to differ")
    import legoesm.parallel.voronoi_mpi as vm
    good = vm.gather_owned_cells_to_root

    def _mis_binned(owned_fields, owned_indices, n_global, root=0):
        return good(owned_fields, np.arange(len(owned_indices)), n_global,
                    root=root)

    monkeypatch.setattr(vm, "gather_owned_cells_to_root", _mis_binned)
    got, ref = _feed(mesh, layout, _global_fields(mesh))
    if layout.rank == 0:
        assert not _all_equal(got, ref), (
            "rank-local (mis-binned) slots did NOT change the CMOR output")


# ---------------------------------------------------------------------------
# 3. Collective safety: a guarded failure must not strand any rank
# ---------------------------------------------------------------------------

def test_guarded_failure_on_one_rank_deadlocks_nobody(mesh, layout):
    """A field-build failure on ONE rank aborts the feed on EVERY rank and
    every rank returns.  A rank left in a collective would hang here forever;
    reaching the barrier below is the proof that none did."""
    if layout.n_ranks < 2:
        pytest.skip("needs >= 2 ranks")
    comm = MPI.COMM_WORLD
    gf = _global_fields(mesh)
    dc = _collector()
    dc.set_cmip_grid_info(grid_type="mpas", grid=mesh, start_year=1979)
    fake = _fake_driver(dc, layout,
                        _local_with_poisoned_halo(gf, layout.partition),
                        raise_on_rank=1)
    ModelDriver._feed_mpas_cmip_accumulators(fake, day=DAY)
    comm.Barrier()
    # Unanimous skip: NO rank committed a partial "global" mean.
    assert dc._spatial_monthly._max_count_ever == 0
    assert dc.monthly_accum._max_count_ever == 0
    assert comm.allreduce(dc._spatial_monthly._max_count_ever, op=MPI.MAX) == 0


def test_field_present_on_only_some_ranks_is_dropped_not_deadlocked(
        mesh, layout):
    """A field missing on ONE rank must not make the per-rank gather counts
    diverge (that deadlocks).  It is intersected away; the run continues."""
    if layout.n_ranks < 2:
        pytest.skip("needs >= 2 ranks")
    comm = MPI.COMM_WORLD
    gf = _global_fields(mesh)
    dc = _collector()
    dc.set_cmip_grid_info(grid_type="mpas", grid=mesh, start_year=1979)
    local = _local_with_poisoned_halo(gf, layout.partition)
    fake = _fake_driver(dc, layout, local,
                        drop_on_rank=("precip", 1))
    ModelDriver._feed_mpas_cmip_accumulators(fake, day=DAY)
    comm.Barrier()
    if layout.rank == 0:
        out = _finalized(dc)
        assert "field_2d_pr" not in out, (
            "a field absent on rank 1 was still published")
        assert "field_2d_tas" in out, "the rest of the feed was lost too"


# ---------------------------------------------------------------------------
# 4. End-to-end: a REAL short AMIP, serial vs MPI, through the whole driver
# ---------------------------------------------------------------------------

def _amip_driver(distributed: bool):
    import tempfile
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=3, nlev=20,
                        vertical_coord="hybrid"),
        dycore=DycoreConfig(discretization="mpas", dt=300.0),
        output=OutputConfig(output_dir="", diag_days=1, cmip_output=True,
                            monthly_means=True, cmip_resolution_deg=10.0),
        days=1, dataset="analytical", radiation="gray", convection="none",
        turbulence="none", precision="fp64", distributed=distributed,
    )
    d = ModelDriver(cfg, output_dir=tempfile.mkdtemp())
    d.setup()
    return d


@pytest.mark.slow
def test_end_to_end_amip_cmor_serial_vs_mpi(layout):
    """A real 1-day MPAS AMIP with ``cmip_output`` produces NON-EMPTY, finite,
    physical global CMOR fields under MPI, close to the serial run.

    Unlike the exact test above this one also runs the DYNAMICS, so the two
    arms' STATES are not identical: the owned-cell mass fixer reduces with
    ``allreduce`` instead of one sequential sum, and that O(eps) difference is
    spread by the dynamics (the same divergence
    ``test_mpas_mpi_amip`` bounds).  The tolerance below is therefore a
    DYNAMICS bound, not a gather bound — the gather itself is proven exact by
    ``test_multirank_cmor_matches_serial_exactly``.
    """
    comm = MPI.COMM_WORLD
    # Serial reference: every rank runs it independently (no collectives).
    ref_d = _amip_driver(False)
    ref_d.run()
    ref = ref_d.diagnostics._spatial_monthly.finalize(min_sample_fraction=0)

    d = _amip_driver(True)
    assert d._voronoi_layout is not None, "MPAS distributed layout not built"
    # Index [0]: the method returns ``(feed_on, wants_cmip)``, and a bare
    # truthiness check on the TUPLE passes even for ``(False, False)``.
    assert d._mpas_cmip_feed_enabled(d.diagnostics)[0], "CMOR feed gated off"
    d.run()
    comm.Barrier()

    if layout.rank != 0:
        return
    got = d.diagnostics._spatial_monthly.finalize(min_sample_fraction=0)
    # 1. The symptom this whole change targets: accumulators must be FED.
    assert d.diagnostics._spatial_monthly._max_count_ever > 0, (
        "CMOR accumulators are EMPTY under MPI — the multi-rank feed did not run")
    for k in ("field_2d_tas", "field_2d_ps", "field_3d_ta"):
        assert k in got, f"{k} missing from the MPI CMOR output"
    # 2. Physical, finite, GLOBAL (not a rank-local subset -> no all-NaN rows).
    ps = np.asarray(got["field_2d_ps"])
    tas = np.asarray(got["field_2d_tas"])
    assert np.isfinite(ps).all() and np.isfinite(tas).all()
    assert 4.0e4 < float(np.mean(ps)) < 1.1e5, float(np.mean(ps))
    assert 150.0 < float(np.mean(tas)) < 350.0, float(np.mean(tas))
    # 3. Close to serial.  Bound = the dynamics divergence, quoted absolute.
    for k, atol in (("field_2d_ps", 50.0), ("field_2d_tas", 1.0),
                    ("field_3d_ta", 1.0)):
        a, b = np.asarray(got[k]), np.asarray(ref[k])
        assert a.shape == b.shape, (k, a.shape, b.shape)
        d_max = float(np.nanmax(np.abs(a - b)))
        assert d_max < atol, f"{k}: max|MPI-serial| = {d_max} >= {atol}"


# ---------------------------------------------------------------------------
# 5. Owned-cell bookkeeping (the invariant the gather leans on)
# ---------------------------------------------------------------------------

def test_owned_cells_partition_the_globe_exactly_once(layout):
    """Every global cell is owned by exactly one rank — the precondition for
    'gather owned only' to be both complete and double-count-free."""
    comm = MPI.COMM_WORLD
    part = layout.partition
    owned = np.asarray(part.local_cells[:part.n_owned_cells])
    counts = np.zeros(part.nCells_global, dtype=np.int64)
    counts[owned] = 1
    comm.Allreduce(MPI.IN_PLACE, counts, op=MPI.SUM)
    assert counts.min() == 1 and counts.max() == 1, (
        f"cells owned {counts.min()}..{counts.max()} times (expected exactly 1)")


# ---------------------------------------------------------------------------
# 6. Phase 0: the edge halo exchange the Perot reconstruction depends on
# ---------------------------------------------------------------------------

def test_halo_edges_are_stale_without_an_exchange_and_fixed_by_one(layout):
    """The reason ``_feed_mpas_cmip_multirank`` exchanges edges before
    building fields (#1517 item 3).

    Under column-local physics (``physics_fn._column_local``, which today only
    the idealized Kessler MPAS forcing sets) ``make_voronoi_mpi_step`` skips
    the pre-physics state exchange and adds no post-physics one, so at the feed
    point every HALO edge holds ``old_halo + dt*local_tendency`` — this rank's
    guess, not its owner's value. Full AMIP physics takes the exchanging path;
    the feed exchanges unconditionally so the reconstruction does not depend on
    which of them ran. ``reconstruct_cell_velocity`` reads the edges
    AROUND a cell, and an OWNED cell on the partition cut is ringed by halo
    edges, so its ``ua``/``va`` would be wrong.

    Poison the halo entries with a rank-dependent value, exchange, and require
    that every halo edge came back holding its OWNER's value. The pre-exchange
    assertion is what makes this non-vacuous: without it a no-op exchange would
    pass on any mesh whose halo happened to be empty.
    """
    part = layout.partition
    n_owned = int(part.n_owned_edges)
    n_local = int(part.n_local_edges)
    if n_local == n_owned:
        pytest.skip("this rank has no halo edges (single-rank or no cut)")

    # Owned edges carry a GLOBAL, rank-independent value; halo entries carry a
    # rank-dependent poison, so a correct exchange must overwrite them.
    owned_ids = np.asarray(part.local_edges[:n_owned])
    field = np.empty((n_local, NLEV), dtype=np.float64)
    field[:n_owned] = owned_ids[:, None].astype(np.float64) * 10.0
    field[n_owned:] = -1000.0 - layout.rank

    halo_ids = np.asarray(part.local_edges[n_owned:])
    expected_halo = np.broadcast_to(
        halo_ids[:, None].astype(np.float64) * 10.0,
        (halo_ids.shape[0], NLEV))
    # Pre-exchange: the halo does NOT hold the owner's value.
    assert not np.allclose(field[n_owned:], expected_halo)

    out = np.asarray(layout.halo_exchange.exchange_edge_field(jnp.asarray(field)))

    assert out.shape == field.shape
    np.testing.assert_allclose(out[:n_owned], field[:n_owned])
    np.testing.assert_allclose(out[n_owned:], expected_halo)


def test_feed_hands_the_exchanged_edges_to_the_field_builder(layout, mesh):
    """Phase 0 must actually reach ``_mpas_cmip_native_kwargs``.

    Exchanging and then not passing the result would leave the defect in place
    while every other test still passed, so pin the hand-off itself.
    """
    dc = _collector()
    dc.set_cmip_grid_info(grid_type="mpas", grid=mesh, start_year=1979)
    gf = _global_fields(mesh)
    drv = _fake_driver(dc, layout,
                       _local_with_poisoned_halo(gf, layout.partition))
    drv._feed_mpas_cmip_multirank(DAY, dc, layout)

    assert drv.seen_u_override, "the field builder was never called"
    u_ov = drv.seen_u_override[-1]
    assert u_ov is not None, (
        "phase 0 passed u_override=None, so the reconstruction would read "
        "state.u with its stale halo edges")
    u_ov = np.asarray(u_ov)
    part = layout.partition
    n_owned = int(part.n_owned_edges)
    assert u_ov.shape[0] == int(part.n_local_edges)

    # The decisive assertion: the HALO rows handed to the field builder must
    # hold their OWNER's value, not this rank's poison. Checking only
    # ``is not None`` would pass on an unexchanged array -- verified by
    # deleting the exchange and watching this file still go green, which is
    # why the check is on the VALUES.
    expected = _edge_owner_values(layout)
    np.testing.assert_allclose(u_ov[:n_owned], expected[:n_owned])
    if u_ov.shape[0] > n_owned:
        np.testing.assert_allclose(u_ov[n_owned:], expected[n_owned:])
        # ...and the poison really was there to begin with.
        raw = np.asarray(drv.state.u.data)
        assert not np.allclose(raw[n_owned:], expected[n_owned:])


def test_gather_promotes_to_the_widest_contributed_dtype(layout):
    """A peer's float64 chunk must not be silently downcast by root's buffer.

    ``gather_owned_cells_to_root`` sizes its output from ``np.result_type`` over
    ALL contributed chunks rather than from root's own. Taking root's alone is
    the silent-precision-loss shape codex flagged in the original design, and it
    only bites when ranks disagree — which is exactly what this arranges.
    """
    from legoesm.parallel.voronoi_mpi import gather_owned_cells_to_root

    part = layout.partition
    n_owned = int(part.n_owned_cells)
    owned_ids = np.asarray(part.local_cells[:n_owned])
    # Rank 0 (the root, and the buffer's natural dtype source) contributes
    # float32; every other rank contributes float64.
    dtype = np.float32 if layout.rank == 0 else np.float64
    # A value float32 cannot hold exactly, so a downcast is detectable.
    vals = (owned_ids.astype(np.float64) + 0.1).astype(dtype)

    out = gather_owned_cells_to_root(
        {"x": vals}, owned_ids, int(part.nCells_global))

    if layout.rank != 0:
        assert out is None
        return
    assert out["x"].dtype == np.float64, (
        f"root buffer took its OWN float32 dtype ({out['x'].dtype}), "
        "silently truncating every peer's float64 chunk")
    if layout.n_ranks > 1:
        # A peer's exact float64 value survived the round trip.
        peer_probe = np.setdiff1d(np.arange(part.nCells_global), owned_ids)
        assert peer_probe.size > 0
        got = out["x"][peer_probe[0]]
        assert got == pytest.approx(float(peer_probe[0]) + 0.1, abs=0.0,
                                    rel=1e-15)
