"""MPI gather / scatter round-trip for the lat-lon band checkpoint path.

Companion to ``tests/parallel/test_latlon_mpi_gather_scatter.py``
(serial, single-rank).  This file is invoked under
``mpirun -n {2,4}`` by ``scripts/run_latlon_mpi_halo_smoke.sbatch``.

Stage 3-D validates that:

* a global state can be deterministically split into rank-local bands
  (via :func:`scatter_field_latlon`),
* the resulting bands gather back to the same global state on rank 0
  (via :func:`gather_field_latlon`),
* the v-face boundary row that lives on TWO neighbouring ranks is
  preserved with the correct value (i.e. the trim-on-send convention
  in :func:`gather_field_latlon` matches the duplicate-on-receive
  convention in :func:`scatter_field_latlon`).

A failure here breaks the AMIP 100-year restart chain: rank 0 would
write a checkpoint whose v-field row at the rank-1 boundary is either
duplicated (corruption) or dropped (NaN gap).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

from legoesm.parallel.latlon_mpi import (  # noqa: E402
    gather_field_latlon,
    make_latlon_band_layout,
    scatter_field_latlon,
)


N_LAT = 24      # divisible by 1, 2, 3, 4, 6
N_LON = 48
NLEV = 4


@pytest.fixture(scope="module")
def comm():
    return MPI.COMM_WORLD


@pytest.fixture(scope="module")
def layout(comm):
    return make_latlon_band_layout(
        rank=comm.Get_rank(),
        n_ranks=comm.Get_size(),
        n_lat=N_LAT,
        n_lon=N_LON,
    )


def _make_global_array(shape, seed):
    """Identical global array on every rank (deterministic seed)."""
    rng = np.random.default_rng(seed=seed)
    return jnp.asarray(rng.standard_normal(shape).astype(np.float64))


@pytest.mark.parametrize("shape_3d", [
    (N_LAT, N_LON),
    (N_LAT, N_LON, NLEV),
])
def test_scatter_then_gather_scalar_matches_global(comm, layout, shape_3d):
    """``gather(scatter(g)) == g`` on rank 0 for scalar fields.

    Other ranks get back ``None`` from gather; assertion-skip there.
    """
    global_arr = _make_global_array(shape_3d, seed=11)

    band = scatter_field_latlon(global_arr, layout)
    assert band.shape[0] == layout.n_lat_local
    assert band.shape[1:] == shape_3d[1:]

    # Slice equivalence: each rank's band should be the global lat-slice.
    s, e = layout.lat_start, layout.lat_end
    np.testing.assert_array_equal(
        np.asarray(band), np.asarray(global_arr[s:e]),
    )

    gathered = gather_field_latlon(band, layout)
    if comm.Get_rank() == 0:
        assert gathered is not None
        assert gathered.shape == global_arr.shape
        np.testing.assert_array_equal(
            np.asarray(gathered), np.asarray(global_arr),
        )
    else:
        assert gathered is None


def test_scatter_then_gather_v_face_matches_global(comm, layout):
    """v-face round-trip: trim-on-send + duplicate-on-receive cancel.

    The v field globally has ``n_lat+1`` rows (lat interfaces).  Each
    rank receives rows ``[lat_start, lat_end+1)`` so the row at
    ``lat_end == neighbour.lat_start`` is held by BOTH ranks.  Gather
    trims every rank except the northernmost.  If the trim convention
    misses or doubles, this test catches it.
    """
    shape = (N_LAT + 1, N_LON, NLEV)
    global_v = _make_global_array(shape, seed=22)

    band = scatter_field_latlon(global_v, layout, is_v_face=True)
    expected_band_rows = layout.n_lat_local + 1
    assert band.shape[0] == expected_band_rows, (
        f"v-band on rank {layout.rank} has {band.shape[0]} rows, "
        f"expected {expected_band_rows}"
    )

    gathered = gather_field_latlon(band, layout, is_v_face=True)
    if comm.Get_rank() == 0:
        assert gathered is not None
        assert gathered.shape == global_v.shape, (
            f"Gathered v has {gathered.shape} rows, expected "
            f"{global_v.shape}.  This is the trim-on-send bug — every "
            "rank except the northernmost should drop its trailing "
            "row before gather."
        )
        np.testing.assert_array_equal(
            np.asarray(gathered), np.asarray(global_v),
        )
    else:
        assert gathered is None


def test_v_face_boundary_row_is_duplicated_across_ranks(comm, layout):
    """The v-face row at ``lat_end`` MUST be identical to the southern
    neighbour's row at ``lat_start = 0``.

    This isn't a gather/scatter property per se — it's what the
    convention is meant to guarantee.  We verify it directly: each
    rank's last v-row (when not the northernmost) must equal the
    neighbour's first v-row.  Without the trim-on-send convention,
    gather would surface a contradiction here on the first non-trivial
    band.
    """
    if comm.Get_size() == 1:
        pytest.skip("Single rank has no neighbour to compare against")

    shape = (N_LAT + 1, N_LON)
    global_v = _make_global_array(shape, seed=33)
    band = scatter_field_latlon(global_v, layout, is_v_face=True)

    # Each rank simultaneously sends its last v-row north and receives
    # the southern neighbour's last v-row.  Use ``comm.sendrecv`` so
    # the test is implementation-independent — MPI_Send's buffering
    # behaviour differs across openmpi / mpich / cray-mpich and a
    # naive ``send`` followed by ``recv`` can deadlock when the
    # message exceeds the eager-protocol threshold.
    send_buf = np.asarray(band[-1]) if layout.north_rank is not None else None
    recv_buf = comm.sendrecv(
        sendobj=send_buf,
        dest=(layout.north_rank if layout.north_rank is not None
              else MPI.PROC_NULL),
        sendtag=0,
        source=(layout.south_rank if layout.south_rank is not None
                else MPI.PROC_NULL),
        recvtag=0,
    )
    if layout.south_rank is not None:
        np.testing.assert_array_equal(
            np.asarray(band[0]), recv_buf,
            err_msg=(
                f"Rank {layout.rank}'s first v-row does not match the "
                f"southern neighbour {layout.south_rank}'s last v-row "
                "— the band-scatter trim convention is broken."
            ),
        )


# NOTE: the "rank 0 passes None to scatter" precondition is covered by
# the *serial* test in tests/parallel/test_latlon_mpi_gather_scatter.py
# ``test_scatter_with_none_global_on_nonzero_rank_when_no_mpi``.  We
# deliberately omit a multi-rank version: in real MPI, rank 0 raising
# before the inner ``comm.bcast`` would deadlock the other ranks (they
# block in bcast forever).  A pytest ``with raises(...)`` on rank 0
# alone cannot rescue them.  The serial pin is sufficient guard.


# ----------------------------------------------------------------------
# Driver-helper assertion tests (Codex review MEDIUM #1, #2 follow-up)
# ----------------------------------------------------------------------
#
# These tests build minimal fake state so they exercise the assertions
# inside ``_gather_state_for_global_checkpoint`` and
# ``_scatter_global_state_to_bands`` without paying the full ModelDriver
# init cost.  The assertions are pure-Python guards that fire BEFORE any
# JAX-side gather call, so a mock state with the right shape +
# ``.replace``-able Field objects is sufficient.

from types import SimpleNamespace  # noqa: E402


class _MockField:
    """Minimal Field-like object with ``.data`` and ``.replace(data=...)``."""

    def __init__(self, data):
        self.data = data

    def replace(self, *, data):
        return _MockField(data)


def _make_mock_state_for_rank(layout):
    """Build a HydrostaticState-like NamedTuple-ish object on this rank.

    Each field is a ``_MockField`` carrying a deterministic band
    derived from a global seed, so neighbouring ranks see consistent
    boundary v-rows by default (the assertion-positive path).
    """
    from collections import namedtuple

    rng = np.random.default_rng(seed=42)
    # Build the GLOBAL fields deterministically, then take this rank's
    # band — so every rank reconstructs the same global state and the
    # duplicated v-rows match across rank pairs.
    g_T = rng.standard_normal((layout.n_lat_global, N_LON, NLEV))
    g_u = rng.standard_normal((layout.n_lat_global, N_LON, NLEV))
    g_v = rng.standard_normal((layout.n_lat_global + 1, N_LON, NLEV))
    g_p = rng.standard_normal((layout.n_lat_global, N_LON))
    g_phi = rng.standard_normal((layout.n_lat_global, N_LON))

    s, e = layout.lat_start, layout.lat_end
    State = namedtuple("MockState", ["T", "u", "v", "p_s", "phis"])
    return State(
        T=_MockField(jnp.asarray(g_T[s:e])),
        u=_MockField(jnp.asarray(g_u[s:e])),
        v=_MockField(jnp.asarray(g_v[s:e + 1])),
        p_s=_MockField(jnp.asarray(g_p[s:e])),
        phis=_MockField(jnp.asarray(g_phi[s:e])),
    )


def _gather_with_mocked_driver(layout, tracers):
    """Run the real ``_gather_state_for_global_checkpoint`` on a fake
    driver shell.  Reuses the production code path (no copy of logic
    into the test) so a regression in the assertion path is caught."""
    from legoesm.driver.model_driver import ModelDriver

    fake_driver = SimpleNamespace(
        _layout=layout,
        _mpi_rank=layout.rank,
        state=_make_mock_state_for_rank(layout),
        tracers=tracers,
    )
    # Bind the unbound method to the fake driver — exactly what
    # ModelDriver(self)._gather_state_for_global_checkpoint() would do.
    return ModelDriver._gather_state_for_global_checkpoint(fake_driver)


def test_gather_assertion_fires_on_divergent_tracer_keys(comm, layout):
    """Rank-dependent tracer key sets MUST raise rather than deadlock.

    Each rank declares a different set of active tracers.  The
    allgather inside ``_gather_state_for_global_checkpoint`` should
    detect the divergence and raise on every rank — no silent gather
    of an asymmetric payload, no MPI hang.
    """
    if comm.Get_size() < 2:
        pytest.skip("Need at least 2 ranks to exhibit divergent keys")

    # Each rank declares a unique tracer.  Together this guarantees
    # ``all_keys[0] != all_keys[1]`` and the assertion must trip.
    my_only_tracer = f"q_rank{layout.rank}"
    tracers = {my_only_tracer: jnp.zeros((layout.n_lat_local, N_LON, NLEV))}

    with pytest.raises(RuntimeError, match="Tracer key sets diverge"):
        _gather_with_mocked_driver(layout, tracers)


def test_gather_succeeds_when_tracer_keys_agree(comm, layout):
    """Sanity counterpart: identical tracer sets across ranks do NOT
    trip the divergence assertion (the assertion shouldn't be a
    false-positive trap for the normal case)."""
    # Every rank declares the same canonical AMIP set.
    tracers = {
        "q_v": jnp.zeros((layout.n_lat_local, N_LON, NLEV)),
        "q_c": jnp.zeros((layout.n_lat_local, N_LON, NLEV)),
        "q_r": jnp.zeros((layout.n_lat_local, N_LON, NLEV)),
    }

    # Should not raise.  Rank 0 returns (state_g, tracers_g), others
    # return (None, None).
    state_g, tracers_g = _gather_with_mocked_driver(layout, tracers)
    if layout.rank == 0:
        assert state_g is not None
        assert set(tracers_g.keys()) == {"q_v", "q_c", "q_r"}
    else:
        assert state_g is None
        assert tracers_g is None


def test_gather_assertion_fires_on_v_face_row_mismatch(comm, layout):
    """If a rank's v[0] disagrees with the southern neighbour's
    v[-1], ``_gather_state_for_global_checkpoint`` must raise
    rather than silently write a corrupt global v field."""
    if comm.Get_size() < 2:
        pytest.skip("Need at least 2 ranks to have a boundary row")

    # Build state where v on rank>0 has its first row perturbed,
    # creating an artificial boundary-row mismatch with rank-1's last
    # row.  Rank 0 keeps the default state so the assertion catches the
    # divergence at the rank-0/rank-1 interface.
    base_state = _make_mock_state_for_rank(layout)
    if layout.rank > 0:
        v = np.asarray(base_state.v.data)
        v[0] = v[0] + 1.0  # deliberate perturbation
        base_state = base_state._replace(v=_MockField(jnp.asarray(v)))

    from legoesm.driver.model_driver import ModelDriver

    fake_driver = SimpleNamespace(
        _layout=layout,
        _mpi_rank=layout.rank,
        state=base_state,
        tracers={"q_v": jnp.zeros((layout.n_lat_local, N_LON, NLEV))},
    )

    # The assertion uses ``allreduce(MAX)`` over every rank's local
    # boundary-row diff, so EVERY rank sees the global max and raises
    # together — no rank slips past into a hung gather collective.
    # This is the collective-safe behaviour we validate here.
    with pytest.raises(RuntimeError, match="v-face boundary row diverges"):
        ModelDriver._gather_state_for_global_checkpoint(fake_driver)
