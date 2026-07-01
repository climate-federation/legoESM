"""Voronoi batched (union-neighbor) halo schedule + pack/unpack tests.

No MPI launcher required: pure schedule math, pack/unpack round-trips
through a simulated exchange, message-count accounting, the np=1
identity degeneracy, and an AST guard that the Voronoi halo module never
calls raw ``mpi4jax.sendrecv`` (every message must route through the
AD-safe ``get_sendrecv_vjp`` wrapper).

Companions:

- tests/distributed/test_voronoi_halo.py — per-entity schedules,
  simulated exchange.
- tests/distributed/test_voronoi_mpi.py — real-MPI halo correctness.
- tests/distributed/test_mpi_differentiability.py::TestVoronoiHaloMPIGrad
  — gradients through the batched exchange under mpirun.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import legoesm.parallel.halo_exchange_voronoi as hev
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.parallel.voronoi_partition import (
    BatchedHaloSchedule,
    build_batched_halo_schedule,
    partition_cells_geometric,
    partition_voronoi_mesh,
    scatter_to_local,
)
from legoesm.parallel.halo_exchange_voronoi import (
    batched_halo_exchange,
    count_batched_messages,
    get_halo_message_count,
    pack_batched_sends,
    reset_halo_message_count,
    unpack_batched_recvs,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    """Exact (bitwise) round-trip assertions need real float64."""
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


@pytest.fixture(scope="module")
def mesh():
    """Small SCVT mesh (162 cells) — same as test_voronoi_halo."""
    return create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)


def _build_all(mesh, n_ranks):
    """All ranks' partitions + batched schedules, in one process."""
    cell_owner = partition_cells_geometric(mesh, n_ranks)
    parts = [
        partition_voronoi_mesh(mesh, n_ranks, r, cell_owner=cell_owner)
        for r in range(n_ranks)
    ]
    scheds = [
        build_batched_halo_schedule(p.cell_comm, p.edge_comm) for p in parts
    ]
    return parts, scheds


def _entity_slices(comm):
    """rank -> (send global-id-ordered local rows, recv rows) per entity."""
    out = {}
    s_off = r_off = 0
    s_idx = np.asarray(comm.send_idx)
    r_idx = np.asarray(comm.recv_idx)
    for i, r in enumerate(comm.neighbor_ranks):
        out[r] = (
            s_idx[s_off:s_off + comm.send_counts[i]],
            r_idx[r_off:r_off + comm.recv_counts[i]],
        )
        s_off += comm.send_counts[i]
        r_off += comm.recv_counts[i]
    return out


# ========================================================================
# Schedule construction
# ========================================================================

class TestBatchedScheduleConstruction:
    """build_batched_halo_schedule joins cell+edge index spaces correctly."""

    @pytest.mark.parametrize("n_ranks", [2, 3, 4])
    def test_union_neighbor_list(self, mesh, n_ranks):
        parts, scheds = _build_all(mesh, n_ranks)
        for p, s in zip(parts, scheds):
            expected = sorted(
                set(p.cell_comm.neighbor_ranks)
                | set(p.edge_comm.neighbor_ranks)
            )
            assert list(s.neighbor_ranks) == expected
            # Union list never contains self.
            assert p.rank not in s.neighbor_ranks

    @pytest.mark.parametrize("n_ranks", [2, 3, 4])
    def test_per_neighbor_slices_match_entity_schedules(self, mesh, n_ranks):
        """Each union neighbor's rows reproduce the entity schedule rows
        (zero-length where the rank is absent from that entity)."""
        parts, scheds = _build_all(mesh, n_ranks)
        for p, s in zip(parts, scheds):
            cell_by = _entity_slices(p.cell_comm)
            edge_by = _entity_slices(p.edge_comm)
            c_s_off = c_r_off = e_s_off = e_r_off = 0
            for i, r in enumerate(s.neighbor_ranks):
                cs, cr = cell_by.get(r, (np.empty(0), np.empty(0)))
                es, er = edge_by.get(r, (np.empty(0), np.empty(0)))
                assert s.cell_send_counts[i] == len(cs)
                assert s.cell_recv_counts[i] == len(cr)
                assert s.edge_send_counts[i] == len(es)
                assert s.edge_recv_counts[i] == len(er)
                np.testing.assert_array_equal(
                    np.asarray(s.cell_send_idx)[
                        c_s_off:c_s_off + len(cs)], cs)
                np.testing.assert_array_equal(
                    np.asarray(s.cell_recv_idx)[
                        c_r_off:c_r_off + len(cr)], cr)
                np.testing.assert_array_equal(
                    np.asarray(s.edge_send_idx)[
                        e_s_off:e_s_off + len(es)], es)
                np.testing.assert_array_equal(
                    np.asarray(s.edge_recv_idx)[
                        e_r_off:e_r_off + len(er)], er)
                c_s_off += len(cs)
                c_r_off += len(cr)
                e_s_off += len(es)
                e_r_off += len(er)
            # Full coverage: nothing dropped, nothing invented.
            assert c_s_off == len(np.asarray(p.cell_comm.send_idx))
            assert c_r_off == len(np.asarray(p.cell_comm.recv_idx))
            assert e_s_off == len(np.asarray(p.edge_comm.send_idx))
            assert e_r_off == len(np.asarray(p.edge_comm.recv_idx))

    @pytest.mark.parametrize("n_ranks", [2, 3, 4])
    def test_recv_idx_concatenation_equals_entity_recv_idx(
            self, mesh, n_ranks):
        """Union-order concatenation of recv rows == entity recv_idx.

        This is what lets ``unpack_batched_recvs`` do a single functional
        scatter per field.  Holds because both the entity neighbor lists
        and the union list are sorted by rank, with zero-length segments
        for union-only ranks.
        """
        parts, scheds = _build_all(mesh, n_ranks)
        for p, s in zip(parts, scheds):
            np.testing.assert_array_equal(
                np.asarray(s.cell_recv_idx), np.asarray(p.cell_comm.recv_idx))
            np.testing.assert_array_equal(
                np.asarray(s.edge_recv_idx), np.asarray(p.edge_comm.recv_idx))
            np.testing.assert_array_equal(
                np.asarray(s.cell_send_idx), np.asarray(p.cell_comm.send_idx))
            np.testing.assert_array_equal(
                np.asarray(s.edge_send_idx), np.asarray(p.edge_comm.send_idx))

    @pytest.mark.parametrize("n_ranks", [2, 3, 4])
    def test_schedule_symmetry_across_ranks(self, mesh, n_ranks):
        """Pack-index correctness for (potentially) asymmetric lists.

        For every pair (A, B): B in A's union iff A in B's union; A's
        send counts to B equal B's recv counts from A per entity; and the
        GLOBAL ids of A's send rows equal the global ids of B's recv rows
        elementwise (both sides sort by global index).  This is the
        uniform-collective-schedule requirement: matching blocking
        sendrecv posts with matching sizes on both ends.
        """
        parts, scheds = _build_all(mesh, n_ranks)
        for a, (pa, sa) in enumerate(zip(parts, scheds)):
            for i, b in enumerate(sa.neighbor_ranks):
                sb = scheds[b]
                assert a in sb.neighbor_ranks, (
                    f"rank {b} missing rank {a} from its union list"
                )
                j = sb.neighbor_ranks.index(a)
                # Count symmetry (message size pairing).
                assert sa.cell_send_counts[i] == sb.cell_recv_counts[j]
                assert sa.cell_recv_counts[i] == sb.cell_send_counts[j]
                assert sa.edge_send_counts[i] == sb.edge_recv_counts[j]
                assert sa.edge_recv_counts[i] == sb.edge_send_counts[j]

                # Content symmetry: global ids of A's sends == B's recvs.
                def _seg(idx, counts, k):
                    off = sum(counts[:k])
                    return np.asarray(idx)[off:off + counts[k]]

                pb = parts[b]
                a_send_cells = np.asarray(pa.local_cells)[
                    _seg(sa.cell_send_idx, sa.cell_send_counts, i)]
                b_recv_cells = np.asarray(pb.local_cells)[
                    _seg(sb.cell_recv_idx, sb.cell_recv_counts, j)]
                np.testing.assert_array_equal(a_send_cells, b_recv_cells)

                a_send_edges = np.asarray(pa.local_edges)[
                    _seg(sa.edge_send_idx, sa.edge_send_counts, i)]
                b_recv_edges = np.asarray(pb.local_edges)[
                    _seg(sb.edge_recv_idx, sb.edge_recv_counts, j)]
                np.testing.assert_array_equal(a_send_edges, b_recv_edges)


# ========================================================================
# Message-count accounting (codex acceptance: 3x reduction)
# ========================================================================

class TestMessageCount:
    """Batched path posts n_union_neighbors messages per dtype group,
    vs the legacy n_edge_neighbors + 2*n_cell_neighbors."""

    @pytest.mark.parametrize("n_ranks", [2, 3, 4])
    def test_single_dtype_group_count(self, mesh, n_ranks):
        parts, scheds = _build_all(mesh, n_ranks)
        nlev = 4
        for p, s in zip(parts, scheds):
            nC, nE = p.n_local_cells, p.n_local_edges
            u = jnp.zeros((nE, nlev), dtype=jnp.float64)
            T = jnp.zeros((nC, nlev), dtype=jnp.float64)
            ps = jnp.zeros((nC,), dtype=jnp.float64)
            qv = jnp.zeros((nC, nlev), dtype=jnp.float64)

            batched = count_batched_messages((u,), (T, ps, qv), s)
            legacy = (len(p.edge_comm.neighbor_ranks)
                      + 2 * len(p.cell_comm.neighbor_ranks))

            # Every union neighbor has cell or edge traffic by
            # construction, and the single group spans both spaces.
            assert batched == len(s.neighbor_ranks)
            assert batched == s.messages_per_exchange(n_dtype_groups=1)
            assert batched < legacy, (
                f"rank {p.rank}: batched {batched} !< legacy {legacy}"
            )
            # Without tracers the legacy count drops by n_cell_neighbors;
            # the batched count is unchanged (same union schedule).
            assert count_batched_messages((u,), (T, ps), s) == batched

    @pytest.mark.parametrize("n_ranks", [2, 3])
    def test_two_dtype_groups_count(self, mesh, n_ranks):
        """Mixed precision: one buffer per dtype group per neighbor, with
        per-group skips when a group has no traffic with a neighbor."""
        parts, scheds = _build_all(mesh, n_ranks)
        nlev = 3
        for p, s in zip(parts, scheds):
            nC, nE = p.n_local_cells, p.n_local_edges
            u32 = jnp.zeros((nE, nlev), dtype=jnp.float32)   # edge group
            T64 = jnp.zeros((nC, nlev), dtype=jnp.float64)   # cell group
            ps64 = jnp.zeros((nC,), dtype=jnp.float64)

            n = len(s.neighbor_ranks)
            edge_traffic = sum(
                1 for i in range(n)
                if s.edge_send_counts[i] + s.edge_recv_counts[i] > 0)
            cell_traffic = sum(
                1 for i in range(n)
                if s.cell_send_counts[i] + s.cell_recv_counts[i] > 0)

            got = count_batched_messages((u32,), (T64, ps64), s)
            assert got == edge_traffic + cell_traffic
            assert got <= s.messages_per_exchange(n_dtype_groups=2)

    def test_counter_accessors(self):
        reset_halo_message_count()
        assert get_halo_message_count() == 0
        reset_halo_message_count()
        assert get_halo_message_count() == 0


# ========================================================================
# Pack / unpack round-trips (no MPI)
# ========================================================================

def _selfloop_schedule():
    """Synthetic schedule whose recv rows EQUAL its send rows, with
    asymmetric per-neighbor counts and union-only neighbors:

    - nbr 1: edge-only (zero cell counts) — the asymmetric union case;
    - nbr 2: both spaces;
    - nbr 3: cell-only (zero edge counts).
    """
    cell_idx = jnp.array([2, 5, 7, 0], dtype=jnp.int32)     # nbr2: 2, nbr3: 2
    edge_idx = jnp.array([1, 3, 8, 12], dtype=jnp.int32)    # nbr1: 3, nbr2: 1
    return BatchedHaloSchedule(
        neighbor_ranks=(1, 2, 3),
        cell_send_counts=(0, 2, 2),
        cell_recv_counts=(0, 2, 2),
        edge_send_counts=(3, 1, 0),
        edge_recv_counts=(3, 1, 0),
        cell_send_idx=cell_idx,
        cell_recv_idx=cell_idx,
        edge_send_idx=edge_idx,
        edge_recv_idx=edge_idx,
    )


class TestPackUnpack:
    """pack -> (identity 'exchange') -> unpack == originals, exactly."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize("with_tracers", [True, False])
    def test_selfloop_identity(self, dtype, with_tracers):
        """Self-loop schedule (recv rows == send rows): receiving one's
        own packed buffers must reproduce every field bit-exactly."""
        sched = _selfloop_schedule()
        nC, nE, nlev = 10, 14, 3
        rng = np.random.default_rng(42)
        u = jnp.asarray(rng.standard_normal((nE, nlev)), dtype=dtype)
        T = jnp.asarray(rng.standard_normal((nC, nlev)), dtype=dtype)
        ps = jnp.asarray(rng.standard_normal((nC,)), dtype=dtype)
        cell_fields = (T, ps)
        if with_tracers:
            qv = jnp.asarray(rng.standard_normal((nC, nlev)), dtype=dtype)
            qc = jnp.asarray(rng.standard_normal((nC, nlev)), dtype=dtype)
            cell_fields = cell_fields + (qv, qc)
        edge_fields = (u,)

        bufs = pack_batched_sends(edge_fields, cell_fields, sched)
        # Identity exchange: recv buffers == own send buffers (valid for
        # the self-loop schedule since send/recv counts match).
        edge_out, cell_out = unpack_batched_recvs(
            edge_fields, cell_fields, sched, bufs)

        for orig, out in zip(edge_fields + cell_fields,
                             edge_out + cell_out):
            assert out.dtype == orig.dtype, "dtype changed in round-trip"
            np.testing.assert_array_equal(np.asarray(out), np.asarray(orig))

    def test_selfloop_identity_mixed_dtypes(self):
        """Two dtype groups in one exchange: no promotion, exact values."""
        sched = _selfloop_schedule()
        nC, nE, nlev = 10, 14, 3
        rng = np.random.default_rng(7)
        u32 = jnp.asarray(rng.standard_normal((nE, nlev)), dtype=jnp.float32)
        T64 = jnp.asarray(rng.standard_normal((nC, nlev)), dtype=jnp.float64)
        ps32 = jnp.asarray(rng.standard_normal((nC,)), dtype=jnp.float32)

        bufs = pack_batched_sends((u32,), (T64, ps32), sched)
        # Group order = first appearance: f32 (u32, ps32) then f64 (T64).
        assert [g[0].dtype for g in bufs] == [
            jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)]
        (u_out,), (T_out, ps_out) = unpack_batched_recvs(
            (u32,), (T64, ps32), sched, bufs)
        assert u_out.dtype == jnp.float32
        assert T_out.dtype == jnp.float64
        assert ps_out.dtype == jnp.float32
        np.testing.assert_array_equal(np.asarray(u_out), np.asarray(u32))
        np.testing.assert_array_equal(np.asarray(T_out), np.asarray(T64))
        np.testing.assert_array_equal(np.asarray(ps_out), np.asarray(ps32))

    @pytest.mark.parametrize("n_ranks", [2, 3])
    @pytest.mark.parametrize("with_tracers", [True, False])
    def test_simulated_exchange_fills_halos_exactly(
            self, mesh, n_ranks, with_tracers):
        """Wire every rank's recv buffers to its neighbors' packed send
        buffers (in one process) and verify halo rows come out equal to
        the global field — the exact value contract of the MPI path."""
        parts, scheds = _build_all(mesh, n_ranks)
        nlev = 3
        rng = np.random.default_rng(2026)
        u_g = jnp.asarray(rng.standard_normal((mesh.nEdges, nlev)))
        T_g = jnp.asarray(rng.standard_normal((mesh.nCells, nlev)))
        ps_g = jnp.asarray(rng.standard_normal((mesh.nCells,)))
        qv_g = jnp.asarray(rng.standard_normal((mesh.nCells, nlev)))

        def _local(arr, p, entity, n_owned):
            loc = scatter_to_local(arr, p, entity)
            # Zero the halo so the test FAILS unless unpack fills it.
            return loc.at[n_owned:].set(0.0)

        edge_fields_all, cell_fields_all = [], []
        for p in parts:
            ef = (_local(u_g, p, "edge", p.n_owned_edges),)
            cf = (_local(T_g, p, "cell", p.n_owned_cells),
                  _local(ps_g, p, "cell", p.n_owned_cells))
            if with_tracers:
                cf = cf + (_local(qv_g, p, "cell", p.n_owned_cells),)
            edge_fields_all.append(ef)
            cell_fields_all.append(cf)

        packed = [
            pack_batched_sends(edge_fields_all[r], cell_fields_all[r], s)
            for r, s in enumerate(scheds)
        ]

        for r, (p, s) in enumerate(zip(parts, scheds)):
            recv_buffers = []
            for g in range(len(packed[r])):
                g_recv = []
                for nbr in s.neighbor_ranks:
                    j = scheds[nbr].neighbor_ranks.index(r)
                    g_recv.append(packed[nbr][g][j])
                recv_buffers.append(g_recv)
            edge_out, cell_out = unpack_batched_recvs(
                edge_fields_all[r], cell_fields_all[r], s, recv_buffers)

            np.testing.assert_array_equal(
                np.asarray(edge_out[0]),
                np.asarray(scatter_to_local(u_g, p, "edge")),
                err_msg=f"rank {r}: u halo mismatch")
            expected_cells = [scatter_to_local(T_g, p, "cell"),
                              scatter_to_local(ps_g, p, "cell")]
            if with_tracers:
                expected_cells.append(scatter_to_local(qv_g, p, "cell"))
            for got, want, name in zip(
                    cell_out, expected_cells,
                    ["T", "p_s", "q_v"][:len(expected_cells)]):
                np.testing.assert_array_equal(
                    np.asarray(got), np.asarray(want),
                    err_msg=f"rank {r}: {name} halo mismatch")

    def test_np1_batched_exchange_is_identity_without_mpi(self, mesh):
        """np=1 degeneracy: empty union schedule short-circuits BEFORE any
        MPI import — bit-identity, runnable on a no-MPI host."""
        parts, scheds = _build_all(mesh, 1)
        p, s = parts[0], scheds[0]
        assert s.neighbor_ranks == ()
        nlev = 3
        rng = np.random.default_rng(5)
        u = jnp.asarray(rng.standard_normal((p.n_local_edges, nlev)))
        T = jnp.asarray(rng.standard_normal((p.n_local_cells, nlev)))
        ps = jnp.asarray(rng.standard_normal((p.n_local_cells,)))

        (u_out,), (T_out, ps_out) = batched_halo_exchange(
            (u,), (T, ps), s, rank=0)
        # Identity — in fact the same array objects (no copies).
        assert u_out is u and T_out is T and ps_out is ps
        assert count_batched_messages((u,), (T, ps), s) == 0


# ========================================================================
# AD routing: no raw mpi4jax.sendrecv left in the Voronoi halo module
# ========================================================================

def _raw_sendrecv_call_lines(source: str) -> list[int]:
    """Line numbers of attribute-style ``*.sendrecv(...)`` calls.

    The AD-safe path calls the wrapper as a bare name
    (``sendrecv = get_sendrecv_vjp(mpi4jax); sendrecv(...)``), so any
    attribute call ``<module>.sendrecv(...)`` is a raw, forward-only
    mpi4jax primitive — a regression of the custom-VJP routing.
    """
    tree = ast.parse(source)
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "sendrecv"
    ]


class TestADRoutingStatic:
    def test_no_raw_mpi4jax_sendrecv_in_voronoi_halo(self):
        source = Path(hev.__file__).read_text()
        hits = _raw_sendrecv_call_lines(source)
        assert hits == [], (
            f"raw *.sendrecv call(s) at line(s) {hits} of "
            f"{hev.__file__} — route through get_sendrecv_vjp instead"
        )
        # And the AD-safe wrapper is actually what the module uses.
        assert "get_sendrecv_vjp" in source

    def test_raw_sendrecv_detector_catches_violation(self):
        """Tripwire self-test: the AST guard is not vacuous."""
        bad = (
            "import mpi4jax\n"
            "def f(a, b, c):\n"
            "    return mpi4jax.sendrecv(a, b, source=c, dest=c)\n"
        )
        assert _raw_sendrecv_call_lines(bad) == [3]
        good = (
            "from legoesm.parallel.halo_exchange import get_sendrecv_vjp\n"
            "def f(mpi4jax, a, b, c):\n"
            "    sendrecv = get_sendrecv_vjp(mpi4jax)\n"
            "    return sendrecv(a, b, c, c, 0, 0, None)\n"
        )
        assert _raw_sendrecv_call_lines(good) == []


class TestRankIndependentTags:
    """Halo MPI tags must be rank-INDEPENDENT and tiny at ANY rank count.

    The old per-entity scheme encoded ``rank * 1000 + nbr`` and hard-raised at
    >= 1000 ranks (and the batched path's ``rank * n_ranks`` + 3e6 base pushed
    tags past MPI_TAG_UB).  mpi4jax ``sendrecv`` already matches on (source,
    dest), so the rank pair is redundant in the tag; dropping it removes the
    ceiling.  These tests lock that the scheme can never reintroduce a
    rank-dependent tag.
    """

    # MPI only guarantees tags up to MPI_TAG_UB >= 32767 (the standard minimum).
    _MPI_GUARANTEED_TAG_UB = 32767

    def test_entity_tag_is_the_entity_type(self):
        assert hev._entity_tag(0) == 0  # cell
        assert hev._entity_tag(1) == 1  # edge
        assert hev._entity_tag(2) == 2  # vertex

    def test_batch_group_tags_disjoint_from_entity_tags(self):
        entity_tags = {hev._entity_tag(e) for e in (0, 1, 2)}
        group_tags = {hev._batch_group_tag(g) for g in range(16)}
        assert entity_tags.isdisjoint(group_tags)
        assert hev._batch_group_tag(0) == hev._BATCH_TAG_BASE

    def test_tags_fit_guaranteed_tag_ub_for_realistic_group_counts(self):
        # Group tags grow with the dtype-group count (NOT unbounded for any
        # count): realistic states have a handful of groups.  256 is already far
        # more than any real model state, and all still fit the guaranteed UB.
        for g in range(256):
            assert hev._batch_group_tag(g) <= self._MPI_GUARANTEED_TAG_UB
        for e in (0, 1, 2):
            assert hev._entity_tag(e) <= self._MPI_GUARANTEED_TAG_UB

    def test_check_tag_bound_raises_on_overflow(self):
        # A pathological group count (tag > MPI_TAG_UB) must FAIL LOUD via the
        # runtime guard, never silently overflow / mis-pair.
        class _FakeComm:
            @staticmethod
            def Get_attr(_key):  # noqa: N802 - mirrors mpi4py's MPI API name
                return 32767

        class _FakeMPI:
            TAG_UB = "tag_ub"
            COMM_WORLD = _FakeComm

        hev._check_tag_bound(32767, _FakeMPI, "test")  # at the bound: ok
        with pytest.raises(ValueError, match="MPI_TAG_UB"):
            hev._check_tag_bound(32768, _FakeMPI, "test")  # over: raise

    def test_tag_helpers_take_no_rank_argument(self):
        # Structural guarantee: a tag is a function of (entity type / group)
        # ONLY, never the rank -> no rank ceiling can be reintroduced.
        import inspect

        assert list(inspect.signature(hev._entity_tag).parameters) == [
            "entity_type"
        ]
        assert list(inspect.signature(hev._batch_group_tag).parameters) == [
            "group"
        ]
