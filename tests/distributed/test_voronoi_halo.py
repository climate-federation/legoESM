"""Tests for Voronoi mesh domain decomposition and halo exchange.

Verifies:
- Partition validity (all cells assigned, no overlap)
- Local mesh connectivity correctness
- Operator equivalence between global and local computation
- Halo exchange correctness
- Communication schedule symmetry

All tests run without MPI by simulating multiple ranks in one process.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.core.operators_voronoi import (
    divergence_cell,
    gradient_edge,
    curl_vertex,
    kinetic_energy_cell,
)
from legoesm.parallel.voronoi_partition import (
    partition_cells_geometric,
    partition_voronoi_mesh,
    build_local_mesh,
    scatter_to_local,
)
from legoesm.parallel.halo_exchange_voronoi import (
    exchange_local_simulated,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    """Tight tolerances require float64 precision."""
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


@pytest.fixture(scope="module")
def mesh():
    """Small SCVT mesh for testing (162 cells)."""
    return create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)


# ========================================================================
# Partition validity
# ========================================================================

class TestPartitioning:
    """Tests for cell partitioning and partition structure."""

    @pytest.mark.parametrize("n_ranks", [2, 3, 4])
    def test_all_cells_assigned(self, mesh, n_ranks):
        """Every cell is assigned to a rank in [0, n_ranks)."""
        owner = partition_cells_geometric(mesh, n_ranks)
        assert owner.shape == (mesh.nCells,)
        assert set(owner.tolist()) == set(range(n_ranks))

    @pytest.mark.parametrize("n_ranks", [2, 3, 4])
    def test_owned_cells_no_overlap(self, mesh, n_ranks):
        """Each cell is owned by exactly one rank."""
        parts = [partition_voronoi_mesh(mesh, n_ranks, r) for r in range(n_ranks)]
        all_owned: set[int] = set()
        for p in parts:
            owned = set(p.local_cells[:p.n_owned_cells].tolist())
            assert len(owned & all_owned) == 0, "Overlap in owned cells"
            all_owned |= owned
        assert all_owned == set(range(mesh.nCells))

    @pytest.mark.parametrize("n_ranks", [2, 3, 4])
    def test_owned_edges_no_overlap(self, mesh, n_ranks):
        """Each edge is owned by exactly one rank."""
        parts = [partition_voronoi_mesh(mesh, n_ranks, r) for r in range(n_ranks)]
        all_owned: set[int] = set()
        for p in parts:
            owned = set(p.local_edges[:p.n_owned_edges].tolist())
            assert len(owned & all_owned) == 0, "Overlap in owned edges"
            all_owned |= owned
        assert all_owned == set(range(mesh.nEdges))

    @pytest.mark.parametrize("n_ranks", [2, 3, 4])
    def test_owned_vertices_no_overlap(self, mesh, n_ranks):
        """Each vertex is owned by exactly one rank."""
        parts = [partition_voronoi_mesh(mesh, n_ranks, r) for r in range(n_ranks)]
        all_owned: set[int] = set()
        for p in parts:
            owned = set(p.local_vertices[:p.n_owned_vertices].tolist())
            assert len(owned & all_owned) == 0, "Overlap in owned vertices"
            all_owned |= owned
        assert all_owned == set(range(mesh.nVertices))

    def test_halo_contains_neighbors(self, mesh):
        """Every owned cell's neighbors are either owned or halo."""
        n_ranks = 3
        coc = np.asarray(mesh.cellsOnCell)
        for r in range(n_ranks):
            part = partition_voronoi_mesh(mesh, n_ranks, r)
            local_set = set(part.local_cells.tolist())
            for c in part.local_cells[:part.n_owned_cells]:
                for k in range(mesh.maxEdges):
                    nbr = int(coc[k, c])
                    if nbr >= 0:
                        assert nbr in local_set, (
                            f"rank {r}: neighbor {nbr} of owned cell {c} "
                            f"not in local domain"
                        )


# ========================================================================
# Local mesh
# ========================================================================

class TestLocalMesh:
    """Tests for local mesh construction."""

    def test_shapes(self, mesh):
        """Local mesh arrays have correct shapes."""
        n_ranks = 3
        for r in range(n_ranks):
            part = partition_voronoi_mesh(mesh, n_ranks, r)
            local = build_local_mesh(mesh, part)
            assert local.nCells == part.n_local_cells
            assert local.nEdges == part.n_local_edges
            assert local.nVertices == part.n_local_vertices
            assert local.areaCell.shape == (part.n_local_cells,)
            assert local.dcEdge.shape == (part.n_local_edges,)
            assert local.areaTriangle.shape == (part.n_local_vertices,)

    def test_connectivity_bounds(self, mesh):
        """All non-padding connectivity entries are valid local indices."""
        n_ranks = 3
        for r in range(n_ranks):
            part = partition_voronoi_mesh(mesh, n_ranks, r)
            local = build_local_mesh(mesh, part)

            coe = np.asarray(local.cellsOnEdge)
            assert np.all(coe >= -1)
            assert np.all(coe < part.n_local_cells)

            eoc = np.asarray(local.edgesOnCell)
            assert np.all(eoc >= -1)
            assert np.all(eoc < part.n_local_edges)

            voe = np.asarray(local.verticesOnEdge)
            assert np.all(voe >= -1)
            assert np.all(voe < part.n_local_vertices)

    def test_geometry_preserved(self, mesh):
        """Cell areas match between global and local mesh."""
        n_ranks = 2
        for r in range(n_ranks):
            part = partition_voronoi_mesh(mesh, n_ranks, r)
            local = build_local_mesh(mesh, part)
            for i in range(part.n_owned_cells):
                g = part.local_cells[i]
                np.testing.assert_allclose(
                    float(local.areaCell[i]),
                    float(mesh.areaCell[g]),
                    rtol=1e-14,
                )

    def test_coordinates_preserved(self, mesh):
        """Coordinates match between global and local mesh."""
        part = partition_voronoi_mesh(mesh, 2, 0)
        local = build_local_mesh(mesh, part)
        np.testing.assert_allclose(
            np.asarray(local.latCell),
            np.asarray(mesh.latCell[part.local_cells]),
            atol=1e-15,
        )
        np.testing.assert_allclose(
            np.asarray(local.latEdge),
            np.asarray(mesh.latEdge[part.local_edges]),
            atol=1e-15,
        )


# ========================================================================
# Operator equivalence
# ========================================================================

class TestOperatorEquivalence:
    """Local operators must match global operators on owned entities."""

    @pytest.mark.parametrize("n_ranks", [2, 3])
    def test_divergence(self, mesh, n_ranks):
        key = jax.random.PRNGKey(42)
        u_global = jax.random.normal(key, (mesh.nEdges,), dtype=jnp.float64)
        div_global = divergence_cell(u_global, mesh)

        for r in range(n_ranks):
            part = partition_voronoi_mesh(mesh, n_ranks, r)
            local_mesh = build_local_mesh(mesh, part)
            u_local = scatter_to_local(u_global, part, "edge")
            div_local = divergence_cell(u_local, local_mesh)

            for i in range(part.n_owned_cells):
                g = part.local_cells[i]
                np.testing.assert_allclose(
                    float(div_local[i]), float(div_global[g]),
                    atol=1e-12,
                    err_msg=f"rank={r}, local cell {i} (global {g})",
                )

    @pytest.mark.parametrize("n_ranks", [2, 3])
    def test_gradient(self, mesh, n_ranks):
        key = jax.random.PRNGKey(7)
        phi_global = jax.random.normal(key, (mesh.nCells,), dtype=jnp.float64)
        grad_global = gradient_edge(phi_global, mesh)

        for r in range(n_ranks):
            part = partition_voronoi_mesh(mesh, n_ranks, r)
            local_mesh = build_local_mesh(mesh, part)
            phi_local = scatter_to_local(phi_global, part, "cell")
            grad_local = gradient_edge(phi_local, local_mesh)

            for i in range(part.n_owned_edges):
                g = part.local_edges[i]
                np.testing.assert_allclose(
                    float(grad_local[i]), float(grad_global[g]),
                    atol=1e-12,
                    err_msg=f"rank={r}, local edge {i} (global {g})",
                )

    @pytest.mark.parametrize("n_ranks", [2, 3])
    def test_curl(self, mesh, n_ranks):
        key = jax.random.PRNGKey(13)
        u_global = jax.random.normal(key, (mesh.nEdges,), dtype=jnp.float64)
        curl_global = curl_vertex(u_global, mesh)

        for r in range(n_ranks):
            part = partition_voronoi_mesh(mesh, n_ranks, r)
            local_mesh = build_local_mesh(mesh, part)
            u_local = scatter_to_local(u_global, part, "edge")
            curl_local = curl_vertex(u_local, local_mesh)

            for i in range(part.n_owned_vertices):
                g = part.local_vertices[i]
                np.testing.assert_allclose(
                    float(curl_local[i]), float(curl_global[g]),
                    atol=1e-12,
                    err_msg=f"rank={r}, local vertex {i} (global {g})",
                )

    @pytest.mark.parametrize("n_ranks", [2, 3])
    def test_kinetic_energy(self, mesh, n_ranks):
        key = jax.random.PRNGKey(99)
        u_global = jax.random.normal(key, (mesh.nEdges,), dtype=jnp.float64)
        ke_global = kinetic_energy_cell(u_global, mesh)

        for r in range(n_ranks):
            part = partition_voronoi_mesh(mesh, n_ranks, r)
            local_mesh = build_local_mesh(mesh, part)
            u_local = scatter_to_local(u_global, part, "edge")
            ke_local = kinetic_energy_cell(u_local, local_mesh)

            for i in range(part.n_owned_cells):
                g = part.local_cells[i]
                np.testing.assert_allclose(
                    float(ke_local[i]), float(ke_global[g]),
                    atol=1e-12,
                    err_msg=f"rank={r}, local cell {i} (global {g})",
                )


# ========================================================================
# Halo exchange
# ========================================================================

class TestHaloExchange:
    """Tests for halo exchange correctness."""

    @pytest.mark.parametrize("n_ranks", [2, 3, 4])
    def test_cell_exchange(self, mesh, n_ranks):
        """After exchange, halo cells have correct values."""
        phi_global = jnp.arange(mesh.nCells, dtype=jnp.float64)
        parts = [
            partition_voronoi_mesh(mesh, n_ranks, r) for r in range(n_ranks)
        ]

        # Start with only owned values; halo = 0.
        local_fields = []
        for p in parts:
            f = jnp.zeros(p.n_local_cells, dtype=jnp.float64)
            f = f.at[:p.n_owned_cells].set(
                phi_global[p.local_cells[:p.n_owned_cells]],
            )
            local_fields.append(f)

        updated = exchange_local_simulated(parts, local_fields, "cell")

        for r, (p, f) in enumerate(zip(parts, updated)):
            expected = phi_global[p.local_cells]
            np.testing.assert_allclose(
                f, expected, atol=1e-12, err_msg=f"rank {r}",
            )

    @pytest.mark.parametrize("n_ranks", [2, 3])
    def test_edge_exchange(self, mesh, n_ranks):
        """After exchange, halo edges have correct values."""
        u_global = jnp.arange(mesh.nEdges, dtype=jnp.float64) * 0.1
        parts = [
            partition_voronoi_mesh(mesh, n_ranks, r) for r in range(n_ranks)
        ]

        local_fields = []
        for p in parts:
            f = jnp.zeros(p.n_local_edges, dtype=jnp.float64)
            f = f.at[:p.n_owned_edges].set(
                u_global[p.local_edges[:p.n_owned_edges]],
            )
            local_fields.append(f)

        updated = exchange_local_simulated(parts, local_fields, "edge")

        for r, (p, f) in enumerate(zip(parts, updated)):
            expected = u_global[p.local_edges]
            np.testing.assert_allclose(
                f, expected, atol=1e-12, err_msg=f"rank {r}",
            )

    @pytest.mark.parametrize("n_ranks", [2, 3])
    def test_vertex_exchange(self, mesh, n_ranks):
        """After exchange, halo vertices have correct values."""
        q_global = jnp.sin(mesh.latVertex)
        parts = [
            partition_voronoi_mesh(mesh, n_ranks, r) for r in range(n_ranks)
        ]

        local_fields = []
        for p in parts:
            f = jnp.zeros(p.n_local_vertices, dtype=jnp.float64)
            f = f.at[:p.n_owned_vertices].set(
                q_global[p.local_vertices[:p.n_owned_vertices]],
            )
            local_fields.append(f)

        updated = exchange_local_simulated(parts, local_fields, "vertex")

        for r, (p, f) in enumerate(zip(parts, updated)):
            expected = q_global[p.local_vertices]
            np.testing.assert_allclose(
                f, expected, atol=1e-12, err_msg=f"rank {r}",
            )

    def test_comm_schedule_symmetry(self, mesh):
        """Send count to rank R must match R's recv count from us."""
        n_ranks = 3
        parts = [
            partition_voronoi_mesh(mesh, n_ranks, r) for r in range(n_ranks)
        ]

        for r in range(n_ranks):
            comm = parts[r].cell_comm
            for i, nbr in enumerate(comm.neighbor_ranks):
                our_send = comm.send_counts[i]
                nbr_comm = parts[nbr].cell_comm
                if r in nbr_comm.neighbor_ranks:
                    j = nbr_comm.neighbor_ranks.index(r)
                    nbr_recv = nbr_comm.recv_counts[j]
                    assert our_send == nbr_recv, (
                        f"rank {r} sends {our_send} cells to {nbr}, "
                        f"but {nbr} expects {nbr_recv} from {r}"
                    )

    def test_multidim_field_exchange(self, mesh):
        """Halo exchange works with multi-dimensional fields (e.g. nlev)."""
        nlev = 5
        phi_global = jnp.ones((mesh.nCells, nlev), dtype=jnp.float64)
        phi_global = phi_global * jnp.arange(mesh.nCells, dtype=jnp.float64)[:, None]

        n_ranks = 2
        parts = [
            partition_voronoi_mesh(mesh, n_ranks, r) for r in range(n_ranks)
        ]

        local_fields = []
        for p in parts:
            f = jnp.zeros((p.n_local_cells, nlev), dtype=jnp.float64)
            f = f.at[:p.n_owned_cells].set(
                phi_global[p.local_cells[:p.n_owned_cells]],
            )
            local_fields.append(f)

        updated = exchange_local_simulated(parts, local_fields, "cell")

        for r, (p, f) in enumerate(zip(parts, updated)):
            expected = phi_global[p.local_cells]
            np.testing.assert_allclose(
                f, expected, atol=1e-12, err_msg=f"rank {r}",
            )


# ========================================================================
# ppermute schedule (for multi-GPU halo exchange)
# ========================================================================

class TestPpermuteSchedule:
    """Tests for the ppermute-based halo exchange schedule builder."""

    @pytest.mark.parametrize("n_ranks", [2, 3, 4])
    def test_schedule_covers_all_halo_cells(self, mesh, n_ranks):
        """Every halo cell is covered by exactly one ppermute round."""
        from legoesm.parallel.voronoi_partition import (
            reorder_voronoi_for_sharding,
        )
        from legoesm.parallel.sharded_dynamics import (
            _build_voronoi_partition_infra,
            _build_ppermute_schedule,
        )

        reordered = reorder_voronoi_for_sharding(mesh, n_ranks)
        nCells = reordered.nCells
        nEdges = reordered.nEdges
        cells_per = nCells // n_ranks
        edges_per = nEdges // n_ranks

        (_, _, _, _, _, max_lc, max_le, partitions, cell_owner,
         ) = _build_voronoi_partition_infra(reordered, n_ranks, halo_depth=2)

        sched = _build_ppermute_schedule(
            partitions, cell_owner, n_ranks,
            cells_per, edges_per, max_lc, max_le,
        )

        # For each device, collect all halo cell positions reached
        for d in range(n_ranks):
            part = partitions[d]
            expected_halo_positions = set(range(
                part.n_owned_cells, part.n_local_cells))

            reached: set[int] = set()
            for r in range(sched['n_rounds']):
                recv_pos = np.asarray(sched['recv_cell_pos'][r])[d]
                for pos in recv_pos:
                    pos = int(pos)
                    if pos < max_lc:  # skip garbage slot
                        reached.add(pos)

            assert expected_halo_positions <= reached, (
                f"device {d}: halo positions "
                f"{expected_halo_positions - reached} "
                f"not covered by ppermute schedule"
            )

    @pytest.mark.parametrize("n_ranks", [2, 3, 4])
    def test_ppermute_halo_matches_simulated_exchange(self, mesh, n_ranks):
        """ppermute schedule produces same halo values as simulated
        exchange."""
        from legoesm.parallel.voronoi_partition import (
            reorder_voronoi_for_sharding,
        )
        from legoesm.parallel.sharded_dynamics import (
            _build_voronoi_partition_infra,
            _build_ppermute_schedule,
        )

        reordered = reorder_voronoi_for_sharding(mesh, n_ranks)
        nCells = reordered.nCells
        cells_per = nCells // n_ranks
        edges_per = reordered.nEdges // n_ranks

        (_, _, _, _, _, max_lc, max_le, partitions, cell_owner,
         ) = _build_voronoi_partition_infra(reordered, n_ranks, halo_depth=2)

        sched = _build_ppermute_schedule(
            partitions, cell_owner, n_ranks,
            cells_per, edges_per, max_lc, max_le,
        )

        # Use global index as value for easy verification
        phi_global = jnp.arange(nCells, dtype=jnp.float64)

        for d in range(n_ranks):
            part = partitions[d]
            owned_data = phi_global[d * cells_per:(d + 1) * cells_per]

            local_arr = jnp.zeros(max_lc + 1, dtype=jnp.float64)
            local_arr = local_arr.at[:cells_per].set(owned_data)

            # Simulate ppermute rounds
            for r in range(sched['n_rounds']):
                perm = sched['ppermute_perms'][r]
                src_dev = None
                for s, dst in perm:
                    if dst == d:
                        src_dev = s
                        break
                if src_dev is None:
                    continue

                src_owned = phi_global[
                    src_dev * cells_per:(src_dev + 1) * cells_per]
                send_idx = np.asarray(
                    sched['send_cell_idx'][r])[src_dev]
                sent_data = src_owned[send_idx]

                recv_pos = np.asarray(
                    sched['recv_cell_pos'][r])[d]
                local_arr = local_arr.at[recv_pos].set(sent_data)

            local_arr = local_arr[:max_lc]

            # Check halo matches global
            for h in range(part.n_owned_cells, part.n_local_cells):
                g = int(part.local_cells[h])
                expected = float(phi_global[g])
                actual = float(local_arr[h])
                assert abs(actual - expected) < 1e-12, (
                    f"device {d}, halo pos {h} (global cell {g}): "
                    f"expected {expected}, got {actual}"
                )

    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_edge_coloring_valid(self, mesh, n_ranks):
        """Each ppermute round has no device appearing as sender twice."""
        from legoesm.parallel.voronoi_partition import (
            reorder_voronoi_for_sharding,
        )
        from legoesm.parallel.sharded_dynamics import (
            _build_voronoi_partition_infra,
            _build_ppermute_schedule,
        )

        reordered = reorder_voronoi_for_sharding(mesh, n_ranks)
        cells_per = reordered.nCells // n_ranks
        edges_per = reordered.nEdges // n_ranks

        (_, _, _, _, _, max_lc, max_le, partitions, cell_owner,
         ) = _build_voronoi_partition_infra(reordered, n_ranks, halo_depth=2)

        sched = _build_ppermute_schedule(
            partitions, cell_owner, n_ranks,
            cells_per, edges_per, max_lc, max_le,
        )

        for r in range(sched['n_rounds']):
            senders = [s for s, _ in sched['ppermute_perms'][r]]
            assert len(senders) == len(set(senders)), (
                f"Round {r}: duplicate senders in ppermute perm"
            )
