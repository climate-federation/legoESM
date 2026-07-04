"""Unit tests for the parallelism infrastructure.

Tests device mesh creation, array sharding, communication topology,
Metal config, and hardware detection.  All tests run on CPU without
requiring multiple devices or MPI.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from unittest.mock import patch
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from legoesm.parallel.mesh import (
    DeviceConfig,
    create_device_mesh,
    get_active_config,
    shard_pytree,
    replicate_pytree,
    N_FACES,
    _best_tile_factorization,
)
from legoesm.parallel.comm import (
    CommTopology,
    build_comm_topology,
    _face_to_rank,
    _rank_to_faces,
    _tile_rank,
    _rank_to_tile,
)
from legoesm.parallel.reductions import (
    require_mpi_stack,
    _validate_mpi_runtime_versions,
)
from legoesm.parallel.metal import get_metal_config, to_cpu
from legoesm.core.hardware import detect_devices, get_backend
from legoesm.core.field import Field
from legoesm.grids.halo import (
    pad_halo,
    set_halo_backend,
    get_halo_backend,
    pad_halo_local,
)


# ==============================================================================
# Phase 1: Device mesh and sharding
# ==============================================================================

class TestDeviceMesh:
    """Tests for create_device_mesh and DeviceConfig."""

    def test_create_device_mesh_single_cpu(self):
        """Single-device (CPU) returns a valid DeviceConfig."""
        config = create_device_mesh(n_devices=1)
        assert isinstance(config, DeviceConfig)
        assert config.n_devices == 1
        assert config.mesh is None
        assert config.face_sharding is None
        assert config.replicated_sharding is None
        assert config.is_distributed is False
        assert isinstance(config.backend, str)

    def test_create_device_mesh_auto(self):
        """auto mode returns a valid config."""
        config = create_device_mesh(n_devices="auto")
        assert isinstance(config, DeviceConfig)
        assert config.n_devices >= 1
        assert config.n_devices <= N_FACES

    def test_create_device_mesh_explicit_devices(self):
        """Explicit device list is accepted and used."""
        config = create_device_mesh(n_devices=1, devices=jax.local_devices())
        assert isinstance(config, DeviceConfig)
        assert config.n_devices == 1

    def test_create_device_mesh_empty_explicit_devices(self):
        """Explicit empty device list raises a clear ValueError."""
        with pytest.raises(ValueError, match="devices must contain at least one"):
            create_device_mesh(n_devices=1, devices=[])

    def test_invalid_n_devices(self):
        """n_devices=0 raises ValueError; 4 and 5 auto-adjust."""
        with pytest.raises(ValueError, match="must be >= 1"):
            create_device_mesh(n_devices=0)

        # 4 and 5 auto-adjust to usable counts (no longer raise)
        config4 = create_device_mesh(n_devices=4)
        assert config4.n_devices in (1, 2, 3)  # rounded down to valid count

        config5 = create_device_mesh(n_devices=5)
        assert config5.n_devices in (1, 2, 3)  # rounded down to valid count

    def test_active_config_set(self):
        """create_device_mesh sets the active config singleton."""
        config = create_device_mesh(n_devices=1)
        assert get_active_config() is config

    def test_auto_falls_back_from_invalid_count(self, caplog):
        """Auto mode with 4 devices warns and rounds to a valid count.

        The pre-iter implementation raised ``ValueError`` here.  The
        current ``create_device_mesh`` (since the iter-N hardening of
        the device-mesh constructor) instead emits a warning and rounds
        the invalid count down to the nearest valid cubed-sphere count
        (3 in this case — divisors of 6 in [1, 4]).  The test now
        matches the actual behaviour: a warning is logged and the
        returned config uses 3 devices.
        """
        fake_devices = [object(), object(), object(), object()]
        with patch("legoesm.parallel.mesh.jax.devices", return_value=fake_devices):
            with patch("legoesm.parallel.mesh.jax.default_backend", return_value="gpu"):
                config = create_device_mesh(n_devices="auto")
        assert config.n_devices == 3
        assert "must divide 6" in caplog.text or "Using 3 device" in caplog.text

    def test_distributed_mode_uses_local_devices(self):
        """When process_count>1 and no backend override, use local devices."""
        local = jax.local_devices()
        with patch("legoesm.parallel.mesh.jax.process_count", return_value=2):
            with patch("legoesm.parallel.mesh.jax.local_devices", return_value=local):
                with patch(
                    "legoesm.parallel.mesh.jax.devices",
                    side_effect=AssertionError("jax.devices should not be used"),
                ):
                    config = create_device_mesh(n_devices=1)
        assert isinstance(config, DeviceConfig)
        assert config.n_devices == 1

    def test_subface_mesh_uses_tile_i_tile_j_axes(self):
        """Sub-face decomposition must shard both horizontal tile directions."""
        dev = jax.devices()[0]
        fake_devices = [dev] * 24  # 6 faces × 2×2 tiles
        with patch("legoesm.parallel.mesh.jax.devices", return_value=fake_devices):
            config = create_device_mesh(n_devices=24)
        assert config.tiling == (2, 2)
        assert config.mesh is not None
        assert config.mesh.axis_names == ("face", "tile_i", "tile_j")
        assert config.face_sharding.spec == P("face", "tile_i", "tile_j")


class TestShardPytree:
    """Tests for shard_pytree and replicate_pytree."""

    def test_shard_noop_single_device(self):
        """On single device, shard_pytree is a no-op."""
        config = create_device_mesh(n_devices=1)
        data = jnp.ones((6, 8, 8))
        result = shard_pytree(data, config)
        assert jnp.array_equal(result, data)

    def test_shard_pytree_preserves_structure(self):
        """shard_pytree preserves the pytree structure."""
        config = create_device_mesh(n_devices=1)
        pytree = {
            "a": jnp.ones((6, 4, 4)),
            "b": jnp.zeros((6, 4, 4, 5)),
            "c": jnp.array(42.0),  # scalar
        }
        result = shard_pytree(pytree, config)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"a", "b", "c"}
        assert result["a"].shape == (6, 4, 4)
        assert result["b"].shape == (6, 4, 4, 5)
        assert result["c"].shape == ()

    def test_shard_field_pytree(self):
        """shard_pytree handles Field objects via pytree."""
        config = create_device_mesh(n_devices=1)
        field = Field(
            data=jnp.ones((6, 8, 8)),
            name="test",
            dims=("face", "x", "y"),
        )
        result = shard_pytree(field, config)
        assert isinstance(result, Field)
        assert result.shape == (6, 8, 8)

    def test_replicate_noop_single_device(self):
        """On single device, replicate_pytree is a no-op."""
        config = create_device_mesh(n_devices=1)
        data = jnp.ones((6, 4, 4))
        result = replicate_pytree(data, config)
        assert jnp.array_equal(result, data)

    def test_shard_tiled_uses_both_horizontal_axes(self):
        """Tiled cubed-sphere mode shards (face, x, y) over (face, tile_i, tile_j)."""
        dev = jax.devices()[0]
        mesh = Mesh(np.array([[[dev]]]), axis_names=("face", "tile_i", "tile_j"))
        config = DeviceConfig(
            mesh=mesh,
            face_sharding=NamedSharding(mesh, P("face", "tile_i", "tile_j")),
            replicated_sharding=NamedSharding(mesh, P()),
            n_devices=1,
            backend="CPU",
            is_distributed=False,
            tiling=(2, 2),
            grid_type="cubed_sphere",
        )
        data = jnp.ones((6, 8, 8), dtype=jnp.float32)
        sharded = shard_pytree(data, config)
        assert sharded.sharding.spec == P("face", "tile_i", "tile_j")

    def test_shard_tiled_face_vector_uses_face_only(self):
        """Lower-rank face-leading arrays should avoid tile-axis mis-sharding."""
        dev = jax.devices()[0]
        mesh = Mesh(np.array([[[dev]]]), axis_names=("face", "tile_i", "tile_j"))
        config = DeviceConfig(
            mesh=mesh,
            face_sharding=NamedSharding(mesh, P("face", "tile_i", "tile_j")),
            replicated_sharding=NamedSharding(mesh, P()),
            n_devices=1,
            backend="CPU",
            is_distributed=False,
            tiling=(2, 2),
            grid_type="cubed_sphere",
        )
        data = jnp.ones((6, 10), dtype=jnp.float32)
        sharded = shard_pytree(data, config)
        assert sharded.sharding.spec == P("face")


# ==============================================================================
# Phase 2: Communication topology
# ==============================================================================

class TestCommTopology:
    """Tests for CommTopology and build_comm_topology."""

    def test_single_process(self):
        """1 process owns all 6 faces."""
        topo = build_comm_topology(rank=0, n_processes=1)
        assert isinstance(topo, CommTopology)
        assert topo.rank == 0
        assert topo.n_processes == 1
        assert topo.local_face_ids == (0, 1, 2, 3, 4, 5)
        # All neighbors are local.
        for face in range(6):
            for edge in range(4):
                assert topo.neighbor_ranks.get((face, edge), 0) == 0

    def test_six_processes(self):
        """6 processes: 1 face per process."""
        for rank in range(6):
            topo = build_comm_topology(rank=rank, n_processes=6)
            assert topo.local_face_ids == (rank,)
            assert topo.rank == rank
            assert topo.n_processes == 6

    def test_three_processes(self):
        """3 processes: 2 faces per process."""
        for rank in range(3):
            topo = build_comm_topology(rank=rank, n_processes=3)
            assert len(topo.local_face_ids) == 2
            assert topo.local_face_ids == (rank * 2, rank * 2 + 1)

    def test_two_processes(self):
        """2 processes: 3 faces per process."""
        for rank in range(2):
            topo = build_comm_topology(rank=rank, n_processes=2)
            assert len(topo.local_face_ids) == 3

    def test_invalid_n_processes(self):
        """n_processes not dividing 6 raises ValueError."""
        with pytest.raises(ValueError, match="does not evenly divide"):
            build_comm_topology(rank=0, n_processes=4)

    def test_face_to_rank_mapping(self):
        """Face-to-rank assignment is contiguous."""
        # 6 processes.
        for face in range(6):
            assert _face_to_rank(face, 6) == face

        # 3 processes.
        assert _face_to_rank(0, 3) == 0
        assert _face_to_rank(1, 3) == 0
        assert _face_to_rank(2, 3) == 1
        assert _face_to_rank(3, 3) == 1
        assert _face_to_rank(4, 3) == 2
        assert _face_to_rank(5, 3) == 2

        # 2 processes.
        for face in range(3):
            assert _face_to_rank(face, 2) == 0
        for face in range(3, 6):
            assert _face_to_rank(face, 2) == 1

    def test_rank_to_faces(self):
        """rank_to_faces is inverse of face_to_rank."""
        assert _rank_to_faces(0, 6) == (0,)
        assert _rank_to_faces(5, 6) == (5,)
        assert _rank_to_faces(0, 3) == (0, 1)
        assert _rank_to_faces(2, 3) == (4, 5)
        assert _rank_to_faces(0, 2) == (0, 1, 2)
        assert _rank_to_faces(1, 2) == (3, 4, 5)

    def test_neighbor_info_symmetric(self):
        """If A→B at edge e, B→A at the matching edge."""
        topo = build_comm_topology(rank=0, n_processes=1)
        for (face, edge), (nbr_face, nbr_edge, _rev) in topo.neighbor_info.items():
            # Check that the reverse mapping exists.
            reverse = topo.neighbor_info.get((nbr_face, nbr_edge))
            assert reverse is not None, (
                f"Missing reverse for face={face}, edge={edge}"
            )
            assert reverse[0] == face, (
                f"Reverse of ({face},{edge})→({nbr_face},{nbr_edge}) "
                f"points to face {reverse[0]}, not {face}"
            )

    def test_remote_edge_order_for_packed_exchange(self):
        """Sender/receiver can agree on packed strip order for 2/3/6 ranks."""
        for n_processes in (2, 3, 6):
            topologies = [
                build_comm_topology(rank=r, n_processes=n_processes)
                for r in range(n_processes)
            ]

            for rank_a, topo_a in enumerate(topologies):
                for rank_b in range(n_processes):
                    if rank_a == rank_b:
                        continue

                    edges_a = [
                        (face, edge, nbr_face, nbr_edge, is_reversed)
                        for (face, edge), (nbr_face, nbr_edge, is_reversed) in topo_a.neighbor_info.items()
                        if topo_a.neighbor_ranks[(face, edge)] == rank_b
                    ]
                    if not edges_a:
                        continue

                    topo_b = topologies[rank_b]
                    edges_b = [
                        (face, edge, nbr_face, nbr_edge, is_reversed)
                        for (face, edge), (nbr_face, nbr_edge, is_reversed) in topo_b.neighbor_info.items()
                        if topo_b.neighbor_ranks[(face, edge)] == rank_a
                    ]

                    send_order_a = sorted(edges_a, key=lambda e: (e[0], e[1]))
                    recv_order_b = sorted(edges_b, key=lambda e: (e[2], e[3]))
                    sent_pairs = [(face, edge) for face, edge, *_ in send_order_a]
                    expected_pairs = [(nbr_face, nbr_edge) for _, _, nbr_face, nbr_edge, _ in recv_order_b]
                    assert sent_pairs == expected_pairs


class TestSubFaceTiling:
    """Tests for sub-face tiling decomposition."""

    def test_square_tile_factorization(self):
        """_best_tile_factorization always produces square tiles for >6 devices."""
        # Perfect squares → square tiles
        assert _best_tile_factorization(24) == (6, 2, 2)   # 6 * 2²
        assert _best_tile_factorization(54) == (6, 3, 3)   # 6 * 3²
        assert _best_tile_factorization(96) == (6, 4, 4)   # 6 * 4²
        assert _best_tile_factorization(150) == (6, 5, 5)  # 6 * 5²
        assert _best_tile_factorization(384) == (6, 8, 8)  # 6 * 8²
        assert _best_tile_factorization(600) == (6, 10, 10)  # 6 * 10²

    def test_non_square_raises(self):
        """Non-perfect-square tiles_per_face now raises ValueError."""
        # 12 devices: tiles_per_face=2, not a perfect square
        with pytest.raises(ValueError, match="perfect square"):
            _best_tile_factorization(12)

        # 48 devices: tiles_per_face=8, not a perfect square
        with pytest.raises(ValueError, match="perfect square"):
            _best_tile_factorization(48)

    def test_tile_rank_mapping(self):
        """_tile_rank and _rank_to_tile are inverses."""
        tx, ty = 3, 3
        for face in range(6):
            for ti in range(tx):
                for tj in range(ty):
                    rank = _tile_rank(face, ti, tj, tx, ty)
                    f, i, j = _rank_to_tile(rank, tx, ty)
                    assert f == face
                    assert i == ti
                    assert j == tj

    def test_tiled_topology_24_ranks(self):
        """24-rank topology: 6 faces × 2×2 tiles, each rank owns one tile."""
        for rank in range(24):
            topo = build_comm_topology(rank, 24)
            assert len(topo.local_face_ids) == 1
            assert topo.tiling == (2, 2)
            # 4 tile neighbors: some same-face, some None (face boundary)
            assert set(topo.tile_neighbors.keys()) == {"west", "east", "south", "north"}

    def test_tiled_topology_tile_neighbors(self):
        """Interior tiles have all 4 same-face neighbors; corner tiles have 2."""
        # 54 ranks: 6 faces × 3×3 tiles
        # Center tile (1,1) on face 0: rank = 0*9 + 1*3 + 1 = 4
        topo = build_comm_topology(rank=4, n_processes=54)
        assert topo.tile_index == (1, 1)
        # Interior tile has all 4 intra-face neighbors
        for d in ("west", "east", "south", "north"):
            assert topo.tile_neighbors[d] is not None

        # Corner tile (0,0) on face 0: rank = 0
        topo = build_comm_topology(rank=0, n_processes=54)
        assert topo.tile_index == (0, 0)
        # Corner tile has 2 face-boundary edges (west, south)
        assert topo.tile_neighbors["west"] is None
        assert topo.tile_neighbors["south"] is None
        # And 2 intra-face neighbors (east, north)
        assert topo.tile_neighbors["east"] is not None
        assert topo.tile_neighbors["north"] is not None

    def test_non_square_tiles_rejected(self):
        """Process counts that yield non-square tiles_per_face raise ValueError."""
        with pytest.raises(ValueError, match="not a perfect square"):
            build_comm_topology(rank=0, n_processes=12)

    def test_tiled_topology_inter_face_neighbors(self):
        """Boundary tiles have inter-face neighbor info."""
        # 24 ranks: 2×2 tiles. Tile (0,0) on face 0 has west and south on face boundary
        topo = build_comm_topology(rank=0, n_processes=24)
        face = topo.local_face_ids[0]
        assert face == 0
        # West boundary → inter-face neighbor (face 3)
        from legoesm.grids.halo import WEST, SOUTH
        assert (face, WEST) in topo.neighbor_info
        assert (face, SOUTH) in topo.neighbor_info


class TestMPIDependencyGuards:
    """Tests for MPI dependency validation and fail-fast behavior."""

    def test_validate_versions_rejects_legacy_mpi4jax(self):
        """mpi4jax<0.8 should fail fast due incompatible token semantics."""
        with pytest.raises(RuntimeError, match="requires mpi4jax"):
            _validate_mpi_runtime_versions("0.9.0", "0.7.5")

    def test_validate_versions_warns_outside_tested_range(self):
        """Out-of-range versions should emit a clear runtime warning."""
        with pytest.warns(RuntimeWarning, match="outside legoESM's tested MPI range"):
            _validate_mpi_runtime_versions("0.10.0", "0.8.1")

    def test_validate_versions_strict_mode_raises(self):
        """Strict mode should convert compatibility warnings to errors."""
        with pytest.raises(RuntimeError, match="outside legoESM's tested MPI range"):
            _validate_mpi_runtime_versions("0.10.0", "0.8.1", strict=True)

    def test_require_mpi_stack_reports_missing_modules(self):
        """Missing mpi4jax/mpi4py should raise a clear ImportError."""
        def _fake_find_spec(name):
            if name in {"mpi4jax", "mpi4py"}:
                return None
            return object()

        with patch("legoesm.parallel.reductions.importlib.util.find_spec", side_effect=_fake_find_spec):
            with pytest.raises(ImportError, match="mpi4jax, mpi4py"):
                require_mpi_stack()

    def test_initialize_distributed_checks_mpi_before_jax_init(self):
        """initialize_distributed should fail before jax.distributed.initialize when MPI deps are absent."""
        import legoesm.parallel.distributed as distributed_mod

        distributed_mod._active_topology = None
        with patch(
            "legoesm.parallel.distributed.require_mpi_stack",
            side_effect=ImportError("missing mpi stack"),
        ):
            with patch(
                "legoesm.parallel.distributed.jax.distributed.initialize",
                side_effect=AssertionError("jax.distributed.initialize should not run"),
            ):
                with pytest.raises(ImportError, match="missing mpi stack"):
                    distributed_mod.initialize_distributed()


# ==============================================================================
# Halo backend dispatch
# ==============================================================================

class TestHaloDispatch:
    """Tests for the halo backend dispatch mechanism."""

    def test_default_backend_is_local(self):
        """Default halo backend is 'local'."""
        # Reset to default.
        set_halo_backend("local")
        assert get_halo_backend() == "local"

    def test_set_invalid_backend(self):
        """Unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown halo backend"):
            set_halo_backend("invalid")

    def test_mpi_requires_topology(self):
        """MPI backend without topology raises ValueError."""
        # The message was reworded to cover both cubed-sphere
        # (CommTopology) and lat-lon (LatLonBandLayout) topologies.
        with pytest.raises(ValueError, match="requires a topology"):
            set_halo_backend("mpi")

    def test_pad_halo_local_matches(self):
        """pad_halo with local backend matches pad_halo_local."""
        set_halo_backend("local")
        data = jnp.ones((6, 8, 8), dtype=jnp.float32)
        result = pad_halo(data)
        expected = pad_halo_local(data)
        assert jnp.allclose(result, expected)

    def teardown_method(self):
        """Reset halo backend after each test."""
        set_halo_backend("local")


# ==============================================================================
# Hardware detection
# ==============================================================================

class TestHardwareDetection:
    """Tests for detect_devices."""

    def test_detect_devices_returns_dict(self):
        """detect_devices returns expected keys."""
        info = detect_devices()
        assert isinstance(info, dict)
        assert "backend" in info
        assert "n_devices" in info
        assert "devices" in info
        assert "supports_f64" in info
        assert "distributed" in info

    def test_backend_is_string(self):
        info = detect_devices()
        assert isinstance(info["backend"], str)
        assert info["backend"] == get_backend()

    def test_n_devices_positive(self):
        info = detect_devices()
        assert info["n_devices"] >= 1

    def test_supports_f64_on_cpu(self):
        """CPU always supports float64."""
        info = detect_devices()
        if info["backend"] == "CPU":
            assert info["supports_f64"] is True


# ==============================================================================
# Metal config
# ==============================================================================

class TestMetalConfig:
    """Tests for Metal detection."""

    def test_metal_config_on_cpu(self):
        """On non-mps backends, is_metal=False."""
        config = get_metal_config()
        if get_backend() != "MPS":
            assert config.is_metal is False
            assert config.metal_device is None

    def test_cpu_device_always_available(self):
        """CPU device is always populated."""
        config = get_metal_config()
        assert config.cpu_device is not None

    def test_to_cpu(self):
        """to_cpu transfers array to CPU."""
        arr = jnp.array([1.0, 2.0, 3.0])
        result = to_cpu(arr)
        assert jnp.allclose(result, arr)


# ==============================================================================
# Integration: sharded computation
# ==============================================================================

class TestShardedComputation:
    """Verify that computations work identically with sharded arrays."""

    def test_pad_halo_sharded_matches_local(self):
        """pad_halo with sharded input matches unsharded."""
        config = create_device_mesh(n_devices=1)
        data = jnp.ones((6, 8, 8), dtype=jnp.float32)
        # Modify some faces so halos are nontrivial.
        data = data.at[0].set(2.0)
        data = data.at[3].set(0.5)

        sharded = shard_pytree(data, config)
        result_sharded = pad_halo(sharded)
        result_local = pad_halo(data)

        assert jnp.allclose(result_sharded, result_local)


class TestMultiprocessSafeDevicePut:
    """multiprocess_safe_device_put (#693): identical to device_put single-
    process; routes through make_array_from_callback under >1 process so the
    cross-process bit-equality assert in device_put cannot trip on host-
    precomputed forcing (Levante 2-node cs_spmd receipt, job 26030677)."""

    def _sharding(self):
        from legoesm.parallel.mesh import multiprocess_safe_device_put  # noqa: F401
        dev = jax.devices()[0]
        mesh = Mesh(np.array([dev]), ("face",))
        return NamedSharding(mesh, P())

    def test_single_process_matches_device_put(self):
        from legoesm.parallel.mesh import multiprocess_safe_device_put
        x = jnp.arange(12.0).reshape(6, 2)
        s = self._sharding()
        out = multiprocess_safe_device_put(x, s)
        ref = jax.device_put(x, s)
        assert np.array_equal(np.asarray(out), np.asarray(ref))
        assert out.sharding == ref.sharding

    def test_non_array_delegates(self):
        from legoesm.parallel.mesh import multiprocess_safe_device_put
        out = multiprocess_safe_device_put(3.5, self._sharding())
        assert float(out) == 3.5

    def test_multiprocess_branch_uses_callback_and_preserves_values(self):
        from legoesm.parallel.mesh import multiprocess_safe_device_put
        x = jnp.arange(24.0).reshape(6, 4)
        s = self._sharding()
        with patch("jax.process_count", return_value=2):
            out = multiprocess_safe_device_put(x, s)
        # values intact and placed on the requested sharding, with NO
        # cross-process equality assertion involved
        assert np.array_equal(np.asarray(out), np.asarray(x))
        assert out.sharding == s
