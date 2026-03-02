"""Unit tests for the parallelism infrastructure.

Tests device mesh creation, array sharding, communication topology,
Metal config, and hardware detection.  All tests run on CPU without
requiring multiple devices or MPI.
"""

import jax
import jax.numpy as jnp
import pytest
from unittest.mock import patch

from legoesm.parallel.mesh import (
    DeviceConfig,
    create_device_mesh,
    get_active_config,
    shard_pytree,
    replicate_pytree,
    _N_FACES,
)
from legoesm.parallel.comm import (
    CommTopology,
    build_comm_topology,
    _face_to_rank,
    _rank_to_faces,
)
from legoesm.parallel.metal import get_metal_config, to_cpu
from legoesm.core.hardware import detect_devices, get_backend
from legoesm.core.field import Field
from legoesm.grids.halo import (
    pad_halo,
    set_halo_backend,
    get_halo_backend,
    _pad_halo_local,
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
        assert config.n_devices <= _N_FACES

    def test_invalid_n_devices(self):
        """n_devices that doesn't divide 6 raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            create_device_mesh(n_devices=0)

        with pytest.raises(ValueError, match="does not evenly divide"):
            create_device_mesh(n_devices=4)

        with pytest.raises(ValueError, match="does not evenly divide"):
            create_device_mesh(n_devices=5)

    def test_active_config_set(self):
        """create_device_mesh sets the active config singleton."""
        config = create_device_mesh(n_devices=1)
        assert get_active_config() is config

    def test_auto_falls_back_from_invalid_count(self):
        """Auto mode should fall back to a valid face-partition count."""
        fake_devices = [object(), object(), object(), object()]
        with patch("legoesm.parallel.mesh.jax.devices", return_value=fake_devices):
            with patch("legoesm.parallel.mesh.jax.default_backend", return_value="gpu"):
                with pytest.warns(RuntimeWarning, match="Falling back to 3 device"):
                    config = create_device_mesh(n_devices="auto")
        assert config.n_devices == 3


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
        with pytest.raises(ValueError, match="CommTopology is required"):
            set_halo_backend("mpi")

    def test_pad_halo_local_matches(self):
        """pad_halo with local backend matches _pad_halo_local."""
        set_halo_backend("local")
        data = jnp.ones((6, 8, 8), dtype=jnp.float32)
        result = pad_halo(data)
        expected = _pad_halo_local(data)
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
        """On non-Metal backends, is_metal=False."""
        config = get_metal_config()
        if get_backend() != "METAL":
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
