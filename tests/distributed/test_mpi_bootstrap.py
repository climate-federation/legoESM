"""MPI bootstrap integration tests.

Run with:
    mpirun -np 2 .venv/bin/python -m pytest tests/distributed/test_mpi_bootstrap.py -v
    mpirun -np 6 .venv/bin/python -m pytest tests/distributed/test_mpi_bootstrap.py -v

These tests verify that the MPI bootstrap path correctly:
- Builds a valid CommTopology for each rank from MPI metadata
- Sets the halo backend to "mpi"
- Creates a DeviceConfig with is_distributed=True
- Assigns faces so that every face is covered and no face overlaps
- Performs a working MPI halo exchange

Note: we replicate the logic of ``initialize_distributed`` without
calling ``jax.distributed.initialize()`` (which requires a gRPC
coordinator and hangs on macOS single-host oversubscribed MPI).
The individual components are the same ones ``initialize_distributed``
uses internally.
"""

import jax
import jax.numpy as jnp
import pytest

# Guard: skip all tests if mpi4jax/mpi4py are not installed.
mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.parallel.comm import CommTopology, build_comm_topology
from legoesm.parallel.mesh import (
    DeviceConfig,
    create_device_mesh,
    set_active_config,
    get_active_config,
)
from legoesm.grids.halo import get_halo_backend, set_halo_backend


# ---------------------------------------------------------------------------
# Module-level MPI bootstrap (runs once per rank, shared by all tests)
# ---------------------------------------------------------------------------
# Replicate the key steps of initialize_distributed() without the gRPC
# coordinator (jax.distributed.initialize) which hangs on macOS.

_comm = MPI.COMM_WORLD
_rank = _comm.Get_rank()
_size = _comm.Get_size()

# 1. Build communication topology.
_topology = build_comm_topology(_rank, _size)

# 2. Set MPI halo backend.
set_halo_backend("mpi", _topology)

# 3. Create device config with distributed flag.
_local_config = create_device_mesh(n_devices=1)
_config = DeviceConfig(
    mesh=_local_config.mesh,
    face_sharding=_local_config.face_sharding,
    replicated_sharding=_local_config.replicated_sharding,
    n_devices=_local_config.n_devices,
    backend=_local_config.backend,
    is_distributed=True,
    tiling=_topology.tiling,
    grid_type="cubed_sphere",
)
set_active_config(_config)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def restore_halo_backend():
    """Ensure halo backend stays 'mpi' across tests."""
    yield
    set_halo_backend("mpi", _topology)


@pytest.fixture
def mpi_info():
    """Return (rank, size) from MPI."""
    return _rank, _size


@pytest.fixture
def topology():
    """Return the CommTopology for this rank."""
    return _topology


# ---------------------------------------------------------------------------
# Tests: DeviceConfig
# ---------------------------------------------------------------------------

class TestDeviceConfig:
    """Verify the DeviceConfig created during MPI bootstrap."""

    def test_config_is_device_config(self):
        """Bootstrap should produce a DeviceConfig instance."""
        assert isinstance(_config, DeviceConfig)

    def test_is_distributed_true(self):
        """DeviceConfig should have is_distributed=True."""
        config = get_active_config()
        assert config is not None
        assert config.is_distributed is True

    def test_grid_type_cubed_sphere(self):
        """Distributed mode should report cubed_sphere grid type."""
        config = get_active_config()
        assert config.grid_type == "cubed_sphere"

    def test_backend_valid(self):
        """Backend should be a known accelerator string."""
        config = get_active_config()
        assert config.backend in ("CPU", "GPU", "TPU", "METAL")

    def test_n_devices_positive(self):
        """At least one device should be available."""
        config = get_active_config()
        assert config.n_devices >= 1

    def test_tiling_valid(self):
        """Tiling should be (1, 1) for face-only decomposition."""
        config = get_active_config()
        assert config.tiling[0] >= 1
        assert config.tiling[1] >= 1


# ---------------------------------------------------------------------------
# Tests: CommTopology
# ---------------------------------------------------------------------------

class TestCommTopology:
    """Verify the CommTopology built from MPI metadata."""

    def test_topology_rank_matches_mpi(self, mpi_info):
        """Topology rank should match MPI rank."""
        rank, _ = mpi_info
        assert _topology.rank == rank

    def test_topology_size_matches_mpi(self, mpi_info):
        """Topology n_processes should match MPI size."""
        _, size = mpi_info
        assert _topology.n_processes == size

    def test_local_face_ids_nonempty(self):
        """Each rank should own at least one face."""
        assert len(_topology.local_face_ids) > 0

    def test_local_face_ids_valid_range(self):
        """All local face IDs should be in [0, 5]."""
        for fid in _topology.local_face_ids:
            assert 0 <= fid <= 5

    def test_all_faces_covered(self):
        """Union of all ranks' faces should cover all 6 cubed-sphere faces."""
        local_faces = list(_topology.local_face_ids)
        all_faces = _comm.allgather(local_faces)
        union = set()
        for faces in all_faces:
            union.update(faces)
        assert union == {0, 1, 2, 3, 4, 5}

    def test_no_face_overlap(self):
        """No face should be assigned to more than one rank."""
        local_faces = list(_topology.local_face_ids)
        all_faces = _comm.allgather(local_faces)
        total = sum(len(fl) for fl in all_faces)
        assert total == 6, f"Face overlap detected: {all_faces}"

    def test_faces_per_rank_balanced(self, mpi_info):
        """Each rank should own 6 / n_processes faces."""
        _, size = mpi_info
        expected = 6 // size
        assert len(_topology.local_face_ids) == expected

    def test_neighbor_ranks_valid(self):
        """Neighbor ranks should be in [0, n_processes)."""
        for (face, edge), nbr_rank in _topology.neighbor_ranks.items():
            assert 0 <= nbr_rank < _topology.n_processes
            assert face in _topology.local_face_ids

    def test_tiling_matches_config(self):
        """Topology tiling should match DeviceConfig tiling."""
        assert _topology.tiling == _config.tiling


# ---------------------------------------------------------------------------
# Tests: Halo backend
# ---------------------------------------------------------------------------

class TestHaloBackend:
    """Verify the halo exchange backend is correctly configured."""

    def test_halo_backend_is_mpi(self):
        """After bootstrap, halo backend should be 'mpi'."""
        assert get_halo_backend() == "mpi"

    def test_set_local_and_restore(self):
        """Setting halo to 'local' and back to 'mpi' should work."""
        set_halo_backend("local")
        assert get_halo_backend() == "local"
        set_halo_backend("mpi", _topology)
        assert get_halo_backend() == "mpi"


# ---------------------------------------------------------------------------
# Tests: MPI halo exchange
# ---------------------------------------------------------------------------

class TestMPIHaloExchange:
    """Verify halo exchange works under MPI with the bootstrapped topology."""

    def test_halo_exchange_produces_finite_result(self, topology):
        """A scalar halo exchange should produce finite results on local faces."""
        from legoesm.grids.halo import pad_halo
        from legoesm.parallel.distributed import partition_state, gather_state

        import warnings

        n = 4
        # Create a known field: face i has value i+1.
        data = jnp.zeros((6, n, n), dtype=jnp.float32)
        for f in range(6):
            data = data.at[f].set(float(f + 1))

        # Partition (zero non-local faces) and exchange halos.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            partitioned = partition_state(data, topology)

        result = pad_halo(partitioned)
        assert result.shape == (6, n + 2, n + 2)

        # Local faces should have all-finite values.
        for f in topology.local_face_ids:
            assert jnp.all(jnp.isfinite(result[f]))

    def test_halo_exchange_roundtrip(self, topology):
        """partition -> halo -> gather should recover local interior data."""
        from legoesm.grids.halo import pad_halo
        from legoesm.parallel.distributed import partition_state, gather_state

        import warnings

        n = 8
        key = jax.random.PRNGKey(42)
        data = jax.random.normal(key, (6, n, n), dtype=jnp.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            partitioned = partition_state(data, topology)

        result = pad_halo(partitioned)
        # Interior of each local face should match original data.
        for f in topology.local_face_ids:
            interior = result[f, 1:-1, 1:-1]
            assert jnp.allclose(interior, data[f], atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: MPI collective consistency
# ---------------------------------------------------------------------------

class TestMPICollective:
    """Verify MPI collective operations across ranks."""

    def test_global_sum(self, topology):
        """Global sum across ranks equals expected value."""
        from legoesm.parallel.reductions import global_sum_mpi

        local = jnp.array(float(topology.rank))
        expected = jnp.array(float(sum(range(topology.n_processes))))
        result = global_sum_mpi(local)
        assert jnp.allclose(result, expected)

    def test_all_ranks_see_same_config(self, mpi_info):
        """All ranks should agree on grid_type and tiling."""
        config = get_active_config()
        local_info = (config.grid_type, config.tiling, config.is_distributed)
        all_info = _comm.allgather(local_info)
        for info in all_info:
            assert info == local_info
