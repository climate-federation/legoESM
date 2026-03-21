"""Tests for the canonical ParallelRuntime and decomposition validation.

These tests run without MPI and verify:
- ParallelRuntime creation in serial mode
- Device count validation (exact, no silent round-down)
- Halo/reduction backend dispatch
- Metadata reporting
"""

import pytest

from legoesm.parallel.runtime import (
    ParallelRuntime,
    HaloBackend,
    ReductionBackend,
    validate_device_count,
    _supported_device_counts,
)
from legoesm.parallel.mesh import _best_tile_factorization


# -----------------------------------------------------------------------
# Device count validation
# -----------------------------------------------------------------------

class TestDeviceCountValidation:
    """validate_device_count must reject unsupported counts and accept valid ones."""

    @pytest.mark.parametrize("n", [1, 2, 3, 6])
    def test_face_only_valid(self, n):
        validate_device_count(n, "cubed_sphere")

    @pytest.mark.parametrize("n", [24, 54, 96, 150, 216, 384, 600])
    def test_tiled_valid(self, n):
        validate_device_count(n, "cubed_sphere")

    @pytest.mark.parametrize("n", [4, 5, 7, 8, 9, 10, 12, 15, 18, 36, 48])
    def test_invalid_rejected(self, n):
        with pytest.raises(ValueError, match="Unsupported device count"):
            validate_device_count(n, "cubed_sphere")

    def test_error_message_has_suggestions(self):
        with pytest.raises(ValueError, match="next"):
            validate_device_count(7, "cubed_sphere")

    def test_latlon_accepts_anything(self):
        """Non-cubed-sphere grids accept any positive count."""
        for n in [1, 4, 7, 13, 100]:
            validate_device_count(n, "latlon")

    def test_zero_rejected(self):
        with pytest.raises(ValueError):
            validate_device_count(0, "cubed_sphere")


class TestTileFactorization:
    """_best_tile_factorization must match exactly, never round down."""

    def test_face_only(self):
        assert _best_tile_factorization(6) == (6, 1, 1)
        assert _best_tile_factorization(3) == (3, 1, 1)
        assert _best_tile_factorization(2) == (2, 1, 1)
        assert _best_tile_factorization(1) == (1, 1, 1)

    def test_tiled(self):
        assert _best_tile_factorization(24) == (6, 2, 2)
        assert _best_tile_factorization(54) == (6, 3, 3)
        assert _best_tile_factorization(96) == (6, 4, 4)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _best_tile_factorization(4)
        with pytest.raises(ValueError):
            _best_tile_factorization(12)  # 6*2, not perfect square
        with pytest.raises(ValueError):
            _best_tile_factorization(7)


class TestSupportedCounts:
    def test_small_counts(self):
        counts = _supported_device_counts(100)
        assert {1, 2, 3, 6, 24, 54, 96} <= counts

    def test_does_not_contain_invalid(self):
        counts = _supported_device_counts(200)
        for invalid in [4, 5, 7, 8, 9, 10, 12]:
            assert invalid not in counts


# -----------------------------------------------------------------------
# ParallelRuntime creation
# -----------------------------------------------------------------------

class TestParallelRuntimeSerial:
    """Test ParallelRuntime.create() in serial (non-MPI) environment."""

    def test_creates_serial_mode(self):
        rt = ParallelRuntime.create(n_devices=1)
        assert rt.mode == "serial"
        assert rt.rank == 0
        assert rt.world_size == 1
        assert rt.local_device_count == 1
        assert rt.halo_backend == HaloBackend.LOCAL
        assert rt.reduction_backend == ReductionBackend.LOCAL

    def test_describe_metadata(self):
        rt = ParallelRuntime.create(n_devices=1)
        meta = rt.describe()
        assert "mode" in meta
        assert "jax_version" in meta
        assert meta["mode"] == "serial"

    def test_repr(self):
        rt = ParallelRuntime.create(n_devices=1)
        s = repr(rt)
        assert "serial" in s
        assert "local" in s

    def test_scatter_gather_identity(self):
        """In serial mode, scatter/gather are identity."""
        import jax.numpy as jnp
        rt = ParallelRuntime.create(n_devices=1)
        x = jnp.ones((6, 4, 4))
        assert rt.scatter(x) is x
        assert rt.gather(x) is x

    def test_global_sum_identity(self):
        """In serial mode, global_sum is identity."""
        import jax.numpy as jnp
        rt = ParallelRuntime.create(n_devices=1)
        x = jnp.array(42.0)
        assert float(rt.global_sum(x)) == 42.0

    def test_grid_type_preserved(self):
        rt = ParallelRuntime.create(n_devices=1, grid_type="latlon")
        assert rt.grid_type == "latlon"


# -----------------------------------------------------------------------
# Deprecation warnings
# -----------------------------------------------------------------------

class TestDeprecation:
    def test_partition_state_warns(self):
        """Legacy partition_state should emit DeprecationWarning."""
        from legoesm.parallel.comm import build_comm_topology
        topo = build_comm_topology(0, 1)
        import jax.numpy as jnp
        state = jnp.zeros((6, 4, 4))

        from legoesm.parallel.distributed import partition_state
        with pytest.warns(DeprecationWarning, match="deprecated"):
            partition_state(state, topo)
