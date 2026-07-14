"""Tests for FV and spectral solvers across backends, dtypes, and parallelism.

Verifies that:
- FV operators work in float32, float64, and float16
- Spectral operators work in float64 (and route to CPU on Apple mps)
- Multi-device sharding produces correct results
- MPI halo exchange is compatible with FV operators

Usage:
    # Float32 (default)
    .venv/bin/python -m pytest tests/unit/test_backend_precision.py -v -k "float32"

    # Float64
    JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/unit/test_backend_precision.py -v

    # MPI (2 ranks)
    mpirun -np 2 .venv/bin/python -m pytest tests/unit/test_backend_precision.py -v -k "mpi"
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.core.operators_fv import (
    ppm_edge_values,
    ppm_limit,
    fv_flux_divergence,
    fv_scalar_advection,
    fv_gradient_x,
    fv_gradient_y,
)
from legoesm.core.hardware import get_backend


# =====================================================================
# Helpers
# =====================================================================

def _sw_to_cdgrid(state, cdgrid):
    """Convert generic ShallowWaterState to CDGridShallowWaterState."""
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import CDGridShallowWaterState
    from legoesm.grids.halo import pad_halo_vector
    h = state.h.data
    u_center = state.u.data
    v_center = state.v.data
    h_s = state.h_s.data
    u_pad, v_pad = pad_halo_vector(
        u_center, v_center,
        cdgrid.base.cos_angle, cdgrid.base.sin_angle,
        cdgrid.base.cos_angle_padded, cdgrid.base.sin_angle_padded,
        interp_offsets=cdgrid.base.halo_interp_offsets,
    )
    u_d = 0.25 * (u_pad[:, :-1, :-1] + u_pad[:, 1:, :-1] +
                   u_pad[:, :-1, 1:] + u_pad[:, 1:, 1:])
    v_d = 0.25 * (v_pad[:, :-1, :-1] + v_pad[:, 1:, :-1] +
                   v_pad[:, :-1, 1:] + v_pad[:, 1:, 1:])
    return CDGridShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)


def _make_fields(grid, dtype):
    """Create test scalar and velocity fields at the given dtype."""
    n = grid.n
    q = (1000.0 + 100.0 * grid.sin_lat).astype(dtype)
    u = (jnp.ones((6, n, n)) * 10.0).astype(dtype)
    v = (jnp.zeros((6, n, n))).astype(dtype)
    return q, u, v


def _requires_x64():
    """Skip if float64 is not enabled."""
    if not jax.config.jax_enable_x64:
        pytest.skip("JAX_ENABLE_X64 not set")


def _requires_metal():
    """Skip if not running on the Apple GPU (mps) backend."""
    if get_backend() != "MPS":
        pytest.skip("Apple GPU (mps) backend not available")


# =====================================================================
# FV — float32
# =====================================================================

class TestFVFloat32:
    """FV operators in float32 (default, works on all backends inc. mps)."""

    @pytest.fixture(scope="class")
    def grid(self):
        return create_cubed_sphere(8)

    def test_flux_divergence(self, grid):
        q, u, v = _make_fields(grid, jnp.float32)
        dq = fv_flux_divergence(q, u, v, grid)
        assert dq.dtype == jnp.float32
        assert dq.shape == q.shape
        assert jnp.all(jnp.isfinite(dq))

    def test_scalar_advection(self, grid):
        q, u, v = _make_fields(grid, jnp.float32)
        dq = fv_scalar_advection(q, u, v, grid)
        assert dq.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(dq))

    def test_gradient_x(self, grid):
        q, _, _ = _make_fields(grid, jnp.float32)
        dq = fv_gradient_x(q, grid)
        assert dq.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(dq))

    def test_gradient_y(self, grid):
        q, _, _ = _make_fields(grid, jnp.float32)
        dq = fv_gradient_y(q, grid)
        assert dq.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(dq))

    def test_ppm_edge_values(self, grid):
        n = grid.n
        q = jnp.ones((6, n + 4, n), dtype=jnp.float32) * 7.0
        edges = ppm_edge_values(q)
        assert edges.dtype == jnp.float32
        assert jnp.allclose(edges, 7.0, atol=1e-5)

    def test_ppm_limiter(self):
        q_bar = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)
        q_L = jnp.array([-0.5, 0.5, 1.5], dtype=jnp.float32)
        q_R = jnp.array([0.5, 1.5, 2.5], dtype=jnp.float32)
        q_L_lim, q_R_lim = ppm_limit(q_bar, q_L, q_R)
        assert q_L_lim.dtype == jnp.float32
        assert q_R_lim.dtype == jnp.float32

    def test_full_sw_step(self, grid):
        """Full FV shallow water step in float32."""
        from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
            CDGridShallowWaterModel,
            CDGridShallowWaterConfig,
        )
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from tests.test_cases.williamson import williamson_test2

        sw_state = williamson_test2(grid)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        state = _sw_to_cdgrid(sw_state, cdgrid)
        # Cast all state fields to float32
        state = jax.tree.map(
            lambda x: x.astype(jnp.float32) if hasattr(x, 'dtype') else x,
            state,
        )
        config = CDGridShallowWaterConfig(
            hyperdiff_coeff=0.0, use_conservation_fixer=False,
        )
        model = CDGridShallowWaterModel(grid, config)
        state_new = model.step(state, dt=300.0)
        assert jnp.all(jnp.isfinite(state_new.h))
        assert jnp.all(jnp.isfinite(state_new.u_d))

    def test_differentiable_float32(self, grid):
        """jax.grad works through FV operators in float32."""
        n = grid.n

        def loss(q):
            u = jnp.ones((6, n, n), dtype=jnp.float32) * 10.0
            v = jnp.zeros((6, n, n), dtype=jnp.float32)
            return jnp.mean(fv_flux_divergence(q, u, v, grid) ** 2)

        q = jnp.ones((6, n, n), dtype=jnp.float32) * 1000.0
        grads = jax.grad(loss)(q)
        assert grads.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(grads))


# =====================================================================
# FV — float64
# =====================================================================

class TestFVFloat64:
    """FV operators in float64 (requires JAX_ENABLE_X64=1)."""

    @pytest.fixture(scope="class")
    def grid(self):
        _requires_x64()
        return create_cubed_sphere(8)

    def test_flux_divergence(self, grid):
        _requires_x64()
        q, u, v = _make_fields(grid, jnp.float64)
        dq = fv_flux_divergence(q, u, v, grid)
        assert dq.dtype == jnp.float64
        assert jnp.all(jnp.isfinite(dq))

    def test_scalar_advection(self, grid):
        _requires_x64()
        q, u, v = _make_fields(grid, jnp.float64)
        dq = fv_scalar_advection(q, u, v, grid)
        assert dq.dtype == jnp.float64
        assert jnp.all(jnp.isfinite(dq))

    def test_gradient_xy(self, grid):
        _requires_x64()
        q, _, _ = _make_fields(grid, jnp.float64)
        dx = fv_gradient_x(q, grid)
        dy = fv_gradient_y(q, grid)
        assert dx.dtype == jnp.float64
        assert dy.dtype == jnp.float64

    def test_conservation_float64(self, grid):
        """Conservation should be tighter in float64."""
        _requires_x64()
        n = grid.n
        key = jax.random.PRNGKey(42)
        q = jax.random.uniform(key, (6, n, n), minval=900.0, maxval=1100.0).astype(
            jnp.float64
        )
        u = jax.random.normal(jax.random.split(key)[0], (6, n, n)).astype(
            jnp.float64
        ) * 5.0
        v = jax.random.normal(jax.random.split(key)[1], (6, n, n)).astype(
            jnp.float64
        ) * 5.0
        dq = fv_flux_divergence(q, u, v, grid)
        global_sum = jnp.sum(dq * grid.area.astype(jnp.float64))
        relative = float(
            jnp.abs(global_sum) / jnp.sum(jnp.abs(dq) * grid.area.astype(jnp.float64))
        )
        assert relative < 0.01, f"Conservation error in float64: {relative:.2e}"

    def test_full_sw_step_float64(self, grid):
        """Full FV shallow water step in float64."""
        _requires_x64()
        from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
            CDGridShallowWaterModel,
            CDGridShallowWaterConfig,
        )
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from tests.test_cases.williamson import williamson_test2

        sw_state = williamson_test2(grid)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        state = _sw_to_cdgrid(sw_state, cdgrid)
        config = CDGridShallowWaterConfig(
            hyperdiff_coeff=0.0, use_conservation_fixer=False,
        )
        model = CDGridShallowWaterModel(grid, config)
        state_new = model.step(state, dt=300.0)
        assert jnp.all(jnp.isfinite(state_new.h))


# =====================================================================
# FV — float16
# =====================================================================

class TestFVFloat16:
    """FV operators in float16 (reduced precision, experimental)."""

    @pytest.fixture(scope="class")
    def grid(self):
        return create_cubed_sphere(8)

    def test_flux_divergence_float16(self, grid):
        """FV flux divergence should produce finite results in float16."""
        n = grid.n
        # Use small values to avoid overflow in float16 (max ~65504)
        q = jnp.ones((6, n, n), dtype=jnp.float16) * 100.0
        u = jnp.ones((6, n, n), dtype=jnp.float16) * 5.0
        v = jnp.zeros((6, n, n), dtype=jnp.float16)
        dq = fv_flux_divergence(q, u, v, grid)
        # float16 may promote to float32 in some JAX ops
        assert dq.dtype in (jnp.float16, jnp.float32)
        assert jnp.all(jnp.isfinite(dq))

    def test_scalar_advection_float16(self, grid):
        n = grid.n
        q = jnp.ones((6, n, n), dtype=jnp.float16) * 100.0
        u = jnp.ones((6, n, n), dtype=jnp.float16) * 5.0
        v = jnp.zeros((6, n, n), dtype=jnp.float16)
        dq = fv_scalar_advection(q, u, v, grid)
        assert dq.dtype in (jnp.float16, jnp.float32)
        assert jnp.all(jnp.isfinite(dq))

    def test_ppm_edge_values_float16(self, grid):
        n = grid.n
        q = jnp.ones((6, n + 4, n), dtype=jnp.float16) * 7.0
        edges = ppm_edge_values(q)
        assert jnp.all(jnp.isfinite(edges))
        assert jnp.allclose(edges, 7.0, atol=0.1)  # looser tolerance for float16

    def test_gradient_float16(self, grid):
        n = grid.n
        q = jnp.ones((6, n, n), dtype=jnp.float16) * 100.0
        dx = fv_gradient_x(q, grid)
        dy = fv_gradient_y(q, grid)
        # Constant field → gradient should be ~0
        assert jnp.all(jnp.isfinite(dx))
        assert jnp.all(jnp.isfinite(dy))


# =====================================================================
# Spectral — float64 (required)
# =====================================================================

class TestSpectralFloat64:
    """Spectral solver requires float64/complex128."""

    @pytest.fixture(scope="class")
    def grid(self):
        _requires_x64()
        from legoesm.grids.gaussian import create_gaussian_grid
        return create_gaussian_grid(n_max=21)

    def test_sh_analysis_synthesis_roundtrip(self, grid):
        """SH analysis → synthesis should be near-identity for smooth fields."""
        _requires_x64()
        from legoesm.grids.gaussian import sh_analysis, sh_synthesis

        # Smooth field: Y_1^0 ≈ sin(lat)
        field = jnp.sin(grid.lat)[:, None] * jnp.ones(grid.n_lon)[None, :]
        coeffs = sh_analysis(grid, field)
        reconstructed = sh_synthesis(grid, coeffs)
        assert reconstructed.dtype == jnp.float64
        assert jnp.allclose(field, reconstructed, atol=1e-10)

    def test_spectral_sw_step(self, grid):
        """Spectral shallow water single step in float64."""
        _requires_x64()
        from legoesm.atmosphere.dynamics.gcm.spectral_sw import (
            SpectralShallowWaterModel,
            williamson_test2_spectral,
        )

        state = williamson_test2_spectral(grid)
        model = SpectralShallowWaterModel(grid)
        state_new = model.step(state, dt=600.0)
        # phi_hat may be a Field or raw array
        phi_data = getattr(state_new.phi_hat, 'data', state_new.phi_hat)
        assert jnp.all(jnp.isfinite(phi_data))

    def test_spectral_sw_differentiable(self, grid):
        """jax.grad through spectral SW step."""
        _requires_x64()
        from legoesm.atmosphere.dynamics.gcm.spectral_sw import (
            SpectralShallowWaterModel,
            williamson_test2_spectral,
        )

        state = williamson_test2_spectral(grid)
        model = SpectralShallowWaterModel(grid)

        # Extract raw data for the prognostic variable
        phi_data = getattr(state.phi_hat, 'data', state.phi_hat)

        def loss(phi_hat_data):
            from legoesm.core.field import Field
            phi_field = state.phi_hat
            if hasattr(phi_field, 'replace'):
                s = state._replace(phi_hat=phi_field.replace(data=phi_hat_data))
            else:
                s = state._replace(phi_hat=phi_hat_data)
            s_new = model.step(s, dt=300.0)
            out = getattr(s_new.phi_hat, 'data', s_new.phi_hat)
            return jnp.sum(jnp.abs(out) ** 2)

        grads = jax.grad(loss)(phi_data)
        assert jnp.all(jnp.isfinite(grads))


# =====================================================================
# Spectral — float32 fallback test
# =====================================================================

class TestSpectralFloat32Guard:
    """Spectral solvers should reject or warn on float32-only backends."""

    def test_check_spectral_backend_no_x64(self):
        """Without X64, spectral backend check should fail or warn."""
        from legoesm.core.hardware import check_spectral_backend
        if jax.config.jax_enable_x64:
            pytest.skip("X64 is enabled, guard won't trigger")
        with pytest.raises(ValueError, match="float64|x64|X64"):
            check_spectral_backend(allow_unsupported=False)


# =====================================================================
# Metal backend tests
# =====================================================================

class TestMetalBackend:
    """Tests for Apple GPU (mps) compatibility."""

    def test_metal_config_detection(self):
        """MetalConfig should detect whether the Apple GPU (mps) is available."""
        from legoesm.parallel.metal import get_metal_config
        config = get_metal_config()
        assert hasattr(config, 'is_metal')
        assert hasattr(config, 'cpu_device')
        # Off the Apple GPU (mps) backend, is_metal is False.
        if get_backend() != "MPS":
            assert config.is_metal is False

    def test_fv_would_work_on_metal(self):
        """FV operators use float32 by default, compatible with mps."""
        grid = create_cubed_sphere(8)
        q, u, v = _make_fields(grid, jnp.float32)
        # Verify inputs are float32
        assert q.dtype == jnp.float32
        # Run the operation to confirm it works
        dq = fv_flux_divergence(q, u, v, grid)
        assert dq.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(dq))

    def test_spectral_metal_cpu_routing(self):
        """Spectral model should detect the Apple GPU (mps) and route to CPU."""
        _requires_metal()
        from legoesm.atmosphere.dynamics.gcm.spectral_sw import SpectralShallowWaterModel
        from legoesm.grids.gaussian import create_gaussian_grid

        grid = create_gaussian_grid(n_lat=32)
        model = SpectralShallowWaterModel(grid)
        assert model._use_cpu_for_spectral is True


# =====================================================================
# Multi-device sharding
# =====================================================================

class TestMultiDeviceSharding:
    """Test that FV and spectral work with device mesh sharding."""

    @pytest.fixture(scope="class")
    def grid(self):
        return create_cubed_sphere(8)

    def test_single_device_mesh(self, grid):
        """Single-device mesh should work without changes."""
        from legoesm.parallel.mesh import create_device_mesh, shard_pytree

        dev_config = create_device_mesh(n_devices=1)
        assert dev_config.n_devices == 1

        n = grid.n
        q = jnp.ones((6, n, n), dtype=jnp.float32) * 1000.0
        # Shard should be a no-op for single device
        q_sharded = shard_pytree(q, dev_config)
        assert jnp.allclose(q, q_sharded)

    def test_fv_with_single_device_mesh(self, grid):
        """FV operators should work after shard_pytree on single device."""
        from legoesm.parallel.mesh import create_device_mesh, shard_pytree

        dev_config = create_device_mesh(n_devices=1)
        q, u, v = _make_fields(grid, jnp.float32)
        q = shard_pytree(q, dev_config)
        u = shard_pytree(u, dev_config)
        v = shard_pytree(v, dev_config)
        dq = fv_flux_divergence(q, u, v, grid)
        assert jnp.all(jnp.isfinite(dq))

    def test_auto_device_mesh(self, grid):
        """Auto mesh creation should succeed."""
        from legoesm.parallel.mesh import create_device_mesh

        dev_config = create_device_mesh(n_devices="auto")
        assert dev_config.n_devices >= 1


# =====================================================================
# MPI compatibility (requires mpirun)
# =====================================================================

class TestMPICompatibility:
    """MPI halo exchange compatibility with FV operators.

    These tests run in single-rank mode by default.
    For full MPI testing, use: mpirun -np 2 python -m pytest -k "mpi"
    """

    def test_mpi4jax_importable(self):
        """mpi4jax should be importable."""
        try:
            import mpi4jax  # noqa: F401
        except ImportError:
            pytest.skip("mpi4jax not installed")

    def test_comm_topology_creation(self):
        """Communication topology for various process counts."""
        from legoesm.parallel.comm import build_comm_topology

        for n_proc in [1, 2, 3, 6]:
            for rank in range(n_proc):
                topo = build_comm_topology(rank, n_proc)
                assert len(topo.local_face_ids) == 6 // n_proc
                assert topo.rank == rank

    def test_fv_after_partition_single_rank(self):
        """FV operators on partitioned state (single rank = full state)."""
        from legoesm.parallel.comm import build_comm_topology

        grid = create_cubed_sphere(8)
        topo = build_comm_topology(rank=0, n_processes=1)

        q, u, v = _make_fields(grid, jnp.float32)
        # Single rank: all faces are local, partition is identity
        dq = fv_flux_divergence(q, u, v, grid)
        assert jnp.all(jnp.isfinite(dq))
        assert dq.shape == q.shape


# =====================================================================
# Cross-dtype consistency
# =====================================================================

class TestCrossDtypeConsistency:
    """Verify FV results are consistent across dtypes."""

    @pytest.fixture(scope="class")
    def grid(self):
        return create_cubed_sphere(8)

    def test_float32_vs_float64_fv_divergence(self, grid):
        """float32 and float64 FV flux divergence should agree approximately."""
        _requires_x64()
        n = grid.n
        q32 = (1000.0 + 100.0 * grid.sin_lat).astype(jnp.float32)
        u32 = jnp.ones((6, n, n), dtype=jnp.float32) * 10.0
        v32 = jnp.zeros((6, n, n), dtype=jnp.float32)

        q64 = q32.astype(jnp.float64)
        u64 = u32.astype(jnp.float64)
        v64 = v32.astype(jnp.float64)

        dq32 = fv_flux_divergence(q32, u32, v32, grid)
        dq64 = fv_flux_divergence(q64, u64, v64, grid)

        # Should agree to ~float32 precision
        rel_err = float(
            jnp.max(jnp.abs(dq32.astype(jnp.float64) - dq64))
            / (jnp.max(jnp.abs(dq64)) + 1e-30)
        )
        assert rel_err < 1e-5, f"f32/f64 relative error: {rel_err:.2e}"

    def test_float32_vs_float64_gradient(self, grid):
        """float32 and float64 FV gradients should agree approximately."""
        _requires_x64()
        q32 = (1000.0 + 100.0 * grid.sin_lat).astype(jnp.float32)
        q64 = q32.astype(jnp.float64)

        dx32 = fv_gradient_x(q32, grid)
        dx64 = fv_gradient_x(q64, grid)

        rel_err = float(
            jnp.max(jnp.abs(dx32.astype(jnp.float64) - dx64))
            / (jnp.max(jnp.abs(dx64)) + 1e-30)
        )
        assert rel_err < 1e-4, f"f32/f64 gradient relative error: {rel_err:.2e}"
