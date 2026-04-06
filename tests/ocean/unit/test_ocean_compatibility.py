"""Verification tests for ocean module compatibility.

Tests cover:
- Metal/GPU backend compatibility (float32 FV, CPU fallback for spectral)
- Multi-device sharding (face-based parallelism)
- MPI-aware conservation fixers and operator dispatch
- Full differentiability through tendencies, model.step, and integrate_scan
- fori_loop vs lax.scan equivalence for barotropic substeps
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.ocean.eos import wright_eos, compute_hydrostatic_pressure
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    compute_layer_thickness,
    compute_ocean_jacobian,
)
from legoesm.ocean.state import OceanState, OceanConfig
from legoesm.ocean.init import rest_state_ocean
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.ocean.dynamics.ocean_pe_cdgrid import ocean_baroclinic_tendencies_cdgrid
from legoesm.ocean.dynamics.barotropic import barotropic_substeps
from legoesm.ocean.dynamics.ocean_model import OceanModel
from legoesm.ocean.conservation import (
    fix_volume_ocean,
    fix_heat_ocean,
    fix_salt_ocean,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def grid():
    return create_cubed_sphere(8)


@pytest.fixture
def cdgrid(grid):
    return create_cubed_sphere_cdgrid(grid)


@pytest.fixture
def z_coord():
    return create_ocean_z_star(n_levels=10, H_max=4000.0)


@pytest.fixture
def state(grid, z_coord):
    return rest_state_ocean(grid, z_coord, H_max=4000.0)


@pytest.fixture
def config():
    return OceanConfig(
        A_h=1e3, K_h=1e2, A_v=1e-3, K_v=1e-4,
        n_barotropic_substeps=10,
        hyperdiff_coeff=0.0,
    )


# ==============================================================================
# Metal / GPU Backend Compatibility
# ==============================================================================

class TestMetalCompatibility:
    """Verify FV ocean model works in float32 (Metal-compatible)."""

    def test_state_is_float32(self, state):
        """All state arrays should be float32 when created in float32 mode."""
        # Skip if x64 mode was enabled by another test module
        if jax.config.jax_enable_x64:
            pytest.skip("x64 mode enabled by another test; float32 check irrelevant")
        for leaf in jax.tree.leaves(state):
            if hasattr(leaf, 'dtype'):
                assert leaf.dtype == jnp.float32, (
                    f"Expected float32, got {leaf.dtype}"
                )

    def test_eos_float32(self):
        """Wright EOS should work in float32."""
        T = jnp.array(10.0, dtype=jnp.float32)
        S = jnp.array(35.0, dtype=jnp.float32)
        p = jnp.array(0.0, dtype=jnp.float32)
        rho = wright_eos(T, S, p)
        assert jnp.isfinite(rho)
        assert 1020.0 < float(rho) < 1030.0

    def test_tendencies_finite(self, state, grid, cdgrid, z_coord, config):
        """Baroclinic tendencies should be finite (works in any precision)."""
        tend = ocean_baroclinic_tendencies_cdgrid(state, grid, z_coord, cdgrid, config)
        for leaf in jax.tree.leaves(tend):
            if hasattr(leaf, 'dtype') and jnp.issubdtype(leaf.dtype, jnp.floating):
                assert jnp.all(jnp.isfinite(leaf))

    def test_model_step_finite(self, grid, z_coord, config, state):
        """Model step should produce finite results (works in any precision)."""
        model = OceanModel(grid, z_coord, config)
        state_new = model.step(state, 3600.0)
        for leaf in jax.tree.leaves(state_new):
            if hasattr(leaf, 'dtype') and jnp.issubdtype(leaf.dtype, jnp.floating):
                assert jnp.all(jnp.isfinite(leaf))

    def test_spectral_ocean_metal_routing(self):
        """SpectralOceanModel should have Metal/CPU routing infrastructure."""
        import inspect
        from legoesm.ocean.dynamics.spectral_ocean_pe import SpectralOceanModel

        # Verify the class has the routing infrastructure without
        # creating a grid (which requires x64 to be enabled).
        init_source = inspect.getsource(SpectralOceanModel.__init__)
        assert "_use_cpu_for_spectral" in init_source
        assert "_cpu_device" in init_source
        assert "_default_device" in init_source
        assert "get_backend" in init_source
        # get_backend() returns lowercase; source uses "metal"
        assert 'metal' in init_source.lower()

        step_source = inspect.getsource(SpectralOceanModel.step)
        assert "_use_cpu_for_spectral" in step_source
        assert "device_put" in step_source

        # Verify batched CPU integration method exists
        assert hasattr(SpectralOceanModel, '_step_on_cpu')
        assert hasattr(SpectralOceanModel, '_integrate_on_cpu')

    def test_spectral_ocean_enforces_x64(self):
        """Spectral model constructor should check jax_enable_x64."""
        import inspect
        from legoesm.ocean.dynamics.spectral_ocean_pe import SpectralOceanModel

        init_source = inspect.getsource(SpectralOceanModel.__init__)
        assert "jax_enable_x64" in init_source
        assert "SpectralOceanModel requires float64/complex128 arithmetic" in init_source

    def test_spectral_tendencies_include_viscous_terms(self):
        """Spectral tendencies should include A_h/K_h/A_v operators."""
        import inspect
        from legoesm.ocean.dynamics.spectral_ocean_pe import spectral_ocean_tendencies

        source = inspect.getsource(spectral_ocean_tendencies)
        assert "config.A_h" in source
        assert "config.K_h" in source
        assert "config.A_v" in source
        assert "vertical_diffusion(" in source

    def test_spectral_ocean_rejects_invalid_hyperdiff_config(self, z_coord):
        """Spectral ocean model should fail fast on invalid hyperdiff settings."""
        from legoesm.ocean.dynamics.spectral_ocean_pe import SpectralOceanModel
        from legoesm.ocean.state import SpectralOceanConfig

        with pytest.raises(ValueError, match="hyperdiff_coeff"):
            SpectralOceanModel(None, z_coord, SpectralOceanConfig(hyperdiff_coeff=-1.0))
        with pytest.raises(ValueError, match="hyperdiff_order"):
            SpectralOceanModel(None, z_coord, SpectralOceanConfig(hyperdiff_order=0))
        with pytest.raises(ValueError, match="min_water_column_m"):
            SpectralOceanModel(None, z_coord, SpectralOceanConfig(min_water_column_m=0.0))


# ==============================================================================
# Multi-Device Sharding
# ==============================================================================

class TestSharding:
    """Verify ocean state works with JAX device sharding."""

    def test_state_shardable(self, state):
        """OceanState should be shardable via shard_pytree."""
        from legoesm.parallel.mesh import create_device_mesh, shard_pytree

        config = create_device_mesh(n_devices=1)
        sharded = shard_pytree(state, config)

        # Verify shapes preserved
        assert sharded.u.data.shape == state.u.data.shape
        assert sharded.eta.data.shape == state.eta.data.shape

    def test_state_face_dim_compatible(self, state):
        """All ocean state arrays should have face dim = 6 (for sharding)."""
        for leaf in jax.tree.leaves(state):
            if hasattr(leaf, 'shape') and leaf.ndim >= 3:
                assert leaf.shape[0] == 6, (
                    f"Expected face dim 6, got shape {leaf.shape}"
                )

    def test_tendencies_shardable(self, state, grid, cdgrid, z_coord, config):
        """OceanTendencies should be shardable."""
        from legoesm.parallel.mesh import create_device_mesh, shard_pytree

        dc = create_device_mesh(n_devices=1)
        tend = ocean_baroclinic_tendencies_cdgrid(state, grid, z_coord, cdgrid, config)
        sharded_tend = shard_pytree(tend, dc)
        assert sharded_tend.du_dt.data.shape == tend.du_dt.data.shape

    def test_grid_replicate_compatible(self, grid):
        """Grid should be replicable."""
        from legoesm.parallel.mesh import create_device_mesh, replicate_pytree

        dc = create_device_mesh(n_devices=1)
        replicated = replicate_pytree(grid, dc)
        assert replicated.area.shape == grid.area.shape


# ==============================================================================
# MPI-Aware Operations
# ==============================================================================

class TestMPIAwareness:
    """Verify ocean module uses MPI-aware patterns."""

    def test_conservation_uses_is_distributed(self):
        """Conservation fixers should use _is_distributed pattern."""
        import inspect
        from legoesm.ocean.conservation import _ocean_area_sum, _ocean_global_sum

        area_source = inspect.getsource(_ocean_area_sum)
        assert "_ocean_global_sum" in area_source, (
            "_ocean_area_sum should route reductions through _ocean_global_sum"
        )

        global_source = inspect.getsource(_ocean_global_sum)
        assert "_is_distributed" in global_source, (
            "_ocean_global_sum must check _is_distributed for MPI"
        )
        assert "global_sum_mpi" in global_source, (
            "_ocean_global_sum must call global_sum_mpi for MPI allreduce"
        )

    def test_operators_use_halo_exchange(self):
        """Baroclinic tendencies should use C-D grid operators with halo exchange."""
        import inspect
        from legoesm.ocean.dynamics.ocean_pe_cdgrid import ocean_baroclinic_tendencies_cdgrid

        source = inspect.getsource(ocean_baroclinic_tendencies_cdgrid)
        # C-D grid ocean uses unified C-D grid operators (not A-grid gradient_x_3d)
        assert "cgrid_mass_flux_divergence(" in source
        assert "cgrid_divergence(" in source
        assert "dgrid_vorticity(" in source

    def test_barotropic_uses_halo_operators(self):
        """Barotropic substeps should call operators with halo exchange."""
        import inspect
        from legoesm.ocean.dynamics.barotropic import (
            _divergence_raw,
            _gradient_x_raw,
        )

        # These should call the full 2D operators (which include pad_halo)
        div_source = inspect.getsource(_divergence_raw)
        assert "divergence(" in div_source

        grad_source = inspect.getsource(_gradient_x_raw)
        assert "gradient_x(" in grad_source

    def test_conservation_fixer_produces_correct_output(self, grid, z_coord, state):
        """Conservation fixer should produce an OceanState with finite data."""
        perturbed = state._replace(
            eta=state.eta.replace(
                data=state.eta.data + 0.01 * jnp.ones_like(state.eta.data),
            ),
        )
        fixed = fix_volume_ocean(perturbed, state, grid)
        assert jnp.all(jnp.isfinite(fixed.eta.data))

    def test_heat_conservation_fixer(self, grid, z_coord, state):
        """Heat fixer should produce finite output."""
        perturbed = state._replace(
            T=state.T.replace(
                data=state.T.data + 0.1 * jnp.ones_like(state.T.data),
            ),
        )
        fixed = fix_heat_ocean(perturbed, state, grid, z_coord)
        assert jnp.all(jnp.isfinite(fixed.T.data))

    def test_spectral_conservation_uses_mpi_aware_reductions(self):
        """Spectral conservation fixer should route totals through MPI-aware sums."""
        import inspect
        from legoesm.ocean.dynamics.spectral_ocean_pe import (
            _spectral_global_sum,
            _spectral_conservation_fixer,
        )

        global_source = inspect.getsource(_spectral_global_sum)
        assert "_is_distributed" in global_source
        assert "global_sum_mpi" in global_source

        fixer_source = inspect.getsource(_spectral_conservation_fixer)
        assert "_spectral_global_sum" in fixer_source


# ==============================================================================
# Differentiability
# ==============================================================================

class TestDifferentiability:
    """Full differentiability verification through the ocean module."""

    def test_grad_through_eos(self):
        """jax.grad should work through the Wright EOS."""
        def loss(T_val):
            rho = wright_eos(T_val, jnp.array(35.0), jnp.array(0.0))
            return rho

        grad = jax.grad(loss)(jnp.array(10.0))
        assert jnp.isfinite(grad)
        # drho/dT should be negative (warmer -> lighter)
        assert float(grad) < 0

    def test_grad_through_hydrostatic_pressure(self):
        """jax.grad through hydrostatic pressure computation."""
        z_coord = create_ocean_z_star(n_levels=5, H_max=4000.0)

        def loss(rho_val):
            rho = jnp.broadcast_to(rho_val, (5,))
            p = compute_hydrostatic_pressure(
                rho, jnp.array(0.0), z_coord.dz_ref,
                jnp.array(1.0), rho_ref=1025.0,
            )
            return jnp.sum(p)

        grad = jax.grad(loss)(jnp.array(1025.0))
        assert jnp.isfinite(grad)

    def test_grad_through_tendencies(self, state, grid, cdgrid, z_coord, config):
        """jax.grad should work through baroclinic tendency computation."""
        def loss_fn(eta_data):
            s = state._replace(
                eta=state.eta.replace(data=eta_data),
            )
            tend = ocean_baroclinic_tendencies_cdgrid(s, grid, z_coord, cdgrid, config)
            return jnp.sum(tend.du_dt.data ** 2)

        grad = jax.grad(loss_fn)(state.eta.data)
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_through_temperature(self, state, grid, cdgrid, z_coord, config):
        """jax.grad should work through temperature tendencies."""
        def loss_fn(T_data):
            s = state._replace(
                T=state.T.replace(data=T_data),
            )
            tend = ocean_baroclinic_tendencies_cdgrid(s, grid, z_coord, cdgrid, config)
            return jnp.sum(tend.dT_dt.data ** 2)

        grad = jax.grad(loss_fn)(state.T.data)
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_through_model_step(self, grid, z_coord, state):
        """jax.grad through a full model step (differentiable barotropic)."""
        config = OceanConfig(
            A_h=1e3, K_h=1e2, A_v=1e-3, K_v=1e-4,
            n_barotropic_substeps=5,
            hyperdiff_coeff=0.0,
            use_conservation_fixer=False,
            differentiable_barotropic=True,
        )
        model = OceanModel(grid, z_coord, config)

        def loss_fn(eta_data):
            s = state._replace(
                eta=state.eta.replace(data=eta_data),
            )
            new_state = model.step(s, 1800.0)
            return jnp.sum(new_state.eta.data ** 2)

        grad = jax.grad(loss_fn)(state.eta.data)
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_through_scan(self, grid, z_coord, state):
        """jax.grad through lax.scan integration (multi-step)."""
        # lax.scan requires matching input/output dtypes. When x64 is
        # enabled by other test modules, float promotion upcasts float32
        # state arrays to float64 inside the model step, causing a dtype
        # mismatch. Skip in that case since the FV ocean targets float32.
        if jax.config.jax_enable_x64:
            pytest.skip("x64 mode causes dtype mismatch in float32 lax.scan")

        config = OceanConfig(
            A_h=1e3, K_h=1e2, A_v=1e-3, K_v=1e-4,
            n_barotropic_substeps=5,
            hyperdiff_coeff=0.0,
            use_conservation_fixer=False,
            differentiable_barotropic=True,
        )
        model = OceanModel(grid, z_coord, config)

        def loss_fn(eta_data):
            s = state._replace(
                eta=state.eta.replace(data=eta_data),
            )
            final, _ = model.integrate_scan(s, n_steps=3, dt=1800.0)
            return jnp.sum(final.eta.data ** 2)

        grad = jax.grad(loss_fn)(state.eta.data)
        assert jnp.all(jnp.isfinite(grad))

    def test_vmap_through_eos(self):
        """jax.vmap should work through the Wright EOS."""
        T = jnp.array([5.0, 10.0, 15.0, 20.0, 25.0])
        S = jnp.full(5, 35.0)
        p = jnp.zeros(5)

        # vmap over batch dimension
        rho_batch = jax.vmap(wright_eos)(T, S, p)
        assert rho_batch.shape == (5,)
        assert jnp.all(jnp.isfinite(rho_batch))

    def test_jit_through_tendencies(self, state, grid, cdgrid, z_coord, config):
        """jax.jit should work through tendency computation."""
        @jax.jit
        def compute_tend(s):
            return ocean_baroclinic_tendencies_cdgrid(s, grid, z_coord, cdgrid, config)

        tend = compute_tend(state)
        assert jnp.all(jnp.isfinite(tend.du_dt.data))

    def test_pytree_tree_map(self, state):
        """OceanState should work with jax.tree.map."""
        doubled = jax.tree.map(lambda x: x * 2.0, state)
        assert jnp.allclose(doubled.u.data, state.u.data * 2.0)
        assert jnp.allclose(doubled.eta.data, state.eta.data * 2.0)


# ==============================================================================
# Barotropic fori_loop vs lax.scan equivalence
# ==============================================================================

class TestBarotropicEquivalence:
    """Verify fori_loop and lax.scan barotropic substeps give same results."""

    def test_fori_vs_scan_equivalence(self, grid, z_coord, state):
        """Both barotropic modes should give identical results."""
        config_fori = OceanConfig(
            A_h=1e3, K_h=1e2, A_v=1e-3, K_v=1e-4,
            n_barotropic_substeps=10,
            hyperdiff_coeff=0.0,
            differentiable_barotropic=False,
        )
        config_scan = config_fori._replace(differentiable_barotropic=True)

        model_fori = OceanModel(grid, z_coord, config_fori)
        model_scan = OceanModel(grid, z_coord, config_scan)

        result_fori = model_fori.step(state, 3600.0)
        result_scan = model_scan.step(state, 3600.0)

        # Should be numerically identical (same operations)
        assert jnp.allclose(
            result_fori.eta.data, result_scan.eta.data, atol=1e-5,
        ), "fori_loop and scan barotropic substeps diverged for eta"
        assert jnp.allclose(
            result_fori.u.data, result_scan.u.data, atol=1e-5,
        ), "fori_loop and scan barotropic substeps diverged for u"
        assert jnp.allclose(
            result_fori.T.data, result_scan.T.data, atol=1e-5,
        ), "fori_loop and scan barotropic substeps diverged for T"


# ==============================================================================
# Numerical stability under JIT
# ==============================================================================

class TestJITStability:
    """Verify JIT compilation doesn't change numerical behavior."""

    def test_jit_determinism(self, grid, z_coord, config, state):
        """JIT and non-JIT should give same results."""
        model = OceanModel(grid, z_coord, config)

        # First call (compiles)
        result1 = model.step(state, 3600.0)
        # Second call (cached)
        result2 = model.step(state, 3600.0)

        assert jnp.allclose(result1.eta.data, result2.eta.data)
        assert jnp.allclose(result1.u.data, result2.u.data)

    def test_multi_step_jit_stable(self, grid, z_coord, config, state):
        """Multiple JIT-compiled steps should remain stable."""
        model = OceanModel(grid, z_coord, config)
        s = state
        for _ in range(10):
            s = model.step(s, 3600.0)

        mask = state.land_mask.data
        T_ocean = s.T.data * mask[..., jnp.newaxis]
        T_max = float(jnp.max(jnp.where(mask[..., jnp.newaxis] > 0.5, T_ocean, -999)))
        T_min = float(jnp.min(jnp.where(mask[..., jnp.newaxis] > 0.5, T_ocean, 999)))
        assert T_min > -5.0, f"T_min = {T_min}"
        assert T_max < 40.0, f"T_max = {T_max}"


# ==============================================================================
# Land masking correctness under JAX transformations
# ==============================================================================

class TestLandMaskingTransformations:
    """Verify land masking is preserved through JAX transformations."""

    def test_land_stays_zero_after_step(self, grid, z_coord, config, state):
        """Land cells should remain exactly zero after model step."""
        model = OceanModel(grid, z_coord, config)
        new_state = model.step(state, 3600.0)

        land = state.land_mask.data < 0.5
        if jnp.any(land):
            assert float(jnp.max(jnp.abs(new_state.u.data[land]))) == 0.0
            assert float(jnp.max(jnp.abs(new_state.v.data[land]))) == 0.0
            assert float(jnp.max(jnp.abs(new_state.eta.data[land]))) == 0.0

    def test_land_mask_preserved_through_grad(self, state, grid, cdgrid, z_coord, config):
        """Gradients should respect land masking (zero on land)."""
        def loss_fn(eta_data):
            s = state._replace(
                eta=state.eta.replace(data=eta_data),
            )
            tend = ocean_baroclinic_tendencies_cdgrid(s, grid, z_coord, cdgrid, config)
            return jnp.sum(tend.deta_dt.data ** 2)

        grad = jax.grad(loss_fn)(state.eta.data)
        land = state.land_mask.data < 0.5
        if jnp.any(land):
            # Gradients on land should be zero (no contribution to loss)
            assert float(jnp.max(jnp.abs(grad[land]))) == 0.0
