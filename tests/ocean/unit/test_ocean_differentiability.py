"""Differentiability tests for recent ocean model changes.

Tests jax.grad through:
1. Smagorinsky biharmonic on MPAS (operators_voronoi.smagorinsky_biharmonic_3d)
2. Bottom drag on full velocity (ocean_pe_latlon_cgrid)
3. Sponge relaxation (ocean_pe_latlon_cgrid with SpongeForcing)
4. Barotropic solver with bottom drag (barotropic_latlon_cgrid)

All tests use small grids (latlon 8x16, MPAS ico2) and JAX_ENABLE_X64=1.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. Smagorinsky biharmonic on MPAS
# ---------------------------------------------------------------------------

class TestSmagorinskyBiharmonicMPASDiff:
    """Test jax.grad through smagorinsky_biharmonic_3d on MPAS mesh."""

    @pytest.fixture(scope="class")
    def mpas_setup(self):
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.core.operators_voronoi import smagorinsky_biharmonic_3d
        mesh = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=10)
        nlev = 3
        nEdges = mesh.nEdges
        key = jax.random.PRNGKey(42)
        u_edge = 0.01 * jax.random.normal(key, (nEdges, nlev), dtype=jnp.float64)
        return mesh, u_edge, smagorinsky_biharmonic_3d

    def test_grad_finite_nonzero(self, mpas_setup):
        """jax.grad of sum(tendency) w.r.t. velocity is finite and nonzero.

        Note: We use sum(tendency) rather than sum(tendency**2) because
        on an Earth-radius MPAS mesh the biharmonic tendency magnitudes
        are O(1e-22) for O(0.01 m/s) velocities, so squaring them
        underflows to ~1e-44 and gradients vanish numerically.
        """
        mesh, u_edge, smag_fn = mpas_setup
        C_smag = 0.1

        def loss(u):
            tendency = smag_fn(u, mesh, C_smag)
            return jnp.sum(tendency)

        grad = jax.grad(loss)(u_edge)
        assert jnp.all(jnp.isfinite(grad)), (
            f"Non-finite gradients in Smagorinsky biharmonic MPAS. "
            f"NaN count: {int(jnp.sum(jnp.isnan(grad)))}, "
            f"Inf count: {int(jnp.sum(jnp.isinf(grad)))}"
        )
        nonzero_frac = float(jnp.mean(jnp.abs(grad) > 1e-30))
        assert nonzero_frac > 0.1, (
            f"Too few nonzero gradient entries: {nonzero_frac:.1%}"
        )

    def test_grad_squared_loss_large_velocity(self, mpas_setup):
        """jax.grad of sum(tendency**2) works with larger velocities.

        On an Earth-radius MPAS mesh (dx ~ 1e6 m), biharmonic tendencies
        scale as u / dx^4 ~ 1e-24, so tendency**2 ~ 1e-48 and gradients
        are ~1e-36.  They are genuinely nonzero (100% at >0 threshold)
        but below any naive 1e-30 cutoff.  We verify finiteness and that
        all entries are strictly nonzero, confirming the gradient flows.
        """
        mesh, _, smag_fn = mpas_setup
        C_smag = 0.1
        key = jax.random.PRNGKey(123)
        u_large = 1.0 * jax.random.normal(
            key, (mesh.nEdges, 3), dtype=jnp.float64,
        )

        def loss(u):
            tendency = smag_fn(u, mesh, C_smag)
            return jnp.sum(tendency ** 2)

        grad = jax.grad(loss)(u_large)
        assert jnp.all(jnp.isfinite(grad)), "Non-finite grad with large velocity"
        # All entries should be strictly nonzero (at double precision)
        nonzero_frac = float(jnp.mean(grad != 0.0))
        assert nonzero_frac > 0.9, (
            f"Too few nonzero gradient entries with large velocity: {nonzero_frac:.1%}"
        )

    def test_grad_jit_consistent(self, mpas_setup):
        """jax.jit(jax.grad(loss)) matches jax.grad(loss)."""
        mesh, u_edge, smag_fn = mpas_setup
        C_smag = 0.1

        def loss(u):
            tendency = smag_fn(u, mesh, C_smag)
            return jnp.sum(tendency ** 2)

        grad_eager = jax.grad(loss)(u_edge)
        grad_jit = jax.jit(jax.grad(loss))(u_edge)
        assert jnp.allclose(grad_eager, grad_jit, atol=1e-10), (
            f"JIT gradient differs from eager. Max diff: "
            f"{float(jnp.max(jnp.abs(grad_eager - grad_jit))):.2e}"
        )


# ---------------------------------------------------------------------------
# 2. Bottom drag on full velocity
# ---------------------------------------------------------------------------

class TestBottomDragDiff:
    """Test jax.grad through baroclinic tendencies with bottom_drag_r > 0."""

    @pytest.fixture(scope="class")
    def latlon_cgrid_setup(self):
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        from legoesm.ocean.state import LatLonCGridOceanConfig
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            latlon_cgrid_ocean_baroclinic_tendencies,
        )

        grid = create_latlon_grid(8, 16)
        z_coord = create_ocean_z_star(n_levels=3, H_max=500.0)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord, H_max=500.0, land_lat_threshold=85.0,
        )
        config = LatLonCGridOceanConfig.from_flat(
            A_h=1000.0,
            K_h=0.0,
            A_v=0.0,
            K_v=0.0,
            bottom_drag_r=1e-3,
            n_barotropic_substeps=2,
            enable_runtime_checks=False,
        )

        # Add small velocity perturbation so drag produces nonzero tendency
        key = jax.random.PRNGKey(7)
        u_pert = 0.05 * jax.random.normal(
            key, state.u.data.shape, dtype=jnp.float64,
        )
        v_pert = 0.05 * jax.random.normal(
            jax.random.PRNGKey(8), state.v.data.shape, dtype=jnp.float64,
        )
        state = state._replace(
            u=state.u.replace(data=state.u.data + u_pert),
            v=state.v.replace(data=state.v.data + v_pert),
        )

        return grid, z_coord, state, config, latlon_cgrid_ocean_baroclinic_tendencies

    def test_grad_wrt_temperature(self, latlon_cgrid_setup):
        """Gradient w.r.t. T through tendencies with bottom drag is finite."""
        grid, z_coord, state, config, tend_fn = latlon_cgrid_setup

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            tendencies = tend_fn(s, grid, z_coord, config)
            return jnp.sum(tendencies.dT_dt.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert jnp.all(jnp.isfinite(grad)), "Non-finite grad w.r.t. T with bottom drag"
        # T gradient may be mostly zero if drag doesn't affect tracers much,
        # but should still be finite.

    def test_grad_wrt_velocity(self, latlon_cgrid_setup):
        """Gradient w.r.t. u through tendencies with bottom drag is finite and nonzero."""
        grid, z_coord, state, config, tend_fn = latlon_cgrid_setup

        def loss(u_data):
            s = state._replace(u=state.u.replace(data=u_data))
            tendencies = tend_fn(s, grid, z_coord, config)
            return jnp.sum(tendencies.du_dt.data ** 2)

        grad = jax.grad(loss)(state.u.data)
        assert jnp.all(jnp.isfinite(grad)), (
            f"Non-finite grad w.r.t. u with bottom drag. "
            f"NaN: {int(jnp.sum(jnp.isnan(grad)))}"
        )
        nonzero_frac = float(jnp.mean(jnp.abs(grad) > 1e-30))
        assert nonzero_frac > 0.05, (
            f"Too few nonzero gradient entries for u: {nonzero_frac:.1%}"
        )

    def test_bottom_drag_changes_gradient(self, latlon_cgrid_setup):
        """Bottom drag actually contributes to the gradient (not a dead code path)."""
        grid, z_coord, state, config, tend_fn = latlon_cgrid_setup

        # Config with no bottom drag
        config_no_drag = config._replace(bottom_drag=config.bottom_drag._replace(bottom_drag_r=0.0))

        def loss(u_data, cfg):
            s = state._replace(u=state.u.replace(data=u_data))
            tendencies = tend_fn(s, grid, z_coord, cfg)
            return jnp.sum(tendencies.du_dt.data ** 2)

        grad_drag = jax.grad(loss)(state.u.data, config)
        grad_no_drag = jax.grad(loss)(state.u.data, config_no_drag)
        diff = float(jnp.max(jnp.abs(grad_drag - grad_no_drag)))
        assert diff > 1e-15, (
            f"Bottom drag has no effect on gradient. Max diff = {diff:.2e}"
        )


# ---------------------------------------------------------------------------
# 3. Sponge relaxation
# ---------------------------------------------------------------------------

class TestSpongeRelaxationDiff:
    """Test jax.grad through sponge tendency gamma * (T_ref - T)."""

    @pytest.fixture(scope="class")
    def sponge_setup(self):
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        from legoesm.ocean.state import LatLonCGridOceanConfig
        from legoesm.ocean.sponge import SpongeForcing
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            latlon_cgrid_ocean_baroclinic_tendencies,
        )

        grid = create_latlon_grid(8, 16)
        z_coord = create_ocean_z_star(n_levels=3, H_max=500.0)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord, H_max=500.0, land_lat_threshold=85.0,
        )
        config = LatLonCGridOceanConfig.from_flat(
            A_h=1000.0,
            K_h=0.0,
            A_v=0.0,
            K_v=0.0,
            bottom_drag_r=0.0,
            n_barotropic_substeps=2,
            enable_runtime_checks=False,
        )

        # Create a sponge forcing with gamma nonzero near edges
        n_lat, n_lon = grid.n_lat, grid.n_lon
        nlev = z_coord.n_levels
        gamma = np.zeros((n_lat, n_lon), dtype=np.float64)
        # Sponge zone at first and last 2 latitude rows
        gamma[:2, :] = 1.0 / 86400.0
        gamma[-2:, :] = 1.0 / 86400.0
        gamma = jnp.array(gamma)

        T_ref = jnp.full((n_lat, n_lon, nlev), 10.0, dtype=jnp.float64)
        S_ref = jnp.full((n_lat, n_lon, nlev), 35.0, dtype=jnp.float64)
        sponge = SpongeForcing(gamma=gamma, T_ref=T_ref, S_ref=S_ref)

        return grid, z_coord, state, config, sponge, latlon_cgrid_ocean_baroclinic_tendencies

    def test_grad_wrt_temperature(self, sponge_setup):
        """Gradient w.r.t. T flows through sponge relaxation."""
        grid, z_coord, state, config, sponge, tend_fn = sponge_setup

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            tendencies = tend_fn(s, grid, z_coord, config, sponge=sponge)
            return jnp.sum(tendencies.dT_dt.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert jnp.all(jnp.isfinite(grad)), "Non-finite grad through sponge relaxation"
        nonzero_frac = float(jnp.mean(jnp.abs(grad) > 1e-30))
        assert nonzero_frac > 0.01, (
            f"Too few nonzero gradient entries through sponge: {nonzero_frac:.1%}"
        )

    def test_sponge_contributes_to_gradient(self, sponge_setup):
        """Sponge relaxation is not a dead code path in gradient computation."""
        grid, z_coord, state, config, sponge, tend_fn = sponge_setup

        def loss_with_sponge(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            tendencies = tend_fn(s, grid, z_coord, config, sponge=sponge)
            return jnp.sum(tendencies.dT_dt.data ** 2)

        def loss_without_sponge(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            tendencies = tend_fn(s, grid, z_coord, config, sponge=None)
            return jnp.sum(tendencies.dT_dt.data ** 2)

        grad_with = jax.grad(loss_with_sponge)(state.T.data)
        grad_without = jax.grad(loss_without_sponge)(state.T.data)
        diff = float(jnp.max(jnp.abs(grad_with - grad_without)))
        assert diff > 1e-15, (
            f"Sponge has no effect on gradient. Max diff = {diff:.2e}"
        )

    def test_sponge_with_velocity_ref(self, sponge_setup):
        """Gradient flows through sponge velocity relaxation."""
        grid, z_coord, state, config, sponge, tend_fn = sponge_setup

        n_lat, n_lon = grid.n_lat, grid.n_lon
        nlev = z_coord.n_levels

        # Add velocity references to the sponge
        from legoesm.ocean.sponge import SpongeForcing
        u_ref = jnp.zeros((n_lat, n_lon + 1, nlev), dtype=jnp.float64)
        v_ref = jnp.zeros((n_lat + 1, n_lon, nlev), dtype=jnp.float64)
        sponge_vel = SpongeForcing(
            gamma=sponge.gamma,
            T_ref=sponge.T_ref,
            S_ref=sponge.S_ref,
            u_ref=u_ref,
            v_ref=v_ref,
        )

        # Add velocity perturbation
        key = jax.random.PRNGKey(99)
        u_pert = 0.05 * jax.random.normal(
            key, state.u.data.shape, dtype=jnp.float64,
        )
        state_pert = state._replace(
            u=state.u.replace(data=state.u.data + u_pert),
        )

        def loss(u_data):
            s = state_pert._replace(u=state_pert.u.replace(data=u_data))
            tendencies = tend_fn(s, grid, z_coord, config, sponge=sponge_vel)
            return jnp.sum(tendencies.du_dt.data ** 2)

        grad = jax.grad(loss)(state_pert.u.data)
        assert jnp.all(jnp.isfinite(grad)), (
            "Non-finite grad through sponge velocity relaxation"
        )


# ---------------------------------------------------------------------------
# 4. Barotropic solver with bottom drag
# ---------------------------------------------------------------------------

class TestBarotropicBottomDragDiff:
    """Test jax.grad through barotropic substeps with bottom_drag_r > 0."""

    @pytest.fixture(scope="class")
    def baro_setup(self):
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        from legoesm.ocean.state import LatLonCGridOceanConfig
        from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
            barotropic_substeps_latlon_cgrid,
        )

        grid = create_latlon_grid(8, 16)
        z_coord = create_ocean_z_star(n_levels=3, H_max=500.0)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord, H_max=500.0, land_lat_threshold=85.0,
        )

        # Add velocity perturbation for nontrivial state
        key = jax.random.PRNGKey(17)
        u_pert = 0.02 * jax.random.normal(
            key, state.u.data.shape, dtype=jnp.float64,
        )
        v_pert = 0.02 * jax.random.normal(
            jax.random.PRNGKey(18), state.v.data.shape, dtype=jnp.float64,
        )
        # Also add eta perturbation
        eta_pert = 0.01 * jax.random.normal(
            jax.random.PRNGKey(19), state.eta.data.shape, dtype=jnp.float64,
        )
        state = state._replace(
            u=state.u.replace(data=state.u.data + u_pert),
            v=state.v.replace(data=state.v.data + v_pert),
            eta=state.eta.replace(data=state.eta.data + eta_pert),
        )

        # Must use differentiable_barotropic=True to use lax.scan
        config = LatLonCGridOceanConfig.from_flat(
            A_h=1000.0,
            K_h=0.0,
            A_v=0.0,
            K_v=0.0,
            bottom_drag_r=1e-3,
            n_barotropic_substeps=5,
            differentiable_barotropic=True,
            barotropic_diffusion_alpha=0.01,
            enable_runtime_checks=False,
        )

        return grid, z_coord, state, config, barotropic_substeps_latlon_cgrid

    def test_grad_eta_wrt_eta(self, baro_setup):
        """Gradient of output eta w.r.t. input eta through barotropic loop."""
        grid, z_coord, state, config, baro_fn = baro_setup
        dt_baroclinic = 600.0
        dt_s = dt_baroclinic / config.n_barotropic_substeps

        def loss(eta_data):
            s = state._replace(eta=state.eta.replace(data=eta_data))
            s_out, _transport = baro_fn(
                s, dt_s, config.n_barotropic_substeps, grid, z_coord, config,
            )
            return jnp.sum(s_out.eta.data ** 2)

        grad = jax.grad(loss)(state.eta.data)
        assert jnp.all(jnp.isfinite(grad)), (
            f"Non-finite grad of eta through barotropic loop with bottom drag. "
            f"NaN: {int(jnp.sum(jnp.isnan(grad)))}"
        )
        nonzero_frac = float(jnp.mean(jnp.abs(grad) > 1e-30))
        assert nonzero_frac > 0.1, (
            f"Too few nonzero eta gradient entries: {nonzero_frac:.1%}"
        )

    def test_grad_u_wrt_u(self, baro_setup):
        """Gradient of output u w.r.t. input u through barotropic loop."""
        grid, z_coord, state, config, baro_fn = baro_setup
        dt_baroclinic = 600.0
        dt_s = dt_baroclinic / config.n_barotropic_substeps

        def loss(u_data):
            s = state._replace(u=state.u.replace(data=u_data))
            s_out, _transport = baro_fn(
                s, dt_s, config.n_barotropic_substeps, grid, z_coord, config,
            )
            return jnp.sum(s_out.u.data ** 2)

        grad = jax.grad(loss)(state.u.data)
        assert jnp.all(jnp.isfinite(grad)), (
            f"Non-finite grad of u through barotropic loop with bottom drag. "
            f"NaN: {int(jnp.sum(jnp.isnan(grad)))}"
        )
        nonzero_frac = float(jnp.mean(jnp.abs(grad) > 1e-30))
        assert nonzero_frac > 0.05, (
            f"Too few nonzero u gradient entries: {nonzero_frac:.1%}"
        )

    def test_bottom_drag_changes_barotropic_gradient(self, baro_setup):
        """Bottom drag in barotropic solver contributes to the gradient."""
        grid, z_coord, state, config, baro_fn = baro_setup
        dt_baroclinic = 600.0
        dt_s = dt_baroclinic / config.n_barotropic_substeps

        config_no_drag = config._replace(bottom_drag=config.bottom_drag._replace(bottom_drag_r=0.0))

        def loss(u_data, cfg):
            s = state._replace(u=state.u.replace(data=u_data))
            s_out, _ = baro_fn(
                s, dt_s, cfg.n_barotropic_substeps, grid, z_coord, cfg,
            )
            return jnp.sum(s_out.u.data ** 2)

        grad_drag = jax.grad(loss)(state.u.data, config)
        grad_no_drag = jax.grad(loss)(state.u.data, config_no_drag)
        diff = float(jnp.max(jnp.abs(grad_drag - grad_no_drag)))
        assert diff > 1e-15, (
            f"Bottom drag in barotropic solver has no effect on gradient. "
            f"Max diff = {diff:.2e}"
        )

    def test_fori_loop_barotropic_no_drag(self, baro_setup):
        """Verify fori_loop path (non-differentiable) runs without error.

        This is not a differentiability test — it just verifies that the
        fori_loop path with bottom drag doesn't crash. jax.grad through
        fori_loop would fail, so we only test forward pass.
        """
        grid, z_coord, state, config, baro_fn = baro_setup
        dt_baroclinic = 600.0
        config_fori = config._replace(differentiable_barotropic=False)
        dt_s = dt_baroclinic / config_fori.n_barotropic_substeps

        s_out, _transport = baro_fn(
            state, dt_s, config_fori.n_barotropic_substeps, grid, z_coord, config_fori,
        )
        assert jnp.all(jnp.isfinite(s_out.eta.data)), "Non-finite eta from fori_loop"


# ---------------------------------------------------------------------------
# Run script for standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    os.environ["JAX_ENABLE_X64"] = "1"

    print("=" * 70)
    print("Ocean Differentiability Tests — Recent Changes")
    print("=" * 70)
    print()

    all_pass = True

    # --- 1. MPAS Smagorinsky biharmonic ---
    print("[1] Smagorinsky biharmonic on MPAS")
    try:
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.core.operators_voronoi import smagorinsky_biharmonic_3d

        mesh = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=10)
        key = jax.random.PRNGKey(42)
        u_edge = 0.01 * jax.random.normal(
            key, (mesh.nEdges, 3), dtype=jnp.float64,
        )

        def loss_smag(u):
            return jnp.sum(smagorinsky_biharmonic_3d(u, mesh, 0.1))

        grad = jax.grad(loss_smag)(u_edge)
        ok = bool(jnp.all(jnp.isfinite(grad)))
        nz = float(jnp.mean(jnp.abs(grad) > 1e-30))
        if ok and nz > 0.1:
            print(f"  PASS  |grad|_max={float(jnp.max(jnp.abs(grad))):.4e}, "
                  f"nonzero={nz:.1%}")
        else:
            print(f"  FAIL  finite={ok}, nonzero_frac={nz:.1%}")
            all_pass = False
    except Exception as e:
        print(f"  ERROR  {type(e).__name__}: {e}")
        all_pass = False
    print()

    # --- 2. Bottom drag ---
    print("[2] Bottom drag on full velocity")
    try:
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        from legoesm.ocean.state import LatLonCGridOceanConfig
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            latlon_cgrid_ocean_baroclinic_tendencies,
        )

        grid = create_latlon_grid(8, 16)
        z_coord = create_ocean_z_star(n_levels=3, H_max=500.0)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord, H_max=500.0, land_lat_threshold=85.0,
        )
        cfg = LatLonCGridOceanConfig.from_flat(
            A_h=1000.0, bottom_drag_r=1e-3, A_v=0.0, K_v=0.0, K_h=0.0,
            enable_runtime_checks=False, n_barotropic_substeps=2,
        )
        k2 = jax.random.PRNGKey(7)
        u_pert = 0.05 * jax.random.normal(k2, state.u.data.shape, dtype=jnp.float64)
        state = state._replace(u=state.u.replace(data=state.u.data + u_pert))

        def loss_bd(u_data):
            s = state._replace(u=state.u.replace(data=u_data))
            t = latlon_cgrid_ocean_baroclinic_tendencies(s, grid, z_coord, cfg)
            return jnp.sum(t.du_dt.data ** 2)

        grad = jax.grad(loss_bd)(state.u.data)
        ok = bool(jnp.all(jnp.isfinite(grad)))
        nz = float(jnp.mean(jnp.abs(grad) > 1e-30))
        if ok and nz > 0.05:
            print(f"  PASS  |grad|_max={float(jnp.max(jnp.abs(grad))):.4e}, "
                  f"nonzero={nz:.1%}")
        else:
            print(f"  FAIL  finite={ok}, nonzero_frac={nz:.1%}")
            all_pass = False
    except Exception as e:
        print(f"  ERROR  {type(e).__name__}: {e}")
        all_pass = False
    print()

    # --- 3. Sponge relaxation ---
    print("[3] Sponge relaxation")
    try:
        from legoesm.ocean.sponge import SpongeForcing

        gamma = np.zeros((8, 16), dtype=np.float64)
        gamma[:2, :] = 1.0 / 86400.0
        gamma[-2:, :] = 1.0 / 86400.0
        gamma = jnp.array(gamma)
        T_ref = jnp.full((8, 16, 3), 10.0, dtype=jnp.float64)
        S_ref = jnp.full((8, 16, 3), 35.0, dtype=jnp.float64)
        sponge = SpongeForcing(gamma=gamma, T_ref=T_ref, S_ref=S_ref)

        # reuse grid/z_coord/state from test 2
        cfg_sp = LatLonCGridOceanConfig.from_flat(
            A_h=1000.0, bottom_drag_r=0.0, A_v=0.0, K_v=0.0, K_h=0.0,
            enable_runtime_checks=False, n_barotropic_substeps=2,
        )

        def loss_sp(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            t = latlon_cgrid_ocean_baroclinic_tendencies(
                s, grid, z_coord, cfg_sp, sponge=sponge,
            )
            return jnp.sum(t.dT_dt.data ** 2)

        grad = jax.grad(loss_sp)(state.T.data)
        ok = bool(jnp.all(jnp.isfinite(grad)))
        nz = float(jnp.mean(jnp.abs(grad) > 1e-30))
        if ok and nz > 0.01:
            print(f"  PASS  |grad|_max={float(jnp.max(jnp.abs(grad))):.4e}, "
                  f"nonzero={nz:.1%}")
        else:
            print(f"  FAIL  finite={ok}, nonzero_frac={nz:.1%}")
            all_pass = False
    except Exception as e:
        print(f"  ERROR  {type(e).__name__}: {e}")
        all_pass = False
    print()

    # --- 4. Barotropic solver with bottom drag ---
    print("[4] Barotropic solver with bottom drag")
    try:
        from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
            barotropic_substeps_latlon_cgrid,
        )

        grid = create_latlon_grid(8, 16)
        z_coord = create_ocean_z_star(n_levels=3, H_max=500.0)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord, H_max=500.0, land_lat_threshold=85.0,
        )
        k3 = jax.random.PRNGKey(17)
        u_pert = 0.02 * jax.random.normal(k3, state.u.data.shape, dtype=jnp.float64)
        v_pert = 0.02 * jax.random.normal(
            jax.random.PRNGKey(18), state.v.data.shape, dtype=jnp.float64,
        )
        eta_pert = 0.01 * jax.random.normal(
            jax.random.PRNGKey(19), state.eta.data.shape, dtype=jnp.float64,
        )
        state = state._replace(
            u=state.u.replace(data=state.u.data + u_pert),
            v=state.v.replace(data=state.v.data + v_pert),
            eta=state.eta.replace(data=state.eta.data + eta_pert),
        )
        cfg_bt = LatLonCGridOceanConfig.from_flat(
            A_h=1000.0, bottom_drag_r=1e-3, n_barotropic_substeps=5,
            differentiable_barotropic=True, barotropic_diffusion_alpha=0.01,
            enable_runtime_checks=False, A_v=0.0, K_v=0.0, K_h=0.0,
        )
        dt_s = 600.0 / 5

        def loss_bt(eta_data):
            s = state._replace(eta=state.eta.replace(data=eta_data))
            s_out, _ = barotropic_substeps_latlon_cgrid(
                s, dt_s, 5, grid, z_coord, cfg_bt,
            )
            return jnp.sum(s_out.eta.data ** 2)

        grad = jax.grad(loss_bt)(state.eta.data)
        ok = bool(jnp.all(jnp.isfinite(grad)))
        nz = float(jnp.mean(jnp.abs(grad) > 1e-30))
        if ok and nz > 0.1:
            print(f"  PASS  |grad|_max={float(jnp.max(jnp.abs(grad))):.4e}, "
                  f"nonzero={nz:.1%}")
        else:
            print(f"  FAIL  finite={ok}, nonzero_frac={nz:.1%}")
            all_pass = False
    except Exception as e:
        print(f"  ERROR  {type(e).__name__}: {e}")
        all_pass = False
    print()

    print("=" * 70)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
