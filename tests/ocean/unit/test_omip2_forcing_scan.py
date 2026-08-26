"""lax.scan-traceable CORE-II forcing kernel for the tripole OMIP loop
(issue #354).

The pure-JAX ``compute_omip2_surface_forcing_jax`` must reproduce the host
``compute_omip2_surface_forcing`` to floating-point tolerance (identical
nearest-neighbour spatial sample, air-sea fluxes, and net-heat formula),
and must be ``jax.jit`` / ``lax.scan`` traceable with a traced record index.
"""

from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.ocean.coupler.omip2_applicator import (
    compute_omip2_surface_forcing,
    compute_omip2_surface_forcing_jax,
    build_core2_forcing_device_stack,
    core2_forcing_nn_indices,
)


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    n_rec, n_lat_f, n_lon_f = 4, 8, 12
    # Forcing grid coords in DEGREES (as the host NN sampler expects).
    forc = types.SimpleNamespace(
        lat=np.linspace(-80.0, 80.0, n_lat_f),
        lon=np.linspace(0.0, 330.0, n_lon_f),
        u10=rng.uniform(-8, 8, (n_rec, n_lat_f, n_lon_f)),
        v10=rng.uniform(-8, 8, (n_rec, n_lat_f, n_lon_f)),
        T_air=rng.uniform(275, 300, (n_rec, n_lat_f, n_lon_f)),
        q_air=rng.uniform(3e-3, 1.5e-2, (n_rec, n_lat_f, n_lon_f)),
        sw_down=rng.uniform(50, 300, (n_rec, n_lat_f, n_lon_f)),
        lw_down=rng.uniform(280, 400, (n_rec, n_lat_f, n_lon_f)),
        precip=rng.uniform(0, 1e-4, (n_rec, n_lat_f, n_lon_f)),
    )
    # Model tripole grid: 2-D lat_T/lon_T in RADIANS.
    n_lat, n_lon, nlev = 5, 7, 3
    lat2d = np.deg2rad(rng.uniform(-70, 70, (n_lat, n_lon)))
    lon2d = np.deg2rad(rng.uniform(0, 360, (n_lat, n_lon)))
    grid = types.SimpleNamespace(lat_T=jnp.asarray(lat2d), lon_T=jnp.asarray(lon2d))
    T = jnp.asarray(rng.uniform(0.0, 25.0, (n_lat, n_lon, nlev)))
    state = types.SimpleNamespace(T=Field(data=T, name="T"))
    return forc, grid, state


def test_nn_indices_match_host_sampler():
    """The extracted (i, j) reproduce the host _nn_interp sample exactly."""
    from legoesm.ocean.coupler.omip2_applicator import _nn_interp_to_points
    forc, grid, _ = _synthetic(1)
    lat_pts = np.degrees(np.asarray(grid.lat_T)).reshape(-1)
    lon_pts = np.degrees(np.asarray(grid.lon_T)).reshape(-1)
    i, j = core2_forcing_nn_indices(forc, lat_pts, lon_pts)
    host = _nn_interp_to_points(forc.u10[2], forc.lat, forc.lon, lat_pts, lon_pts)
    np.testing.assert_allclose(np.asarray(forc.u10[2])[i, j], host, rtol=0, atol=0)


def test_jax_kernel_matches_host():
    forc, grid, state = _synthetic(2)
    idx = 2
    sf_host = compute_omip2_surface_forcing(
        state, forcing=forc, idx_t=idx, grid=grid, grid_type="tripole")
    stack, nn_i, nn_j, nn_w, gshape = build_core2_forcing_device_stack(
        forc, grid, "tripole")
    sf_jax = compute_omip2_surface_forcing_jax(
        state, forcing_stack=stack, nn_i=nn_i, nn_j=nn_j, nn_w=nn_w,
        grid_shape=gshape, idx_t=idx)
    for f in ("tau_x", "tau_y", "q_net", "sw_down"):
        np.testing.assert_allclose(
            np.asarray(getattr(sf_jax, f)), np.asarray(getattr(sf_host, f)),
            rtol=1e-9, atol=1e-9, err_msg=f"{f} kernel != host")


def test_kernel_is_jit_traceable_with_traced_index():
    """The record index may be a traced JAX int (required for lax.scan)."""
    forc, grid, state = _synthetic(3)
    stack, nn_i, nn_j, nn_w, gshape = build_core2_forcing_device_stack(
        forc, grid, "tripole")

    @jax.jit
    def go(it):
        sf = compute_omip2_surface_forcing_jax(
            state, forcing_stack=stack, nn_i=nn_i, nn_j=nn_j, nn_w=nn_w,
            grid_shape=gshape, idx_t=it)
        return sf.q_net

    out = go(jnp.int32(1))
    assert out.shape == (5, 7)
    assert bool(jnp.all(jnp.isfinite(out)))
    # Different record -> different forcing (index is actually used).
    assert not bool(jnp.allclose(out, go(jnp.int32(3))))


def test_non_tripole_raises():
    forc, grid, _ = _synthetic(4)
    with pytest.raises(NotImplementedError, match="tripole"):
        build_core2_forcing_device_stack(forc, grid, "latlon")


def test_scan_block_equals_manual_steps():
    """End-to-end: build_omip2_scan_block_fn over N steps == N manual
    (kernel -> model._step_impl) steps on a small lat-lon C-grid model."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.coupler.omip2_applicator import (
        build_omip2_scan_block_fn,
    )

    n_lat, n_lon, nlev = 12, 24, 4
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=18.0, T_deep=2.0,
        S_uniform=35.0, H_max=4000.0,
    )
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=1e4, B_h=0.0, K_h=0.0, K_bih=0.0,
        bottom_drag_r=0.0, n_barotropic_substeps=3,
        barotropic_solver="explicit_substep",
    )
    model = LatLonCGridOceanModel(grid, z_coord, cfg)

    # Synthetic forcing on the MODEL grid (identity NN gather).
    rng = np.random.default_rng(7)
    n_rec = 5
    stack = {
        name: jnp.asarray(rng.uniform(lo, hi, (n_rec, n_lat, n_lon)))
        for name, (lo, hi) in {
            "u10": (-6, 6), "v10": (-6, 6), "T_air": (278, 298),
            "q_air": (4e-3, 1.2e-2), "sw_down": (50, 250),
            "lw_down": (290, 390),
            # NEMO-parity channels: the scan kernel samples these
            # unconditionally from the device stack (q_ns heat-content).
            "precip": (0, 1e-4), "snow": (0, 2e-5),
            "slp": (99000.0, 103000.0),
        }.items()
    }
    nn_i = jnp.asarray(np.repeat(np.arange(n_lat), n_lon).astype(np.int32))
    nn_j = jnp.asarray(np.tile(np.arange(n_lon), n_lat).astype(np.int32))
    gshape = (n_lat, n_lon)
    dt = 300.0
    idx = [1, 3]

    block_fn = build_omip2_scan_block_fn(model, dt, gshape)
    s_scan = block_fn(
        state, stack, nn_i, nn_j, None, jnp.asarray(idx, dtype=jnp.int32),
        jnp.int32(1))

    s = state
    for it in idx:
        sf = compute_omip2_surface_forcing_jax(
            s, forcing_stack=stack, nn_i=nn_i, nn_j=nn_j,
            grid_shape=gshape, idx_t=it)
        s = model._step_impl(s, dt, surface_forcing=sf)

    # The jitted lax.scan fuses/reorders ops differently from the eager
    # manual loop, so on the float32 C-grid model they agree only to
    # ~float32 epsilon (~1e-7).  A real logic bug (wrong forcing index,
    # missed step, dropped term) would be O(0.1)+, far above this.
    for f in ("T", "S", "u", "v", "eta"):
        np.testing.assert_allclose(
            np.asarray(getattr(s_scan, f).data),
            np.asarray(getattr(s, f).data),
            rtol=1e-5, atol=1e-5, err_msg=f"scan {f} != manual {f}")
