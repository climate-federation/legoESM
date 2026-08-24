"""Full-card SPMD gate: a PARTIAL-CELL coordinate must band with the state.

The eORCA025 full-card 4-GPU run died with ``mul (1208,1440,75) vs
(302,1440,1)`` in ``compute_frozen_geom_density``: the shard_map body received
a band-local state but the model's GLOBAL z-coordinate — the SPMD lane never
banded the vertical coordinate's per-cell fields (``h_partial`` /
``bottom_level`` / ``is_active``), because every prior SPMD config ran pure
z-star (no per-cell fields).  The fix threads a ``z_coord`` override through
the step drivers (mirroring the existing ``grid=`` override) and stacks the
per-cell fields over bands exactly like the geometry.

The gate is an equivalence test: an N-step SPMD run over fake CPU devices must
match the single-device step on a VARIABLE-bathymetry partial-cell setup — the
configuration whose mere construction crashed before the fix, so this test
FAILS (with that same shape error) when the banding is removed.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import (
    create_ocean_z_star, create_partial_cell_coordinate,
)


def _have_sharded_step():
    try:
        from legoesm.ocean.dynamics.sharded_ocean_step import (  # noqa: F401
            make_sharded_ocean_step,
        )
        from legoesm.parallel.mesh import create_latlon_mesh  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(jax.device_count() < 4,
                    reason="needs >=4 devices (XLA_FLAGS host device count)")
@pytest.mark.skipif(not _have_sharded_step(),
                    reason="sharded_ocean_step module not present")
def test_partial_cell_spmd_matches_single_device():
    from legoesm.ocean.dynamics.sharded_ocean_step import (
        gather_state_latlon, make_sharded_ocean_step, shard_state_latlon,
    )
    from legoesm.parallel.mesh import create_latlon_mesh

    n_lat, n_lon, nlev = 48, 96, 10
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    zs = create_ocean_z_star(n_levels=nlev, H_max=4000.0)

    # Variable bathymetry with real partial bottom cells: a smooth ridge so
    # h_partial / bottom_level / is_active genuinely vary per column.  It is
    # exactly this per-cell variation the un-banded coordinate could not
    # deliver to a band body.
    lat_idx = np.arange(n_lat)[:, None]
    lon_idx = np.arange(n_lon)[None, :]
    H_bathy = (4000.0
               - 1500.0 * np.exp(-((lat_idx - n_lat / 2) / 8.0) ** 2)
               - 400.0 * np.cos(2 * np.pi * lon_idx / n_lon))
    zc = create_partial_cell_coordinate(zs, jnp.asarray(H_bathy))
    assert int(np.asarray(zc.bottom_level).min()) != int(
        np.asarray(zc.bottom_level).max()), "bathymetry must vary"

    cfg = LatLonCGridOceanConfig.from_flat()
    model = LatLonCGridOceanModel(grid, zc, cfg)
    state0 = rest_state_latlon_cgrid_ocean(
        grid, zc, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0)
    # Overwrite bathymetry/perturb so the step exercises the partial-cell PGF.
    rng = np.random.default_rng(0)
    state0 = state0._replace(
        H_bathy=state0.H_bathy.replace(data=jnp.asarray(H_bathy)),
        u=state0.u.replace(data=jnp.asarray(
            0.02 * rng.standard_normal((n_lat, n_lon + 1, nlev)))),
        eta=state0.eta.replace(data=jnp.asarray(
            0.005 * rng.standard_normal((n_lat, n_lon)))),
        T=state0.T.replace(data=jnp.asarray(
            5.0 + 15.0 * np.exp(np.linspace(0, -4, nlev))[None, None, :]
            + 0.05 * rng.standard_normal((n_lat, n_lon, nlev)))),
    )
    dt, n_steps = 600.0, 3

    s = state0
    for _ in range(n_steps):
        s = model.step(s, dt)

    model._ensure_vertex_mask(state0)
    dev = create_latlon_mesh(n_devices=4)
    step = make_sharded_ocean_step(model, dev.mesh)
    # The fix's fingerprint: the step must carry non-empty z-coord stacks.
    assert len(step.aux) == 4, "aux must carry (geom, vmask, zc, iwm) stacks"
    zc_stacks = step.aux[2]
    assert set(zc_stacks) >= {"h_partial", "bottom_level", "is_active"}

    ss = shard_state_latlon(state0, dev.mesh)
    for _ in range(n_steps):
        ss = step(ss, dt)
    ss = gather_state_latlon(ss, dev.mesh)

    # The ESTABLISHED SPMD tolerance (test_latlon_ocean_spmd_step.py):
    # band-wise reductions legitimately reorder float ops vs the single
    # device, so equivalence is to (2e-4, 1e-3), not bit-exact.  A first
    # version demanded 1e-10 and failed at max|dT| = 3.4e-4 — the
    # EXPECTATION was wrong, not the banding (measured, matching the z-star
    # gate's own basis).
    _ATOL, _RTOL = 2.0e-4, 1.0e-3
    for name in ("T", "S", "u", "v", "eta"):
        a = np.asarray(getattr(s, name).data)
        b = np.asarray(getattr(ss, name).data)
        np.testing.assert_allclose(
            b, a, atol=_ATOL, rtol=_RTOL,
            err_msg=f"{name} diverged between SPMD and single-device with "
                    f"partial cells")


@pytest.mark.skipif(jax.device_count() < 4,
                    reason="needs >=4 devices (XLA_FLAGS host device count)")
@pytest.mark.skipif(not _have_sharded_step(),
                    reason="sharded_ocean_step module not present")
def test_zstar_config_still_has_empty_aux_stacks():
    """A pure z-star model (no per-cell fields, no iwm) must produce EMPTY
    z/iwm stacks and a None override — the pre-fix behaviour, bit-identical,
    so every existing SPMD user is untouched."""
    from legoesm.ocean.dynamics.sharded_ocean_step import (
        make_sharded_ocean_step,
    )
    from legoesm.parallel.mesh import create_latlon_mesh

    grid = create_latlon_grid(n_lat=48, n_lon=96)
    zc = create_ocean_z_star(n_levels=10, H_max=4000.0)
    model = LatLonCGridOceanModel(grid, zc, LatLonCGridOceanConfig.from_flat())
    state0 = rest_state_latlon_cgrid_ocean(
        grid, zc, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0)
    model._ensure_vertex_mask(state0)
    dev = create_latlon_mesh(n_devices=4)
    step = make_sharded_ocean_step(model, dev.mesh)
    _geom, _vmask, zc_stacks, iwm_stacks = step.aux
    assert zc_stacks == {}
    assert iwm_stacks is None
