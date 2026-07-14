"""FV3_3D iter 563: heat_source_del2_coeff sweep.

iter-562 found ``heat_source_del2_iters=8`` gives -89% edge_std
at default coeff=0.20.  Sweep coeff for additional optimization.

Tests
-----

1. ``test_heat_source_del2_coeff_sweep`` — coeff ∈ {0.05,
   0.10, 0.20, 0.40, 0.80} at iters=2 (FV3 default).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerModel,
    make_legoesm_nh_min_edge_config,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import (
    create_cubed_sphere,
    rotate_winds_geo_to_grid,
)
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
)


def _edge_std(field_data):
    n_face, n_x, n_y, n_lev = field_data.shape
    edge_mask = np.zeros((n_x, n_y), dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    edge_mask_b = np.broadcast_to(
        edge_mask[None, :, :, None], field_data.shape,
    )
    return float(np.asarray(field_data)[edge_mask_b].std())


def _build_sbr_state(n=16, U_0=20.0):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    lat = grid.lat
    u_east = U_0 * jnp.cos(lat)
    v_north = jnp.zeros_like(lat)
    u_grid, v_grid = rotate_winds_geo_to_grid(
        u_east, v_north, grid.angle,
    )
    u_p = jnp.broadcast_to(
        u_grid[..., None], (6, n, n, nlev),
    ).astype(jnp.float64)
    v_p = jnp.broadcast_to(
        v_grid[..., None], (6, n, n, nlev),
    ).astype(jnp.float64)
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=u_p, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v_p, name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1), dtype=jnp.float64),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float64),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev), dtype=jnp.float64),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n), dtype=jnp.float64),
                   name="phis", dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0), dtype=jnp.float64),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, hc, tm, state


def test_heat_source_del2_coeff_sweep(capsys):
    grid, hc, tm, state = _build_sbr_state(n=16)
    base_kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
        heat_source_del2_iters=2,
    )
    coeffs = [0.05, 0.10, 0.20, 0.40, 0.80]
    results = []
    for coeff in coeffs:
        kw = {**base_kw, "heat_source_del2_coeff": coeff}
        cfg = make_legoesm_nh_min_edge_config(**kw)
        m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
        step = jax.jit(m.step)
        s = state
        for _ in range(10):
            s = step(s, 10.0)
        e = _edge_std(s.theta_prime.data)
        results.append((coeff, e))
    with capsys.disabled():
        print(
            f"\n[iter-563 heat_source_del2_coeff sweep "
            f"(iters=2) @ C16 SBR, 10 steps]"
        )
        for coeff, e in results:
            print(f"  coeff={coeff:.2f}: edge_std={e:.3e}")
        finite = [(c, e) for c, e in results if np.isfinite(e)]
        if finite:
            best = min(finite, key=lambda x: x[1])
            print(f"\n  Best: coeff={best[0]:.2f} → edge_std={best[1]:.3e}")
    assert all(np.isfinite(e) for _, e in results)
