"""FV3_3D iter 464: parametric sweep of
``heat_source_del2_iters`` (iter-457/458 del-2 smoothing) to
test whether spatial smoothing of the d_con heat source
affects cube-edge artifact metric.

Hypothesis (per iter-463 insight): per-level scaling can't
move the edge metric.  ``heat_source_del2_iters`` is a
genuine SPATIAL Laplacian smoothing of θ-tendency per level
— THIS should change the spatial distribution within levels
if any of our knobs does.

Sweep iters in {0, 1, 2, 4, 8} at C8 (3 steps, 3 seeds) for
NH + PE.

Tests
-----

1. ``test_nh_heat_source_del2_iters_sweep``.
2. ``test_pe_heat_source_del2_iters_sweep``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerModel,
    make_fv3_faithful_nh_config,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationModel,
    hydrostatic_to_fv3,
    make_fv3_faithful_pe_config,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
    standard_hybrid_levels,
)


def _edge_interior_ratio(field_data: jnp.ndarray) -> float:
    n_face, n_x, n_y, n_lev = field_data.shape
    edge_mask = np.zeros((n_x, n_y), dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    edge_mask_b = np.broadcast_to(
        edge_mask[None, :, :, None], field_data.shape,
    )
    interior_mask = ~edge_mask_b
    arr = np.asarray(field_data)
    if arr[interior_mask].std() == 0.0:
        return 0.0
    return float(arr[edge_mask_b].std() / arr[interior_mask].std())


def _build_nh_state(n, seed):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=False)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    rng = np.random.default_rng(seed=seed)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)), name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, hc, tm, state


def test_nh_heat_source_del2_iters_sweep(capsys):
    seeds = [464, 465, 466]
    sweep_values = [0, 1, 2, 4, 8]
    results_v = {}
    results_theta = {}
    for iters in sweep_values:
        ratios_v = []
        ratios_theta = []
        for seed in seeds:
            grid, hc, tm, state = _build_nh_state(8, seed=seed)
            cfg = make_fv3_faithful_nh_config(
                n_acoustic_substeps=4,
                damp_v=0.030, damp_v_d_con=1.0,
                corner_div_damp_d2_bg=5e-4,
                corner_div_damp_d_con=1.0,
                div_damp_coeff=1e6, div_damp_d_con=1.0,
                damp_w=0.030, damp_w_d_con=1.0,
                heat_source_del2_iters=iters,
            )
            m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
            s = state
            for _ in range(3):
                s = m.step(s, dt=10.0)
            ratios_v.append(_edge_interior_ratio(s.v.data))
            ratios_theta.append(_edge_interior_ratio(s.theta_prime.data))
        results_v[iters] = float(np.mean(ratios_v))
        results_theta[iters] = float(np.mean(ratios_theta))
    with capsys.disabled():
        print(
            f"\n[iter-464 NH heat_source_del2_iters sweep, "
            f"3 seeds @ C8, 3 steps]"
        )
        print(f"  v field:")
        for n_it, r in results_v.items():
            print(f"    iters = {n_it}: mean ratio = {r:.4f}")
        print(f"  theta_prime field:")
        for n_it, r in results_theta.items():
            print(f"    iters = {n_it}: mean ratio = {r:.4f}")
    assert all(np.isfinite(r) for r in results_v.values())
    assert all(np.isfinite(r) for r in results_theta.values())


def _build_pe_state(n, seed):
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state = hydrostatic_to_fv3(held_suarez_init(grid, coord), cdgrid)
    rng = np.random.default_rng(seed=seed)
    n_corners = n + 1
    u_p = rng.uniform(-5.0, 5.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-5.0, 5.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )
    return grid, coord, state


def test_pe_heat_source_del2_iters_sweep(capsys):
    seeds = [464, 465, 466]
    sweep_values = [0, 1, 2, 4, 8]
    results_v = {}
    results_T = {}
    for iters in sweep_values:
        ratios_v = []
        ratios_T = []
        for seed in seeds:
            grid, coord, state = _build_pe_state(8, seed=seed)
            cfg = make_fv3_faithful_pe_config(
                damp_v=0.030, damp_v_d_con=1.0,
                corner_div_damp_d2_bg=5e-4,
                corner_div_damp_d_con=1.0,
                div_damp_coeff=1e6, div_damp_d_con=1.0,
                heat_source_del2_iters=iters,
            )
            m = CDGridPrimitiveEquationModel(grid, coord, cfg)
            s = state
            for _ in range(3):
                s = m.step(s, dt=10.0)
            ratios_v.append(_edge_interior_ratio(s.v_d.data))
            ratios_T.append(_edge_interior_ratio(s.T.data))
        results_v[iters] = float(np.mean(ratios_v))
        results_T[iters] = float(np.mean(ratios_T))
    with capsys.disabled():
        print(
            f"\n[iter-464 PE heat_source_del2_iters sweep, "
            f"3 seeds @ C8, 3 steps]"
        )
        print(f"  v_d field:")
        for n_it, r in results_v.items():
            print(f"    iters = {n_it}: mean ratio = {r:.4f}")
        print(f"  T field:")
        for n_it, r in results_T.items():
            print(f"    iters = {n_it}: mean ratio = {r:.4f}")
    assert all(np.isfinite(r) for r in results_v.values())
    assert all(np.isfinite(r) for r in results_T.values())
