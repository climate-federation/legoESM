"""FV3_3D iter 463: parametric sweep of ``rf_tau_days``
(Rayleigh friction timescale) to find calibration that
reduces cube-edge artifact ratio.

iter-462 showed corner-div sponge boost (``d2_bg_k1``) is
INSENSITIVE at C8.  Try a different mechanism: column
Rayleigh friction at model top.

Sweep ``rf_tau_days`` in {0 (off), 1, 5, 10, 30} days at C8
(3 steps, 3 seeds) and report mean edge_var/interior_var
ratio for NH + PE.

Tests
-----

1. ``test_nh_rf_tau_sweep``.
2. ``test_pe_rf_tau_sweep``.
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


def test_nh_rf_tau_sweep(capsys):
    seeds = [463, 464, 465]
    sweep_values = [0.0, 1.0, 5.0, 30.0]
    results = {}
    for tau in sweep_values:
        ratios = []
        for seed in seeds:
            grid, hc, tm, state = _build_nh_state(8, seed=seed)
            cfg = make_fv3_faithful_nh_config(
                n_acoustic_substeps=4,
                damp_v=0.030, damp_v_d_con=1.0,
                corner_div_damp_d2_bg=5e-4,
                corner_div_damp_d_con=1.0,
                div_damp_coeff=1e6, div_damp_d_con=1.0,
                damp_w=0.030, damp_w_d_con=1.0,
                rf_tau_days=tau,
                rf_cutoff_pa=1.0e5,  # high cutoff → most levels damped
            )
            m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
            s = state
            for _ in range(3):
                s = m.step(s, dt=10.0)
            ratios.append(_edge_interior_ratio(s.v.data))
        results[tau] = float(np.mean(ratios))
    with capsys.disabled():
        print(f"\n[iter-463 NH rf_tau_days sweep, 3 seeds @ C8, 3 steps]")
        for tau, r in results.items():
            print(f"  rf_tau_days = {tau:>5.1f}: mean ratio = {r:.4f}")
    assert all(np.isfinite(r) for r in results.values())


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


def test_pe_rf_tau_sweep(capsys):
    seeds = [463, 464, 465]
    sweep_values = [0.0, 1.0, 5.0, 30.0]
    results = {}
    for tau in sweep_values:
        ratios = []
        for seed in seeds:
            grid, coord, state = _build_pe_state(8, seed=seed)
            cfg = make_fv3_faithful_pe_config(
                damp_v=0.030, damp_v_d_con=1.0,
                corner_div_damp_d2_bg=5e-4,
                corner_div_damp_d_con=1.0,
                div_damp_coeff=1e6, div_damp_d_con=1.0,
                rf_tau_days=tau,
                rf_cutoff_pa=1.0e5,
            )
            m = CDGridPrimitiveEquationModel(grid, coord, cfg)
            s = state
            for _ in range(3):
                s = m.step(s, dt=10.0)
            ratios.append(_edge_interior_ratio(s.v_d.data))
        results[tau] = float(np.mean(ratios))
    with capsys.disabled():
        print(f"\n[iter-463 PE rf_tau_days sweep, 3 seeds @ C8, 3 steps]")
        for tau, r in results.items():
            print(f"  rf_tau_days = {tau:>5.1f}: mean ratio = {r:.4f}")
    assert all(np.isfinite(r) for r in results.values())
