"""FV3_3D iter 461: empirical comparison of the iter-456
edge_var/interior_var metric across configurations.

Quantifies whether the iter-431..460 FV3-faithful stack
reduces the cube-edge artifact signature vs the default config.

Methodology:
* C8 grid (only 6 of 8 cells are interior per dim → small but
  comparable edge/interior populations).
* Multiple seeds (5) → average ratio.
* 3 steps (longer than iter-456's 2 → more time for edge
  artifacts to develop).
* 4 configurations:
    A. NH default (no FV3 flags)
    B. NH factory (all FV3 defaults from iter-431..459) +
       d2_bg_k1=1e-4 (legoESM-scale)
    C. PE default
    D. PE factory + d2_bg_k1=1e-4

Test asserts the MEAN ratio over seeds, for diagnostic
purposes — does NOT gate on factory-beats-default because:
* Smaller damping in default may produce LESS spatial
  structure → smaller ratio even though artifacts are
  qualitatively worse.
* Edge ratio is just one proxy for visual artifact.

The MEAN ratio values are returned via pytest output so a
user can observe the empirical signature.

Tests
-----

1. ``test_nh_edge_metric_default_vs_factory`` — record both
   mean ratios; assert both finite + bounded.
2. ``test_pe_edge_metric_default_vs_factory`` — same for PE.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    CDGridCompressibleEulerModel,
    make_fv3_faithful_nh_config,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
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
    edge_mask_b = edge_mask[None, :, :, None]
    edge_mask_b = np.broadcast_to(edge_mask_b, field_data.shape)
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


def _run_nh_steps(grid, hc, tm, state, cfg, n_steps):
    m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    s = state
    for _ in range(n_steps):
        s = m.step(s, dt=10.0)
    return s


def test_nh_edge_metric_default_vs_factory(capsys):
    seeds = [461, 462, 463]
    cfg_default = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    factory_kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d2_bg_k1=1e-4,   # legoESM-scale
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    default_ratios = []
    factory_ratios = []
    for seed in seeds:
        grid, hc, tm, state = _build_nh_state(8, seed=seed)
        s_def = _run_nh_steps(grid, hc, tm, state, cfg_default, 3)
        s_fac = _run_nh_steps(
            grid, hc, tm, state,
            make_fv3_faithful_nh_config(**factory_kw), 3,
        )
        default_ratios.append(_edge_interior_ratio(s_def.v.data))
        factory_ratios.append(_edge_interior_ratio(s_fac.v.data))
    mean_def = float(np.mean(default_ratios))
    mean_fac = float(np.mean(factory_ratios))
    with capsys.disabled():
        print(
            f"\n[iter-461 NH edge metric, 3 seeds @ C8, 3 steps]"
            f"\n  default mean ratio: {mean_def:.4f}"
            f"\n  factory mean ratio: {mean_fac:.4f}"
            f"\n  factory/default:    {mean_fac/mean_def:.4f}"
        )
    assert np.isfinite(mean_def) and mean_def > 0.0
    assert np.isfinite(mean_fac) and mean_fac > 0.0


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


def _run_pe_steps(grid, coord, state, cfg, n_steps):
    m = CDGridPrimitiveEquationModel(grid, coord, cfg)
    s = state
    for _ in range(n_steps):
        s = m.step(s, dt=10.0)
    return s


def test_pe_edge_metric_default_vs_factory(capsys):
    seeds = [461, 462, 463]
    cfg_default = CDGridPrimitiveEquationConfig(
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
    )
    factory_kw = dict(
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d2_bg_k1=1e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
    )
    default_ratios = []
    factory_ratios = []
    for seed in seeds:
        grid, coord, state = _build_pe_state(8, seed=seed)
        s_def = _run_pe_steps(grid, coord, state, cfg_default, 3)
        s_fac = _run_pe_steps(
            grid, coord, state,
            make_fv3_faithful_pe_config(**factory_kw), 3,
        )
        default_ratios.append(_edge_interior_ratio(s_def.v_d.data))
        factory_ratios.append(_edge_interior_ratio(s_fac.v_d.data))
    mean_def = float(np.mean(default_ratios))
    mean_fac = float(np.mean(factory_ratios))
    with capsys.disabled():
        print(
            f"\n[iter-461 PE edge metric, 3 seeds @ C8, 3 steps]"
            f"\n  default mean ratio: {mean_def:.4f}"
            f"\n  factory mean ratio: {mean_fac:.4f}"
            f"\n  factory/default:    {mean_fac/mean_def:.4f}"
        )
    assert np.isfinite(mean_def) and mean_def > 0.0
    assert np.isfinite(mean_fac) and mean_fac > 0.0
