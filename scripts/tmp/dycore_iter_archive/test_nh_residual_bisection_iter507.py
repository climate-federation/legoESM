"""FV3_3D iter 507: bisect the 1.13× NH residual under min-edge + clip.

iter-504 showed min-edge factory + monotone_halo_clip_context
brings NH edge ratio to ~1.135× (65.7% reduction).  This
test identifies which FV3-fidelity flag still ON in
``make_legoesm_nh_min_edge_config`` drives the residual 13%
leak: bisect by toggling one flag OFF at a time on top of
the iter-504 stack.

Tests
-----

1. ``test_per_flag_residual_drop`` — for each of the 4 still-
   on flags, measure edge ratio with that flag forced OFF
   under clip context; print delta vs baseline.
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
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import monotone_halo_clip_context
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
)


def _edge_and_interior_std(field_data):
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
    return (
        float(arr[edge_mask_b].std()),
        float(arr[interior_mask].std()),
    )


def _build_nh_state(n, seed, use_duogrid):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=use_duogrid)
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


def _measure_ratio(seeds, extra_overrides):
    base_kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    base_kw.update(extra_overrides)
    ratios = []
    for seed in seeds:
        grid_on, hc, tm, state_on = _build_nh_state(8, seed, True)
        grid_off, _, _, state_off = _build_nh_state(8, seed, False)
        cfg = make_legoesm_nh_min_edge_config(**base_kw)
        with monotone_halo_clip_context(slack=0.5):
            m_on = CDGridCompressibleEulerModel(grid_on, hc, tm, cfg)
            m_off = CDGridCompressibleEulerModel(grid_off, hc, tm, cfg)
            s_on = m_on.step(state_on, dt=10.0)
            s_off = m_off.step(state_off, dt=10.0)
        e_on, _ = _edge_and_interior_std(s_on.theta_prime.data)
        e_off, _ = _edge_and_interior_std(s_off.theta_prime.data)
        if e_off > 1e-30:
            ratios.append(e_on / e_off)
    return float(np.mean(ratios)) if ratios else float("nan")


def test_per_flag_residual_drop(capsys):
    seeds = [507, 508]

    flag_variants = [
        ("baseline (all flags on)", {}),
        ("use_fv3_d_con_cv=False", {"use_fv3_d_con_cv": False}),
        ("use_fv3_vector_halo_uv=False", {"use_fv3_vector_halo_uv": False}),
        ("use_fv3_dynamic_exner=False", {"use_fv3_dynamic_exner": False}),
        ("use_fv3_cross_face_du_proj=False",
         {"use_fv3_cross_face_du_proj": False}),
        ("damp_w=0.0", {"damp_w": 0.0, "damp_w_d_con": 0.0}),
        ("damp_v=0.0", {"damp_v": 0.0, "damp_v_d_con": 0.0}),
        ("corner_div_damp=0.0", {"corner_div_damp_d2_bg": 0.0,
                                  "corner_div_damp_d_con": 0.0}),
    ]
    results = []
    for name, overrides in flag_variants:
        r = _measure_ratio(seeds, overrides)
        results.append((name, r))
    baseline_r = results[0][1]
    with capsys.disabled():
        print(
            f"\n[iter-507 NH residual bisection under min-edge + clip "
            f"(2 seeds, C8, 1 step)]"
        )
        print(f"  {'config':40s}: {'ratio':>8s}  Δ vs baseline")
        for name, r in results:
            delta = r - baseline_r
            print(f"  {name:40s}: {r:7.3f}×  ({delta:+.3f})")
        sorted_results = sorted(results[1:], key=lambda x: x[1])
        best_name, best_r = sorted_results[0]
        print(
            f"\n  Lowest ratio:  {best_name} → {best_r:.3f}× "
            f"(Δ {best_r - baseline_r:+.3f})"
        )
    assert all(np.isfinite(r) for _, r in results)
