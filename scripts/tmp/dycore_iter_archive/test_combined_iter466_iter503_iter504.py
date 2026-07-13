"""FV3_3D iter 504: combine iter-466 (flag drops) + iter-503
(halo clip) — does it compose to a new low?

iter-466 alone: 50% reduction (4.54 → 2.25 at C8+duogrid)
iter-482 + d2_bg boost: 60% reduction (3.50 → 1.42)
iter-503 alone: 46% reduction (3.72 → 2.006)

Question: does iter-466+iter-503 yield additional reduction
beyond either alone, possibly approaching the floor?

Tests
-----

1. ``test_combined_flag_drops_and_halo_clip``.
"""
from __future__ import annotations

from unittest.mock import patch
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerModel,
    make_fv3_faithful_nh_config,
    make_legoesm_nh_min_edge_config,
    make_legoesm_nh_min_edge_aggressive_config,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import (
    pad_halo_4d as _real_pad_halo_4d,
    pad_halo_vector_4d as _real_pad_halo_vec,
)
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


def test_combined_flag_drops_and_halo_clip(capsys):
    seeds = [504, 505]
    base_kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    patch_modules = [
        "legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid._pad_halo_4d_module",
        "legoesm.core.operators_3d.pad_halo_4d",
        "legoesm.core.operators_cdgrid.pad_halo_4d",
        "legoesm.core.operators_fc.pad_halo_4d",
    ]
    vec_patch_modules = [
        "legoesm.core.operators_cdgrid.pad_halo_vector_4d",
        "legoesm.core.operators_3d.pad_halo_vector_4d",
    ]
    configs = [
        ("A. factory full",         make_fv3_faithful_nh_config, False),
        ("B. min-edge (iter-466)",  make_legoesm_nh_min_edge_config, False),
        ("C. aggressive (iter-483)", make_legoesm_nh_min_edge_aggressive_config, False),
        ("D. factory + clip",       make_fv3_faithful_nh_config, True),
        ("E. min-edge + clip",      make_legoesm_nh_min_edge_config, True),
        ("F. aggressive + clip",    make_legoesm_nh_min_edge_aggressive_config, True),
    ]
    results = []
    for name, factory, use_clip in configs:
        ratios = []
        for seed in seeds:
            grid_on, hc, tm, state_on = _build_nh_state(8, seed, True)
            grid_off, _, _, state_off = _build_nh_state(8, seed, False)
            cfg = factory(**base_kw)
            patches = []
            if use_clip:
                clipped_scalar = functools.partial(
                    _real_pad_halo_4d,
                    monotone_clip=True,
                    monotone_clip_slack=0.5,
                )
                clipped_vec = functools.partial(
                    _real_pad_halo_vec,
                    monotone_clip=True,
                    monotone_clip_slack=0.5,
                )
                for tgt in patch_modules:
                    try:
                        patches.append(patch(tgt, clipped_scalar))
                    except AttributeError:
                        pass
                for tgt in vec_patch_modules:
                    try:
                        patches.append(patch(tgt, clipped_vec))
                    except AttributeError:
                        pass
            with __import__("contextlib").ExitStack() as stk:
                for p in patches:
                    try:
                        stk.enter_context(p)
                    except (AttributeError, ModuleNotFoundError):
                        pass
                m_on = CDGridCompressibleEulerModel(grid_on, hc, tm, cfg)
                m_off = CDGridCompressibleEulerModel(grid_off, hc, tm, cfg)
                s_on = m_on.step(state_on, dt=10.0)
                s_off = m_off.step(state_off, dt=10.0)
            e_on, _ = _edge_and_interior_std(s_on.theta_prime.data)
            e_off, _ = _edge_and_interior_std(s_off.theta_prime.data)
            if e_off > 1e-30:
                ratios.append(e_on / e_off)
        results.append((name, float(np.mean(ratios))))
    with capsys.disabled():
        print(
            f"\n[iter-504 combined factory + halo clip, 2 seeds @ C8, 1 step]"
        )
        baseline = results[0][1]
        for name, r in results:
            reduction = (1 - r / baseline) * 100
            print(f"  {name:35s}: {r:.3f}×  ({reduction:+.1f}%)")
    assert all(np.isfinite(r) for _, r in results)
