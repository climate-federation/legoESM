"""FV3_3D iter 466: apply iter-465 findings — compare "factory
minimal-for-edge" vs "factory full" at C8 + duogrid.

Per iter-465, in factory + duogrid + C8 the edge-helping flags
are ``use_fv3_vector_halo_uv`` (Δ=+5) and ``use_fv3_d_con_cv``
(Δ=+1.65); the hurting flags are ``use_fv3_metric_aware_d_con``
(Δ=−1.6), ``heat_source_del2_iters`` (Δ=−1.4), and
``d_con_top_zero_levels`` (Δ=−0.58).

This test constructs a "minimized" config keeping only the
edge-helping flags + verifies the ratio is actually lower.

Tests
-----

1. ``test_minimal_factory_lower_edge_ratio`` — minimal-for-
   edge config produces lower θ′ edge ratio than full factory.
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
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
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
    grid = create_cubed_sphere(n, use_duogrid=True)
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


def test_minimal_factory_lower_edge_ratio(capsys):
    """Build two configs:
    A. ``factory_full``: all FV3 fidelity flags default (factory)
    B. ``factory_min_edge``: only edge-helping flags from
       iter-465 (vector_halo_uv + d_con_cv)
    Verify B produces strictly LOWER NH θ′ edge ratio than A.
    """
    seeds = [466, 467, 468, 469, 470]
    base_kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    # Configuration A: factory full (all FV3 flags ON)
    kw_full = dict(**base_kw)
    # Configuration B: minimal-for-edge (only iter-465 helpers ON,
    # hurters OFF)
    kw_min_edge = dict(
        **base_kw,
        # KEEP: vector_halo_uv (Δ+5), d_con_cv (Δ+1.65)
        use_fv3_metric_aware_d_con=False,    # Δ-1.6 HURT
        heat_source_del2_iters=0,            # Δ-1.4 HURT
        d_con_top_zero_levels=0,             # Δ-0.58 HURT
    )
    ratios_full = []
    ratios_min = []
    for seed in seeds:
        grid, hc, tm, state = _build_nh_state(8, seed=seed)
        cfg_full = make_fv3_faithful_nh_config(**kw_full)
        cfg_min = make_fv3_faithful_nh_config(**kw_min_edge)
        m_full = CDGridCompressibleEulerModel(grid, hc, tm, cfg_full)
        m_min = CDGridCompressibleEulerModel(grid, hc, tm, cfg_min)
        s_full = state
        s_min = state
        for _ in range(3):
            s_full = m_full.step(s_full, dt=10.0)
            s_min = m_min.step(s_min, dt=10.0)
        ratios_full.append(
            _edge_interior_ratio(s_full.theta_prime.data)
        )
        ratios_min.append(
            _edge_interior_ratio(s_min.theta_prime.data)
        )
    mean_full = float(np.mean(ratios_full))
    mean_min = float(np.mean(ratios_min))
    with capsys.disabled():
        print(
            f"\n[iter-466 factory minimal-for-edge comparison, "
            f"5 seeds @ C8 + duogrid, 3 steps]"
        )
        print(f"  factory full  θ′ edge ratio: {mean_full:.4f}")
        print(f"  factory min   θ′ edge ratio: {mean_min:.4f}")
        print(f"  min/full ratio:              {mean_min/mean_full:.4f}")
        print(f"  Reduction:                   {(1 - mean_min/mean_full)*100:.1f}%")
    assert np.isfinite(mean_full)
    assert np.isfinite(mean_min)
    assert mean_min < mean_full, (
        f"iter-466: expected minimal-for-edge config to produce "
        f"LOWER edge ratio than full factory.  Got "
        f"full={mean_full:.4f}, min={mean_min:.4f}.  iter-465 "
        f"finding may not be robust."
    )
