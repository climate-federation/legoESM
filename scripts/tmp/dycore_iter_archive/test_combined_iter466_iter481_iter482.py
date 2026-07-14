"""FV3_3D iter 482: test combined iter-466 + iter-481 paths.

iter-466 dropped 3 hurting flags → 2.25× (50% reduction).
iter-481 boosted corner_div_damp 100× → 2.25× (36% reduction).
Both reach ~2.25×.  Question: does the COMBINATION go BELOW
2.25×?

If multiplicative: 0.50 × 0.64 = 0.32 of original 4.5 = ~1.4×.
If both hit the same floor: combined stays at ~2.25.

Tests
-----

1. ``test_combined_min_edge_factory_with_boosted_corner_div``.
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
    make_legoesm_nh_min_edge_config,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
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


def test_combined_min_edge_factory_with_boosted_corner_div(capsys):
    """Compare 3 configurations at C8 + duogrid:
    A. factory full (iter-466 baseline)
    B. min-edge factory (iter-467, drops 3 flags)
    C. min-edge factory + d2_bg=5e-2 (iter-481 boost)
    """
    seeds = [482, 483, 484]
    common = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    configs = {
        "A. factory full": (
            make_fv3_faithful_nh_config,
            dict(**common, corner_div_damp_d2_bg=5e-4),
        ),
        "B. min-edge": (
            make_legoesm_nh_min_edge_config,
            dict(**common, corner_div_damp_d2_bg=5e-4),
        ),
        "C. min-edge + d2_bg=5e-2": (
            make_legoesm_nh_min_edge_config,
            dict(**common, corner_div_damp_d2_bg=5e-2),
        ),
    }
    results = {}
    for name, (factory, kw) in configs.items():
        ratios = []
        for seed in seeds:
            grid_on, hc, tm, state_on = _build_nh_state(8, seed, True)
            grid_off, _, _, state_off = _build_nh_state(8, seed, False)
            cfg = factory(**kw)
            m_on = CDGridCompressibleEulerModel(grid_on, hc, tm, cfg)
            m_off = CDGridCompressibleEulerModel(grid_off, hc, tm, cfg)
            s_on = m_on.step(state_on, dt=10.0)
            s_off = m_off.step(state_off, dt=10.0)
            e_on, _ = _edge_and_interior_std(s_on.theta_prime.data)
            e_off, _ = _edge_and_interior_std(s_off.theta_prime.data)
            if e_off > 1e-30:
                ratios.append(e_on / e_off)
        results[name] = float(np.mean(ratios))
    with capsys.disabled():
        print(
            f"\n[iter-482 combined iter-466 + iter-481 paths, "
            f"3 seeds @ C8, 1 step]"
        )
        for name, r in results.items():
            print(f"  {name:30s}: edge ratio = {r:.3f}×")
        a, b, c = results["A. factory full"], results["B. min-edge"], results["C. min-edge + d2_bg=5e-2"]
        print(f"  A → B reduction: {(1 - b/a)*100:.1f}%")
        print(f"  A → C reduction: {(1 - c/a)*100:.1f}%")
        print(f"  B → C additional: {(1 - c/b)*100:.1f}%")
    assert all(np.isfinite(r) for r in results.values())
