"""FV3_3D iter 534: aggressive factory + make_clipped_step.

Combines iter-483's ``make_legoesm_nh_min_edge_aggressive_config``
(boosted ``corner_div_damp_d2_bg``) with iter-526's JIT-safe
``make_clipped_step`` helper.

Confirms the user-facing stack works end-to-end:

    aggressive_cfg → model → make_clipped_step → loop

Compares vs the default min-edge factory under same conditions.

Tests
-----

1. ``test_aggressive_vs_min_edge_at_10_steps``.
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
    make_legoesm_nh_min_edge_aggressive_config,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import make_clipped_step
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


def _build_state(n, seed):
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


def test_aggressive_vs_min_edge_at_10_steps(capsys):
    grid, hc, tm, state = _build_state(8, 534)
    kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    results = []
    for label, factory in [
        ("min_edge",        make_legoesm_nh_min_edge_config),
        ("aggressive",      make_legoesm_nh_min_edge_aggressive_config),
    ]:
        cfg = factory(**kw)
        m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
        step = make_clipped_step(m, state, dt=10.0, slack=0.5)
        s = state
        for _ in range(10):
            s = step(s, 10.0)
        e, i = _edge_and_interior_std(s.theta_prime.data)
        ratio = e / i if i > 1e-30 else float("nan")
        results.append((label, ratio, e, i))
    with capsys.disabled():
        print(
            f"\n[iter-534 factory + make_clipped_step combinations, "
            f"10 steps]"
        )
        for label, r, e, i in results:
            print(
                f"  {label:12s}: e/i = {r:.3f}×, "
                f"edge_std = {e:.3e}, int_std = {i:.3e}"
            )
    assert all(np.isfinite(r) for _, r, _, _ in results)
