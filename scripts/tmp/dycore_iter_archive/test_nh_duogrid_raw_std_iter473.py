"""FV3_3D iter 473: decompose iter-471 duogrid effect — is the
duogrid penalty driven by HIGHER edge std, LOWER interior
std, or BOTH?

iter-471/472 found duogrid INCREASES ``edge_var/interior_var``
ratio by 4-5× at C8/C16.  Ratio doesn't tell us which side
moved.  iter-473 logs RAW std of edge cells vs interior cells
separately for duogrid ON vs OFF.

If duogrid mostly REDUCES interior std (smoother interior),
the ratio increase is benign — visual edge artifacts may
actually be unchanged or improved.

If duogrid mostly INCREASES edge std (rougher edges), the
ratio increase reflects real worsening of edge artifacts.

Tests
-----

1. ``test_nh_duogrid_raw_std_decomposition``.
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


def _edge_and_interior_std(field_data: jnp.ndarray):
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


def test_nh_duogrid_raw_std_decomposition(capsys):
    seeds = [473, 474, 475]
    base_kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    edge_stds_on = []
    interior_stds_on = []
    edge_stds_off = []
    interior_stds_off = []
    for seed in seeds:
        grid_on, hc, tm, state_on = _build_nh_state(8, seed, True)
        grid_off, _, _, state_off = _build_nh_state(8, seed, False)
        cfg = make_fv3_faithful_nh_config(**base_kw)
        m_on = CDGridCompressibleEulerModel(grid_on, hc, tm, cfg)
        m_off = CDGridCompressibleEulerModel(grid_off, hc, tm, cfg)
        s_on, s_off = state_on, state_off
        for _ in range(3):
            s_on = m_on.step(s_on, dt=10.0)
            s_off = m_off.step(s_off, dt=10.0)
        e_on, i_on = _edge_and_interior_std(s_on.theta_prime.data)
        e_off, i_off = _edge_and_interior_std(s_off.theta_prime.data)
        edge_stds_on.append(e_on)
        interior_stds_on.append(i_on)
        edge_stds_off.append(e_off)
        interior_stds_off.append(i_off)
    me_on = float(np.mean(edge_stds_on))
    mi_on = float(np.mean(interior_stds_on))
    me_off = float(np.mean(edge_stds_off))
    mi_off = float(np.mean(interior_stds_off))
    with capsys.disabled():
        print(
            f"\n[iter-473 NH θ′ raw std decomposition, "
            f"3 seeds @ C8, 3 steps]"
        )
        print(f"  duogrid=ON:")
        print(f"    edge std:     {me_on:.4e}")
        print(f"    interior std: {mi_on:.4e}")
        print(f"    ratio:        {me_on / max(mi_on, 1e-30):.4f}")
        print(f"  duogrid=OFF:")
        print(f"    edge std:     {me_off:.4e}")
        print(f"    interior std: {mi_off:.4e}")
        print(f"    ratio:        {me_off / max(mi_off, 1e-30):.4f}")
        edge_change = me_on / max(me_off, 1e-30)
        int_change = mi_on / max(mi_off, 1e-30)
        print(f"  duogrid effect: edge std × {edge_change:.2f}, "
              f"interior std × {int_change:.2f}")
        if edge_change > int_change:
            print(
                f"  Conclusion: duogrid INCREASES edge std "
                f"more than interior std → genuinely "
                f"worsening edge structure."
            )
        elif int_change < 0.5 * edge_change:
            print(
                f"  Conclusion: duogrid DECREASES interior std "
                f"more than edge → ratio rises because "
                f"interior gets smoother, edges unchanged → "
                f"benign metric artifact."
            )
        else:
            print(
                f"  Conclusion: duogrid affects both sides "
                f"comparably."
            )
    assert np.isfinite(me_on) and np.isfinite(me_off)
    assert np.isfinite(mi_on) and np.isfinite(mi_off)
