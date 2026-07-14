"""FV3_3D iter 492: end-to-end test of monotone_clip in NH
dycore — does enabling iter-490's clip reduce iter-471's
duogrid edge ratio?

iter-489 localized the overshoot to cube vertices.
iter-490 added the clip option to ``pad_halo_4d``.
iter-491 showed clip eliminates the 76% random-field
overshoot at the operator level.

iter-492 monkey-patches the NH dycore's ``pad_halo_4d``
reference to always use ``monotone_clip=True`` and measures
the resulting edge ratio.

Tests
-----

1. ``test_nh_duogrid_edge_ratio_with_clip``.
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
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import pad_halo_4d as _real_pad_halo_4d
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
)


_clipped_pad_halo_4d = functools.partial(
    _real_pad_halo_4d, monotone_clip=True,
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


def test_nh_duogrid_edge_ratio_with_clip(capsys):
    """Patch _pad_halo_4d_module in compressible_euler_cdgrid
    with the always-clipped version, then measure ratio."""
    seeds = [492, 493]
    base_kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    ratios_no_clip = []
    ratios_with_clip = []
    for seed in seeds:
        grid_on, hc, tm, state_on = _build_nh_state(8, seed, True)
        grid_off, _, _, state_off = _build_nh_state(8, seed, False)
        cfg = make_fv3_faithful_nh_config(**base_kw)

        # Run baseline (no clip)
        m_on = CDGridCompressibleEulerModel(grid_on, hc, tm, cfg)
        m_off = CDGridCompressibleEulerModel(grid_off, hc, tm, cfg)
        s_on = m_on.step(state_on, dt=10.0)
        s_off = m_off.step(state_off, dt=10.0)
        e_on, _ = _edge_and_interior_std(s_on.theta_prime.data)
        e_off, _ = _edge_and_interior_std(s_off.theta_prime.data)
        if e_off > 1e-30:
            ratios_no_clip.append(e_on / e_off)

        # Run with clip (patched)
        with patch(
            "legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid."
            "_pad_halo_4d_module",
            _clipped_pad_halo_4d,
        ):
            m_on_c = CDGridCompressibleEulerModel(grid_on, hc, tm, cfg)
            m_off_c = CDGridCompressibleEulerModel(grid_off, hc, tm, cfg)
            s_on_c = m_on_c.step(state_on, dt=10.0)
            s_off_c = m_off_c.step(state_off, dt=10.0)
            e_on_c, _ = _edge_and_interior_std(s_on_c.theta_prime.data)
            e_off_c, _ = _edge_and_interior_std(s_off_c.theta_prime.data)
            if e_off_c > 1e-30:
                ratios_with_clip.append(e_on_c / e_off_c)

    mean_no_clip = float(np.mean(ratios_no_clip))
    mean_with_clip = float(np.mean(ratios_with_clip))
    with capsys.disabled():
        print(
            f"\n[iter-492 NH dycore monotone_clip end-to-end test, "
            f"2 seeds @ C8, 1 step]"
        )
        print(
            f"  no clip   : edge ratio = {mean_no_clip:.3f}×"
        )
        print(
            f"  with clip : edge ratio = {mean_with_clip:.3f}×"
        )
        print(
            f"  reduction : {(1 - mean_with_clip / mean_no_clip) * 100:.1f}%"
        )
    assert np.isfinite(mean_no_clip)
    assert np.isfinite(mean_with_clip)
