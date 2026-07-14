"""FV3_3D iter 533: corner-excited IC — does clip protect the
24 cube-vertex cells specifically?

iter-489 showed the duogrid overshoot is confined to the 24
cube-vertex cells.  Stress test: drop a temperature
perturbation AT one of the cube corners and let it propagate.
With clip, vertex cells should not amplify the disturbance;
without clip, they should overshoot.

Tests
-----

1. ``test_corner_excited_clip_vs_noclip`` — Gaussian θ′
   perturbation at the (0,0) corner of face 0.  Measure
   max(|θ′|) at the 4 corner cells of each face vs interior
   max, with and without clip.
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


def _vertex_max(field_data):
    """Max |field| at the 4 cube corners of each face."""
    arr = np.abs(np.asarray(field_data))
    # corners: (0, 0), (0, -1), (-1, 0), (-1, -1)
    return max(
        float(arr[:, 0, 0, :].max()),
        float(arr[:, 0, -1, :].max()),
        float(arr[:, -1, 0, :].max()),
        float(arr[:, -1, -1, :].max()),
    )


def _interior_max(field_data):
    arr = np.abs(np.asarray(field_data))
    return float(arr[:, 1:-1, 1:-1, :].max())


def _build_state(n, seed=533):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    # All zero state with theta_prime perturbation at face=0 corner
    theta_p = np.zeros((6, n, n, nlev))
    theta_p[0, 0, 0, :] = 5.0  # 5 K spike at one corner
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev)),
                name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)),
                name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.asarray(theta_p),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)),
                   name="phis", dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers", dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, hc, tm, state


def test_corner_excited_clip_vs_noclip(capsys):
    import contextlib
    n = 8
    grid, hc, tm, state = _build_state(n)
    kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    cfg = make_legoesm_nh_min_edge_config(**kw)
    init_vmax = _vertex_max(state.theta_prime.data)
    init_imax = _interior_max(state.theta_prime.data)
    results = []
    for label, use_clip in [("no clip", False), ("clip slack=0.5", True)]:
        ctx = (monotone_halo_clip_context(slack=0.5)
               if use_clip else contextlib.nullcontext())
        with ctx:
            m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
            s = state
            for _ in range(5):
                s = m.step(s, dt=10.0)
        vmax = _vertex_max(s.theta_prime.data)
        imax = _interior_max(s.theta_prime.data)
        results.append((label, vmax, imax))
    with capsys.disabled():
        print(
            f"\n[iter-533 corner-excited θ′ stress, 5 steps]"
        )
        print(
            f"  initial vertex max:   {init_vmax:.3f} K"
        )
        print(
            f"  initial interior max: {init_imax:.3f} K"
        )
        for label, vmax, imax in results:
            print(
                f"  {label:18s}: vertex max = {vmax:.3f}, "
                f"interior max = {imax:.3f}, "
                f"v/i = {(vmax / max(imax, 1e-30)):.2f}×"
            )
    assert all(np.isfinite(v) and np.isfinite(i) for _, v, i in results)
