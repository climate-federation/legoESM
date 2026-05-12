"""FV3_3D iter 535: example script — using the FV3 clip helper stack.

Demonstrates the iter-466/505/526 user-facing API for FV3-
faithful 3D cubed-sphere runs with edge-artifact suppression:

1. ``make_legoesm_nh_min_edge_config`` (iter-466) — factory
   with 50% edge-ratio reduction defaults.
2. ``make_clipped_step`` (iter-526) — JIT-safe wrapper that
   bakes the iter-505 monotone halo clip into the compiled
   graph for an additional 50% long-term reduction.

Run with::

    python scripts/example_fv3_clip_helper.py

(This is documentation as runnable code — NOT a CI test.)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.compressible_euler_cdgrid import (
    CDGridCompressibleEulerModel,
    make_legoesm_nh_min_edge_config,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import (
    create_cubed_sphere,
    rotate_winds_geo_to_grid,
)
from legoesm.grids.halo import make_clipped_step
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
)


def build_sbr_state(n: int = 16, U_0: float = 20.0):
    """Solid-body-rotation initial state (Williamson 2-like)."""
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    lat = grid.lat
    u_east = U_0 * jnp.cos(lat)
    v_north = jnp.zeros_like(lat)
    u_grid, v_grid = rotate_winds_geo_to_grid(
        u_east, v_north, grid.angle,
    )
    u_p = jnp.broadcast_to(u_grid[..., None], (6, n, n, nlev))
    v_p = jnp.broadcast_to(v_grid[..., None], (6, n, n, nlev))
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=u_p, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v_p, name="v", dims=dims_3d, units="m/s"),
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


def main():
    # === 1. Build state ===
    print("Building C16 SBR initial state...")
    grid, hc, tm, state = build_sbr_state(n=16)
    print(f"  state.u.data.shape = {state.u.data.shape}")

    # === 2. Build FV3-faithful config (min-edge variant) ===
    print("\nBuilding edge-min config (50% reduction baseline)...")
    cfg = make_legoesm_nh_min_edge_config(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    model = CDGridCompressibleEulerModel(grid, hc, tm, cfg)

    # === 3. JIT-safe clipped step (additional 50% reduction) ===
    print("\nCompiling JIT-clipped step (iter-526 make_clipped_step)...")
    step = make_clipped_step(model, state, dt=10.0, slack=0.5)

    # === 4. Run 10 steps ===
    print("\nRunning 10 steps...")
    s = state
    for k in range(10):
        s = step(s, 10.0)
    print(f"  final state.theta_prime.data.max() = "
          f"{float(jnp.abs(s.theta_prime.data).max()):.3e}")

    # === 5. Edge / interior std diagnostic ===
    arr = np.asarray(s.theta_prime.data)
    n_face, n_x, n_y, n_lev = arr.shape
    edge_mask = np.zeros((n_x, n_y), dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    edge_mask_b = np.broadcast_to(
        edge_mask[None, :, :, None], arr.shape,
    )
    interior_mask = ~edge_mask_b
    edge_std = float(arr[edge_mask_b].std())
    int_std = float(arr[interior_mask].std())
    print(f"\nθ′ statistics after 10 steps:")
    print(f"  edge_std     = {edge_std:.3e}")
    print(f"  interior_std = {int_std:.3e}")
    print(f"  ratio        = {edge_std / max(int_std, 1e-30):.3f}×")
    print(f"\nExpected at C16 SBR (iter-521): edge_std ~ 6e-2, "
          f"int_std ~ 1e-2.")
    print("The clip helper reduces both vs the unclipped baseline.")


if __name__ == "__main__":
    main()
