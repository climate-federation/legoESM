"""Probe: KPP A_v interior floor uses A_bg, not K_bg.

After codex iter-1 fix #6, the interior momentum floor is A_bg (1e-4),
not K_bg (1e-5).  This probe confirms the ratio.
"""
from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
from legoesm.ocean.eos import wright_eos
from legoesm.ocean.vertical import create_ocean_z_star


def main() -> None:
    nlev = 20
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=1000.0)
    shape = (6, 4, 4, nlev)
    shape_2d = (6, 4, 4)

    # Stable column with weak shear → interior should be at the floor.
    T = jnp.broadcast_to(
        jnp.linspace(15.0, 5.0, nlev)[None, None, None, :], shape,
    ).astype(jnp.float64)
    S = jnp.full(shape, 35.0, dtype=jnp.float64)
    p = jnp.full(shape, 1e7, dtype=jnp.float64)
    rho = wright_eos(T, S, p)

    u = jnp.full(shape, 0.001, dtype=jnp.float64)  # very weak
    v = jnp.zeros(shape, dtype=jnp.float64)
    eta = jnp.zeros(shape_2d, dtype=jnp.float64)
    J = jnp.ones(shape_2d, dtype=jnp.float64)

    # Stable forcing, small h_bl_prev so most of column is "interior".
    B_f = jnp.full(shape_2d, -1e-7, dtype=jnp.float64)
    h_bl_prev = jnp.full(shape_2d, 5.0, dtype=jnp.float64)
    cfg = KPPConfig()

    out = kpp_vertical_mixing(
        u, v, T, S, rho, eta, z_coord, J, cfg,
        B_f=B_f, h_bl_prev=h_bl_prev,
    )

    # Bottom 5 layers: should be deep in the interior, away from BL.
    K_v_bot = float(jnp.min(out.K_v[..., -5:]))
    A_v_bot = float(jnp.min(out.A_v[..., -5:]))
    K_v_max = float(jnp.max(out.K_v))
    A_v_max = float(jnp.max(out.A_v))
    print(f"K_v interior min = {K_v_bot:.3e}  (expect K_bg = {cfg.K_bg})")
    print(f"A_v interior min = {A_v_bot:.3e}  (expect A_bg = {cfg.A_bg})")
    print(f"K_v max          = {K_v_max:.3e}")
    print(f"A_v max          = {A_v_max:.3e}")
    print()
    if abs(K_v_bot - cfg.K_bg) / cfg.K_bg < 0.5:
        print("[PASS] K_v interior floor matches K_bg.")
    else:
        print("[FAIL] K_v interior floor differs from K_bg.")
    if abs(A_v_bot - cfg.A_bg) / cfg.A_bg < 0.5:
        print("[PASS] A_v interior floor matches A_bg.")
    else:
        print(f"[FAIL] A_v interior floor differs from A_bg "
              f"(ratio {A_v_bot / cfg.A_bg:.2f}).")


if __name__ == "__main__":
    main()
