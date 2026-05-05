"""Probe: KPP non-local tracer transport column conservation.

After the codex iter-1 fix (drop the in_bl_full mask, keep only
is_unstable_col), the column-integrated non-local tendency should
sum to zero exactly (within machine precision):

    sum_k dT_nonlocal[k] * dz[k] = -F_top_flux + F_bot_flux = 0 - 0 = 0

(zero-flux BCs at surface and bottom).
"""
from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
from legoesm.ocean.eos import wright_eos
from legoesm.ocean.vertical import create_ocean_z_star


def main() -> None:
    nlev = 8
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=200.0)
    shape = (6, 2, 2, nlev)
    shape_2d = (6, 2, 2)

    # Unstable column: cold over warm so it convects.
    T = jnp.broadcast_to(
        jnp.array([2.0, 5.0, 10.0, 15.0, 18.0, 20.0, 22.0, 23.0])[None, None, None, :],
        shape,
    ).astype(jnp.float64)
    S = jnp.full(shape, 35.0, dtype=jnp.float64)
    p = jnp.full(shape, 1e6, dtype=jnp.float64)
    rho = wright_eos(T, S, p)

    u = jnp.full(shape, 0.05, dtype=jnp.float64)
    v = jnp.zeros(shape, dtype=jnp.float64)
    eta = jnp.zeros(shape_2d, dtype=jnp.float64)
    J = jnp.ones(shape_2d, dtype=jnp.float64)

    # Strong unstable B_f to ensure non-local is active.
    B_f = jnp.full(shape_2d, 1e-6, dtype=jnp.float64)
    Q_sfc_T = jnp.full(shape_2d, 1e-3, dtype=jnp.float64)  # K m/s

    # h_bl_prev: pick a value that places the BL boundary INSIDE a layer
    # so the previous mask would have leaked.
    h_bl_prev = jnp.full(shape_2d, 75.0, dtype=jnp.float64)  # mid-column

    cfg = KPPConfig()
    out = kpp_vertical_mixing(
        u, v, T, S, rho, eta, z_coord, J, cfg,
        B_f=B_f, Q_sfc_T=Q_sfc_T, h_bl_prev=h_bl_prev,
    )

    # Reproduce the diffusion-only run for subtraction.
    out_no_qsfc = kpp_vertical_mixing(
        u, v, T, S, rho, eta, z_coord, J, cfg,
        B_f=B_f, Q_sfc_T=jnp.zeros(shape_2d), h_bl_prev=h_bl_prev,
    )

    # The non-local tendency = full - (no Q_T).
    dT_nonlocal = out.dT_dt - out_no_qsfc.dT_dt

    # Column integral with layer thickness dz_actual = dz_ref * J.
    dz_actual = z_coord.dz_ref * J[..., jnp.newaxis]
    column_integral = jnp.sum(dT_nonlocal * dz_actual, axis=-1)
    max_abs = float(jnp.max(jnp.abs(column_integral)))
    norm = float(jnp.max(jnp.abs(dT_nonlocal)))
    rel = max_abs / max(norm, 1e-30)
    print(f"max|column integral of dT_nonlocal|: {max_abs:.3e}")
    print(f"max|dT_nonlocal|:                    {norm:.3e}")
    print(f"relative drift:                      {rel:.3e}")
    if rel < 1e-12:
        print("[PASS] non-local transport conserves column tracer.")
    else:
        print("[FAIL] non-local transport does NOT conserve column tracer.")


if __name__ == "__main__":
    main()
