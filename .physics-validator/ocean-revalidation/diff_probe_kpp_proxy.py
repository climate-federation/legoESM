"""Probe: B_f proxy sign in kpp.py:204-207 when B_f is None.

The proxy: ``B_f = -g/rho_0 * K_bg * drho_dz_sfc`` with ``drho_dz_sfc =
(rho[0] - rho[1]) / dz``.

Convention check:
- Stable column: rho_top < rho_below → drho_dz_sfc < 0 → B_f > 0.
- Unstable column: rho_top > rho_below → drho_dz_sfc > 0 → B_f < 0.

But KPP convention is ``B_f > 0 = unstable``. The proxy thus has the
wrong sign — stable columns are labelled unstable.

Compare against the proper formulation: ``B_f`` should equal the
*surface* buoyancy flux (something forced from outside; cooling →
unstable → B_f > 0).  A diffusive proxy from interior gradient
should be ``B_f ~ +g/rho_0 * K_bg * drho_dz_sfc`` (so unstable
column with drho_dz_sfc > 0 gives B_f > 0). The leading minus is
incorrect.
"""
from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.eos import wright_eos
from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
from legoesm.ocean.vertical import create_ocean_z_star


def main() -> None:
    nlev = 8
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=1000.0)
    shape = (6, 4, 4, nlev)
    shape_2d = (6, 4, 4)

    # Build a STABLE column: T decreases with depth (lighter at top).
    T_stable = jnp.broadcast_to(
        jnp.linspace(20.0, 5.0, nlev)[None, None, None, :], shape,
    ).astype(jnp.float64)
    # Build an UNSTABLE column: top colder than below (heavier on top).
    T_unstable = jnp.broadcast_to(
        jnp.linspace(2.0, 20.0, nlev)[None, None, None, :], shape,
    ).astype(jnp.float64)
    S = jnp.full(shape, 35.0, dtype=jnp.float64)
    p = jnp.full(shape, 1e7, dtype=jnp.float64)
    rho_stable = wright_eos(T_stable, S, p)
    rho_unstable = wright_eos(T_unstable, S, p)

    print(f"Stable column: rho[0]={float(rho_stable[0,0,0,0]):.3f}, rho[1]={float(rho_stable[0,0,0,1]):.3f}")
    print(f"Unstable column: rho[0]={float(rho_unstable[0,0,0,0]):.3f}, rho[1]={float(rho_unstable[0,0,0,1]):.3f}")

    u = jnp.full(shape, 0.1, dtype=jnp.float64)
    v = jnp.zeros(shape, dtype=jnp.float64)
    eta = jnp.zeros(shape_2d, dtype=jnp.float64)
    J = jnp.ones(shape_2d, dtype=jnp.float64)
    cfg = KPPConfig()

    # Run KPP with B_f=None to exercise the proxy.
    out_stable = kpp_vertical_mixing(u, v, T_stable, S, rho_stable, eta, z_coord, J, cfg)
    out_unstable = kpp_vertical_mixing(u, v, T_unstable, S, rho_unstable, eta, z_coord, J, cfg)

    # The non-local transport ONLY fires for unstable columns
    # (is_unstable_col = B_f > 0.0).  If the proxy has the right sign,
    # out_unstable.dT_dt should be NONZERO (non-local active) and
    # out_stable.dT_dt should match the diffusion-only solution
    # (no non-local).
    nl_stable = float(jnp.max(jnp.abs(out_stable.dT_dt - jnp.mean(out_stable.dT_dt, keepdims=True))))
    nl_unstable = float(jnp.max(jnp.abs(out_unstable.dT_dt - jnp.mean(out_unstable.dT_dt, keepdims=True))))
    print(f"Stable-column max|dT/dt| (should be SMALL — diffusion only): {float(jnp.max(jnp.abs(out_stable.dT_dt))):.3e}")
    print(f"Unstable-column max|dT/dt| (should be LARGE — non-local + diff): {float(jnp.max(jnp.abs(out_unstable.dT_dt))):.3e}")
    print()

    # The TELL: if proxy is correct, K_v_unstable.max() > K_v_stable.max() (convective enhancement).
    K_v_stable_max = float(jnp.max(out_stable.K_v))
    K_v_unstable_max = float(jnp.max(out_unstable.K_v))
    print(f"K_v max (stable column):   {K_v_stable_max:.3e}")
    print(f"K_v max (unstable column): {K_v_unstable_max:.3e}")
    print()
    if K_v_unstable_max > K_v_stable_max + 1e-4:
        print("Looks like physics is right (unstable mixes harder).")
    else:
        print("WARNING: stable column appears to have similar/higher K_v —")
        print("         consistent with the proxy sign-flip hypothesis.")


if __name__ == "__main__":
    main()
