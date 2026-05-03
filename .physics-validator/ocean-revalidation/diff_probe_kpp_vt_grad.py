"""Probe: KPP V_t² gradient through sqrt(|N²|) at N²=0.

The line:
    V_t2 = Cv * sqrt(max(|N²_full|, 0)) / sqrt(...) * ...

If any layer has exactly N²=0 (rare but possible), the backward
pass evaluates `d sqrt(|0|)/d|0| = inf` chained with `d|0|/d0 = 0`
(JAX subgradient), producing 0*inf = NaN.

Setup an N²=0 column to test.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
from legoesm.ocean.vertical import create_ocean_z_star


def main() -> None:
    nlev = 6
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=1000.0)
    shape = (6, 4, 4, nlev)
    shape_2d = (6, 4, 4)

    # Density EXACTLY constant in middle layers (N²=0).
    rho = jnp.full(shape, 1029.0, dtype=jnp.float64)

    T = jnp.full(shape, 15.0, dtype=jnp.float64)
    S = jnp.full(shape, 35.0, dtype=jnp.float64)
    u = jnp.full(shape, 0.1, dtype=jnp.float64)
    v = jnp.zeros(shape, dtype=jnp.float64)
    eta = jnp.zeros(shape_2d, dtype=jnp.float64)
    J = jnp.ones(shape_2d, dtype=jnp.float64)
    cfg = KPPConfig()

    def loss(rho_in):
        out = kpp_vertical_mixing(u, v, T, S, rho_in, eta, z_coord, J, cfg)
        return jnp.sum(out.dT_dt)

    g = jax.grad(loss)(rho)
    has_nan = bool(jnp.any(jnp.isnan(g)))
    print(f"Constant-rho column → N²=0 → V_t² formula triggered.")
    print(f"  any NaN in dL/drho: {has_nan}")
    if has_nan:
        print("CONFIRMED: NaN gradient through KPP V_t² at N²=0.")


if __name__ == "__main__":
    main()
