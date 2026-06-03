"""Probe: Visbeck kappa AD gradient for unstable column.

The line:
    N = jnp.sqrt(jnp.maximum(N2, 0.0))

is forward-correct but the backward pass through `sqrt(0)` yields
`1/(2*sqrt(0)) = inf`. The `maximum(N2, 0)` masks the gradient on the
N2<0 side to 0, so the multiplication 0*inf produces NaN.

This probe checks whether jax.grad of a Visbeck-using ocean step
produces NaN for an unstable column.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.ocean.eos import wright_eos
from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
    compute_visbeck_kappa_gm,
)
from legoesm.ocean.physics.lateral_mixing.config import VisbeckConfig
from legoesm.ocean.vertical import create_ocean_z_star


def main() -> None:
    nlev = 6
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=1000.0)
    shape = (4, nlev)

    # Profile with N² < 0 in mid column (unstable layer):
    T = jnp.array([20.0, 5.0, 18.0, 8.0, 3.0, 1.0]).astype(jnp.float64)
    T = jnp.broadcast_to(T, shape)
    S = jnp.full(shape, 35.0, dtype=jnp.float64)
    p = jnp.full(shape, 1e7, dtype=jnp.float64)
    rho = wright_eos(T, S, p)
    print(f"Density profile: {rho[0]}")

    S_x = jnp.full((4, nlev - 1), 1e-3, dtype=jnp.float64)
    S_y = jnp.zeros((4, nlev - 1), dtype=jnp.float64)
    J = jnp.ones(4, dtype=jnp.float64)
    f_coriolis = jnp.full(4, 1e-4, dtype=jnp.float64)
    cfg = VisbeckConfig()

    def loss(rho_in):
        kappa = compute_visbeck_kappa_gm(
            rho_in, S_x, S_y, z_coord, J, f_coriolis, cfg,
        )
        return jnp.sum(kappa)

    g = jax.grad(loss)(rho)
    has_nan = bool(jnp.any(jnp.isnan(g)))
    has_inf = bool(jnp.any(jnp.isinf(g)))
    print(f"Visbeck kappa gradient w.r.t. rho:")
    print(f"  any NaN: {has_nan}")
    print(f"  any Inf: {has_inf}")
    print(f"  max abs grad: {float(jnp.max(jnp.abs(jnp.where(jnp.isfinite(g), g, 0.0)))):.3e}")
    if has_nan or has_inf:
        print(f"  raw: {g[0]}")
    if has_nan:
        print("CONFIRMED: NaN gradient through sqrt(max(N²,0)) in unstable layer.")


if __name__ == "__main__":
    main()
