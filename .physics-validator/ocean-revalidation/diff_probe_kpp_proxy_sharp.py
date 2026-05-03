"""Sharp probe: B_f proxy SIGN — detects flip without convective branch hiding it.

Setup: a STABLE column (so the interior N² is positive throughout, no
convective branch firing). The non-local transport flag in
``kpp_vertical_mixing`` is gated on ``B_f > 0.0``.

If the proxy has the wrong sign, then a STABLE column produces
``B_f > 0`` (proxy bug), and KPP's non-local transport fires when it
should not. This is detectable by comparing the column-mean
non-local contribution.
"""
from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.eos import wright_eos, rho_0 as _RHO_0
from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
from legoesm.ocean.vertical import create_ocean_z_star


def main() -> None:
    nlev = 8
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=1000.0)
    shape = (6, 4, 4, nlev)
    shape_2d = (6, 4, 4)

    # STABLE column with no static instability: T monotone decreasing.
    T_data = jnp.broadcast_to(
        jnp.linspace(20.0, 5.0, nlev)[None, None, None, :], shape,
    ).astype(jnp.float64)
    S = jnp.full(shape, 35.0, dtype=jnp.float64)
    p = jnp.full(shape, 1e7, dtype=jnp.float64)
    rho = wright_eos(T_data, S, p)

    u = jnp.full(shape, 0.1, dtype=jnp.float64)
    v = jnp.zeros(shape, dtype=jnp.float64)
    eta = jnp.zeros(shape_2d, dtype=jnp.float64)
    J = jnp.ones(shape_2d, dtype=jnp.float64)

    # Provide a known-good Q_sfc_T and Q_sfc_S so the non-local code path
    # (which only fires for B_f > 0) takes the imposed flux.
    Q_sfc_T = jnp.full(shape_2d, 1e-3, dtype=jnp.float64)
    Q_sfc_S = jnp.zeros(shape_2d, dtype=jnp.float64)

    cfg = KPPConfig()

    # Case A: explicit STABLE B_f (correct).  Non-local should NOT fire.
    B_f_stable = jnp.full(shape_2d, -1e-7, dtype=jnp.float64)
    out_explicit = kpp_vertical_mixing(
        u, v, T_data, S, rho, eta, z_coord, J, cfg,
        B_f=B_f_stable, Q_sfc_T=Q_sfc_T, Q_sfc_S=Q_sfc_S,
    )

    # Case B: let kpp use the proxy (B_f=None).
    out_proxy = kpp_vertical_mixing(
        u, v, T_data, S, rho, eta, z_coord, J, cfg,
        Q_sfc_T=Q_sfc_T, Q_sfc_S=Q_sfc_S,
    )

    # Manually compute what the proxy returns:
    dz_actual = z_coord.dz_ref * J[..., jnp.newaxis]
    dz_half0 = 0.5 * (dz_actual[..., 0] + dz_actual[..., 1])
    drho_dz_sfc = (rho[..., 0] - rho[..., 1]) / dz_half0
    B_f_from_proxy = constants.g / _RHO_0 * cfg.K_bg * drho_dz_sfc
    print(f"drho_dz_sfc (stable, top - below): {float(drho_dz_sfc[0, 0, 0]):+.3e}  (expect NEGATIVE for stable)")
    print(f"Proxy B_f for STABLE column:       {float(B_f_from_proxy[0, 0, 0]):+.3e}")
    print(f"Expected B_f for stable column:     NEGATIVE (B_f > 0 = unstable)")
    print()

    # Both should be similar IF the proxy gives the correct STABLE sign.
    # If the proxy gives positive B_f for stable column, the "out_proxy"
    # case will incorrectly trigger non-local transport.
    diff = float(jnp.max(jnp.abs(out_proxy.dT_dt - out_explicit.dT_dt)))
    print(f"|dT/dt(proxy) - dT/dt(explicit-stable)|.max(): {diff:.3e}")
    if diff > 1e-8:
        print("BUG CONFIRMED: proxy disagrees with explicit-stable B_f for")
        print("              what should be the same physical configuration.")
        print("              Proxy is producing the WRONG SIGN of B_f.")


if __name__ == "__main__":
    main()
