"""What does the #1690 surface-face metric fix actually change? (one variable)

The `nemo_z0` surface boundary condition couples a virtual z=0 row to the first
interior W-point across a face distance that must be NEMO's `e3t(1)` -- the
full top-cell thickness -- because both rows are W-points
(`zzd_lw(jk=2)` denominator `e3t(1)*e3w(2)`, zdftke.F90:403-410).  Before #1690
the orchestrator handed the solver the top cell's MIDPOINT depth instead, which
is half of that on a midpoint grid, so the coupling was DOUBLED.

This is the A/B.  Everything about the column is held fixed and the ONLY thing
that differs between the arms is that face distance:

    arm PRE  : dz_face_surface = e3t(1) / 2     (the midpoint metric, the bug)
    arm POST : dz_face_surface = e3t(1)         (NEMO's, the fix)

Both arms call the production solver `_solve_tke_backward_euler` on operands
built by the model's own helpers (`compute_N2`, `compute_mixing_lengths`,
`compute_K_from_tke`, `vertical_shear_squared`, `nemo_surface_avm`), so nothing
is re-derived and nothing but the metric moves.

WHY THE SIGN IS NOT OBVIOUS, and why this is measured rather than asserted
(GLM review): the doubled coefficient is a doubled implicit TKE FLUX across the
top interface.  It acts as a spurious SOURCE where the held surface value sits
ABOVE the interior equilibrium and as a spurious SINK where it sits below.  The
OMIP/ORCA1 card described in #1690 holds a surface value far above the interior
floor, so there it over-mixed; on another card the sign can flip.  The arms
below therefore sweep the wind stress, which is what sets the held surface
value.

Prints numbers only.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
from legoesm.ocean.physics.vertical_mixing.tke import (
    _NEMO_TKE_EBB, _NEMO_TKE_EMIN0, _safe_stress_modulus,
    _solve_tke_backward_euler, compute_K_from_tke, compute_mixing_lengths,
)
from legoesm.ocean.physics.vertical_mixing._shared import (
    compute_N2, vertical_shear_squared,
)

_RHO0 = constants.rho_ocean


def _column(nlev, dz_m, du_dz, dT_dz):
    """A wind-forced, stably stratified near-surface column."""
    shape = (1, 1, nlev)
    T = jnp.asarray(20.0 - dT_dz * dz_m * np.arange(nlev))[None, None, :] * jnp.ones(shape)
    S = jnp.full(shape, 35.0)
    # Linear-in-depth density consistent with the temperature ramp.
    rho = jnp.asarray(1025.0 + 0.2 * dT_dz * dz_m * np.arange(nlev))[None, None, :] \
        * jnp.ones(shape)
    u = jnp.asarray(du_dz * dz_m * np.arange(nlev))[None, None, :] * jnp.ones(shape)
    v = jnp.zeros(shape)
    dz_half = jnp.full((1, 1, nlev - 1), dz_m)
    return u, v, T, S, rho, dz_half


def arms(*, tau, nlev=12, dz_m=10.0, dt=1800.0, du_dz=-0.004, dT_dz=0.06):
    """(pre-fix, post-fix) solved interior TKE profiles for one wind stress."""
    u, v, T, S, rho, dz_half = _column(nlev, dz_m, du_dz, dT_dz)
    cfg = TKEConfig(surface_bc="nemo_dirichlet", tke_surface_bc_level="nemo_z0")
    tke_old = jnp.full((1, 1, nlev - 1), cfg.tke_background)

    N2 = compute_N2(rho, dz_half, _RHO0, constants.g)
    shear_sq = vertical_shear_squared(u, v, dz_half)
    l_k, l_eps = compute_mixing_lengths(tke_old, N2, dz_half, cfg, signed_n2=False)
    K_M, K_H = compute_K_from_tke(tke_old, l_k, cfg)

    tau_x = jnp.full((1, 1), tau)
    taum = _safe_stress_modulus(tau_x, jnp.zeros((1, 1)))
    e_sfc = jnp.maximum(jnp.asarray(_NEMO_TKE_EMIN0),
                        _NEMO_TKE_EBB / _RHO0 * taum)

    def solve(face):
        return _solve_tke_backward_euler(
            e_old=tke_old, K_M_old=K_M, K_H_old=K_H,
            P_s=K_M * shear_sq, N2=N2, l_eps=l_eps, dz_half=dz_half,
            surface_flux=jnp.zeros((1, 1)), dt=dt, cfg=cfg,
            dz_face_surface=jnp.full((1, 1), face),
            surface_dirichlet=e_sfc, surface_bc_level="nemo_z0")

    e3t1 = dz_m
    return (np.asarray(solve(0.5 * e3t1))[0, 0],   # PRE: midpoint metric
            np.asarray(solve(e3t1))[0, 0],         # POST: NEMO e3t(1)
            float(np.asarray(e_sfc)[0, 0]),
            cfg, l_k)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dz", type=float, default=10.0, help="top-cell thickness [m]")
    p.add_argument("--dt", type=float, default=1800.0)
    args = p.parse_args(argv)

    print(f"top-cell e3t(1) = {args.dz:g} m, dt = {args.dt:g} s; "
          "PRE = midpoint metric (the bug), POST = NEMO e3t(1)")
    hdr = (f"{'tau [Pa]':>9s} {'e_sfc held':>11s} {'TKE@if0 PRE':>12s} "
           f"{'TKE@if0 POST':>13s} {'ratio':>7s} {'TKE@if1 ratio':>14s}")
    print(hdr)
    print("-" * len(hdr))
    for tau in (0.02, 0.05, 0.1, 0.2, 0.4):
        pre, post, e_sfc, cfg, _ = arms(tau=tau, dz_m=args.dz, dt=args.dt)
        r0 = pre[0] / post[0]
        r1 = pre[1] / post[1]
        print(f"{tau:9.3f} {e_sfc:11.3e} {pre[0]:12.4e} {post[0]:13.4e} "
              f"{r0:7.3f} {r1:14.3f}")

    print()
    print("Interior TKE profile at tau = 0.1 Pa [m^2/s^2], interface 0 = "
          f"{args.dz:g} m depth:")
    pre, post, _, _, _ = arms(tau=0.1, dz_m=args.dz, dt=args.dt)
    print(f"{'interface':>10s} {'PRE':>12s} {'POST':>12s} {'PRE/POST':>9s}")
    for k in range(min(6, len(pre))):
        print(f"{k:10d} {pre[k]:12.4e} {post[k]:12.4e} "
              f"{pre[k] / post[k]:9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
