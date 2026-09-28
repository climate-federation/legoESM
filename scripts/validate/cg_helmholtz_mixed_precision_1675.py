"""#1675 site 1: what does promoting the Helmholtz metrics actually buy?

The issue's claim is that in ``mixed`` the conjugate-gradient Helmholtz solve
floors its residual near 1e-7 because ``cdgrid_scalar_laplacian`` reads the
GRID's metric arrays, which carry the grid's float32 storage dtype -- and that
at the semi-implicit condition number kappa ~ 1e6 this is an O(0.1) forward
error in the gravity-wave-damping correction, not the "benign backward error"
the in-code comment claimed.

Two quantities are NOT the same and this probe reports both, because the fix
moves one of them and cannot move the other:

* **residual** -- how well CG inverted the operator it was handed.  Float32
  ARITHMETIC inside the Laplacian makes that operator non-repeatable at ~1e-7
  relative to itself, which is what stalls CG.  Promoting the arrays to the
  control dtype fixes this: the operator becomes exactly repeatable.
* **forward error** -- how far the answer sits from the solve the fp64 grid
  would have produced.  The promoted arrays still hold float32 VALUES, so a
  ~1e-7 backward error in the metric coefficients survives the promotion.  If
  kappa amplifies that to O(0.1), the metrics must be REBUILT at control
  precision, not upcast, and this probe is what says so.

Reference arm is a grid built under ``PrecisionPolicy.fp64()``; the mixed arms
are built under ``PrecisionPolicy.mixed()``.  Nothing else differs.

Run: see scripts/cluster/issue_sweep_0923/cg_metrics_1675.sbatch
"""
from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm import semi_implicit_cdgrid as sic
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


def _build(policy, n):
    orig = get_policy()
    try:
        set_policy(policy)
        return create_cubed_sphere_cdgrid(create_cubed_sphere(n))
    finally:
        set_policy(orig)


def _with_promoted_metrics(cdgrid):
    """The #1675 site-1 fix, implemented HERE rather than imported.

    The fix it tests is a five-array upcast inside ``cg_helmholtz_solve``.  It
    measured as a no-op and was never landed, so importing it would make this
    probe un-runnable against main -- and a probe that cannot run against main
    cannot be re-run when the model changes, which is the whole point of
    committing it.  Building the promoted grid here keeps both arms available
    forever and keeps the comparison honest: the ONLY difference between the
    arms is which of these two grids the solve is handed.
    """
    def _c(x):
        return jnp.asarray(x, dtype=jnp.float64)

    return cdgrid._replace(
        rdxc=_c(cdgrid.rdxc),
        rdyc=_c(cdgrid.rdyc),
        dy_edge_x=_c(cdgrid.dy_edge_x),
        dx_edge_y=_c(cdgrid.dx_edge_y),
        base=cdgrid.base._replace(area=_c(cdgrid.base.area)),
    )


def _f64_operator_residual(sol, rhs, coeff, cdgrid_ref):
    """Residual against the fp64-BUILT operator -- the physics, not the f32 one."""
    s = jnp.asarray(sol, dtype=jnp.float64)
    r = jnp.asarray(rhs, dtype=jnp.float64)
    A = s - coeff * sic.cdgrid_scalar_laplacian(s, cdgrid_ref)
    return float(jnp.linalg.norm((r - A).ravel()) / jnp.linalg.norm(r.ravel()))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=24, help="cubed-sphere face size")
    p.add_argument("--kappa", type=float, default=1.0e6,
                   help="target Helmholtz condition number")
    p.add_argument("--tol", type=float, default=1.0e-10)
    p.add_argument("--maxiter", type=int, default=400)
    a = p.parse_args(argv)

    n = a.n
    # kappa ~ 1 + coeff * |lambda_max| and |lambda_max| ~ 8 / dx^2 for this
    # 5-point FV Laplacian, so coeff = (kappa - 1) * dx^2 / 8 puts the solve at
    # the requested condition number rather than at a hand-picked coefficient.
    dx_min = 6.371e6 * (np.pi / 2) / n / np.sqrt(3)
    coeff = (a.kappa - 1.0) * dx_min ** 2 / 8.0

    cd_ref = _build(PrecisionPolicy.fp64(), n)
    cd_mix = _build(PrecisionPolicy.mixed(), n)
    print(f"n={n}  kappa_target={a.kappa:.3e}  coeff={coeff:.6e}")
    print(f"  fp64 grid   rdxc dtype = {cd_ref.rdxc.dtype}")
    print(f"  mixed grid  rdxc dtype = {cd_mix.rdxc.dtype}")

    np.random.seed(0)
    rhs64 = jnp.asarray(np.random.randn(6, n, n))

    # Reference: everything fp64.
    set_policy(PrecisionPolicy.fp64())
    sol_ref, res_ref = sic.cg_helmholtz_solve(
        rhs64, coeff=coeff, cdgrid=cd_ref, tol=a.tol,
        maxiter=a.maxiter, return_residual=True)
    print(f"\nREFERENCE fp64 grid + fp64 policy")
    print(f"  reported rel_res      = {float(res_ref):.6e}")

    # Mixed arms.  The control is handed the raw mixed grid (float32 metrics);
    # the other arm is handed the same grid with those five arrays upcast.
    # That is the non-vacuity control: the fix has to move this number or it
    # does nothing.
    for label, promote in (("mixed, metrics PROMOTED (the fix)", True),
                           ("mixed, metrics f32 (pre-fix control)", False)):
        set_policy(PrecisionPolicy.mixed())
        cd_arm = _with_promoted_metrics(cd_mix) if promote else cd_mix
        rhs32 = jnp.asarray(rhs64, dtype=jnp.float32)
        sol, res = sic.cg_helmholtz_solve(
            rhs32, coeff=coeff, cdgrid=cd_arm, tol=a.tol,
            maxiter=a.maxiter, return_residual=True)
        fwd = float(jnp.linalg.norm(
            (jnp.asarray(sol, dtype=jnp.float64) - sol_ref).ravel())
            / jnp.linalg.norm(sol_ref.ravel()))
        res_true = _f64_operator_residual(sol, rhs64, coeff, cd_ref)
        print(f"\n{label}")
        print(f"  reported rel_res      = {float(res):.6e}   "
              f"(vs the operator it was handed)")
        print(f"  rel_res vs fp64 op    = {res_true:.6e}   "
              f"(vs the fp64-BUILT operator)")
        print(f"  forward error vs fp64 = {fwd:.6e}")

    set_policy(PrecisionPolicy.fp64())
    print("\nREADING RULE (pre-registered):")
    print("  the fix WORKS if the promoted arm's reported rel_res drops to the")
    print("  requested tol while the f32 control floors near 1e-6/1e-7.")
    print("  the fix is INSUFFICIENT if the promoted arm's FORWARD ERROR stays")
    print("  O(0.1) -- that would mean the metric VALUES, not the arithmetic,")
    print("  carry the error and the metrics must be rebuilt at fp64.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
