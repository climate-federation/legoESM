"""Probe: implicit-CN Helmholtz preconditioners on the synthetic tripole.

Serial, x64 operator. Same production pieces as the solver step:
_h_total_at_faces (min-depth faces, fold-aware), coeff = th_eta*th_pgf*dt^2*g,
_helmholtz_inv_diag, _make_helmholtz, _select_preconditioner,
solve_helmholtz_implicit(distributed=True) (the fixed-M PCG the GPU lane runs).

Prints, per grid size:
  * Gershgorin lambda_max (Chebyshev window top) vs power-iteration lambda_max
  * lambda_min estimate (window bottom assumed = 1)
  * W-symmetry defect of A and of each M_inv
  * Mmin/Mtop: ESTIMATE of M_inv's extreme W-Rayleigh quotients by power
    iteration (not a proof of definiteness; negative = indefinite witness)
  * true relative residual after M fixed iterations, per preconditioner, with
    PCG vectors in f32 ('f32v': operator and preconditioner still evaluated in
    f64, so NOT the GPU f32 operator) and in f64
"""
from __future__ import annotations

import argparse
import json
import subprocess

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.grids.tripole import create_synthetic_tripole
from legoesm.ocean.dynamics.barotropic_common import solve_helmholtz_implicit
from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import (
    _h_total_at_faces, _helmholtz_inv_diag, _make_helmholtz, _select_preconditioner,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import BarotropicConfig
from legoesm.ocean.vertical import create_ocean_z_star


def wdot(x, y, w):
    return float(jnp.sum(x * y * w))


def power_max(op, x, w, n, mask):
    # Iterate stays on the wet subspace: land cells carry weight 1 in w.
    for _ in range(n):
        y = op(x) * mask
        lam = wdot(x, y, w) / wdot(x, x, w)
        x = y / np.sqrt(wdot(y, y, w))
    return lam


def jacobi_cheb(A, inv_diag, mask, degree, lmin, lmax):
    """Probe-only: Chebyshev polynomial in D^-1 A on [lmin, lmax] (Barrett Templates)."""
    c, d = (lmax - lmin) / 2.0, (lmax + lmin) / 2.0
    def M(r):
        b = r * inv_diag
        x = jnp.zeros_like(b); res = b; p = jnp.zeros_like(b)
        for i in range(degree):
            if i == 0:
                p = res; alpha = 1.0 / d
            else:
                beta = 0.5 * (c * alpha) ** 2 if i == 1 else (c * alpha / 2.0) ** 2
                alpha = 1.0 / (d - beta / alpha); p = res + beta * p
            x = x + alpha * p
            res = res - alpha * (A(p) * inv_diag)
        return x * mask
    return M


def build(n_lat, n_lon, dt, nlev):
    grid = create_synthetic_tripole(n_lat=n_lat, n_lon=n_lon)
    z = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    st = rest_state_latlon_cgrid_ocean(grid, z, T_water_init_C=20.0, T_deep=2.0,
                                       S_uniform=35.0, H_max=4000.0)
    mask = jnp.asarray(np.asarray(st.land_mask.data, np.float64))  # 1 = ocean (state docstring)
    h_k = jnp.asarray(np.asarray(st.h.data, np.float64)) if hasattr(st, "h") else None
    if h_k is None or h_k.ndim != 3:
        h_k = jnp.broadcast_to((4000.0 / nlev) * mask[..., None], mask.shape + (nlev,))
    H_u, H_v = _h_total_at_faces(h_k, jnp.asarray(1.0), mask, grid)
    u_mask = (H_u > 0).astype(jnp.float64)
    v_mask = (H_v > 0).astype(jnp.float64)
    cfg = BarotropicConfig()
    coeff = jnp.asarray(cfg.barotropic_implicit_theta_eta * cfg.barotropic_implicit_theta_pgf
                        * dt * dt * constants.g)
    return grid, mask, H_u, H_v, u_mask, v_mask, coeff


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="384x544,768x1088")
    p.add_argument("--dt", type=float, default=600.0)
    p.add_argument("--nlev", type=int, default=5)
    p.add_argument("--iters", default="15,30,60,120,200")
    p.add_argument("--power-iters", type=int, default=400)
    p.add_argument("--variant", default="standard", choices=["standard", "single_reduce"])
    a = p.parse_args()
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(json.dumps({"probe": "tripole_helmholtz_precond", "sha": sha, **vars(a)}), flush=True)
    for s in a.sizes.split(","):
        n_lat, n_lon = map(int, s.split("x"))
        grid, mask, H_u, H_v, u_mask, v_mask, coeff = build(n_lat, n_lon, a.dt, a.nlev)
        w = grid.area * mask + (1.0 - mask)
        A = jax.jit(_make_helmholtz(H_u, H_v, coeff, grid, mask, u_mask, v_mask))
        inv_diag = _helmholtz_inv_diag(H_u, H_v, coeff, grid, mask)
        diag = jnp.where(mask > 0.5, 1.0 / inv_diag, 0.0)
        gersh = float(jnp.max(jnp.where(mask > 0.5, 2 * diag - 1, -jnp.inf)))
        rng = np.random.default_rng(0)
        x0 = jnp.asarray(rng.standard_normal(mask.shape)) * mask
        lmax = power_max(A, x0, w, a.power_iters, mask)
        lmin = lmax - power_max(lambda v: (lmax * v - A(v)) * mask, x0, w, a.power_iters, mask)
        y0 = jnp.asarray(rng.standard_normal(mask.shape)) * mask
        def wsym(op):
            l, r = wdot(x0, op(y0), w), wdot(op(x0), y0, w)
            return abs(l - r) / max(abs(l), abs(r))
        print(f"\n== tripole {n_lat}x{n_lon} dt={a.dt} wet={int(mask.sum())} "
              f"diag range [{float(jnp.min(jnp.where(mask>0.5,diag,jnp.inf))):.3g}, {float(jnp.max(diag)):.3g}]")
        print(f"lambda_max: gershgorin {gersh:.6g}  power {lmax:.6g}  ratio power/gersh {lmax/gersh:.4f}")
        print(f"lambda_min (power on lmax-A): {lmin:.6g}  (Chebyshev window assumes wet lambda_min >= 1)")
        print(f"A W-symmetry defect: {wsym(A):.2e}")
        pcs = {"jacobi": dict(name="jacobi", deg=0), "zonal_line": dict(name="zonal_line", deg=0)}
        for d in (2, 4, 8):
            pcs[f"cheb{d}"] = dict(name="chebyshev", deg=d)
        DA = lambda v: A(v) * inv_diag
        jmax = power_max(DA, x0, w, a.power_iters, mask)
        jmin = jmax - power_max(lambda v: (jmax * v - DA(v)) * mask, x0, w, a.power_iters, mask)
        print(f"D^-1 A spectrum (power): [{jmin:.4g}, {jmax:.4g}]  Gershgorin top 2")
        for d in (2, 4, 8):
            pcs[f"jcheb{d}_win2/30"] = dict(name="jcheb", deg=d, win=(2.0 / 30.0, 2.0))
            pcs[f"jcheb{d}_measured"] = dict(name="jcheb", deg=d, win=(jmin, 2.0))
        b = jnp.asarray(rng.standard_normal(mask.shape)) * mask
        for key, pc in pcs.items():
            if pc["name"] == "jcheb":
                M = jacobi_cheb(A, inv_diag, mask, pc["deg"], *pc["win"])
            else:
                M = _select_preconditioner(pc["name"], inv_diag, H_u, H_v, coeff, grid, mask,
                                           A_op=A, cheby_degree=max(pc["deg"], 1))
            M = jax.jit(M)
            # most negative W-Rayleigh quotient: power iteration on (s*I - M), s = top of M
            mtop = power_max(lambda v: M(v) * mask, x0, w, 100, mask)
            mmin = mtop - power_max(lambda v: (mtop * v - M(v)) * mask, x0, w, a.power_iters, mask)
            row = [f"{key:20s} Msym {wsym(M):.1e} Mmin/Mtop {mmin/mtop:+.2e}"]
            for prec, dt_ in (("f32v", jnp.float32), ("f64", jnp.float64)):
                for M_it in map(int, a.iters.split(",")):
                    A_c = (lambda v, dt_=dt_: A(v.astype(jnp.float64)).astype(dt_))
                    M_c = (lambda v, dt_=dt_: M(v.astype(jnp.float64)).astype(dt_))
                    x, _ = solve_helmholtz_implicit(
                        A_c, b.astype(dt_), M_c, jnp.zeros_like(b, dt_), distributed=True,
                        fixed_iters=M_it, residual_tol=0.0, stock_cg_tol=0.0,
                        stock_cg_maxiter=1, pcg_variant=a.variant, dot_weight=w.astype(dt_))
                    r = float(jnp.linalg.norm(b - A(x.astype(jnp.float64))) / jnp.linalg.norm(b))
                    row.append(f"{prec}@{M_it}={r:.1e}")
            print("  ".join(row), flush=True)


if __name__ == "__main__":
    main()
