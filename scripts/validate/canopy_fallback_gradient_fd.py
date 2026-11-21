"""AD vs central finite differences for two-leaf canopy columns stopped at the cap.

Columns that stop at the iteration cap keep their last iterate
(``CanopyConfig.fallback_from_last_iterate``).  This compares d(loss)/d(eps)
by reverse mode and by central differences, where eps scales the shortwave
(or shifts the air temperature) and loss = sum of SH + LE over the accepted
unconverged columns, with the solver's adjoint on unconverged columns ON
(``adjoint_unconverged``) and OFF (zero cotangent, the old behaviour).
Second sweep ("near root"): the convergence gate is disabled (rtol = atol =
0), so every column stops at the cap close to its root, as the held columns
of a real run do; the residual left (squared, relative to the seed) is
printed.  float64.  Prints one line per (max_iters, knob, adjoint): AD, FD, AD/FD.

Run: JAX_ENABLE_X64=1 python scripts/validate/canopy_fallback_gradient_fd.py
"""
from __future__ import annotations

import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land import multilayer_land as ml
from legoesm.land.canopy import solver as canopy_solver
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.soil_grid import SoilGridConfig
from legoesm.land.surface_scheme import TwoLeafCanopyConfig

assert jax.config.read("jax_enable_x64"), "run with JAX_ENABLE_X64=1"
_N = 4
_ORIG = canopy_solver._make_implicit_newton_solver


def _forcing(eps_sw, eps_t, T=300.0):
    o = jnp.ones(_N)
    ps = 1e5 * o
    Ta = (T + eps_t) * o
    return AtmToSurface(
        sw_down=900.0 * (1.0 + eps_sw) * o, lw_down=400.0 * o, precip_total=0 * o,
        precip_snow=0 * o, T_lowest=Ta, q_lowest=0.008 * o,
        u_lowest=jnp.array([0.7, 1.4, 2.8, 3.6]), v_lowest=0 * o,
        p_lowest=0.99 * ps, p_surface=ps, rho_lowest=ps / (constants.R_d * Ta),
        cos_zenith=0.85 * o, co2_ppmv=340 * o, has_radiation=o,
        has_precipitation=o)


def _setup(max_iters, T_init):
    sch = TwoLeafCanopyConfig(max_iters=max_iters, fallback_from_last_iterate=True,
                              implicit_ground_flux=False)
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=10, dz_top=0.025, growth_factor=2.0,
                                 total_depth=3.0),
        surface_scheme=sch)
    st = ml.init_multilayer_land_state(_N, cfg, T_init=T_init, theta_init=0.12)

    def run(eps_sw, eps_t):
        return ml.step_multilayer_land_with_diagnostics(
            st, _forcing(eps_sw, eps_t), cfg, 1.0, 1800.0, lat=jnp.full(_N, -0.55))
    return run


def _measure(max_iters, T_init, adjoint, near_root=False):
    tol0 = (canopy_solver._LM_RTOL, canopy_solver._LM_ATOL)
    if near_root:
        canopy_solver._LM_RTOL, canopy_solver._LM_ATOL = 0.0, 0.0

    def patched(*a, **k):
        k["adjoint_unconverged"] = adjoint
        return _ORIG(*a, **k)
    canopy_solver._make_implicit_newton_solver = patched
    try:
        run = _setup(max_iters, T_init)
        _, r, _, so = jax.jit(run)(0.0, 0.0)
        mask = (~np.asarray(so.converged).astype(bool)
                & ~np.asarray(so.held).astype(bool))
        if not mask.any():
            return None
        rel = getattr(so, "canopy_resid_rel", None)
        rel = float(np.max(np.asarray(rel)[mask])) if rel is not None else float("nan")
        m = jnp.asarray(mask, dtype=jnp.float64)

        def loss(eps_sw, eps_t):
            _, rr, _, _ = run(eps_sw, eps_t)
            return jnp.sum(m * (rr.shflx + rr.lhflx))
        g = jax.jit(jax.grad(loss, argnums=(0, 1)))(0.0, 0.0)
        lf = jax.jit(loss)
        out = []
        for i, h in ((0, 1e-6), (1, 1e-4)):
            e = [0.0, 0.0]
            e[i] = h
            fp = lf(*e)
            e[i] = -h
            fm = lf(*e)
            fd = float((fp - fm) / (2 * h))
            out.append((("sw_scale", "T_air")[i], float(g[i]), fd, int(mask.sum()), rel))
        return out
    finally:
        canopy_solver._make_implicit_newton_solver = _ORIG
        canopy_solver._LM_RTOL, canopy_solver._LM_ATOL = tol0


def sweep(n=64, max_iters=1):
    """Far from root: many columns stopped one LM step from the cold seed.

    Per-column d(SH+LE)/d(T_air) by reverse mode (the columns are independent,
    so a jacobian of the per-column loss is diagonal) against central
    differences; prints the largest |AD|, |FD| and the AD/FD spread, to see
    whether the last-iterate adjoint ever blows up where it is least exact.
    """
    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.uniform(0.3, 6.0, n))
    ta = jnp.asarray(rng.uniform(285.0, 315.0, n))
    sw = jnp.asarray(rng.uniform(100.0, 1000.0, n))
    sch = TwoLeafCanopyConfig(max_iters=max_iters, fallback_from_last_iterate=True,
                              implicit_ground_flux=False)
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=10, dz_top=0.025, growth_factor=2.0,
                                 total_depth=3.0),
        surface_scheme=sch)
    st = ml.init_multilayer_land_state(n, cfg, T_init=307.0, theta_init=0.12)
    o = jnp.ones(n)
    ps = 1e5 * o

    def run(dT):
        Ta = ta + dT
        f = AtmToSurface(
            sw_down=sw, lw_down=400.0 * o, precip_total=0 * o, precip_snow=0 * o,
            T_lowest=Ta, q_lowest=0.008 * o, u_lowest=u, v_lowest=0 * o,
            p_lowest=0.99 * ps, p_surface=ps, rho_lowest=ps / (constants.R_d * Ta),
            cos_zenith=0.85 * o, co2_ppmv=340 * o, has_radiation=o,
            has_precipitation=o)
        return ml.step_multilayer_land_with_diagnostics(
            st, f, cfg, 1.0, 1800.0, lat=jnp.full(n, -0.55))

    _, _, _, so = jax.jit(run)(jnp.zeros(n))
    acc = (~np.asarray(so.converged).astype(bool)
           & ~np.asarray(so.held).astype(bool))

    def per_col(dT):
        _, r, _, _ = run(dT)
        return r.shflx + r.lhflx
    ad = np.diag(np.asarray(jax.jit(jax.jacrev(per_col))(jnp.zeros(n))))
    f = jax.jit(per_col)
    h = 1e-4
    fd = np.asarray((f(jnp.full(n, h)) - f(jnp.full(n, -h))) / (2 * h))
    a, d = ad[acc], fd[acc]
    r = a / np.where(d == 0.0, np.nan, d)
    print(f"sweep n={n} max_iters={max_iters} accepted_unconverged={acc.sum()} "
          f"max|AD|={np.max(np.abs(a)):.3e} max|FD|={np.max(np.abs(d)):.3e} "
          f"AD/FD min={np.nanmin(r):.3f} median={np.nanmedian(r):.3f} "
          f"max={np.nanmax(r):.3f}", flush=True)


def main():
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"git {sha}  x64  N={_N}  T_init=307 K  loss=sum(SH+LE) accepted unconverged")
    for near_root, iters in ((False, (1, 2)), (True, (1, 2, 3))):
        for max_iters in iters:
            for adjoint in (False, True):
                res = _measure(max_iters, 307.0, adjoint, near_root)
                if res is None:
                    print(f"near_root={near_root} max_iters={max_iters}: "
                          "no accepted unconverged column")
                    break
                for knob, ad, fd, n, rel in res:
                    ratio = ad / fd if fd != 0.0 else float("nan")
                    print(f"near_root={near_root} max_iters={max_iters} n_cols={n} "
                          f"max_resid_rel={rel:.2e} adjoint={'on ' if adjoint else 'off'} "
                          f"{knob:8s} AD={ad: .6e} FD={fd: .6e} AD/FD={ratio: .4f}",
                          flush=True)


if __name__ == "__main__":
    import sys
    sweep() if "--sweep" in sys.argv else main()
