"""AD vs central finite differences for two-leaf canopy columns stopped at the cap.

Columns that stop at the iteration cap keep their last iterate
(``CanopyConfig.fallback_from_last_iterate``).  This compares d(loss)/d(eps)
by reverse mode and by central differences, where eps scales the shortwave
(or shifts the air temperature) and loss = sum of SH + LE over the accepted
unconverged columns, with the solver's adjoint on unconverged columns ON
(``adjoint_unconverged``) and OFF (zero cotangent, the old behaviour).
float64.  Prints one line per (max_iters, knob, adjoint): AD, FD, AD/FD.

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


def _measure(max_iters, T_init, adjoint):
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
            out.append((("sw_scale", "T_air")[i], float(g[i]), fd, int(mask.sum())))
        return out
    finally:
        canopy_solver._make_implicit_newton_solver = _ORIG


def main():
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"git {sha}  x64  N={_N}  T_init=307 K  loss=sum(SH+LE) accepted unconverged")
    for max_iters in (1, 2, 3, 5, 8):
        for adjoint in (False, True):
            res = _measure(max_iters, 307.0, adjoint)
            if res is None:
                print(f"max_iters={max_iters}: no accepted unconverged column")
                break
            for knob, ad, fd, n in res:
                ratio = ad / fd if fd != 0.0 else float("nan")
                print(f"max_iters={max_iters} n_cols={n} adjoint={'on ' if adjoint else 'off'} "
                      f"{knob:8s} AD={ad: .6e} FD={fd: .6e} AD/FD={ratio: .4f}")


if __name__ == "__main__":
    main()
