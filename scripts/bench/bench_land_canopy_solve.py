#!/usr/bin/env python
"""Steady-state wall time of one two-leaf canopy call on synthetic columns.

Times the jitted ``compute_two_leaf_canopy_fluxes`` (6 Picard passes, grouped
Newton solves) on ``--ncol`` synthetic columns with the production iteration cap,
excluding compile, and optionally writes a jax.profiler trace of one call.
Prints the per-call time and a digest of the outputs so two code versions can be
compared for both speed and answer.

USE (JAX_ENABLE_X64=1, on a GPU node):
  python scripts/bench/bench_land_canopy_solve.py --ncol 7403 --cap 10 \
      [--reps 5] [--trace DIR] [--save out.npz]
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Synthetic soil response for the Picard coupling: dT_soil/dG [K per W m-2].
# Only gives the outer loop something to relax; not a physical soil model.
SYNTH_SOIL_DT_PER_G = 1.0e-3


def build_call(ncol: int, cap: int):
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.surface_scheme import TwoLeafCanopyConfig
    from legoesm.land.surface_scheme.two_leaf_canopy import (
        compute_two_leaf_canopy_fluxes)

    r = np.random.default_rng(3)
    f = lambda a: jnp.asarray(a, dtype=jnp.float64)
    Ta = 290.0 + 8.0 * r.random(ncol)
    forcing = AtmToSurface(
        sw_down=f(200.0 + 400.0 * r.random(ncol)),
        lw_down=f(300.0 + 60.0 * r.random(ncol)),
        precip_total=f(np.zeros(ncol)), precip_snow=f(np.zeros(ncol)),
        T_lowest=f(Ta), q_lowest=f(0.006 + 0.006 * r.random(ncol)),
        u_lowest=f(1.0 + 3.0 * r.random(ncol)), v_lowest=f(np.zeros(ncol)),
        p_lowest=f(np.full(ncol, 99000.0)), p_surface=f(np.full(ncol, 100000.0)),
        rho_lowest=f(np.full(ncol, 1.2)), cos_zenith=f(0.3 + 0.5 * r.random(ncol)),
        co2_ppmv=f(np.full(ncol, 400.0)), has_radiation=f(np.ones(ncol)),
        has_precipitation=f(np.ones(ncol)))
    T_soil = f(288.0 + 8.0 * r.random(ncol))
    w_frac = f(0.3 + 0.6 * r.random(ncol))
    cfg = TwoLeafCanopyConfig(max_iters=cap)
    land_cfg = MultiLayerLandConfig()

    @jax.jit
    def call(forcing, T_soil, w_frac):
        out = compute_two_leaf_canopy_fluxes(
            T_soil_top=T_soil, forcing=forcing, canopy_config=cfg,
            land_config=land_cfg, canopy_params=None, w_frac_rz=w_frac,
            wind_speed=jnp.abs(forcing.u_lowest),
            wind_dir_x=jnp.ones_like(w_frac), wind_dir_y=jnp.zeros_like(w_frac),
            soil_thermal_fn=lambda G, dt_: T_soil + SYNTH_SOIL_DT_PER_G * G, dt=1800.0)
        return {k: getattr(out, k) for k in
                ("shflx", "lhflx", "T_surface", "gpp", "converged", "canopy_x")
                if getattr(out, k, None) is not None}
    return call, (forcing, T_soil, w_frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncol", type=int, default=7403)
    ap.add_argument("--cap", type=int, default=10)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--trace", default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--group", type=int, default=None,
                    help="override the canopy solve's column group size")
    a = ap.parse_args()
    if a.group is not None:
        from legoesm.land.surface_scheme import two_leaf_canopy
        two_leaf_canopy._CANOPY_SOLVE_GROUP = a.group
    from legoesm.land.surface_scheme import two_leaf_canopy as _tlc
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], text=True,
                         capture_output=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], text=True,
                           capture_output=True).stdout.count("\n")
    print(f"commit {rev} dirty {dirty} jax {jax.__version__} "
          f"group {_tlc._CANOPY_SOLVE_GROUP}")
    call, args = build_call(a.ncol, a.cap)
    t0 = time.perf_counter()
    out = jax.block_until_ready(call(*args))
    print(f"compile+first call {time.perf_counter() - t0:.1f} s "
          f"device {jax.devices()[0]}")
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        jax.block_until_ready(call(*args))
        ts.append(time.perf_counter() - t0)
    print(f"canopy call ncol={a.ncol} cap={a.cap}: median {np.median(ts):.3f} s "
          f"(min {min(ts):.3f}, reps {a.reps})")
    host = {k: np.asarray(v) for k, v in out.items()}
    h = hashlib.sha256(b"".join(host[k].tobytes() for k in sorted(host)))
    print(f"converged {host['converged'].mean():.4f}  digest {h.hexdigest()[:16]}")
    if a.save:
        np.savez(a.save, **host)
    if a.trace:
        jax.profiler.start_trace(a.trace, create_perfetto_trace=True)
        jax.block_until_ready(call(*args))
        jax.profiler.stop_trace()
        print(f"trace written to {a.trace}")


if __name__ == "__main__":
    main()
