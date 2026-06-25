"""Strong/weak scaling bench for the lat-band SPMD lat-lon C-grid hydrostatic
atm step (make_sharded_atm_latlon_step / run_atm_latlon_spmd, the A1 work).

Times the SHARDED step across an N-device ("lat",) mesh and reports per-step
wall time + speedup vs 1 device. Per-step granularity exposes whether the
shard_map is being RE-TRACED every call (the make_sharded_atm_latlon_step
sharded_step rebuilds shard_map per call): if steps 1.. are as slow as step 0,
the cost is host tracing, not device compute, and the scaling number is
meaningless until the shard_map is built once.

  strong: fixed (n_lat, n_lon, nlev), vary n_devices -> speedup = t(1)/t(n).
  weak:   n_lat = nlat_per_dev * n_devices (fixed per-device rows) -> ideal flat.

Device count is fixed at process start, so each n_devices runs as a SEPARATE
process (one sbatch step per count); this script benches ONE n_devices and
appends a JSON line. JAX_PLATFORMS=cpu with --xla_force_host_platform_device_count
gives virtual CPU devices (communication-overhead characterization, NOT a real
speedup); a real number needs one GPU per band.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np


def _build(n_lat, n_lon, nlev):
    from legoesm import constants
    from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig, CGridLatLonPrimitiveEquationModel)
    from legoesm.atmosphere.held_suarez import held_suarez_init_latlon
    from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
        hydrostatic_to_cgrid)
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon, radius=constants.R_earth,
                              omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=nlev)
    cfg = CGridLatLonPrimitiveEquationConfig(
        fix_mass=True, use_polar_filter=False, use_ppm_transport=True,
        time_integrator="ssp_rk3")
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)
    hs0 = held_suarez_init_latlon(grid, sigma)
    c0 = hydrostatic_to_cgrid(hs0, grid)
    return model, c0


def _block(state):
    jax.block_until_ready(jax.tree.leaves(state))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-lat", type=int, default=128)
    p.add_argument("--n-lon", type=int, default=256)
    p.add_argument("--nlev", type=int, default=30)
    p.add_argument("--n-devices", type=int, required=True)
    p.add_argument("--mode", choices=["strong", "weak"], default="strong")
    p.add_argument("--nlat-per-dev", type=int, default=32,
                   help="weak mode: lat rows per device")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--physics", choices=["none", "held_suarez"], default="none")
    p.add_argument("--dt", type=float, default=60.0)
    p.add_argument("--out", type=str, default="results/a1/spmd_scaling.jsonl")
    args = p.parse_args()

    from legoesm.atmosphere.dynamics.sharded_atm_latlon_step import (
        make_sharded_atm_latlon_step, shard_state_atm_latlon)
    physics_fn = None
    if args.physics == "held_suarez":
        from legoesm.atmosphere.held_suarez import held_suarez_forcing_latlon
        physics_fn = held_suarez_forcing_latlon

    nd = args.n_devices
    avail = len(jax.devices())
    if avail < nd:
        raise SystemExit(f"need {nd} devices, have {avail} "
                         f"(set --xla_force_host_platform_device_count)")
    n_lat = args.n_lat if args.mode == "strong" else args.nlat_per_dev * nd
    if n_lat % nd != 0:
        raise SystemExit(f"n_lat {n_lat} not divisible by n_devices {nd}")

    model, c0 = _build(n_lat, args.n_lon, args.nlev)

    if nd == 1:
        mesh = None
        step = make_sharded_atm_latlon_step(model, None, physics_fn=physics_fn)
        c = c0
    else:
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:nd]),
                                 axis_names=("lat",))
        step = make_sharded_atm_latlon_step(model, mesh, physics_fn=physics_fn)
        c = shard_state_atm_latlon(c0, mesh)

    # Per-step timing: step 0 includes compile; record each step so re-trace
    # (every step slow) is visible vs steady-state (steps 1.. fast).
    per_step_ms = []
    for i in range(args.steps):
        t0 = time.perf_counter()
        c = step(c, args.dt)
        _block(c)
        per_step_ms.append((time.perf_counter() - t0) * 1e3)

    steady = per_step_ms[args.warmup:]
    med = float(np.median(steady))
    rec = dict(
        mode=args.mode, n_devices=nd, n_lat=n_lat, n_lon=args.n_lon,
        nlev=args.nlev, physics=args.physics, steps=args.steps,
        platform=jax.default_backend(),
        compile_ms=round(per_step_ms[0], 1),
        steady_median_ms=round(med, 2),
        steady_min_ms=round(float(np.min(steady)), 2),
        per_step_ms=[round(x, 1) for x in per_step_ms],
        cells=n_lat * args.n_lon * args.nlev,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    print(f"[nd={nd} {args.mode} {n_lat}x{args.n_lon}x{args.nlev}] "
          f"compile={rec['compile_ms']}ms steady_median={med:.2f}ms/step "
          f"(per-step: {rec['per_step_ms']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
