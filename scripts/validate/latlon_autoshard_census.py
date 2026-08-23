"""What does the compiler emit if it partitions the WHOLE lat-lon step?

A microbenchmark on a single latitude halo found automatic partitioning
producing exactly the same two neighbour exchanges as the hand-written
``shard_map`` path, with identical values and slightly less time at 32, 64 and
128 GPUs. That is one exchange. The production step runs thirteen, on a C-grid
with a staggered meridional velocity carried on ``n_lat+1`` rows, a pole fold,
and a global mass fixer.

This asks the compiler directly, on the real step, before anything is ported.
The serial model step -- the same code the hand-written path is bit-tested
against -- is jitted with a latitude sharding on every state leaf, and the
compiled module's communication is counted.

The staggered velocity is carried as the ``n_lat``-row lower faces, exactly as
the hand-written path carries it, so no leaf has a row count that fails to
divide the device count. Without that the comparison would be measuring an
unequal split rather than the decomposition.

The number to compare against is the hand-written step's own census, taken from
a real run's dumped module: THIRTEEN collective-permutes and ONE all-reduce, at
32, 64 and 128 devices alike.

Runs on host CPU devices; the collective STRUCTURE is a property of the program
and not of the device type or the grid size.

    XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
    python scripts/validate/latlon_autoshard_census.py --n-devices 8
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np

KINDS = ("collective-permute", "all-gather", "all-reduce", "all-to-all",
         "reduce-scatter", "collective-broadcast", "send", "recv")
#: The hand-written path, censused from the module a real run compiled
#: (job 27102260), identical at 32, 64 and 128 devices.
MANUAL_REFERENCE = {"collective-permute": 13, "all-reduce": 1}


def census(text):
    out = {}
    for kind in KINDS:
        n = len(re.findall(rf"(?<![\w-]){kind}(?:-start)?\(", text))
        if n:
            out[kind] = n
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-devices", type=int, default=8)
    ap.add_argument("--n-lat", type=int, default=32)
    ap.add_argument("--n-lon", type=int, default=64)
    ap.add_argument("--nlev", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from legoesm import constants
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig, CGridLatLonPrimitiveEquationModel,
        hydrostatic_to_cgrid)
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_latlon)
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    nd, nlat = args.n_devices, args.n_lat
    if nd > len(jax.devices()):
        raise SystemExit(f"--n-devices {nd} > {len(jax.devices())} visible")
    if nlat % nd:
        raise SystemExit(f"--n-lat {nlat} must divide {nd} devices")

    grid = create_latlon_grid(n_lat=nlat, n_lon=args.n_lon,
                             radius=constants.R_earth, omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=args.nlev)
    model = CGridLatLonPrimitiveEquationModel(
        grid, sigma,
        CGridLatLonPrimitiveEquationConfig(
            fix_mass=True, use_polar_filter=False, use_ppm_transport=True,
            time_integrator="ssp_rk3"))
    c0 = hydrostatic_to_cgrid(held_suarez_init_latlon(grid, sigma), grid)
    mesh = Mesh(np.array(jax.devices()[:nd]), ("lat",))

    lower = c0._replace(v=c0.v[:nlat])
    v_top = c0.v[nlat:]

    def spec(x):
        return (P("lat", *([None] * (x.ndim - 1)))
                if getattr(x, "ndim", 0) and x.shape[0] == nlat else P())

    leaves, treedef = jax.tree.flatten(lower)
    if any(spec(l) == P() for l in leaves):
        raise SystemExit(
            "a state leaf could not be sharded on latitude; the census would "
            "then be measuring a replicated leaf rather than the decomposition")
    shard = jax.tree.unflatten(
        treedef, [NamedSharding(mesh, spec(l)) for l in leaves])
    rep = NamedSharding(mesh, P())

    def step(s, top):
        full = s._replace(v=jnp.concatenate([s.v, top], axis=0))
        out = model.step(full, 60.0)
        return out._replace(v=out.v[:nlat]), out.v[nlat:]

    fn = jax.jit(step, in_shardings=(shard, rep), out_shardings=(shard, rep))
    got = census(fn.lower(jax.device_put(lower, shard),
                          jax.device_put(v_top, rep)).compile().as_text())

    rec = {
        "component": "latlon_autoshard_census",
        "n_devices": nd, "n_lat": nlat, "n_lon": args.n_lon,
        "nlev": args.nlev, "backend": jax.default_backend(),
        "auto_partitioned": got,
        "hand_written_reference": MANUAL_REFERENCE,
        "auto_total": sum(got.values()),
        "hand_written_total": sum(MANUAL_REFERENCE.values()),
        "emits_gathers": bool(got.get("all-gather") or got.get("all-to-all")),
    }
    print(json.dumps(rec, indent=2))
    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    # No verdict is printed. The interpretation belongs in the analysis, and a
    # probe that prints its own conclusion gets that conclusion quoted back.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
