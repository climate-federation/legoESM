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


def _build(jax, jnp, P, NamedSharding, Mesh, np, nd, nlat, nlon, nlev, over):
    """Model, latitude-sharded state, and the shardings, for one ablation."""
    from legoesm import constants
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig, CGridLatLonPrimitiveEquationModel,
        hydrostatic_to_cgrid)
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_latlon)
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(n_lat=nlat, n_lon=nlon,
                              radius=constants.R_earth, omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=nlev)
    kw = dict(fix_mass=True, use_polar_filter=False, use_ppm_transport=True,
              time_integrator="ssp_rk3")
    kw.update(over)
    model = CGridLatLonPrimitiveEquationModel(
        grid, sigma, CGridLatLonPrimitiveEquationConfig(**kw))
    c0 = hydrostatic_to_cgrid(held_suarez_init_latlon(grid, sigma), grid)
    mesh = Mesh(np.array(jax.devices()[:nd]), ("lat",))
    # The staggered meridional velocity has nlat+1 rows. JAX REFUSES to shard
    # that across a device count that does not divide it -- an IndivisibleError,
    # not a slow path -- so carrying it as the nlat-row lower faces is not a
    # convenience, it is the only representation that can be sharded at all.
    low = c0._replace(v=c0.v[:nlat])
    top = jnp.asarray(np.asarray(c0.v[nlat:]))
    spec = lambda x: P("lat", *([None] * (x.ndim - 1)))
    leaves, treedef = jax.tree.flatten(low)
    if any(getattr(l, "shape", (0,))[0] != nlat for l in leaves):
        raise SystemExit("a state leaf is not nlat rows; it would replicate "
                         "and the census would measure that instead")
    shard = jax.tree.unflatten(
        treedef, [NamedSharding(mesh, spec(l)) for l in leaves])
    return model, low, top, shard, mesh


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
    import numpy as _np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    nd, nlat = args.n_devices, args.n_lat
    if nd > len(jax.devices()):
        raise SystemExit(f"--n-devices {nd} > {len(jax.devices())} visible")
    if nlat % nd:
        raise SystemExit(f"--n-lat {nlat} must divide {nd} devices")

    def census_of(over, tendency_only=False, replicate=False):
        model, low, top, shard, mesh = _build(
            jax, jnp, P, NamedSharding, Mesh, _np, nd, nlat, args.n_lon,
            args.nlev, over)
        if replicate:
            # Control: with nothing sharded there is nothing to communicate.
            # Any collective here would mean the compiler partitioned something
            # we never asked it to, and every other row would be suspect.
            shard = jax.tree.unflatten(jax.tree.structure(low),
                                       [NamedSharding(mesh, P())
                                        for _ in jax.tree.leaves(low)])

        def whole_step(s):
            out = model.step(
                s._replace(v=jnp.concatenate([s.v, top], axis=0)), 60.0)
            return out._replace(v=out.v[:nlat])

        def one_tendency(s):
            out = model.tendencies(
                s._replace(v=jnp.concatenate([s.v, top], axis=0)))
            return jax.tree.map(
                lambda x: x[:nlat] if getattr(x, "ndim", 0)
                and x.shape[0] in (nlat, nlat + 1) else x, out)

        fn = jax.jit(one_tendency if tendency_only else whole_step,
                     in_shardings=(shard,))
        return census(fn.lower(jax.device_put(low, shard)).compile().as_text())

    # Each rung changes exactly one thing against the baseline above it.
    ladder = [
        ("whole step, everything replicated", {}, False, True),
        ("whole step, default", {}, False, False),
        ("ONE tendency evaluation", {}, True, False),
        ("whole step, mass fixer off", dict(fix_mass=False), False, False),
        ("whole step, mean surface-pressure tendency off",
         dict(zero_mean_ps_tendency=False), False, False),
        ("whole step, PPM transport off",
         dict(use_ppm_transport=False), False, False),
        ("whole step, pole velocity condition off",
         dict(pole_v_bc=(False, False)), False, False),
    ]
    rows = []
    print(f"{'arm':<46} {'perm':>5} {'a2a':>5} {'agath':>6} {'aredu':>6} "
          f"{'total':>6}")
    for name, over, tend, repl in ladder:
        try:
            c = census_of(over, tendency_only=tend, replicate=repl)
        except Exception as exc:  # a rung that cannot build is reported, not hidden
            print(f"{name:<46} FAILED {type(exc).__name__}: {str(exc)[:50]}")
            rows.append({"arm": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        print(f"{name:<46} {c.get('collective-permute', 0):>5} "
              f"{c.get('all-to-all', 0):>5} {c.get('all-gather', 0):>6} "
              f"{c.get('all-reduce', 0):>6} {sum(c.values()):>6}")
        rows.append({"arm": name, "overrides": {k: str(v) for k, v in over.items()},
                     "tendency_only": tend, "replicated": repl,
                     "census": c, "total": sum(c.values())})
    ref = MANUAL_REFERENCE
    print(f"{'hand-written whole step (job 27102260)':<46} "
          f"{ref['collective-permute']:>5} {0:>5} {0:>6} "
          f"{ref['all-reduce']:>6} {sum(ref.values()):>6}")

    rec = {"component": "latlon_autoshard_census", "n_devices": nd,
           "n_lat": nlat, "n_lon": args.n_lon, "nlev": args.nlev,
           "backend": jax.default_backend(),
           "hand_written_reference": ref, "ladder": rows}
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
