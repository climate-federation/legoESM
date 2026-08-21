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

Runs on host CPU devices. The collective structure is NOT assumed to be
device-count invariant -- that assumption was made once here and measured to be
false for the automatic path, which is itself the finding: the hand-written
path's census does not move between 2, 4 and 8 devices and the automatic one
grows. Both paths are therefore censused at every device count rather than at
one.

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
#: The hand-written path is CENSUSED HERE, through this same function and at
#: the same device counts, rather than quoted from a production run. A literal
#: taken at 32-128 GPUs and compared against rows measured on a handful of CPU
#: devices is a cross-protocol comparison, which is what the first version of
#: this file did.


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

    def census_at(nd_, over, tendency_only=False, replicate=False):
        model, low, top, shard, mesh = _build(
            jax, jnp, P, NamedSharding, Mesh, _np, nd_, nlat, args.n_lon,
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

    def manual_census(nd_):
        """The hand-written path, through this same census function."""
        from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
            make_sharded_atm_latlon_step, shard_state_atm_latlon)
        model, low, top, _shard, mesh = _build(
            jax, jnp, P, NamedSharding, Mesh, _np, nd_, nlat, args.n_lon,
            args.nlev, {})
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
            hydrostatic_to_cgrid)
        from legoesm.atmosphere.forcing.idealized.held_suarez import (
            held_suarez_init_latlon)
        full = hydrostatic_to_cgrid(
            held_suarez_init_latlon(model.grid, model.sigma_coord), model.grid)
        step = make_sharded_atm_latlon_step(model, mesh)
        laid = shard_state_atm_latlon(full, mesh)
        return census(jax.jit(lambda s: step(s, 60.0))
                      .lower(laid).compile().as_text())

    # Both paths at every device count. The device sweep is not decoration: the
    # first version of this file asserted the structure was device-independent
    # and the sweep is what showed that to be false for one of the two paths.
    print(f"{'devices':>7}  {'path':<38} {'perm':>5} {'a2a':>5} {'agath':>6} "
          f"{'aredu':>6} {'total':>6}")
    rows = []
    sweep = [d for d in (2, 4, 8, 16, 32) if d <= nd and nlat % d == 0]
    for d in sweep:
        for name, fn in (("hand-written (shard_map + ppermute)",
                          lambda d_=d: manual_census(d_)),
                         ("automatic (GSPMD)",
                          lambda d_=d: census_at(d_, {}))):
            try:
                c = fn()
            except Exception as exc:
                print(f"{d:>7}  {name:<38} FAILED "
                      f"{type(exc).__name__}: {str(exc)[:40]}")
                rows.append({"devices": d, "path": name,
                             "error": f"{type(exc).__name__}: {exc}"})
                continue
            print(f"{d:>7}  {name:<38} {c.get('collective-permute', 0):>5} "
                  f"{c.get('all-to-all', 0):>5} {c.get('all-gather', 0):>6} "
                  f"{c.get('all-reduce', 0):>6} {sum(c.values()):>6}")
            rows.append({"devices": d, "path": name, "census": c,
                         "total": sum(c.values())})

    # Ablations, automatic path only, at the largest device count. Each changes
    # exactly one configuration field. Two caveats found in review and kept
    # visible: turning the mass fixer off also turns the mean surface-pressure
    # tendency ON (they share one gate), so that row moves two things; and with
    # the fixer left on, the mean-tendency row is a byte-identical no-op.
    print()
    for name, over in (("mass fixer off (also enables the mean tendency)",
                        dict(fix_mass=False)),
                       ("PPM transport off", dict(use_ppm_transport=False)),
                       ("pole velocity condition off",
                        dict(pole_v_bc=(False, False)))):
        try:
            c = census_at(nd, over)
        except Exception as exc:
            print(f"{nd:>7}  {name:<38} FAILED {type(exc).__name__}")
            continue
        print(f"{nd:>7}  {name:<38} {c.get('collective-permute', 0):>5} "
              f"{c.get('all-to-all', 0):>5} {c.get('all-gather', 0):>6} "
              f"{c.get('all-reduce', 0):>6} {sum(c.values()):>6}")
        rows.append({"devices": nd, "path": f"automatic, {name}",
                     "census": c, "total": sum(c.values())})

    rec = {"component": "latlon_autoshard_census", "n_devices": nd,
           "n_lat": nlat, "n_lon": args.n_lon, "nlev": args.nlev,
           "backend": jax.default_backend(),
           "ladder": rows}
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
