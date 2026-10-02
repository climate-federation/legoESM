"""Where and when does the lat-lon C-grid benchmark state go non-finite?

Runs the benchmark's own model (``bench_atm_latlon_spmd_scaling._build_model``
and its Held-Suarez-IC builders) either serially or on N host CPU devices in
the lat-band SPMD step, checks every prognostic field after every step, and
prints: per-step max|u|, max|v|, min/max T, min/max p_s (with the latitude
row of the max|u| / max|v|), then the first step and field that is
non-finite.  Config overrides (polar filter, Laplacian viscosity A_h) are
applied with ``_replace`` on the benchmark's config, so every other setting is
the benchmark's.

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
    python scripts/validate/latlon_first_nonfinite.py --n-lat 128 --n-lon 256 \\
        --dt 60 --steps 200 --devices 4 --polar-filter on
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]


def _bench():
    spec = importlib.util.spec_from_file_location(
        "bench_atm_latlon_spmd_scaling",
        REPO / "scripts/bench/bench_atm_latlon_spmd_scaling.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stats(c, n_lat):
    """Host-side summary; v has n_lat+1 rows (v-faces), u has n_lat rows."""
    u, v, T, ps = (np.asarray(c.u), np.asarray(c.v), np.asarray(c.T), np.asarray(c.p_s))
    out = {}
    for name, a in (("u", u), ("v", v), ("T", T), ("p_s", ps)):
        bad = ~np.isfinite(a)
        out[name + "_bad"] = int(bad.sum())
        out[name + "_bad_rows"] = sorted(set(np.nonzero(bad)[0].tolist()))[:6]
    au, av = np.abs(np.nan_to_num(u)), np.abs(np.nan_to_num(v))
    out["umax"], out["urow"] = float(au.max()), int(np.unravel_index(au.argmax(), au.shape)[0])
    out["vmax"], out["vrow"] = float(av.max()), int(np.unravel_index(av.argmax(), av.shape)[0])
    out["T"] = (float(np.nanmin(T)), float(np.nanmax(T)))
    out["ps"] = (float(np.nanmin(ps)), float(np.nanmax(ps)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-lat", type=int, required=True)
    p.add_argument("--n-lon", type=int, required=True)
    p.add_argument("--nlev", type=int, default=26)
    p.add_argument("--dt", type=float, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--polar-filter", choices=["on", "off"], required=True)
    p.add_argument("--A-h", type=float, default=None, help="override Laplacian viscosity [m^2/s]")
    p.add_argument("--cutoff-deg", type=float, default=None, help="override polar_filter_cutoff_deg")
    p.add_argument("--every", type=int, default=10)
    args = p.parse_args()

    import jax
    b = _bench()
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationModel, hydrostatic_to_cgrid)
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_latlon
    base = b._build_model(args.n_lat, args.n_lon, args.nlev, args.dt)
    over = {"use_polar_filter": args.polar_filter == "on"}
    if args.A_h is not None:
        over["A_h"] = args.A_h
    if args.cutoff_deg is not None:
        over["polar_filter_cutoff_deg"] = args.cutoff_deg
    model = CGridLatLonPrimitiveEquationModel(
        base.grid, base.sigma_coord, base.config._replace(**over), dt=args.dt)
    print(f"config: n_lat={args.n_lat} n_lon={args.n_lon} nlev={args.nlev} dt={args.dt} "
          f"devices={args.devices} overrides={over}", flush=True)

    if args.devices == 1:
        c = hydrostatic_to_cgrid(held_suarez_init_latlon(model.grid, model.sigma_coord), model.grid)
        step = jax.jit(lambda s: model.step(s, args.dt))
        gather = lambda s: s  # noqa: E731
    else:
        from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
            build_sharded_held_suarez_state_atm_latlon, make_sharded_atm_latlon_step)
        if len(jax.devices()) < args.devices:
            raise SystemExit(f"needs {args.devices} devices, have {len(jax.devices())}")
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:args.devices]), axis_names=("lat",))
        c = build_sharded_held_suarez_state_atm_latlon(model.grid, model.sigma_coord, mesh)
        sstep = make_sharded_atm_latlon_step(model, mesh)
        step = lambda s: sstep(s, args.dt)  # noqa: E731
        gather = lambda s: s  # noqa: E731  (single process: np.asarray gathers)

    st = _stats(gather(c), args.n_lat)
    print(f"step 0: umax={st['umax']:.3e}@row{st['urow']} vmax={st['vmax']:.3e}@row{st['vrow']} "
          f"T={st['T'][0]:.1f}..{st['T'][1]:.1f} ps={st['ps'][0]:.0f}..{st['ps'][1]:.0f}", flush=True)
    for k in range(1, args.steps + 1):
        c = step(c)
        st = _stats(gather(c), args.n_lat)
        nbad = sum(st[f + "_bad"] for f in ("u", "v", "T", "p_s"))
        other = [kp for kp, x in jax.tree_util.tree_flatten_with_path(c)[0]
                 if x is not None and np.asarray(x).dtype.kind == "f"
                 and not np.all(np.isfinite(np.asarray(x)))]
        if other and not nbad:
            print("FIRST NON-FINITE at step", k, "in", [jax.tree_util.keystr(kp) for kp in other])
            return 1
        if nbad or k % args.every == 0 or k <= 3:
            print(f"step {k}: umax={st['umax']:.3e}@row{st['urow']} vmax={st['vmax']:.3e}@row{st['vrow']} "
                  f"T={st['T'][0]:.1f}..{st['T'][1]:.1f} ps={st['ps'][0]:.0f}..{st['ps'][1]:.0f}", flush=True)
        if nbad:
            print("FIRST NON-FINITE at step", k, {f: (st[f + "_bad"], st[f + "_bad_rows"])
                                                  for f in ("u", "v", "T", "p_s") if st[f + "_bad"]})
            return 1
    print(f"FINITE through {args.steps} steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
