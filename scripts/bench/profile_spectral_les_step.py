"""Per-component cost of ONE serial spectral-LES step vs the bandwidth roofline.

Builds the LES-suite production dry-CBL configuration exactly as the scaling
bench does (``bench_spectral_les_dd_scaling``: ``cbl_les_config`` = 3/2
de-aliasing, SSP-RK3, buoyancy, the case's first SGS variant), then times the
full jitted step and each operator it is built from, each jitted on its own:

* ``rhs``            one full right-hand side (called 3x per RK3 step)
* ``advection``      rotational advection with 3/2 pad/truncate
* ``eddy_viscosity`` SGS coefficient (LASD dynamic procedure when selected)
* ``sgs_and_wall``   SGS stress divergence + wall model
* ``scalar_rhs``     theta advection + SGS flux
* ``project``        pressure projection (called 3x per step)
* ``filt_state``     end-of-step spectral cutoff
* ``fft_pair``       one rfft2 + irfft2 of a (ny, nx, nz) field

For each: median ms over ``--reps`` calls, XLA ``cost_analysis`` flops and
bytes accessed, the number of FFT ops in the optimised HLO, and the
bandwidth-bound time ``bytes / --bw-gbs``. XLA's ``bytes accessed`` counts every
HLO operand/result once (an upper bound on DRAM traffic when fusions keep data
in cache, a lower bound when they spill), so the roofline column is an estimate
of the memory floor, not a measurement.

    JAX_PLATFORMS=cpu python scripts/bench/profile_spectral_les_step.py \\
        --precision float64 --nx 64 --ny 64 --nz 128 --output prof.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--precision", choices=["float32", "float64"],
                   default="float64")
    p.add_argument("--case", default="cbl_nieuwstadt")
    p.add_argument("--nx", type=int, required=True)
    p.add_argument("--ny", type=int, required=True)
    p.add_argument("--nz", type=int, required=True)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--bw-gbs", type=float, default=21.0,
                   help="memory bandwidth for the roofline column [GB/s]")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--trace", type=str, default=None,
                   help="also record a perfetto trace of --reps full steps "
                        "here and print XLA op time by category")
    return p


def trace_categories(trace_dir: str, n_calls: int) -> list[tuple[str, float, int]]:
    """``(category, ms per call, ops per call)`` from a perfetto trace: the
    XLA HLO-op events (those carrying an ``hlo_op`` argument) grouped by HLO
    name with the numeric suffix dropped (``fft.12`` -> ``fft``); host frames,
    executor/thread-pool wrappers and ``end:`` markers carry no ``hlo_op`` and
    are excluded."""
    import collections
    import glob
    import gzip
    paths = glob.glob(f"{trace_dir}/**/perfetto_trace.json.gz", recursive=True)
    if not paths:
        raise FileNotFoundError(f"no perfetto trace under {trace_dir}")
    ev = json.load(gzip.open(sorted(paths)[-1]))
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    dur, cnt = collections.Counter(), collections.Counter()
    for e in ev:
        name = e.get("name", "")
        if e.get("ph") != "X" or "hlo_op" not in (e.get("args") or {}):
            continue
        c = re.sub(r"\.\d+$", "", name)
        dur[c] += e.get("dur", 0)
        cnt[c] += 1
    return [(c, t / 1e3 / n_calls, cnt[c] / n_calls)
            for c, t in dur.most_common()]


def fft_flops(ny: int, nx: int, nz: int) -> float:
    """Nominal real 2-D FFT flops, 2.5 N log2 N per transform of N=ny*nx
    points per level (half of the 5 N log2 N complex count)."""
    n = ny * nx
    return 2.5 * n * math.log2(n) * nz


def _cost(compiled) -> tuple[float, float, int]:
    ca = compiled.cost_analysis()
    if isinstance(ca, list):
        ca = ca[0]
    hlo = compiled.as_text()
    n_fft = len(re.findall(r"\bfft\(", hlo))
    return float(ca.get("flops", 0.0)), float(ca.get("bytes accessed", 0.0)), n_fft


def main(argv=None):
    pargs, rest = build_parser().parse_known_args(argv)
    if pargs.precision == "float32" and "--f32" not in sys.argv:
        sys.argv.append("--f32")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "run"))
    import jax
    import jax.numpy as jnp

    import run_les_suite as les
    from bench_plane_crm_dd_scaling import provenance
    from bench_spectral_les_dd_scaling import local_cbl_state, scaled_grid
    from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl
    from legoesm.atmosphere.les_suite import (
        get_case, list_cases, register_default_catalog)
    from legoesm.timestepping.split_explicit import select_dt

    jax.config.update("jax_enable_x64", pargs.precision == "float64")
    dtype = jnp.float32 if pargs.precision == "float32" else jnp.float64
    args = les.build_arg_parser().parse_args(["--case", pargs.case] + rest)
    if not list_cases():
        register_default_catalog()
    case = get_case(args.case)
    if case.regime != "dry_convective":
        raise SystemExit(f"{case.name}: profiler wires dry_convective only")
    if args.theta0 is None:
        args.theta0 = les._CBL_THETA0_K
    u_geo = args.Ug if args.Ug is not None else (case.geostrophic_wind_m_s or 0.0)
    lat = args.lat if args.lat is not None else les._CBL_LAT_DEG
    f_cor = les._coriolis_f(lat) if u_geo > 0.0 else 0.0
    sgs = args.sgs if args.sgs is not None else case.sgs_variants[0]
    Q0 = args.q0 if args.q0 is not None else case.surface_theta_flux_K_m_s
    grid = scaled_grid(case.grid, pargs.nx, pargs.ny, pargs.nz)
    cfg = les.cbl_les_config(grid, args, sgs)
    g = sl.make_grid(cfg, dtype=dtype)
    st = local_cbl_state(sl, g, args, 0, pargs.ny, u_geo, dtype)
    step = les._make_emit_step(g, case.regime, (u_geo, 0.0), f_cor, Q0,
                               grid.Lz_m, dtype)
    dt0 = (float(args.dt) if args.dt is not None else
           select_dt(g.dx, max_wind_safe=args.max_wind, cfl_safe=args.cfl,
                     dt_cap=args.dt_max))
    dt = jnp.asarray(dt0, dtype)
    # A few steps so the strain / LASD fields are not the IC's degenerate ones.
    st, _ = step(st, dt=dt, first=True)
    for _ in range(2):
        st, _ = step(st, dt=dt, first=False)
    jax.block_until_ready(st)
    u, v, w, th = st.u, st.v, st.w, st.theta
    nu = jax.jit(lambda u, v, w: sl.eddy_viscosity(u, v, w, g))(u, v, w)
    ug = (u_geo, 0.0)

    comps = {
        "step": (lambda s: step(s, dt=dt, first=False), (st,), 1),
        "rhs": (lambda u, v, w, th: sl.rhs(u, v, w, g, ug, f_cor, theta=th,
                                           sfc_theta_flux=Q0),
                (u, v, w, th), 3),
        "advection": (lambda u, v, w: sl.advection(u, v, w, g), (u, v, w), 3),
        "eddy_viscosity": (lambda u, v, w: sl.eddy_viscosity(u, v, w, g),
                           (u, v, w), 3),
        "sgs_and_wall": (lambda u, v, w, nu: sl.sgs_and_wall(u, v, w, nu, g, ug),
                         (u, v, w, nu), 3),
        "scalar_rhs": (lambda th, u, v, w, nu: sl.scalar_rhs(th, u, v, w, nu, g, Q0),
                       (th, u, v, w, nu), 3),
        "project": (lambda u, v, w: sl.project(u, v, w, dt, g), (u, v, w), 3),
        "filt_state": (lambda u, v, w, th: sl._filt_state(u, v, w, th, None, g),
                       (u, v, w, th), 1),
        "fft_pair": (lambda u: sl._ifft(sl._fft(u, g), g, u.shape[-1]), (u,), 0),
    }
    rows = []
    for name, (fn, a, per_step) in comps.items():
        if name == "step":
            jf = step
            # Cost of the SAME executable that is timed (dt dynamic, not folded).
            compiled = step.lower(*a, dt=dt, first=False).compile()
        else:
            jf = jax.jit(fn)
            compiled = jf.lower(*a).compile()
        flops, nbytes, n_fft = _cost(compiled)
        jax.block_until_ready(jf(*a) if name != "step" else fn(*a))
        ts = []
        for _ in range(pargs.reps):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*a) if name == "step" else jf(*a))
            ts.append(time.perf_counter() - t0)
        ms = 1e3 * statistics.median(ts)
        rows.append(dict(component=name, calls_per_step=per_step, ms=ms,
                         flops=flops, bytes=nbytes, n_fft_ops=n_fft,
                         roofline_ms=1e3 * nbytes / (pargs.bw_gbs * 1e9),
                         gflops_s=flops / (ms * 1e6) if ms > 0 else None))
    rec = {"model": "spectral_les", "case": case.name, "sgs": sgs,
           "precision": pargs.precision, "nx": pargs.nx, "ny": pargs.ny,
           "nz": pargs.nz, "dt": dt0, "bw_gbs": pargs.bw_gbs,
           "fft_flops_per_pair_nominal": 2 * fft_flops(pargs.ny, pargs.nx, pargs.nz),
           "field_bytes": pargs.ny * pargs.nx * pargs.nz * jnp.dtype(dtype).itemsize,
           "components": rows, "provenance": provenance()}
    print(json.dumps(rec), flush=True)
    for r in rows:
        print(f"{r['component']:>15s} x{r['calls_per_step']} {r['ms']:10.2f} ms "
              f"roof {r['roofline_ms']:9.2f} ms  {r['flops']/1e9:8.2f} GF "
              f"{r['bytes']/1e9:8.2f} GB  fft_ops={r['n_fft_ops']}", flush=True)
    if pargs.trace:
        with jax.profiler.trace(pargs.trace, create_perfetto_trace=True):
            for _ in range(pargs.reps):
                jax.block_until_ready(step(st, dt=dt, first=False))
        cats = trace_categories(pargs.trace, pargs.reps)
        rec["trace_categories"] = cats
        for c, ms, n in cats[:15]:
            print(f"   trace {c:>32s} {ms:10.2f} ms/step  n={n:g}", flush=True)
    if pargs.output:
        out = Path(pargs.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
