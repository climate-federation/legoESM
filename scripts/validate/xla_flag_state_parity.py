"""Byte-compare the model state after N steps with and without an XLA flag.

XLA_FLAGS is read once per process at backend initialisation, so each arm runs
in its own subprocess; the parent compares every state leaf by shape, dtype and
raw bytes.  Lanes are the benchmark lanes the flag would be wired into, built
with the benchmarks' own model/state builders on 4 host CPU devices:

    python scripts/validate/xla_flag_state_parity.py \\
        --flag=--xla_cpu_use_fusion_emitters=false --lanes latlon mpas_atm

Exit 0: every lane bit-identical.  Exit 1: some lane differs (the max
absolute difference per leaf is printed).  Non-vacuity: every lane must
compare at least MIN_ELEMENTS elements, each leaf must be finite, and the
state must have moved from its initial condition (a model that did not step
compares equal whatever the flag does).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
MIN_ELEMENTS = 100_000
N_DEV = 4


def _load_bench(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts/bench" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _leaves(state):
    import jax
    flat, _ = jax.tree_util.tree_flatten_with_path(state)
    return {jax.tree_util.keystr(k): np.asarray(v) for k, v in flat if v is not None}


def _child(lane, steps, out):
    import jax
    if lane == "latlon":
        b = _load_bench("bench_atm_latlon_spmd_scaling")
        from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
            build_sharded_held_suarez_state_atm_latlon, make_sharded_atm_latlon_step)
        model = b._build_model(64, 128, 26, 60.0)
        if len(jax.devices()) < N_DEV:
            raise SystemExit(f"needs {N_DEV} devices, have {len(jax.devices())}")
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:N_DEV]), axis_names=("lat",))
        s = build_sharded_held_suarez_state_atm_latlon(model.grid, model.sigma_coord, mesh)
        step = make_sharded_atm_latlon_step(model, mesh)
        dt = 60.0
        run = lambda st: step(st, dt)  # noqa: E731
    elif lane == "mpas_atm":
        b = _load_bench("bench_mpas_spmd_scaling")
        from legoesm.parallel.sharded_dynamics import make_voronoi_sharded_step
        _, model, s, dev_config = b.build_model_and_state(
            5, 26, N_DEV, N_DEV, "sfc", lloyd_iterations=0)
        os.environ["LEGOESM_MPAS_WIDE_HALO"] = "1"
        step = make_voronoi_sharded_step(model, dev_config, halo_strategy="auto")
        dt = max(600.0 * 4.0 ** (4 - 5), 30.0)
        run = lambda st: step(st, dt)  # noqa: E731
    else:
        raise ValueError(f"unknown lane {lane!r}")
    s0 = _leaves(s)
    for _ in range(steps):
        s = run(s)
    s1 = _leaves(s)
    moved = sum(int(not np.array_equal(s0[k], s1[k])) for k in s1 if k in s0)
    np.savez(out, __moved__=np.array(moved), **s1)


def _compare(lane, a_path, b_path):
    a, b = np.load(a_path), np.load(b_path)
    keys = [k for k in a.files if k != "__moved__"]
    if sorted(keys) != sorted(k for k in b.files if k != "__moved__"):
        print(f"{lane}: leaf sets differ"); return False
    n_el = 0; same = True
    for k in keys:
        x, y = a[k], b[k]
        if x.dtype.kind in "fc" and not np.all(np.isfinite(x)):
            raise SystemExit(f"{lane}: non-finite values in {k} (default arm)")
        n_el += x.size if x.dtype.kind in "fc" else 0
        if x.shape != y.shape or x.dtype != y.dtype or x.tobytes() != y.tobytes():
            same = False
            d = np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))) if x.shape == y.shape else float("nan")
            scale = np.max(np.abs(x.astype(np.float64))) if x.size else 0.0
            print(f"{lane}: DIFFERS {k} max|diff|={d:.3e} (max|x|={scale:.3e})")
    if n_el < MIN_ELEMENTS:
        raise SystemExit(f"{lane}: only {n_el} elements compared (< {MIN_ELEMENTS})")
    if int(a["__moved__"]) == 0:
        raise SystemExit(f"{lane}: state did not move from its initial condition")
    print(f"{lane}: {'IDENTICAL' if same else 'DIFFERENT'} over {len(keys)} leaves, "
          f"{n_el} elements, {int(a['__moved__'])} leaves moved")
    return same


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flag", required=True, help="the XLA flag of the second arm")
    p.add_argument("--lanes", nargs="+", default=["latlon", "mpas_atm"])
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--child", nargs=2, metavar=("LANE", "OUT"), help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.child:
        _child(args.child[0], args.steps, args.child[1]); return 0
    base_flags = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in base_flags:
        base_flags = f"{base_flags} --xla_force_host_platform_device_count={N_DEV}".strip()
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for lane in args.lanes:
            outs = []
            for tag, flags in (("default", base_flags), ("flag", f"{base_flags} {args.flag}")):
                out = f"{tmp}/{lane}_{tag}.npz"
                env = dict(os.environ, XLA_FLAGS=flags, JAX_PLATFORMS="cpu")
                subprocess.run([sys.executable, __file__, f"--flag={args.flag}", "--steps",
                                str(args.steps), "--child", lane, out], env=env, check=True)
                outs.append(out)
            ok &= _compare(lane, *outs)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
