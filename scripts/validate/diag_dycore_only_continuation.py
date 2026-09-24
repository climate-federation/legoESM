"""Dycore-only continuation of an AMIP run: does a feature grow with physics OFF?

    diag_dycore_only_continuation.py --config <deck> [--restart <checkpoint.npz>] --steps N

Answers "is this the dynamical core or the physics" for a growing local feature
(a hot column, a wind maximum) that a forward run shows.  Used 2026-09-22 to
attribute the CAM6 AMIP arm's stratospheric warming: the peak grew 340.4 -> 359.2 K
in six simulated hours with physics_fn=None, while the production-grid arm under the
same treatment was flat, which put the cause in the dycore rather than in radiation.

Builds the driver EXACTLY as run_amip does (same deck, same mesh, same vertical
coordinate, same dycore config), restores the checkpoint, then integrates with
``model.step(state, dt)`` and physics_fn=None.  Prints, every --stride steps,
the global T max and the temperature of the columns that were already hot in
the restart state.  NUMBERS ONLY -- no verdict.
"""
from __future__ import annotations
import argparse, importlib.util, os, sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]


def _load_run_amip():
    path = _ROOT / "scripts" / "run" / "run_amip.py"
    spec = importlib.util.spec_from_file_location("run_amip_probe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_amip_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--restart", default=None)
    ap.add_argument("--steps", type=int, default=192)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--hot-level", type=int, default=3)
    ap.add_argument("--hot-threshold", type=float, default=300.0)
    ap.add_argument("--thermo-terms", action="store_true",
                    help="decompose dT/dt at the restart state and exit (no integration)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="run_amip path flags (see config/amip/amip_production.sh)")
    return ap


def main():
    a = build_arg_parser().parse_args()

    ra = _load_run_amip()
    argv = ["--config", a.config] + list(a.extra)
    parser = ra.build_arg_parser()
    from legoesm.driver.run_config_yaml import load_yaml_config
    _cfg_keys = load_yaml_config(a.config, parser)
    parser.set_defaults(**_cfg_keys)
    parser.set_defaults(_config_keys=frozenset(_cfg_keys))
    args = parser.parse_args(argv)
    args = ra._postprocess_args(args, parser, argv)
    ra._apply_spectral_scheme_fallback(args, argv, parser)
    config = ra.build_config_from_args(args)

    from legoesm.driver.model_driver import ModelDriver
    driver = ModelDriver(config)
    driver.setup()
    if a.restart:
        step0, day0 = driver.load_checkpoint(a.restart)
        print(f"restored step={step0} day={day0}", flush=True)
    else:
        print("cold start from the deck's initial condition (no --restart)", flush=True)

    state = driver.state
    dt = float(config.dycore.dt)
    T0 = np.asarray(state.T.data)
    hot = np.where(T0[:, a.hot_level] > a.hot_threshold)[0]
    print(f"dt={dt} nlev={T0.shape[1]} n_hot(k={a.hot_level},>{a.hot_threshold})={hot.size}", flush=True)

    def report(n, s):
        T = np.asarray(s.T.data)
        u = np.asarray(s.u.data)
        h = T[hot, a.hot_level] if hot.size else np.array([np.nan])
        print(f"step {n:5d}  t={n*dt/3600:7.3f} h  Tmax={T.max():8.2f}  "
              f"Tmax(k={a.hot_level})={T[:, a.hot_level].max():8.2f}  "
              f"hot_mean={h.mean():8.2f}  |u|max={np.abs(u).max():8.2f}", flush=True)

    report(0, state)

    if a.thermo_terms:
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
            mpas_hydrostatic_tendencies,
        )
        m = driver.model
        _, th = mpas_hydrostatic_tendencies(
            state, m.mesh, m.sigma_coord, m.config, return_thermo_terms=True)
        day = 86400.0
        cols = [("horiz_adv", th.horiz_adv), ("horiz_diff", th.horiz_diff),
                ("vert_adv", th.vert_adv), ("adiabatic_ps", th.adiabatic_ps),
                ("sigma_dot", th.sigma_dot)]
        print("  dT/dt decomposition [K/day], hot columns vs all columns", flush=True)
        for name, arr in cols:
            v = np.asarray(arr)
            sc = 1.0 if name == "sigma_dot" else day
            print(f"   {name:13s} hot(k={a.hot_level}) mean={v[hot, a.hot_level].mean()*sc:11.4f} "
                  f"max={v[hot, a.hot_level].max()*sc:11.4f}   all mean={v[:, a.hot_level].mean()*sc:11.4f}",
                  flush=True)
        return

    for n in range(1, a.steps + 1):
        state = driver.model.step(state, dt)
        if n % a.stride == 0 or n == a.steps:
            report(n, state)
            if not np.isfinite(np.asarray(state.T.data)).all():
                print("NON-FINITE", flush=True)
                break


if __name__ == "__main__":
    main()
