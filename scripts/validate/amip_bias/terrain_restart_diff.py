#!/usr/bin/env python
"""Where does a relaunch's rebuilt terrain differ from a run's stored terrain?

#2303: replaying jan1979_main1027_ncaero from its checkpoints is refused by
the restart terrain gate (rebuilt product vs the checkpoint's ``phis``).
Relaunches the run with its own logged arguments (``launch_argv``) and dumps,
per mode:
  restart  single process from --day's checkpoint: product at the gate
  mpi      under srun with N ranks (--distributed restored): global product
  cold     single process, day 0, no restart: first-step state phis + product
  steps    single process, day 0, no restart: state phis before/after each of
           the first --n-steps model steps vs the product
then prints max|d| of each against the day-``--day`` checkpoint's phis.
Not a test: prints, exit 1 on failure.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mpas_onestep_param_grad as H  # noqa: E402


class _Done(BaseException):
    pass


def terrain_diff(a, b) -> dict:
    """max|a-b|, its index, and how many cells differ at all (exact)."""
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    return {"max": float(d.max()), "argmax": int(d.argmax()), "n_diff": int((d > 0).sum())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--mode", choices=("restart", "mpi", "cold", "steps"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-steps", type=int, default=200)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    ckpt = np.load(H.ROOT / args.run / f"checkpoint_day_{args.day:04d}.npz")["phis"]
    import legoesm.driver.model_driver as md
    got = {}

    if args.mode == "restart":
        def hook(self):
            got["product"] = np.asarray(self._phis_data)
            got["state"] = np.asarray(self.state.phis.data)
            raise _Done()
        md.ModelDriver._check_restart_terrain = hook
    elif args.mode == "mpi":
        real = md.ModelDriver._create_topography

        def hook(self):
            real(self)
            if getattr(self, "_phis_global", None) is None:
                raise SystemExit("no _phis_global: not the Voronoi MPI lane")
            got["product"] = np.asarray(self._phis_global)
            raise _Done()
        md.ModelDriver._create_topography = hook
    elif args.mode == "steps":
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
            MPASPrimitiveEquationModel)
        real_step = MPASPrimitiveEquationModel.step
        rows = got.setdefault("rows", [])

        def hook(self, state, *a, **kw):
            if "product" not in got:
                drv = [o for o in gc.get_objects() if isinstance(o, md.ModelDriver)]
                if len(drv) != 1:
                    raise SystemExit(f"expected one ModelDriver, found {len(drv)}")
                got["product"] = np.asarray(drv[0]._phis_data)
            before = np.asarray(state.phis.data)
            new = real_step(self, state, *a, **kw)
            after = np.asarray(new.phis.data)
            n = len(rows)
            rows.append((n, terrain_diff(after, before)["max"],
                         terrain_diff(after, got["product"])["max"], str(after.dtype)))
            print(f"steps: step {n}: max|after-before| {rows[-1][1]:.3e}, "
                  f"max|after-product| {rows[-1][2]:.3e} m2/s2, dtype {rows[-1][3]}", flush=True)
            if n + 1 >= args.n_steps:
                got["state"] = after
                raise _Done()
            return new
        MPASPrimitiveEquationModel.step = hook
    else:
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
            MPASPrimitiveEquationModel)

        def hook(self, state, *a, **kw):
            drv = [o for o in gc.get_objects() if isinstance(o, md.ModelDriver)]
            if len(drv) != 1:
                raise SystemExit(f"expected one ModelDriver, found {len(drv)}")
            got["product"] = np.asarray(drv[0]._phis_data)
            got["state"] = np.asarray(state.phis.data)
            raise _Done()
        MPASPrimitiveEquationModel.step = hook

    argv_run = H.launch_argv(args.run, args.day, out / f"_launch_{args.mode}_r{rank}")
    if args.mode == "mpi":
        argv_run.append("--distributed")
    if args.mode in ("cold", "steps"):
        i = argv_run.index("--restart-from")
        del argv_run[i:i + 2]
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run"))
    import run_amip
    try:
        run_amip.main(argv_run)
    except _Done:
        pass
    if "product" not in got:
        raise SystemExit("the relaunch never reached the terrain hook")
    if rank != 0:
        return 0
    rows = got.pop("rows", None)
    np.savez(out / f"terrain_{args.mode}_d{args.day}.npz", ckpt=ckpt, **got)
    if rows is not None and len(rows) < args.n_steps:
        raise SystemExit(f"run ended after {len(rows)} of {args.n_steps} steps")
    for k, v in got.items():
        r = terrain_diff(v, ckpt)
        print(f"{args.mode}: {k} vs day-{args.day} checkpoint phis: max|d| {r['max']:.3e} m2/s2 "
              f"(cell {r['argmax']}), {r['n_diff']} cells differ")
    if "state" in got:
        r = terrain_diff(got["state"], got["product"])
        print(f"{args.mode}: state vs product: max|d| {r['max']:.3e} m2/s2, {r['n_diff']} cells differ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
