"""Production-setup validation: bit-exact restart chains with stateful physics.

Ported from fix/persist-physics onto main's #413 carry architecture
(FIX_RESTART_TIME, docs/dev-notes/FIX_RESTART_TIME.md).

Runs the SAME
experiment twice — once straight through, once as a checkpoint/restart
chain — with stateful physics schemes (prognostic TKE turbulence and,
where supported, the prognostic-spectral GWD spectrum), and asserts the
final prognostic state AND the physics carries agree BITWISE.

This is the production-scale companion of
``tests/unit/test_persist_physics_state.py::test_*_bitexact_restart_continuation``
— same comparison, configurable resolution/length, suitable for a batch
job (see ``scripts/cluster/persist_physics/``).

Bitwise-equality preconditions (documented in the tests):
* MPAS: none — its loop's forcing is daily-cadence + per-step traced
  scalars, segmentation-free.
* cubed_sphere (compiled loop): restarts must land on GCD segment
  boundaries, which whole-day checkpoint cadences guarantee.

Usage::

    python scripts/validate/validate_persist_physics_restart.py \
        --grid mpas --resolution 4 --nlev 20 --days-per-link 1 --links 2 \
        --turbulence tke --gwd prognostic_spectral
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np


def _build_driver(outdir, args, days):
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
    )
    from legoesm.driver.model_driver import ModelDriver

    if args.grid == "mpas":
        grid = GridConfig(grid_type="mpas", resolution=args.resolution,
                          nlev=args.nlev, vertical_coord="hybrid")
        dycore = DycoreConfig(discretization="mpas", dt=args.dt)
    else:
        grid = GridConfig(grid_type="cubed_sphere",
                          resolution=args.resolution, nlev=args.nlev)
        dycore = DycoreConfig(dt=args.dt)
    cfg = ExperimentConfig(
        grid=grid, dycore=dycore,
        output=OutputConfig(output_dir="", diag_days=1, checkpoint_days=1),
        days=days, dataset="analytical", radiation="gray",
        rad_update_steps=args.rad_update_steps,
        convection="none", turbulence=args.turbulence,
        gravity_wave_drag=args.gwd, precision="fp64",
    )
    d = ModelDriver(cfg, output_dir=str(outdir))
    d.setup()
    return d


def _final_fields(driver, grid):
    """Prognostic state + physics carries as a name->ndarray dict."""
    out = {
        "T": np.asarray(driver.state.T.data),
        "u": np.asarray(driver.state.u.data),
        "p_s": np.asarray(driver.state.p_s.data),
    }
    if grid == "mpas":
        ps = getattr(driver, "_mpas_phys_state", None)
        if ps is not None:
            for f in type(ps)._fields:
                out[f"phys.{f}"] = np.asarray(getattr(ps, f))
    else:
        out["v"] = np.asarray(driver.state.v.data)
        out["q_v"] = np.asarray(driver.q_v)
        for key in ("tke", "qke", "gwd_spectrum"):
            if key in driver._carry_aux:
                out[key] = np.asarray(driver._carry_aux[key])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", choices=("mpas", "cubed_sphere"),
                   default="mpas")
    p.add_argument("--resolution", type=int, default=4)
    p.add_argument("--nlev", type=int, default=20)
    p.add_argument("--dt", type=float, default=300.0)
    p.add_argument("--days-per-link", type=float, default=1.0)
    p.add_argument("--links", type=int, default=2)
    p.add_argument("--turbulence", default="tke")
    p.add_argument("--gwd", default="none")  # prognostic_spectral raises on MPAS (main)
    p.add_argument("--rad-update-steps", type=int, default=2)
    p.add_argument("--workdir", default=None)
    args = p.parse_args()

    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp())
    work.mkdir(parents=True, exist_ok=True)
    total_days = args.days_per_link * args.links

    print(f"=== straight run: {total_days} days "
          f"(grid={args.grid} res={args.resolution} nlev={args.nlev} "
          f"turbulence={args.turbulence} gwd={args.gwd}) ===", flush=True)
    dA = _build_driver(work / "straight", args, total_days)
    status = dA.run()
    assert status == "COMPLETED", status

    print(f"=== chained run: {args.links} links x "
          f"{args.days_per_link} days ===", flush=True)
    d = _build_driver(work / "link0", args, args.days_per_link)
    assert d.run() == "COMPLETED"
    for link in range(1, args.links):
        prev_out = work / f"link{link - 1}"
        ckpts = sorted(prev_out.glob("checkpoint_day_*.npz"))
        assert ckpts, f"link {link - 1} wrote no checkpoint"
        if args.grid == "mpas":
            # MPAS restart contract: cfg.days = days THIS link advances.
            d = _build_driver(work / f"link{link}", args,
                              args.days_per_link)
        else:
            # Cube loops treat --days as TOTAL days since the epoch.
            d = _build_driver(work / f"link{link}", args,
                              args.days_per_link * (link + 1))
        step, day = d.load_checkpoint(ckpts[-1])
        print(f"  link {link}: resumed step={step} day={day}", flush=True)
        assert d.run(start_step=step, start_day=day) == "COMPLETED"

    a, b = _final_fields(dA, args.grid), _final_fields(d, args.grid)
    failed = []
    for name in sorted(a):
        if a[name].dtype.kind in ("U", "S"):
            equal = bool(np.all(a[name] == b[name]))
            delta = "n/a"
        elif a[name].dtype.kind == "f":
            delta = float(np.nanmax(np.abs(a[name] - b[name]))) \
                if a[name].size else 0.0
            # equal_nan: the surface_T_sfc_override slot uses a NaN
            # sentinel for "no override" — NaN==NaN counts as equal.
            equal = np.array_equal(a[name], b[name], equal_nan=True)
        else:
            delta = float(np.max(np.abs(a[name] - b[name])))
            equal = np.array_equal(a[name], b[name])
        marker = "OK " if equal else "FAIL"
        print(f"  [{marker}] {name:24s} max|straight-chained| = {delta}")
        if not equal:
            failed.append(name)
    if failed:
        print(f"FAIL: non-bitwise fields: {failed}")
        return 1
    print("PASS: chained restart is bit-identical to the straight run "
          "(prognostic state + physics carries).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
