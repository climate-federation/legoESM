#!/usr/bin/env python
"""WHICH skipped statement owned legoESM's first-step error? Measure, don't guess.

#1729 replaced the standalone DINO card's forward-Euler start -- an early
return out of ``_leapfrog_step`` -- with the same body every other step runs.
That closed SEVERAL gaps at once, and the step-1 gate then showed temperature
improving by ~310x.  Attributing that to ``mlf_baro_corr`` (the gap the code's
own warning named) would be wrong: that routine is called AFTER ``tra_zdf``
(stpmlf.f90:534 vs :507) and writes only the velocity arrays, so it cannot
move temperature at all.

This probe decomposes the step-1 residual by turning ONE thing off at a time,
each arm scored against NEMO's kt=1 now-level:

  A  production      the shipped card, as run_dino.py runs it
  B  no surface rate the same, with the external surface-tracer tendency
                     withheld from model.step -- exactly what the old early
                     return did by never forwarding ``external_tracer_rate``
  C  no reconcile    the same, with barotropic_after_reconcile="off"

B-vs-A sizes the dropped surface forcing; C-vs-A sizes ``mlf_baro_corr``.
Whichever moves temperature owns the temperature row; a term that moves it by
nothing owns none of it, however plausible the story.

GPU only (the card's XLA CPU compile crashes on this machine):

    CUDA_VISIBLE_DEVICES=<uuid> JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \\
    python scripts/validate/ocean_fidelity/dino_1226/\\
        step1_euler_term_attribution.py
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from rebuild_nemo_restart import rebuild  # noqa: E402

DEFAULT_RESTART = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
                   "RUN_FROMREST_KT1/DINO_00000001_restart_*.nc")
_DT = 2700.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart-glob", default=DEFAULT_RESTART)
    args = ap.parse_args()

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    set_policy(PrecisionPolicy.fp64())                       # Rule 1c
    from legoesm.ocean.experiments import dino as dm
    from legoesm.ocean.fidelity import nemo_dino_mesh as ndm
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)

    if not glob.glob(args.restart_glob):
        raise SystemExit(f"no restart tiles match {args.restart_glob}")
    R = rebuild(args.restart_glob, ["tn", "sn", "sshn", "un", "vn"])

    def O3(k):
        return np.moveaxis(R[k], 0, -1)

    cfg = dm.nemo_faithful_dino_config(
        base=dm.dino_config_for_recipe("nemo_dino_kamm_mlf"))
    grid = dm.dino_lat_lon_grid(cfg)
    z = dm.dino_lat_lon_vertical(grid, cfg)
    state0 = dm.dino_lat_lon_state(grid, z, cfg)
    forcing = dm.dino_lat_lon_surface_forcing_arrays(grid, cfg)
    sf_step = (dm.dino_step_surface_forcing(forcing)
               if getattr(cfg, "wind_through_step", False) else None)
    mc, _ = dm.dino_lat_lon_model_config(grid, cfg, physics=True)
    print(f"card: outer={mc.flat_get('outer_integrator')!r} "
          f"after_reconcile="
          f"{mc.barotropic.barotropic_after_reconcile!r} "
          f"placement={cfg.surface_tendency_placement!r} "
          f"precision={get_policy().storage.__name__}")

    mc_off = mc._replace(
        barotropic=mc.barotropic._replace(barotropic_after_reconcile="off"))

    def one_step(model_config, *, with_rate: bool):
        model = LatLonCGridOceanModel(grid, z, model_config)
        st, rate = dm.apply_dino_lat_lon_surface_forcing(
            state0, forcing, z, cfg, _DT, t_seconds=_DT, return_rate=True)
        return model.step(st, dt=_DT, surface_forcing=sf_step,
                          external_tracer_rate=(rate if with_rate else None))

    g = ndm.nemo_dino_mesh()
    wet3, uwet, vwet = g.tmask > 0.5, g.umask > 0.5, g.vmask > 0.5

    def score(s):
        out = {}
        for name, built, oracle, msk in (
                ("T", np.asarray(s.T.data), O3("tn"), wet3),
                ("S", np.asarray(s.S.data), O3("sn"), wet3),
                ("eta", np.asarray(s.eta.data), R["sshn"], wet3[:, :, 0]),
                ("u", np.asarray(s.u.data)[:, 1:, :], O3("un"), uwet),
                ("v", np.asarray(s.v.data)[1:, :, :], O3("vn"), vwet)):
            d = (built - oracle)[msk]
            out[name] = float(np.sqrt(np.mean(d ** 2)))
        return out

    arms = {
        "A production": one_step(mc, with_rate=True),
        "B no surface rate": one_step(mc, with_rate=False),
        "C no reconcile": one_step(mc_off, with_rate=True),
    }
    sc = {k: score(v) for k, v in arms.items()}

    print("\nrms residual vs NEMO kt=1 now-level (wet cells)")
    fields = ("T", "S", "eta", "u", "v")
    print(f"{'arm':20s}" + "".join(f"{f:>13s}" for f in fields))
    for k in arms:
        print(f"{k:20s}" + "".join(f"{sc[k][f]:13.4e}" for f in fields))

    print("\nWHAT EACH TERM OWNS -- how far the arm moves from production")
    print(f"{'term':34s}" + "".join(f"{f:>13s}" for f in fields))
    for label, arm in (("the surface tracer tendency", "B no surface rate"),
                       ("mlf_baro_corr", "C no reconcile")):
        row = []
        for f in fields:
            a = np.asarray(getattr(arms["A production"], f).data)
            b = np.asarray(getattr(arms[arm], f).data)
            row.append(float(np.max(np.abs(a - b))))
        print(f"{label:34s}" + "".join(f"{x:13.4e}" for x in row)
              + "   (max|A-arm|)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
