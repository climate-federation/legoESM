"""Day-1 gate for the layered-snow A/B (run by bundle.sbatch between stage 1
and stage 2).  Usage: check_day1.py <run_dir> <restart_day> <bulk|layered> <log>

Pass (exit 0) needs, at checkpoint day restart+1:
- every floating-point array finite;
- the RESOLVED configuration (experiment_config.json) selecting the expected
  snow scheme with soil freeze/thaw on;
- layered arms: the four snow-layer fields present, the layers' water equal to
  the snow water (rtol 1e-5, float32 storage), and the run log carrying the
  "pack built from its snow water" warning (the bulk restart was seeded);
- every arm: the summed snow water at day 1 at least half the restart's (a
  pack that seeding or the first day erased cannot pass).
Held land column-steps (budgets not closed) are printed, not gated.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np

_LAYERS = ("snow_ice_layers", "snow_liq_layers", "snow_T_layers", "snow_rho_layers")
_SEEDED = "LAYERED pack is built from its snow water"


def day1_problems(run_dir, restart_day, scheme, log_text):
    run_dir = pathlib.Path(run_dir)
    ck = run_dir / f"checkpoint_day_{restart_day + 1:04d}.npz"
    if not ck.exists():
        return [f"no {ck.name}"]
    z = np.load(ck, allow_pickle=False)
    out = [f"non-finite {k}" for k in z.files
           if z[k].dtype.kind in "fc" and not np.all(np.isfinite(z[k]))]
    rst = run_dir / f"checkpoint_day_{restart_day:04d}.npz"
    if rst.exists():
        s0 = float(np.sum(np.load(rst, allow_pickle=False)["land_ml_snow_depth"], dtype=np.float64))
        s1 = float(np.sum(z["land_ml_snow_depth"], dtype=np.float64))
        if s1 < 0.5 * s0:
            out.append(f"snow water {s1:.4g} after day 1 vs {s0:.4g} at restart")
    else:
        out.append(f"no restart {rst.name}")
    cfg = json.loads((run_dir / "experiment_config.json").read_text())
    if cfg.get("land_snow_scheme") != scheme:
        out.append(f"resolved land_snow_scheme {cfg.get('land_snow_scheme')!r}, want {scheme!r}")
    if cfg.get("land_soil_freeze_thaw") is not True:
        out.append(f"resolved land_soil_freeze_thaw {cfg.get('land_soil_freeze_thaw')!r}")
    if scheme == "layered":
        missing = [f for f in _LAYERS if f"land_ml_{f}" not in z.files]
        if missing:
            out.append(f"missing {missing}")
        else:
            tot = (z["land_ml_snow_ice_layers"].astype(np.float64).sum(-1)
                   + z["land_ml_snow_liq_layers"].astype(np.float64).sum(-1))
            if not np.allclose(tot, z["land_ml_snow_depth"], rtol=1e-5, atol=1e-6):
                out.append("layer water != snow water")
        if _SEEDED not in log_text:
            out.append("no pack-seeding warning in the log")
    return out


def main(argv=None):
    run_dir, day, scheme, log = (argv or sys.argv[1:])
    text = pathlib.Path(log).read_text(errors="replace")
    held = re.findall(r"land: (\d+) column-steps held", text)
    print(f"{pathlib.Path(run_dir).name}: held column-step reports {held[:4]}")
    probs = day1_problems(run_dir, int(day), scheme, text)
    print("DAY1 PASS" if not probs else f"DAY1 FAIL: {probs}")
    return 1 if probs else 0


if __name__ == "__main__":
    raise SystemExit(main())
