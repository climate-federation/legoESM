#!/usr/bin/env python
"""Per-step surface-pressure adjustment in the first hours of a forecast.

WHAT THIS DOES AND DOES NOT MEASURE — read before quoting a number.

It runs the dycore with ZERO physics and records the area-weighted RMS of
``p_s(t) - p_s(0)`` every step. That quantity is the model's EVOLUTION away from
its initial state. It is NOT an error: the real atmosphere also moves, and over
6 h the true change is ~258 Pa (persistence error at 6 h with the representation
floor removed). Measured here, the zero-physics dycore moves 313 Pa in 6 h —
the same order — so this probe CANNOT separate correct evolution from wrong
evolution. Only a comparison against ERA5 at the verification time can, and that
is what the WB2 scorecard already does.

It was written to test whether the arms' ~1000 Pa 6 h mslp error is a purely
dynamical initialisation shock. It does not answer that, and the first version
of this docstring claimed it would. What it DOES establish, and these stand:

  * **No land/ocean asymmetry.** 311.9 Pa over land vs 314.6 Pa over ocean at
    6 h, and equal at every step. The ``p_s * exp(delta_phis/(R_d T_sfc))``
    reconciliation to smoothed topography acts ONLY where topography was
    smoothed, so it cannot be the mechanism. REFUTED.
  * **Global mean p_s is conserved** to +0.15 Pa over 6 h, confirming the
    dry-mass anchor behaves in a free run.
  * **The evolution magnitude is not anomalous** — 313 Pa of dynamical motion
    against a true 6 h change of ~258 Pa. The dycore is not flinging the state
    around; whatever produces the ~1000 Pa error is not gross over-activity.

To actually attribute the 6 h error, the next probe must compare against the
ERA5 verification and decompose into amplitude vs phase/pattern — not measure
displacement from t=0.

Committed rather than run as a heredoc: this number decides where the next
GPU-hours go, so it has to be re-runnable against a changed dycore.

Usage:
    JAX_ENABLE_X64=1 python scripts/validate/aimip_pressure_spinup.py \
        --suite config/aimip/ace2/suite_curriculum_v2.yaml --variant column_nn \
        --hours 6 --n-cases 2 --out results/aimip_ps_spinup.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "run"))


def _load_run_aimip():
    spec = importlib.util.spec_from_file_location(
        "run_aimip_spinup", _REPO / "scripts" / "run" / "run_aimip.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_aimip_spinup"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", required=True)
    p.add_argument("--variant", default="column_nn",
                   choices=("classical", "column_nn", "sfno_full"))
    p.add_argument("--eval-year", type=int, default=2017, dest="eval_year")
    p.add_argument("--n-cases", type=int, default=2, dest="n_cases")
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--out", default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    import numpy as np
    import jax
    import jax.numpy as jnp

    ra = _load_run_aimip()
    from evaluations.wb_era5_cases import build_forecast_cases
    from legoesm.grids.gaussian import create_gaussian_grid, sh_synthesis
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.era5_to_state import TrainingERA5Config
    from legoesm.training.neural_gcm_spectral import spectral_rollout

    suite = ra._load_yaml(Path(args.suite))
    cfg = ra._merge(ra._load_yaml(Path(suite["base"])),
                    suite.get("cfg_overrides") or {})
    cfg = ra._merge(cfg, ra._load_yaml(
        Path(args.suite).parent / f"variant_{args.variant}.yaml"))
    cfg["aimip_variant"] = args.variant
    spec_cfg = ra._build_spectral_config(cfg)

    grid = create_gaussian_grid(spec_cfg.n_max, dealiasing="quadratic")
    sigma = create_sigma_coordinate(spec_cfg.n_levels)
    dt = float(spec_cfg.dt)
    n_steps = int(round(args.hours * 3600.0 / dt))

    cadence = int(cfg.get("era5_cadence_hours", 6) or 6)
    cases = build_forecast_cases(
        TrainingERA5Config(dt_hours=cadence), grid, sigma,
        leads_hours=(cadence,), eval_year=args.eval_year,
        n_inits=args.n_cases, init_stride_hours=24, resolution_deg=1.5,
    )

    def zero_physics(state, grid_, sigma_, **_kw):
        """No physics at all — the whole point of the probe."""
        return jax.tree.map(jnp.zeros_like, state)

    area = jnp.asarray(grid.grid_area, dtype=jnp.float64)
    # Land/ocean split from the model's own surface geopotential: the p_s
    # reconciliation only acts where topography was smoothed, so an ocean signal
    # cannot come from it. phis is on the case's initial state.
    def _ps(state):
        return jnp.exp(sh_synthesis(grid, state.lnps_hat.data))

    def _rms(d, mask=None):
        w = area if mask is None else area * mask
        return float(jnp.sqrt(jnp.sum(w * d ** 2) / jnp.maximum(jnp.sum(w), 1e-30)))

    series = []
    for case in cases:
        s0 = case.init_state
        # The state carries ``phis_hat`` (SPECTRAL), not ``phis``. A
        # getattr(s0, "phis", None) returns None and silently disables the
        # land/ocean split — the exact silent-degradation pattern this campaign
        # keeps tripping over — so synthesise it and fail loudly if absent.
        if not hasattr(s0, "phis_hat"):
            raise AttributeError(
                "initial state has no phis_hat; the land/ocean split cannot be "
                "built and a whole-globe number alone would not discriminate "
                "the orography-reconciliation mechanism.")
        phis = sh_synthesis(grid, s0.phis_hat.data)
        land = (phis > 1.0).astype(jnp.float64)
        ocean = 1.0 - land
        p0 = _ps(s0)
        rows = []
        state = s0
        for k in range(1, n_steps + 1):
            state = spectral_rollout(
                state, zero_physics, grid, sigma, spec_cfg.pe_config,
                dt, 1, None, None)
            d = _ps(state) - p0
            rows.append({
                "step": k,
                "hours": k * dt / 3600.0,
                "rms_dps_pa": _rms(d),
                "rms_dps_land_pa": _rms(d, land),
                "rms_dps_ocean_pa": _rms(d, ocean),
                "mean_dps_pa": float(
                    jnp.sum(area * d) / jnp.sum(area)),
            })
        series.append(rows)

    # Average the per-step curves across cases.
    n = len(series[0])
    avg = []
    for i in range(n):
        row = {"step": series[0][i]["step"], "hours": series[0][i]["hours"]}
        for key in ("rms_dps_pa", "rms_dps_land_pa", "rms_dps_ocean_pa",
                    "mean_dps_pa"):
            vals = [s[i][key] for s in series if s[i][key] is not None]
            row[key] = float(np.mean(vals)) if vals else None
        avg.append(row)

    out = {
        "meta": {
            "what": "area-weighted RMS of p_s(t) - p_s(0) with ZERO physics; "
                    "isolates the purely dynamical adjustment of an ERA5 "
                    "analysis onto this dycore's balanced state.",
            "suite": args.suite, "variant": args.variant,
            "eval_year": args.eval_year, "n_cases": len(series),
            "dt_s": dt, "n_steps": n_steps, "hours": args.hours,
            "n_max": int(spec_cfg.n_max), "n_levels": int(spec_cfg.n_levels),
        },
        "series": avg,
    }
    text = json.dumps(out, indent=2)
    print(json.dumps(out["meta"], indent=2))
    for r in avg:
        if r["step"] <= 6 or r["step"] % 6 == 0:
            print(f"  t={r['hours']:5.2f} h  rms={r['rms_dps_pa']:9.1f} Pa"
                  f"  land={r['rms_dps_land_pa']:9.1f}"
                  f"  ocean={r['rms_dps_ocean_pa']:9.1f}"
                  f"  mean={r['mean_dps_pa']:+9.2f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
