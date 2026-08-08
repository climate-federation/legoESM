#!/usr/bin/env python
"""Lead-0 error: what the ERA5 -> model-state -> WB2 round-trip costs BEFORE
any model runs.

WHY THIS EXISTS. Every WB2 arm is scored by converting an ERA5 slice into the
spectral model state, integrating, diagnosing the headline fields
(``wb_forecast.diagnose_and_regrid``) and comparing to ERA5 on the WB2 1.5 deg
grid. That pipeline has an error of its OWN — 8 sigma levels instead of ERA5's
pressure levels, a T63 spectral truncation, a hydrostatic z500 diagnosis, a
regrid — and it is charged to the model at every lead.

Nothing in the campaign had ever measured it, and two live claims depend on it:

* sfno_full scores 11.6 m z500 after ONE 6 h step, which was read as
  "single-step accuracy is the binding constraint". If the round-trip alone
  costs ~10 m, that reading is wrong and the model is near the floor.
* column_nn scores 82 m at 6 h from the same initial conditions. If the floor
  is ~10 m, the extra ~70 m is genuinely the dycore, not representation.

The measurement is a rollout of ZERO steps: diagnose the initial state and
score it against the ERA5 verification at lead 0 — the same diagnosis, the same
regrid, the same cos-lat weights, the same masks the real scorecard uses. The
CLI refuses lead 0 (``--leads must be positive integers``), which is why this
is a separate probe rather than a flag.

Committed, not a heredoc: a number quoted in an argument about where the error
comes from has to be re-runnable against a changed model.

Usage:
    JAX_ENABLE_X64=1 python scripts/validate/wb2_representation_floor.py \
        --suite config/aimip/ace2/suite_curriculum_v2.yaml --variant column_nn \
        --eval-year 2017 --n-inits 8 --out results/wb2_repr_floor.json
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
    entry = _REPO / "scripts" / "run" / "run_aimip.py"
    spec = importlib.util.spec_from_file_location("run_aimip_floor", entry)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_aimip_floor"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", required=True)
    p.add_argument("--variant", required=True,
                   choices=("classical", "column_nn", "sfno_full",
                            "sfno_physics"))
    p.add_argument("--eval-year", type=int, default=2017, dest="eval_year")
    p.add_argument("--n-inits", type=int, default=8, dest="n_inits")
    p.add_argument("--init-stride-hours", type=int, default=24,
                   dest="init_stride_hours")
    p.add_argument("--resolution-deg", type=float, default=1.5,
                   dest="resolution_deg")
    # Overrides so the floor can be DECOMPOSED. The floor is set by what the
    # pipeline throws away, and the two candidates are the vertical (8 sigma
    # levels vs ERA5's pressure levels) and the horizontal (T63 truncation).
    # Sweeping them separately says which one to spend on.
    p.add_argument("--n-levels", type=int, default=None, dest="n_levels")
    p.add_argument("--n-max", type=int, default=None, dest="n_max")
    p.add_argument("--out", default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    import numpy as np
    import jax.numpy as jnp

    ra = _load_run_aimip()
    from evaluations.wb_era5_cases import build_forecast_cases
    from evaluations.wb_forecast import diagnose_and_regrid, score_forecast
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.era5_to_state import TrainingERA5Config

    suite = ra._load_yaml(Path(args.suite))
    cfg = ra._merge(ra._load_yaml(Path(suite["base"])),
                    suite.get("cfg_overrides") or {})
    cfg = ra._merge(cfg, ra._load_yaml(
        Path(args.suite).parent / f"variant_{args.variant}.yaml"))
    cfg["aimip_variant"] = args.variant
    if args.n_levels is not None:
        cfg["n_levels"] = int(args.n_levels)
        cfg.pop("nlev", None)          # the builder accepts either key
    if args.n_max is not None:
        cfg["n_max"] = int(args.n_max)
    spec_cfg = ra._build_spectral_config(cfg)

    grid = create_gaussian_grid(spec_cfg.n_max, dealiasing="quadratic")
    sigma = create_sigma_coordinate(spec_cfg.n_levels)

    # The SHORTEST lead the store supports, purely so build_forecast_cases has
    # something to verify against; we score the lead-0 verification, which the
    # case builder also has to fetch as the IC's own timestamp.
    cadence = int(cfg.get("era5_cadence_hours", 6) or 6)
    era5_cfg = TrainingERA5Config(dt_hours=cadence)
    if cfg.get("era5_zarr"):
        era5_cfg = era5_cfg._replace(zarr_store=cfg["era5_zarr"])

    # Lead 0 needs the ERA5 truth AT the init time. ``ForecastCase`` exposes
    # only ``verif_by_lead``, and ``_build_verification`` is private (this repo
    # forbids importing private symbols across modules). So use consecutive
    # inits spaced exactly one cadence apart: case k's verification at lead
    # ``cadence`` IS the ERA5 truth at case k+1's init time. Pairing them gives
    # a genuine lead-0 comparison out of public fields only.
    if args.init_stride_hours != cadence:
        raise SystemExit(
            f"--init-stride-hours must equal the ERA5 cadence ({cadence} h) "
            "for this probe: it pairs case k's lead-{cadence} verification with "
            f"case k+1's initial state, which is only the same timestamp when "
            "the inits are one cadence apart.")

    cases = build_forecast_cases(
        era5_cfg, grid, sigma,
        leads_hours=(cadence,), eval_year=args.eval_year,
        n_inits=args.n_inits + 1, init_stride_hours=cadence,
        resolution_deg=args.resolution_deg,
    )

    per_case: dict[str, list[float]] = {}
    for k in range(len(cases) - 1):
        verif = cases[k].verif_by_lead[cadence]
        fields_wb2, valid_wb2, wb2_lat, _ = diagnose_and_regrid(
            cases[k + 1].init_state, grid, sigma,
            resolution_deg=args.resolution_deg)
        # clim_fields is only used for ACC; pass the verification so the call is
        # well-formed and read RMSE only.
        scores = score_forecast(
            fields_wb2, verif["fields"], verif["fields"], wb2_lat,
            valid=valid_wb2)
        for key, s in scores.items():
            r = float(s["rmse"])
            if r == r:                      # drop NaN fields (below-ground etc.)
                per_case.setdefault(key, []).append(r)

    out = {
        "meta": {
            "what": "lead-0 ERA5 -> model state -> WB2 round-trip RMSE; no "
                    "model integration. This is the FLOOR every arm is scored "
                    "against at every lead.",
            "suite": args.suite, "variant": args.variant,
            "eval_year": args.eval_year, "n_inits": len(cases) - 1,
            "n_max": int(spec_cfg.n_max), "n_levels": int(spec_cfg.n_levels),
            "resolution_deg": args.resolution_deg,
        },
        "rmse": {k: float(np.mean(v)) for k, v in sorted(per_case.items())},
    }
    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
