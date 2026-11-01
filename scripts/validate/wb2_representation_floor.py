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
    # Two config sources: an AIMIP suite (--suite/--variant) or a WeatherBench
    # training deck (--config/--mode), the latter built exactly as
    # run_weatherbench_eval.py builds it and split by land/ocean and UTC hour.
    p.add_argument("--suite", default=None)
    p.add_argument("--variant", default=None,
                   choices=("classical", "column_nn", "sfno_full",
                            "sfno_physics"))
    p.add_argument("--config", default=None,
                   help="WeatherBench training deck (as run_weatherbench_eval).")
    p.add_argument("--mode", default=None,
                   choices=("physics", "neural_gcm", "sfno"))
    p.add_argument("--training-core", default="spectral",
                   dest="training_core", choices=("spectral",))
    # Default: 2017 for --suite (unchanged); the deck's eval_years[0] for
    # --config, as run_weatherbench_eval resolves it.
    p.add_argument("--eval-year", type=int, default=None, dest="eval_year")
    # Defaults 8 / 24 h for --suite; --config must name the scorecard's sample.
    p.add_argument("--n-inits", type=int, default=None, dest="n_inits")
    p.add_argument("--init-stride-hours", type=int, default=None,
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


def split_scores(fields, verif, valid, lat_deg, land):
    """Lead-0 RMSE/bias per field over all, land and ocean cells.

    ``land`` is a WB2-grid bool mask; each region intersects it with the
    model's own validity mask, so the cos-lat weights and below-ground masking
    are the scorecard's (``score_forecast``) and only the region differs.
    """
    from evaluations.wb_forecast import score_forecast

    out = {}
    for region, sel in (("all", None), ("land", land), ("ocean", ~land)):
        vr = {k: (v if sel is None else (v & sel)) for k, v in valid.items()}
        for key, s in score_forecast(fields, verif, verif, lat_deg,
                                     valid=vr).items():
            out[(region, key)] = (float(s["rmse"]), float(s["bias"]))
    return out


def _aggregate(per_case):
    """Mean over cases of per-case RMSE and bias (the orchestrator's average).
    A NaN score is FATAL rather than dropped: a silently shorter mean is a
    different sample than the one the label claims."""
    import numpy as np

    res = {}
    for (group, region, key), vals in sorted(per_case.items()):
        arr = np.asarray(vals, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise SystemExit(f"non-finite floor score for {group}/{region}/{key}")
        res.setdefault(group, {}).setdefault(region, {})[key] = {
            "rmse": float(arr[:, 0].mean()), "bias": float(arr[:, 1].mean()),
            "n": int(len(arr))}
    return res


def deck_init_windows(times, eval_year, n_inits, init_stride_hours, cadence):
    """Per scorecard date, the ``(start, stop)`` time-index slice and the UTC
    hours of the paired lead-0 inits.

    Dates: run_weatherbench_eval's sampling (anchor eval_year-01-01, step
    init_stride_hours). Each slice starts one cadence BEFORE the date so the
    pairing (case j's lead-cadence truth = case j+1's init) yields the date's
    own 24 h of inits, one per cadence hour.
    """
    import numpy as np

    base = int(np.searchsorted(times, np.datetime64(f"{eval_year}-01-01")))
    stride = init_stride_hours // cadence
    n_sub = 24 // cadence
    out = []
    for k in range(n_inits):
        i0 = base + k * stride
        if i0 - 1 < 0 or i0 + n_sub >= len(times):
            raise SystemExit(f"date {k} window falls outside the ERA5 store")
        hours = [int(str(times[i0 + j])[11:13]) for j in range(n_sub)]
        out.append(((i0 - 1, i0 + n_sub + 1), hours))
    return out


def main_deck(args):
    """Floor for a WeatherBench deck: the scorecard's 24 dates, each expanded
    to one init per UTC cadence hour, split by land/ocean and UTC hour."""
    import types

    import numpy as np
    import yaml

    from evaluations.wb_era5_cases import build_forecast_cases
    from evaluations.wb_forecast import diagnose_and_regrid
    from evaluations.wb_regrid import regrid_to_wb2
    from legoesm.training.era5_to_state import open_era5_zarr, resolve_var
    from legoesm.training.scale_build import (
        build_mode_components, wb_era5_config, wb_ic_microphysics)

    import jax
    jax.config.update("jax_enable_x64", True)

    import subprocess
    sha = subprocess.run(["git", "-c", "safe.directory=*", "rev-parse",
                          "--short", "HEAD"], capture_output=True, text=True,
                         cwd=_REPO).stdout.strip()
    yml = yaml.safe_load(open(args.config))
    mode_cfg = types.SimpleNamespace(mode=args.mode,
                                     training_core=args.training_core)
    _m, grid, sigma, *_ = build_mode_components(mode_cfg, yml)
    era5_cfg = wb_era5_config(mode_cfg, yml)
    cadence = int(era5_cfg.dt_hours)
    if 24 % cadence or args.init_stride_hours % cadence:
        raise SystemExit("cadence must divide 24 h and the init stride")
    n_sub = 24 // cadence                  # one init per UTC cadence hour

    ds = open_era5_zarr(era5_cfg.zarr_store)
    times = np.asarray(ds.time.values, dtype="datetime64[ns]")
    lsm_name = resolve_var(ds, "land_sea_mask")
    if lsm_name is None:
        raise SystemExit("no land_sea_mask in the ERA5 store; cannot split")
    lsm = ds[lsm_name]
    if "time" in lsm.dims:
        lsm = lsm.isel(time=0)
    lsm = np.asarray(lsm.values, dtype=np.float64)
    if not np.all(np.isfinite(lsm)):
        raise SystemExit("land_sea_mask has non-finite values")
    land, _, _ = regrid_to_wb2(
        lsm,
        np.asarray(ds.lat.values), np.asarray(ds.lon.values),
        resolution_deg=args.resolution_deg, mask=True, mask_threshold=0.5)

    eval_year = (args.eval_year if args.eval_year is not None
                 else int(yml["eval_years"][0]))
    per_case: dict = {}
    dates = []
    for (a, b), hours in deck_init_windows(times, eval_year, args.n_inits,
                                           args.init_stride_hours, cadence):
        sub = ds.isel(time=slice(a, b))
        t_sub = np.asarray(sub.time.values, dtype="datetime64[ns]")
        # build_forecast_cases anchors at Jan 1 of eval_year; the slice's own
        # first year makes that anchor the slice start (also across 31 Dec;
        # the IC forcing's day-of-year is then off by a year, unused here).
        cases = build_forecast_cases(
            era5_cfg, grid, sigma, leads_hours=(cadence,),
            eval_year=int(str(t_sub[0])[:4]), n_inits=n_sub + 1,
            init_stride_hours=cadence, resolution_deg=args.resolution_deg,
            ds=sub, microphysics=wb_ic_microphysics(args.mode, yml))
        # build_forecast_cases puts case j at its anchor + j cadences; the
        # pairing is right only if that anchor is the slice's first time.
        anchor = int(np.searchsorted(
            t_sub, np.datetime64(f"{str(t_sub[0])[:4]}-01-01")))
        if anchor != 0 or len(cases) != n_sub + 1:
            raise SystemExit(f"window {a}:{b}: case anchor {anchor} != 0")
        dates.append(str(t_sub[1]))
        for j in range(n_sub):
            verif = cases[j].verif_by_lead[cadence]
            fields, valid, lat, _ = diagnose_and_regrid(
                cases[j + 1].init_state, grid, sigma,
                resolution_deg=args.resolution_deg)
            hour = f"{hours[j]:02d}"
            for (region, key), sc in split_scores(
                    fields, verif["fields"], valid, lat, land).items():
                per_case.setdefault(("overall", region, key), []).append(sc)
                per_case.setdefault((f"utc{hour}", region, key), []).append(sc)
    return {
        "meta": {
            "what": "lead-0 ERA5 -> model state -> WB2 round-trip RMSE/bias "
                    "(mean over inits), no model integration; split by "
                    "land/ocean (ERA5 land_sea_mask >= 0.5 on the WB2 grid) "
                    "and UTC hour",
            "config": args.config, "mode": args.mode, "git": sha,
            "eval_year": eval_year, "dates": dates,
            "inits_per_date": n_sub, "init_stride_hours": args.init_stride_hours,
            "resolution_deg": args.resolution_deg,
            "land_fraction_of_cells": float(land.mean()),
        },
        "floor": _aggregate(per_case),
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.config is not None:
        if args.suite is not None or args.mode is None:
            raise SystemExit("--config needs --mode and excludes --suite")
        if args.n_inits is None or args.init_stride_hours is None:
            raise SystemExit("--config needs explicit --n-inits and "
                             "--init-stride-hours (the scorecard's sample)")
        if args.n_inits < 1 or args.init_stride_hours < 1:
            raise SystemExit("--n-inits and --init-stride-hours must be >= 1")
        out = main_deck(args)
        text = json.dumps(out, indent=2)
        print(text)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text)
        return 0
    if args.suite is None or args.variant is None:
        raise SystemExit("--suite and --variant are required without --config")
    if args.eval_year is None:
        args.eval_year = 2017
    if args.n_inits is None:
        args.n_inits = 8
    if args.init_stride_hours is None:
        args.init_stride_hours = 24

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
    # sigma_top from the config: training/eval run 0.05, the factory default
    # is 0.01 — the floor must be measured on the SAME vertical coordinate.
    sigma = create_sigma_coordinate(
        spec_cfg.n_levels, sigma_top=spec_cfg.sigma_top)

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
