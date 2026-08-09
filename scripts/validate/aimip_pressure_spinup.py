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
    p.add_argument(
        "--vs-era5", action="store_true", dest="vs_era5",
        help="Instead of displacement from t=0, compare the 6 h forecast "
             "against the ERA5 verification and split the mslp error into a "
             "CASE-INVARIANT pattern and per-case residual. A large invariant "
             "fraction means a systematic base-state offset (correctable); a "
             "small one means genuine per-case forecast error.")
    p.add_argument(
        "--land-phis-threshold", type=float, default=1000.0,
        dest="land_phis_threshold",
        help="Surface geopotential [m2/s2] above which a WB2 cell counts as "
             "LAND. NOT 0: the spectral state's phis rings under truncation "
             "(min -1801 m2/s2 at T63) and bilinear regridding spreads it, so "
             "a >0 test calls 68%% of the AREA land against a true ~29%%. "
             "1000 m2/s2 (~100 m) reproduces 0.298 area-weighted.")
    p.add_argument(
        "--vs-reconciled", action="store_true", dest="vs_reconciled",
        help="Compare the forecast p_s DIRECTLY against the ERA5 state AFTER "
             "the same phis reconciliation the initial condition went through "
             "(case k+1's init_state, one cadence later). No sea-level "
             "reduction is involved, so this separates a real p_s base-state "
             "offset from amplification by the mslp diagnostic — and because "
             "the reconciliation is present on BOTH sides it cancels, so a "
             "fixed offset that VANISHES here is caused by the reconciliation "
             "while one that PERSISTS is the dycore's own.")
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

    if args.vs_reconciled:
        # Consecutive inits one cadence apart: case k+1's init_state IS the
        # ERA5 truth at case k's verification time, carried through the SAME
        # ERA5 -> spectral path (including the phis reconciliation) as the IC.
        cases_c = build_forecast_cases(
            TrainingERA5Config(dt_hours=cadence), grid, sigma,
            leads_hours=(cadence,), eval_year=args.eval_year,
            n_inits=args.n_cases + 1, init_stride_hours=cadence,
            resolution_deg=1.5)
        errs = []
        for k in range(len(cases_c) - 1):
            fc = spectral_rollout(
                cases_c[k].init_state, zero_physics, grid, sigma,
                spec_cfg.pe_config, dt, n_steps, None, None)
            e = np.asarray(_ps(fc)) - np.asarray(_ps(cases_c[k + 1].init_state))
            errs.append(e)
        E = np.stack(errs)
        mean_pat = E.mean(axis=0)
        v_tot = float(np.mean(E ** 2))
        v_fix = float(np.mean(np.repeat(mean_pat[None], len(E), 0) ** 2))
        phis_g = np.asarray(sh_synthesis(grid, cases_c[0].init_state.phis_hat.data))
        thr = float(args.land_phis_threshold)
        land = phis_g > thr
        out = {
            "meta": {
                "what": "6 h zero-physics forecast p_s vs the RECONCILED ERA5 "
                        "p_s one cadence later (case k+1's init state). No "
                        "sea-level reduction; the reconciliation is on both "
                        "sides and cancels.",
                "suite": args.suite, "variant": args.variant,
                "eval_year": args.eval_year, "n_cases": len(errs),
                "hours": args.hours, "land_phis_threshold": thr,
            },
            "ps_rms_total_pa": float(np.sqrt(v_tot)),
            "ps_rms_fixed_pattern_pa": float(np.sqrt(v_fix)),
            "ps_rms_residual_pa": float(np.sqrt(np.mean((E - mean_pat) ** 2))),
            "fixed_fraction_of_variance": v_fix / max(v_tot, 1e-30),
            "ps_fixed_land_pa": float(np.sqrt(np.mean(mean_pat[land] ** 2))),
            "ps_fixed_ocean_pa": float(np.sqrt(np.mean(mean_pat[~land] ** 2))),
        }
        print(json.dumps(out, indent=2))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out, indent=2))
        return 0

    if args.vs_era5:
        from evaluations.wb_forecast import diagnose_and_regrid

        errs = []
        for case in cases:
            state = spectral_rollout(
                case.init_state, zero_physics, grid, sigma,
                spec_cfg.pe_config, dt, n_steps, None, None)
            fields, valid, _lat, _lon = diagnose_and_regrid(
                state, grid, sigma, resolution_deg=1.5)
            verif = case.verif_by_lead[cadence]
            e = np.asarray(fields["mslp"]) - np.asarray(verif["fields"]["mslp"])
            m = np.asarray(valid["mslp"]).astype(bool)
            errs.append(np.where(m, e, np.nan))
        E = np.stack(errs)                       # (n_cases, n_lat, n_lon)
        mean_pat = np.nanmean(E, axis=0)         # case-invariant component
        resid = E - mean_pat[None]
        # Variance split, over cells valid in every case.
        # LAND/OCEAN SPLIT OF THE FIXED PATTERN. This is the test that
        # implicates or clears the orography reconciliation
        # (``p_s * exp(delta_phis/(R_d T_sfc))``), which acts ONLY where
        # topography was smoothed. An earlier version of this analysis "refuted"
        # that mechanism using the land/ocean symmetry of p_s(t) - p_s(0) — the
        # WRONG QUANTITY, because the reconciliation happens AT t=0 and so
        # cancels out of a displacement from t=0 entirely. The ERA5-referenced
        # error is where its signature lives.
        ok = np.all(np.isfinite(E), axis=0)
        v_tot = float(np.nanmean(E[:, ok] ** 2))
        v_fix = float(np.nanmean(np.repeat(mean_pat[ok][None], len(E), 0) ** 2))
        v_res = float(np.nanmean(resid[:, ok] ** 2))
        # Model orography regridded to the WB2 grid, so land can be masked on
        # the same grid the error lives on.
        phis_g = np.asarray(sh_synthesis(grid, cases[0].init_state.phis_hat.data))
        from evaluations.wb_regrid import regrid_to_wb2 as _rg
        # regrid_to_wb2 returns (field, tgt_lat, tgt_lon) — a 3-tuple, not an
        # array — and takes 1-D source coordinates in DEGREES while grid.lat /
        # grid.lon are in RADIANS. Both read from the source, after np.asarray
        # on the tuple raised "inhomogeneous shape (3,)".
        phis_wb2, _tlat, _tlon = _rg(
            jnp.asarray(phis_g),
            np.rad2deg(np.asarray(grid.lat)),
            np.rad2deg(np.asarray(grid.lon)),
            resolution_deg=1.5)
        phis_wb2 = np.asarray(phis_wb2)
        _thr = float(args.land_phis_threshold)
        land_m = (phis_wb2 > _thr) & ok
        sea_m = (phis_wb2 <= _thr) & ok
        fix_land = float(np.sqrt(np.mean(mean_pat[land_m] ** 2))) if land_m.any() else None
        fix_sea = float(np.sqrt(np.mean(mean_pat[sea_m] ** 2))) if sea_m.any() else None

        out = {
            "mslp_fixed_pattern_land_pa": fix_land,
            "mslp_fixed_pattern_ocean_pa": fix_sea,
            "land_cells": int(land_m.sum()), "ocean_cells": int(sea_m.sum()),
            "land_phis_threshold": float(args.land_phis_threshold),
            "meta": {
                "what": "6 h ZERO-PHYSICS forecast vs ERA5: mslp error split "
                        "into a case-invariant pattern and a per-case "
                        "residual. High invariant fraction = systematic "
                        "base-state offset, not a skill failure.",
                "suite": args.suite, "variant": args.variant,
                "eval_year": args.eval_year, "n_cases": len(errs),
                "hours": args.hours, "n_max": int(spec_cfg.n_max),
                "n_levels": int(spec_cfg.n_levels),
            },
            "mslp_rms_total_pa": float(np.sqrt(v_tot)),
            "mslp_rms_fixed_pattern_pa": float(np.sqrt(v_fix)),
            "mslp_rms_residual_pa": float(np.sqrt(v_res)),
            "fixed_fraction_of_variance": v_fix / max(v_tot, 1e-30),
        }
        print(json.dumps(out, indent=2))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out, indent=2))
        return 0

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
