#!/usr/bin/env python
"""AMIP physics-toggle stability screen (Tier-1 of the physics campaign).

Submits one short `run_amip.py` per case — a baseline (all physics on = the
reference config) plus one-parameterization-off variants and structural A/Bs —
then aggregates the per-run stability signals into one table so we can see WHICH
physics destabilizes the model.

Design notes (why these metrics): the model has many safety nets that silently
MASK instability (q_v/q_c floors, the JOINT vapour donor clamp, and the
multiplicative moisture fixer `fix_moisture_hydrostatic`).  The only non-masking
honesty signals are the host-side blow-up detector (the `Status:` line in
results.txt) and the residual diagnostics.  So this screen keys off the run
STATUS (blow-up day), `max_wind`, and the moisture/energy residuals — NOT just
"did it NaN".  Reuses `validate_amip_run.validate` for the pass/fail verdict.

Usage::

    # submit the Tier-1 screen (one sbatch per case, 5-day, latlon reference)
    python -m scripts.validate.physics_toggle_screen --submit --days 5 \
        --base /scratch/b/b309178/phys_screen_$(date +%s)

    # after the jobs finish, tabulate
    python -m scripts.validate.physics_toggle_screen --aggregate \
        /scratch/b/b309178/phys_screen_XXXX

Off-toggles require `--allow-disabled-physics` (run_amip rejects a disabled
physics slot otherwise).  Radiation has no 'none' -> the minimal leg is 'gray'.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

# --- Tier-1 case list: (label, extra run_amip args) --------------------------
# Baseline = the reference config verbatim.  Each off-toggle isolates one
# parameterization's role in stability.  Structural A/Bs test the new stable-BL
# MOST and the Rayleigh drag.
_ALLOW = ["--allow-disabled-physics"]
TIER1_CASES: list[tuple[str, list[str]]] = [
    ("baseline",        []),
    ("convection_off",  ["--convection", "none", *_ALLOW]),
    ("gwd_off",         ["--gravity-wave-drag", "none", *_ALLOW]),
    ("turbulence_off",  ["--turbulence", "none", *_ALLOW]),
    ("clouds_off",      ["--clouds", "none", *_ALLOW]),
    ("microphysics_off", ["--microphysics", "none", *_ALLOW]),
    ("radiation_gray",  ["--radiation", "gray"]),          # no 'none' for radiation
    ("stability_dyer",  ["--surface-stability-scheme", "dyer1974"]),  # vs beljaars A/B
]


# --- Tier-2 cloud-albedo levers (the reference config is over-reflective:
# rsut +130, clt +19, netTOA -69) — rank each lever's albedo (rsut) effect. -----
TIER2_CLOUD_CASES: list[tuple[str, list[str]]] = [
    ("baseline",     []),                                         # convective_cloud=true (ref)
    ("cc_off",       ["--no-convective-cloud"]),                  # biggest albedo lever
    ("qc_low",       ["--q-c-diagnostic", "1.5e-4"]),             # thinner cloud (less LWP)
    ("ccmax_low",    ["--cloud-conv-cloud-max", "0.05"]),         # cap convective cloud cover
    ("rhcrit_hi",    ["--rh-crit", "0.85"]),                      # less stratiform cloud
    ("cc_off_qc_low", ["--no-convective-cloud", "--q-c-diagnostic", "1.5e-4"]),
]


# --- Tier-2 precipitation-efficiency matrix (run ALL in parallel, rank by which
# raises precip toward ~2.8 and drops clwvi/rsut).  Root cause of the over-
# reflection is under-precipitation -> cloud water piles up -> optically thick.
# Levers: subgrid autoconversion (in-cloud q_c -> faster rain, CLI), the warm-rain
# rate (k_au / autoconversion_rate via --params), and the Bechtold convective
# precip efficiency (--params).  Param YAMLs live in /scratch/.../tuning_params. --
# NOTE: only CLI-reachable levers are used.  --params scheme fields (k_au,
# autoconversion_rate, precip_efficiency) target nested *Config NamedTuples that
# the AMIP driver's config object does not expose to the --params class-router
# ("scheme not built into this driver's config") — dropped; those need a
# top-level ExperimentConfig scalar + CLI flag (as Pierre did for
# conv_cloud_condensate #840) or a scheme-default edit.  Run WITH the sponge on
# (config sponge_enabled=true) — the no-sponge sweep is the control.
TIER2_PRECIP_CASES: list[tuple[str, list[str]]] = [
    ("baseline",       []),
    ("subgrid_auto",   ["--subgrid-autoconversion"]),                       # in-cloud autoconv (#1)
    ("subgrid_qc_low", ["--subgrid-autoconversion", "--q-c-diagnostic", "1.5e-4"]),
    ("ccond_low",      ["--conv-cloud-condensate", "3.0e-5"]),               # #840 thin anvil
    ("subgrid_ccond",  ["--subgrid-autoconversion", "--conv-cloud-condensate", "3.0e-5"]),
]


# --- Tier-2 evaporation + cloud-fraction levers.  Diagnosis (#847): the config
# equilibrates into an over-reflective / weak-hydrological-cycle state — ocean
# hfls 40 vs ~110 W/m2, precip 0.75 vs 2.8, albedo 0.65.  Two independent
# hypotheses, tested in parallel:
#   (a) EVAP source: gustiness_zi is 300 in-config vs coare3-native 600 (halved)
#       -> raise it to boost low-wind ocean evaporation.
#   (b) CLOUD overcast: Xu-Randall p_xr HIGHER / alpha_xr LOWER directly flatten
#       the "moisture-driven overcast runaway" (the clt-85% / albedo lever).
# Run WITH the sponge (config sponge_enabled=true) for stability. --
TIER2_EVAP_CASES: list[tuple[str, list[str]]] = [
    ("baseline",   []),
    ("gust_600",   ["--gustiness-zi", "600"]),                    # coare3-native (vs 300)
    ("gust_1500",  ["--gustiness-zi", "1500"]),                   # aggressive evap boost
    ("pxr_hi",     ["--cloud-p-xr", "0.6"]),                      # flatten overcast runaway
    ("alphaxr_lo", ["--cloud-alpha-xr", "30"]),                   # cloud grows slower w/ condensate
    ("gust_pxr",   ["--gustiness-zi", "600", "--cloud-p-xr", "0.6"]),  # combined evap+cloud
]


def build_cases(tier: str = "tier1") -> list[tuple[str, list[str]]]:
    """Return the (label, extra-args) case list for a screen tier."""
    if tier == "tier1":
        return list(TIER1_CASES)
    if tier == "tier2_cloud":
        return list(TIER2_CLOUD_CASES)
    if tier == "tier2_precip":
        return list(TIER2_PRECIP_CASES)
    if tier == "tier2_evap":
        return list(TIER2_EVAP_CASES)
    raise ValueError(
        f"unknown screen tier {tier!r} "
        "(tier1 | tier2_cloud | tier2_precip | tier2_evap)")


def _run_status(run_dir: Path) -> str:
    """Read the model-side Status line from results.txt ('COMPLETED' / 'BLOWUP ...')."""
    rt = run_dir / "results.txt"
    if not rt.exists():
        return "MISSING"
    for line in rt.read_text().splitlines():
        if line.strip().startswith("Status:"):
            return line.split(":", 1)[1].strip()
    return "NO_STATUS"


def _blowup_day(status: str) -> float:
    """Extract the blow-up day from a status string, or +inf if it completed."""
    m = re.search(r"day\s+(\d+)", status)
    if m:
        return float(m.group(1))
    return float("inf") if status.upper().startswith("COMPLETED") else float("nan")


def extract_run_metrics(run_dir: Path) -> dict:
    """Per-run stability metrics from results.txt Status + timeseries.npz.

    Returns status, blowup_day, max_wind, |moisture_residual| max, and the final
    energy_toa_net (all NaN-safe; missing timeseries -> NaN metrics)."""
    run_dir = Path(run_dir)
    status = _run_status(run_dir)
    out = {"label": run_dir.name, "status": status,
           "blowup_day": _blowup_day(status),
           "max_wind": float("nan"), "moisture_resid": float("nan"),
           "energy_toa_net": float("nan"), "rsut": float("nan"),
           "olr": float("nan"), "precip": float("nan"),
           "hfls": float("nan"), "sw_net_sfc": float("nan")}
    ts = run_dir / "timeseries.npz"
    if ts.exists():
        z = np.load(ts, allow_pickle=True)

        def _last(key):
            return float(z[key][-1]) if key in z.files and z[key].size else float("nan")

        if "max_wind" in z.files and z["max_wind"].size:
            out["max_wind"] = float(np.nanmax(z["max_wind"]))
        if "moisture_residual" in z.files and z["moisture_residual"].size:
            out["moisture_resid"] = float(np.nanmax(np.abs(z["moisture_residual"])))
        out["energy_toa_net"] = _last("energy_toa_net")
        # Global-mean TOA fluxes (already area-mean diagnostics): rsut = reflected
        # SW (the albedo proxy), olr = outgoing LW.  Used to rank cloud levers.
        out["rsut"] = _last("sw_up_toa")
        out["olr"] = _last("lw_up_toa")
        # Global-mean surface precip [mm/day] — the primary rank key for the
        # precip-efficiency sweep (raise toward GPCP ~2.8; drains cloud water).
        out["precip"] = _last("precip")
        # Latent heat flux (evaporation, W/m2; obs ~88) + net surface SW (W/m2;
        # obs ~165) — the hydrological-cycle source + the over-reflection proxy.
        out["hfls"] = _last("hfls")
        out["sw_net_sfc"] = _last("sw_net_sfc")
    return out


def format_table(rows: list[dict]) -> str:
    """Tabulate the screen results, most-stable (largest blowup_day) first."""
    rows = sorted(rows, key=lambda r: (-(r["blowup_day"] if np.isfinite(r["blowup_day"]) else 1e9),
                                        r["label"]))
    lines = [f"{'case':16s} {'blowup':>7s} {'precip':>7s} {'hfls':>6s} "
             f"{'swnsfc':>7s} {'rsut':>7s} {'olr':>7s} {'toa_net':>8s} "
             f"{'maxwind':>8s}  status",
             "-" * 108]
    for r in rows:
        bd = r["blowup_day"]
        bd_s = "surv" if np.isinf(bd) else ("n/a" if np.isnan(bd) else f"{bd:.0f}")
        lines.append(
            f"{r['label']:16s} {bd_s:>7s} {r.get('precip', float('nan')):7.2f} "
            f"{r.get('hfls', float('nan')):6.1f} "
            f"{r.get('sw_net_sfc', float('nan')):7.1f} "
            f"{r.get('rsut', float('nan')):7.1f} {r.get('olr', float('nan')):7.1f} "
            f"{r['energy_toa_net']:8.1f} {r['max_wind']:8.1f}  {r['status']}")
    return "\n".join(lines)


def _submit(base: Path, days: int, sbatch: Path, tier: str) -> int:
    base.mkdir(parents=True, exist_ok=True)
    for label, extra in build_cases(tier):
        env = (f"ALL,SCREEN_LABEL={label},SCREEN_DAYS={days},"
               f"SCREEN_BASE={base},SCREEN_EXTRA={' '.join(extra)}")
        cmd = ["sbatch", f"--export={env}",
               f"--job-name=phys_{label}", str(sbatch)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(f"  submit {label:16s} -> {r.stdout.strip() or r.stderr.strip()}")
    print(f"screen base: {base}")
    return 0


def _aggregate(base: Path, tier: str) -> int:
    rows = []
    for label, _ in build_cases(tier):
        d = base / label
        if d.is_dir():
            rows.append(extract_run_metrics(d))
    if not rows:
        print(f"no case dirs under {base}", file=sys.stderr)
        return 2
    print(format_table(rows))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submit", action="store_true", help="submit one sbatch per case")
    ap.add_argument("--aggregate", type=Path, default=None,
                    help="tabulate an existing screen base dir")
    ap.add_argument("--base", type=Path, default=None, help="scratch base dir for --submit")
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--tier", type=str, default="tier1")
    ap.add_argument("--sbatch", type=Path,
                    default=Path("scripts/tmp/cluster_oneoffs/amip/physics_toggle_run.sbatch"))
    args = ap.parse_args(argv)
    if args.aggregate is not None:
        return _aggregate(args.aggregate, args.tier)
    if args.submit:
        if args.base is None:
            print("--submit requires --base", file=sys.stderr)
            return 2
        return _submit(args.base, args.days, args.sbatch, args.tier)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
