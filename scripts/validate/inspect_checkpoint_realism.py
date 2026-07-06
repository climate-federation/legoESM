#!/usr/bin/env python
"""Fast physical-realism gate on a saved model checkpoint's RAW state.

Loads an AMIP/coupled ``checkpoint_day_*.npz`` and checks every field against a
physical sanity range — temperatures in a plausible atmospheric band, humidity /
condensate / soil-water / snow non-negative, surface pressure and winds bounded,
and NO ``NaN``/``Inf`` anywhere — returning a pass/fail realism report plus the
global-mean context values (T, land skin-T, precip, OLR, planetary-albedo proxy)
that tell you whether a *stable* run is also *realistic*.

This is the LIVE-STATE complement to the CMOR-climatology scorecard
(``plot_amip_cmor_diagnostics.py``): it needs only a single daily checkpoint, so
it flags a blow-up or an unphysical field within a day instead of waiting for the
first monthly mean.  It distinguishes a STABILITY defect (NaN / runaway T /
negative water — a real bug to fix now) from an expected coarse-resolution /
spin-up bias (low OLR, high albedo — quantified later at the climatology).

Pure ``numpy`` — imports no ``legoesm`` and runs on any run's checkpoint.

The bounds below are physical VALIDITY ranges (widest plausible Earth-atmosphere
values), NOT model physical constants — they parameterise a sanity gate only and
are deliberately not sourced from ``legoesm.constants``.

Usage::

    python scripts/validate/inspect_checkpoint_realism.py <run_dir_or_ckpt> [--gate]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np

# --- Per-field physical sanity bounds: (lo, hi).  ``hi=None`` means "no upper
#     bound beyond finiteness"; a ``lo`` of 0.0 enforces non-negativity.  Applied
#     by field NAME, so the gate is grid-agnostic (cubed-sphere, lat-lon,
#     spectral all share names).  Ranges are the widest plausible Earth values —
#     a violation is a genuine unphysical state, not a tuning bias. ---
_FIELD_BOUNDS = {
    # atmospheric prognostic state
    "T": (150.0, 340.0),          # K: cold mesosphere top .. hot surface
    "u": (-150.0, 150.0),         # m/s: |wind| blow-up guard (jets < ~120)
    "v": (-150.0, 150.0),
    "p_s": (3.0e4, 1.10e5),       # Pa: high-topography .. strong high
    "q_v": (0.0, 0.05),           # kg/kg: no negative, no absurd supersat
    "q_c": (0.0, 0.02),
    "q_r": (0.0, 0.02),
    # land / surface carries
    "carry_T_land": (180.0, 345.0),   # K: polar-winter ice .. hot desert
    "carry_w_land": (0.0, None),       # mm soil water
    "carry_snow": (0.0, None),         # mm snow water-equivalent
    "carry_seg_precip": (0.0, None),   # accumulated precip (>=0)
    # two-moment microphysics carries
    "carry_dmtr_q_i": (0.0, 0.02),
    "carry_dmtr_q_s": (0.0, 0.02),
    "carry_dmtr_q_g": (0.0, 0.02),
    "carry_dmtr_N_c": (0.0, None),
    "carry_dmtr_N_r": (0.0, None),
    "carry_dmtr_N_i": (0.0, None),
}

# Fields whose global mean is reported as realism CONTEXT (not gated — a value
# outside the "realistic" hint is a coarse-res/spin-up bias, not an error).
# (key, label, realistic-hint low, high, unit)
_CONTEXT_FIELDS = (
    ("T", "air_T", 240.0, 275.0, "K"),           # 40-level volume mean (cold)
    ("carry_T_land", "land_T", 275.0, 295.0, "K"),
    ("carry_seg_precip", "seg_precip", 1.5, 5.0, "mm/day~"),
    ("carry_held_lw_up_toa", "OLR", 230.0, 250.0, "W/m2"),
)


def _find_checkpoint(path: str | Path) -> str:
    """Resolve a run directory (-> latest ``checkpoint_day_*.npz``) or a direct
    ``.npz`` file to a concrete checkpoint path."""
    p = str(path)
    if p.endswith(".npz"):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        return p
    cks = glob.glob(os.path.join(p, "checkpoint_day_*.npz"))
    if not cks:
        raise FileNotFoundError(f"no checkpoint_day_*.npz under {p}")
    # Select by NUMERIC day, not lexicographically (else day_9999 > day_10000).
    def _day(f):
        m = re.search(r"checkpoint_day_(\d+)\.npz$", os.path.basename(f))
        return int(m.group(1)) if m else -1
    return max(cks, key=_day)


def inspect_checkpoint_realism(path: str | Path) -> dict:
    """Load a checkpoint and gate every numeric field on physical sanity.

    Returns::

        {"checkpoint": <basename>, "day": float|None,
         "fields": {name: {min,max,mean,n_nan,n_inf,within, reason}},
         "context": {label: {value, lo, hi, unit, in_hint}},
         "n_violations": int, "passed": bool}

    ``passed`` is True iff NO field has a NaN/Inf and every bounded field lies
    within its range.  A field present in the checkpoint but absent from
    ``_FIELD_BOUNDS`` is still finiteness-checked (NaN/Inf fails) but not
    range-gated.  Pure + deterministic — the unit-tested realism gate."""
    ckpt = _find_checkpoint(path)
    z = np.load(ckpt, allow_pickle=True)

    fields: dict = {}
    n_violations = 0
    for name in z.files:
        arr = np.asarray(z[name])
        kind = arr.dtype.kind
        if kind not in "fiuc":       # skip strings/objects (config_json, scheme)
            continue
        # Complex spectral coefficients (vor_hat/T_hat/...) MUST be NaN/Inf-checked
        # too; report magnitude stats (they carry no physical range -> no bounds).
        is_complex = kind == "c"
        a = arr.astype(np.complex128 if is_complex else np.float64)
        n_nan = int(np.isnan(a).sum())      # complex NaN = NaN in real OR imag
        n_inf = int(np.isinf(a).sum())
        finite = a[np.isfinite(a)]          # complex finite = both parts finite
        mag = np.abs(finite) if is_complex else finite
        amin = float(mag.min()) if mag.size else float("nan")
        amax = float(mag.max()) if mag.size else float("nan")
        amean = float(mag.mean()) if mag.size else float("nan")

        reason = ""
        within = (n_nan == 0 and n_inf == 0)
        # Normalise a "trc_" tracer-array prefix (MPAS/spectral store q_v as
        # trc_q_v) to the base name so tracer bounds apply on every grid.
        base = name[4:] if name.startswith("trc_") else name
        if not within:
            reason = f"{n_nan} NaN, {n_inf} Inf"
        elif not is_complex and base in _FIELD_BOUNDS:
            lo, hi = _FIELD_BOUNDS[base]
            if amin < lo:
                within = False; reason = f"min {amin:.4g} < {lo}"
            elif hi is not None and amax > hi:
                within = False; reason = f"max {amax:.4g} > {hi}"
        fields[name] = {"min": amin, "max": amax, "mean": amean,
                        "n_nan": n_nan, "n_inf": n_inf,
                        "within": bool(within), "reason": reason}
        n_violations += int(not within)

    context: dict = {}
    for key, label, lo, hi, unit in _CONTEXT_FIELDS:
        if key in fields:
            v = fields[key]["mean"]
            context[label] = {"value": v, "lo": lo, "hi": hi, "unit": unit,
                              "in_hint": bool(lo <= v <= hi)}

    day = float(z["day"]) if "day" in z.files else None
    return {"checkpoint": os.path.basename(ckpt), "day": day,
            "fields": fields, "context": context,
            "n_violations": n_violations, "passed": bool(n_violations == 0)}


def format_realism_report(report: dict) -> str:
    """Multi-line human summary: PASS/FAIL, offending fields, context values."""
    head = "PASS" if report["passed"] else "FAIL"
    day = report["day"]
    lines = [f"{head} realism  {report['checkpoint']}"
             f"{'' if day is None else f'  day={day:g}'}"
             f"  violations={report['n_violations']}"]
    for name, d in report["fields"].items():
        if not d["within"]:
            lines.append(f"  VIOLATION {name}: {d['reason']}  "
                         f"[min={d['min']:.4g} max={d['max']:.4g}]")
    ctx = "  ".join(
        f"{lab}={c['value']:.4g}{c['unit']}{'' if c['in_hint'] else '(!)'}"
        for lab, c in report["context"].items())
    if ctx:
        lines.append(f"  context: {ctx}   ((!)=outside realistic hint, not gated)")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="run directory (uses latest checkpoint) or a .npz")
    p.add_argument("--gate", action="store_true",
                   help="exit nonzero if the realism gate does not pass")
    args = p.parse_args(argv)
    report = inspect_checkpoint_realism(args.path)
    print(format_realism_report(report))
    if args.gate and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
