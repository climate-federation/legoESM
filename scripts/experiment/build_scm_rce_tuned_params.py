#!/usr/bin/env python
"""Distil the 5-seed corrected-protocol campaign into per-scheme parameter files.

Emits one YAML per convection scheme under
``config/params/scm_rce_tuned_f0/<scheme>.yaml``, keyed by the
``param_collector`` qualified name ``scheme_key.field`` that ``--params``
consumes on every MIP driver (SCM and AMIP).  This is the CLAUDE.md-sanctioned
"recommended trained JSON" path: it NEVER mutates a production ``*Config``
default; a run opts in with ``--params``.

Per parameter the value written is the MEDIAN across the five seeds, because the
campaign measured the multi-parameter schemes to be non-identifiable (tuned
values differ up to 90% of range at equal score).  The median is a defensible
point estimate, not a calibration; the per-seed spread is written as a comment
so the reader sees how load-bearing each number is.

A parameter is emitted only if its median moved from the shipped default by more
than a small fraction of its range — an unmoved parameter in a flat basin is
noise, and writing it would imply a precision the search did not establish.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

# Below this fraction of (upper-lower), the tuned median is indistinguishable
# from the default and is NOT written — it would assert a calibration the flat
# basin does not support.
_MOVED_FRAC = 0.02


def _records(seed_dirs, scheme):
    per_param = {}
    meta = {}
    for d in seed_dirs:
        p = d / f"scheme_{scheme}.json"
        if not p.exists():
            continue
        j = json.loads(p.read_text())
        for rec in j.get("records", []):
            key = f"{rec['scheme_key']}.{rec['parameter']}"
            per_param.setdefault(key, []).append(float(rec["tuned"]))
            meta[key] = rec
    return per_param, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-dir", type=Path, required=True)
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--arm-suffix", default="f0")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--schemes", nargs="*", default=None)
    a = ap.parse_args(argv)

    seed_dirs = [a.arm_dir / f"arm_thermo_physical_{a.arm_suffix}_seed{s}"
                 for s in a.seeds]
    schemes = a.schemes or sorted({
        Path(p).name[len("scheme_"):-len(".json")]
        for d in seed_dirs for p in glob.glob(str(d / "scheme_*.json"))})

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for scheme in schemes:
        per_param, meta = _records(seed_dirs, scheme)
        if not per_param:
            continue
        lines = [
            f"# SCM-RCE tuned parameters for convection={scheme}",
            f"# corrected nonrotating protocol (coriolis=0), median of "
            f"{len(a.seeds)} seeds.",
            "# NON-IDENTIFIABLE for multi-parameter schemes: the spread comment "
            "on each",
            "# line is how far the seeds disagreed. A large spread means the "
            "median is one",
            "# point in a flat basin, not a calibrated value. Opt in with "
            "--params; this",
            "# file never mutates a production *Config default.",
            "",
        ]
        n_written = 0
        for key in sorted(per_param):
            vals = per_param[key]
            rec = meta[key]
            lo, hi = float(rec["lower"]), float(rec["upper"])
            med = statistics.median(vals)
            rng = hi - lo if hi > lo else 1.0
            if abs(med - float(rec["default"])) < _MOVED_FRAC * rng:
                continue  # unmoved from default; do not assert it
            spread = (max(vals) - min(vals)) / rng * 100.0
            units = rec.get("units", "")
            lines.append(
                f"{key}: {med:.6g}  # default {rec['default']:.6g}, "
                f"seed-spread {spread:.0f}% of range"
                + (f", {units}" if units and units != "1" else ""))
            n_written += 1
        (a.out_dir / f"{scheme}.yaml").write_text("\n".join(lines) + "\n")
        print(f"{scheme}: {n_written} moved params written "
              f"({len(per_param)} tuned total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
