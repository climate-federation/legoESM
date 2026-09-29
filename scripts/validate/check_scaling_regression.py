"""Flag scaling-receipt rows that got slower than a baseline set.

Loads both receipt sets through the scaling figure's own loader
(``scripts/plot/plot_nature_scaling.load``: same validity, precision, level,
solver, NCCL and CPU-affinity filters, best-of per key), pairs rows on the
full key (component, grid, backend, precision, mode, resolution, devices) and
prints candidate/baseline per pair.

Exit 1: a pair is slower than ``1 + tol_pct/100``. Exit 3: a baseline row has
no candidate row (a dropped or refused row is not a pass), or a pair's raw
problem size / level / step count differs (the figure's resolution label
rounds, so two meshes can share a key). Exit 2: no common rows.

Scope: the loader keeps the FASTEST replicate per key on both sides, so a
slowdown that shows up in only some replicates is not detected.

The tolerance has no default: it must be at least the run-to-run spread of
the lane being checked, which differs by lane and is the caller's to state.

    python scripts/validate/check_scaling_regression.py --tol-pct 5 \\
        --baseline /scratch/.../nature_v5_* --candidate /scratch/.../new_ladder
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

_RAW = ("n_lat", "n_lon", "n_cells", "subdivision", "resolution", "kt", "nlev",
        "n_levels", "nu_del4", "dt", "dt_seconds", "use_polar_filter")


def _plotter():
    path = os.path.join(os.path.dirname(__file__), "..", "plot", "plot_nature_scaling.py")
    spec = importlib.util.spec_from_file_location("plot_nature_scaling", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _raw(path: str, ms: float, n_devices: int):
    """Raw size fields of the receipt the loader kept: same file, timing and
    device count. Several such receipts with DIFFERENT sizes -> None (the
    kept one cannot be identified, which the caller treats as a mismatch)."""
    found = []
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (r.get("steady_median_ms") is not None and float(r["steady_median_ms"]) == ms
                and r.get("n_devices") is not None
                and int(r["n_devices"]) == n_devices):
            found.append({k: r.get(k) for k in _RAW})
    if not found:
        raise LookupError(f"no receipt with steady_median_ms={ms} in {path}")
    return found[0] if all(f == found[0] for f in found) else None


def compare(base: dict, cand: dict, tol_pct: float):
    """-> (rows, n_regressed, n_mismatched); rows = (key, base_ms, cand_ms,
    ratio, regressed, raw_diff)."""
    rows = []
    for key in sorted(set(base) & set(cand), key=str):
        b, c = base[key][0], cand[key][0]
        nd = key[-1]
        rb, rc = _raw(base[key][2], b, nd), _raw(cand[key][2], c, nd)
        if rb is None or rc is None:
            diff = {"receipt": "ambiguous: tied timings with different sizes"}
        else:
            diff = {k: (rb[k], rc[k]) for k in _RAW if rb[k] != rc[k]}
        if base[key][3] != cand[key][3]:      # steps as the loader resolved them
            diff["steps"] = (base[key][3], cand[key][3])
        r = c / b
        rows.append((key, b, c, r, r > 1.0 + tol_pct / 100.0, diff))
    return rows, sum(row[4] for row in rows), sum(bool(row[5]) for row in rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--candidate", nargs="+", required=True)
    ap.add_argument("--tol-pct", type=float, required=True,
                    help="allowed slowdown in percent (>= the lane's replicate spread)")
    ap.add_argument("--atm-nlev", type=int, required=True,
                    help="atmosphere level count the receipts must carry")
    args = ap.parse_args(argv)
    if not (math.isfinite(args.tol_pct) and args.tol_pct >= 0):
        ap.error("--tol-pct must be finite and >= 0")

    plot = _plotter()
    plot.NLEV = plot._nlev_map(args.atm_nlev)
    base, cand = plot.load(args.baseline), plot.load(args.candidate)
    rows, bad, mism = compare(base, cand, args.tol_pct)
    if not rows:
        print("NO COMMON ROWS: nothing compared (check keys / filters).")
        return 2
    for key, b, c, r, reg, diff in rows:
        tag = "SIZE?" if diff else ("SLOWER" if reg else "ok")
        print(f"{tag:6s} {r:6.3f}x  {b:10.3f} -> {c:10.3f} ms  "
              + " ".join(str(k) for k in key) + (f"  differs: {diff}" if diff else ""))
    missing = sorted(set(base) - set(cand), key=str)
    for key in missing:
        print("MISSING in candidate: " + " ".join(str(k) for k in key))
    print(f"{len(rows)} rows compared (fastest replicate per key), {bad} slower than "
          f"+{args.tol_pct}%, {mism} with a different raw size, {len(missing)} "
          f"baseline rows missing, {len(set(cand) - set(base))} candidate-only")
    if missing or mism:
        return 3
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
