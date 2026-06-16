"""Aggregate baroclinic-wave scaling JSONs into one tidy CSV for plotting.

Walks a results root (default ``results/bcw_scaling``), reads every flat
per-case JSON written by ``run_cpu_mpi_scaling.py`` (and the equivalent
single-process ``run_levante_gpu_scaling.py`` rows), and emits one tidy row per
measured point with the columns the publication plotter consumes:

    backend, grid, case, precision, mode, n_devices, resolution,
    resolution_km, n_levels, sypd, time_per_step_ms, total_cells,
    mcells_per_s, scaling_efficiency, dt_seconds, physics_level,
    compile_time_s, source

``backend`` (CPU vs GPU) is inferred from the output-directory name
(``..._gpu_...`` / ``..._cpu_...``) because the flat CPU-MPI JSON does not
record a device field.  ``case`` maps ``physics_level`` ("none"->"dry",
"moist"->"moist", others kept verbatim).  ``resolution_km`` is the nominal
horizontal grid spacing for the grid family.  Duplicate keys
(same backend/grid/case/precision/mode/n_devices/resolution) keep the row with
the highest SYPD (best of repeated measurements); the number dropped is logged.

Pure stdlib + ``legoesm.constants`` (for R_earth) — no JAX — so it runs in a
few seconds on a login node.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from legoesm import constants

FIELDS = [
    "component", "backend", "grid", "case", "precision", "mode", "n_devices",
    "resolution", "resolution_km", "n_levels", "sypd", "time_per_step_ms",
    "total_cells", "mcells_per_s", "scaling_efficiency", "dt_seconds",
    "physics_level", "compile_time_s", "source",
]

# Default result roots: atmosphere baroclinic-wave campaign + ocean campaign.
DEFAULT_ROOTS = ("results/bcw_scaling", "results/scaling_cpu_ocean",
                 "results/scaling_ocean")

_CASE = {"none": "dry", "moist": "moist"}


def resolution_km(grid_type: str, resolution) -> float:
    """Nominal horizontal grid spacing [km] for a grid family + resolution."""
    R_km = constants.R_earth / 1000.0
    r = float(resolution)
    if grid_type == "icosahedral":
        n_cells = 10.0 * 4.0 ** r + 2.0  # SCVT cell count at subdivision r
        return R_km * math.sqrt(4.0 * math.pi / n_cells)
    if grid_type == "latlon":
        return R_km * math.pi / r  # meridional spacing dy = R * dlat, dlat=pi/n_lat
    if grid_type in ("cubed-sphere", "cubed_sphere"):
        return (math.pi / 2.0) * R_km / r  # face arc / cells-per-face
    if grid_type == "spectral":
        return R_km * math.pi / r  # ~ R*pi/T (nominal)
    return float("nan")


def backend_from_path(path: Path) -> str:
    s = str(path)
    if "_gpu_" in s or "/gpu_" in s:
        return "GPU"
    if "_cpu_" in s or "/cpu_" in s:
        return "CPU"
    return "UNKNOWN"


def resolve_backend(d: dict, source: Path) -> str:
    """Prefer the JAX backend recorded in the JSON; fall back to the path.

    Newer runs serialize ``backend`` (cpu/gpu/cuda/tpu); older ones don't, so
    we still infer CPU-vs-GPU from the output-dir name as a fallback.
    """
    jb = str(d.get("backend", "") or "").lower()
    if jb in ("gpu", "cuda", "rocm"):
        return "GPU"
    if jb == "cpu":
        return "CPU"
    if jb == "tpu":
        return "TPU"
    return backend_from_path(source)


def _row_from_json(d: dict, source: Path) -> dict | None:
    """Build a tidy row from one flat per-case JSON, or None if not a case."""
    if "sypd" not in d or "grid_type" not in d:
        return None
    grid = d["grid_type"]
    n_dev = d.get("n_ranks", d.get("n_gpus"))
    if n_dev is None:
        return None
    phys = d.get("physics_level", "none")
    res = d.get("resolution")
    return {
        "component": "atm",
        "backend": resolve_backend(d, source),
        "grid": grid,
        "case": _CASE.get(phys, phys),
        "precision": d.get("precision", ""),
        "mode": d.get("mode", ""),
        "n_devices": int(n_dev),
        "resolution": res,
        "resolution_km": round(resolution_km(grid, res), 3) if res is not None else "",
        "n_levels": d.get("n_levels", ""),
        "sypd": d.get("sypd"),
        "time_per_step_ms": d.get("time_per_step_ms"),
        "total_cells": d.get("total_cells"),
        "mcells_per_s": d.get("mcells_per_s"),
        "scaling_efficiency": d.get("scaling_efficiency"),
        "dt_seconds": d.get("dt_seconds"),
        "physics_level": phys,
        "compile_time_s": d.get("compile_time_s"),
        "source": str(source),
    }


def _rows_from_nested(d: dict, source: Path) -> list[dict]:
    """Ocean-campaign schema: ``{backend, results:[TimingResult,...]}``.

    Flattens each inner result into a component='ocean' tidy row so the
    atmosphere (flat per-case JSON) and ocean (nested) campaigns land in one
    unified table.
    """
    backend = str(d.get("backend", "")).upper() or backend_from_path(source)
    if backend not in ("CPU", "GPU", "TPU"):
        backend = backend_from_path(source)
    out = []
    for r in d.get("results") or []:
        if not isinstance(r, dict) or r.get("sypd") is None:
            continue
        n_dev = r.get("n_ranks", r.get("n_gpus"))
        if n_dev is None:
            continue
        grid = r.get("grid_type", "latlon")
        res = r.get("resolution")
        out.append({
            "component": "ocean",
            "backend": backend,
            "grid": grid,
            "case": "ocean",
            "precision": r.get("precision", ""),
            "mode": str(r.get("mode", "strong")).replace("ocean_", "") or "strong",
            "n_devices": int(n_dev),
            "resolution": res,
            "resolution_km": round(resolution_km(grid, res), 3) if res is not None else "",
            "n_levels": r.get("n_levels", ""),
            "sypd": r.get("sypd"),
            "time_per_step_ms": r.get("time_per_step_ms"),
            "total_cells": r.get("total_cells"),
            "mcells_per_s": r.get("mcells_per_s"),
            "scaling_efficiency": r.get("scaling_efficiency"),
            "dt_seconds": r.get("dt_seconds"),
            "physics_level": r.get("physics_level", ""),
            "compile_time_s": r.get("compile_time_s"),
            "source": str(source),
        })
    return out


def _key(row: dict) -> tuple:
    # Include component + n_levels so otherwise-identical rows (e.g. atm vs
    # ocean latlon, or L26 vs L40) do NOT collapse to one row (codex review).
    return (row["component"], row["backend"], row["grid"], row["case"],
            row["precision"], row["mode"], row["n_devices"], row["resolution"],
            row["n_levels"])


def collect(roots) -> tuple[list[dict], int]:
    """Return (deduped rows, n_dropped_duplicates) across one or more roots.

    ``roots`` may be a single path or an iterable of paths.  Flat per-case
    JSONs (atmosphere) and nested ``{results:[...]}`` JSONs (ocean) are both
    ingested.
    """
    if isinstance(roots, (str, Path)):
        roots = [roots]
    best: dict[tuple, dict] = {}
    dropped = 0

    def _add(row):
        nonlocal dropped
        if row is None or row.get("sypd") is None:
            return
        k = _key(row)
        if k in best:
            dropped += 1
            if row["sypd"] > best[k]["sypd"]:
                best[k] = row
        else:
            best[k] = row

    for root in roots:
        rp = Path(root)
        if not rp.exists():
            continue
        for jf in sorted(rp.rglob("*.json")):
            # Skip validation/smoke output dirs so their small fixed-resolution
            # probe points do not contaminate the production scaling curves.
            if "/val_" in str(jf) or "/_ab_" in str(jf):
                continue
            try:
                d = json.loads(jf.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(d, dict) and isinstance(d.get("results"), list):
                for row in _rows_from_nested(d, jf):
                    _add(row)
            else:
                _add(_row_from_json(d, jf))
    rows = sorted(
        best.values(),
        key=lambda r: (r["backend"], r["grid"], r["case"], r["precision"],
                       r["resolution"], r["n_devices"]),
    )
    return rows, dropped


def write_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def summarize(rows: list[dict]) -> str:
    from collections import Counter
    c = Counter((r["component"], r["backend"], r["grid"], r["case"], r["precision"])
                for r in rows)
    lines = [f"{n:3d}  {comp:5s} {b:3s} {g:12s} {case:9s} {prec}"
             for (comp, b, g, case, prec), n in sorted(c.items())]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", nargs="+", default=list(DEFAULT_ROOTS),
                   help="One or more result roots (atm flat + ocean nested).")
    p.add_argument("--out", default="results/bcw_scaling/bcw_scaling_tidy.csv")
    args = p.parse_args()

    rows, dropped = collect(args.root)
    write_csv(rows, Path(args.out))
    print(f"Collected {len(rows)} unique rows ({dropped} duplicate(s) dropped) "
          f"from {args.root}")
    print(f"Wrote {args.out}")
    print("count  comp  bk  grid         case      precision")
    print(summarize(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
