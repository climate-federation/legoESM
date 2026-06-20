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
import sys
from pathlib import Path

from legoesm import constants

FIELDS = [
    "component", "backend", "grid", "case", "precision", "mode",
    "n_devices", "cpus_per_task", "n_cores", "n_resource",
    "resolution", "resolution_km", "n_levels", "sypd", "time_per_step_ms",
    "total_cells", "mcells_per_s", "scaling_efficiency", "dt_seconds",
    "physics_level", "fix_mass", "compile_time_s", "source",
]


def _resource_count(backend: str, n_dev: int, n_cores: int) -> int:
    """The scaling x-axis: CPU -> cores (hybrid 8r x 4c = 32 cores, not 8),
    GPU/TPU -> device (=rank) count. Packed CPU (cpus_per_task=1) has
    n_cores == n_ranks so this is unchanged for the existing ladder."""
    return n_cores if backend == "CPU" else n_dev

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


# Grid families recognised in an output-dir name (longest/hyphen variants first
# so 'cubed-sphere' matches before any shorter token).  The nested GPU
# ScalingReport JSON does NOT serialize grid_type, so when one sub-dir holds a
# single grid (``.../atm/latlon_ng2/...``, ``.../ocean/mpas_f64/...``) the grid
# is recovered from that sub-dir name.
_KNOWN_GRIDS = ("cubed-sphere", "cubed_sphere", "icosahedral", "latlon",
                "spectral", "gaussian", "mpas")
# Short aliases that show up in ad-hoc sub-dir names (e.g. the GPU smoke dir
# ``atm_cube_1gpu``); mapped to the canonical grid family.
_GRID_ALIASES = {"cube": "cubed-sphere", "cubedsphere": "cubed-sphere",
                 "cs": "cubed-sphere", "ll": "latlon", "ico": "icosahedral"}


def _canon_grid(g: str) -> str:
    return "cubed-sphere" if g == "cubed_sphere" else g


def grid_from_path(source) -> str | None:
    """Infer the grid family from a result path when the JSON omits grid_type.

    Two passes per path PART (the GPU harnesses name sub-dirs ``<grid>_<suffix>``
    — ``latlon_ng2``, ``cubed-sphere_ng1``, ``mpas_f64``):
      1. PART equal to a known grid token or beginning ``<grid>_`` (canonical).
      2. underscore/hyphen sub-tokens matched against known grids + short
         aliases (catches ad-hoc names like ``atm_cube_1gpu`` -> cubed-sphere).
    Returns the canonical hyphen form, or None if no known grid token is found
    (the caller then SKIPS the row rather than emit a colliding empty grid)."""
    for part in Path(source).parts:
        for g in _KNOWN_GRIDS:
            if part == g or part.startswith(g + "_"):
                return _canon_grid(g)
        for tok in part.replace("-", "_").split("_"):
            if tok in _GRID_ALIASES:
                return _GRID_ALIASES[tok]
            if tok in _KNOWN_GRIDS:
                return _canon_grid(tok)
    return None


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
    backend = resolve_backend(d, source)
    cpt = int(d.get("cpus_per_task") or 1)
    n_cores = int(d.get("n_cores") or (int(n_dev) * cpt))
    return {
        "component": "atm",
        "backend": backend,
        "grid": grid,
        "case": _CASE.get(phys, phys),
        "precision": d.get("precision", ""),
        "mode": d.get("mode", ""),
        "n_devices": int(n_dev),
        "cpus_per_task": cpt,
        "n_cores": n_cores,
        "n_resource": _resource_count(backend, int(n_dev), n_cores),
        "fix_mass": d.get("fix_mass", True),
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


def _is_ocean_schema(d: dict) -> bool:
    """True only for the OCEAN nested report (mode starts 'ocean_').

    The atmosphere GPU harness (run_levante_gpu_scaling.py) writes nested
    ``{results:[...]}`` too, with mode 'weak'/'strong' — those must NOT be
    misrouted to ocean (codex review).  Gate on the ocean mode prefix at the
    top level or in the first result.
    """
    if str(d.get("mode", "")).startswith("ocean"):
        return True
    res = d.get("results")
    if isinstance(res, list) and res and isinstance(res[0], dict):
        return str(res[0].get("mode", "")).startswith("ocean")
    return False


def _rows_from_nested(d: dict, source: Path, component: str) -> list[dict]:
    """Nested ScalingReport schema ``{backend, results:[TimingResult,...]}``.

    Written by BOTH the ocean bench (``mode`` 'ocean_*') and the atmosphere GPU
    harness ``run_levante_gpu_scaling.py`` (``mode`` 'weak'/'strong').
    ``component`` is 'ocean' or 'atm' (decided by the caller from the mode
    prefix).  Flattens each inner result into one tidy row so the atmosphere
    (flat per-case JSON) and the nested GPU/ocean campaigns land in one unified
    table.  Neither nested report serializes ``grid_type``, so the grid is taken
    from the inner result if present, else inferred from the output-dir path
    (``grid_from_path``); ocean falls back to 'latlon' only as a last resort.
    """
    backend = str(d.get("backend", "")).upper() or backend_from_path(source)
    if backend not in ("CPU", "GPU", "TPU"):
        backend = backend_from_path(source)
    path_grid = grid_from_path(source)
    out = []
    skipped = 0
    for r in d.get("results") or []:
        if not isinstance(r, dict) or r.get("sypd") is None:
            continue
        n_dev = r.get("n_ranks", r.get("n_gpus"))
        if n_dev is None:
            continue
        phys = r.get("physics_level", "none")
        # Ocean keeps the historical 'latlon' fallback; atm has no safe default
        # (an empty grid would collide distinct grids in _key()), so an
        # unresolvable atm grid is SKIPPED + warned, never silently mislabelled.
        grid = r.get("grid_type") or path_grid or (
            "latlon" if component == "ocean" else "")
        if component != "ocean" and not grid:
            skipped += 1
            continue
        if component == "ocean":
            case = "ocean"
            mode = str(r.get("mode", "strong")).replace("ocean_", "") or "strong"
        else:
            case = _CASE.get(phys, phys)
            mode = str(r.get("mode", ""))
        res = r.get("resolution")
        cpt = int(r.get("cpus_per_task") or 1)
        n_cores = int(r.get("n_cores") or (int(n_dev) * cpt))
        out.append({
            "component": component,
            "backend": backend,
            "grid": grid,
            "case": case,
            "precision": r.get("precision", ""),
            "mode": mode,
            "n_devices": int(n_dev),
            "cpus_per_task": cpt,
            "n_cores": n_cores,
            "n_resource": _resource_count(backend, int(n_dev), n_cores),
            "fix_mass": r.get("fix_mass", True),
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
    if skipped:
        print(f"  [warn] {source}: skipped {skipped} nested-{component} "
              f"result(s) with unresolvable grid (no grid_type, no grid token "
              f"in the output-dir path)", file=sys.stderr)
    return out


def _key(row: dict) -> tuple:
    # Key on the RESOURCE count (CPU cores / GPU devices) so a hybrid
    # 8r x 4c run does not collide with a packed 32r x 1c run, and include
    # component + n_levels + physics_level + fix_mass so otherwise-identical
    # rows (atm vs ocean latlon, L26 vs L40, ocean baro solver, or a
    # NO_MASS_FIX ablation) never collapse to one row (codex review/audit).
    return (row["component"], row["backend"], row["grid"], row["case"],
            row["precision"], row["mode"], row["n_resource"], row["resolution"],
            row["n_levels"], row.get("physics_level", ""), row.get("fix_mass", True))


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
            # Skip validation/smoke/AB dirs (any path PART, incl. the root
            # itself) so their probe points never contaminate the curves.
            if any(part.startswith(("val_", "_ab_")) for part in jf.parts):
                continue
            try:
                d = json.loads(jf.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(d, dict) and isinstance(d.get("results"), list):
                # Nested ScalingReport.  Ocean (inner mode 'ocean_*') -> the
                # 'ocean' component; the atmosphere GPU harness
                # (run_levante_gpu_scaling.py) writes the SAME nested shape with
                # mode 'weak'/'strong' -> the 'atm' component (previously this
                # nested-atm report was silently dropped).  Grid is recovered
                # via grid_from_path since neither serializes grid_type.
                component = "ocean" if _is_ocean_schema(d) else "atm"
                for row in _rows_from_nested(d, jf, component):
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
