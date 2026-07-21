"""Aggregate baroclinic-wave scaling JSONs into one tidy CSV for plotting.

Walks a results root (default ``results/bcw_scaling``), reads every flat
per-case JSON written by ``run_cpu_mpi_scaling.py`` (and the equivalent
single-process ``run_levante_gpu_scaling.py`` rows), plus the ``.jsonl``
records appended by the SPMD bench lanes (``bench_atm_latlon_spmd_scaling``,
``bench_mpas_spmd_scaling``, ``bench_ocean_latlon_spmd_scaling``,
``bench_cube_tiled_step_scaling``), and emits one tidy row per
measured point with the columns the publication plotter consumes:

    backend, grid, case, precision, mode, n_devices, resolution,
    resolution_km, n_levels, sypd, time_per_step_ms, total_cells,
    mcells_per_s, scaling_efficiency, dt_seconds, physics_level,
    compile_time_s, source

``backend`` (CPU vs GPU) is inferred from the output-directory name
(``..._gpu_...`` / ``..._cpu_...``) because the flat CPU-MPI JSON does not
record a device field.  ``case`` maps ``physics_level`` ("none"->"dry",
"moist"->"moist", others kept verbatim).  ``resolution_km`` is the nominal
horizontal grid spacing for the grid family.  Duplicate keys (the full
``_key`` tuple: component, backend, grid, case, precision, mode, n_resource,
n_devices, resolution, n_levels, physics_level, fix_mass) keep the row with
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


def resolve_backend(d: dict, source: Path) -> str:
    """Prefer the JAX backend recorded in the JSON; fall back to the path.

    Newer runs serialize ``backend`` (cpu/gpu/cuda/tpu); older ones don't, so
    we still infer CPU-vs-GPU from the output-dir name as a fallback.
    """
    jb = str(d.get("backend", "") or d.get("platform", "") or "").lower()
    if jb in ("gpu", "cuda", "rocm"):
        return "GPU"
    if jb == "cpu":
        return "CPU"
    if jb == "tpu":
        return "TPU"
    return backend_from_path(source)


def _row_from_json(d: dict, source: Path) -> dict | None:
    """Build a tidy row from one flat per-case JSON, or None if not a case.

    Serves BOTH the run_cpu_mpi_scaling flat JSONs (n_ranks semantics) and
    the SPMD bench-lane JSONL records (n_devices semantics; single-process
    multi-device, so n_ranks would collapse the whole ladder to 1).
    Virtual-CPU-device proxy rows (forced host-platform devices) are
    communication-overhead probes, NOT hardware scaling — skipped so they
    never contaminate a CPU curve.
    """
    if "sypd" not in d or "grid_type" not in d:
        return None
    md = d.get("metadata")
    if isinstance(md, dict) and md.get("virtual_cpu_devices"):
        return None
    grid = d["grid_type"]
    n_dev = d.get("n_devices", d.get("n_ranks", d.get("n_gpus")))
    if n_dev is None:
        return None
    phys = d.get("physics_level", "none")
    res = d.get("resolution")
    backend = resolve_backend(d, source)
    cpt = int(d.get("cpus_per_task") or 1)
    n_cores = int(d.get("n_cores") or (int(n_dev) * cpt))
    is_ocean = str(d.get("component", "")).lower() == "ocean"
    return {
        "component": "ocean" if is_ocean else "atm",
        "backend": backend,
        "grid": grid,
        "case": "ocean" if is_ocean else _CASE.get(phys, phys),
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


def _is_atm_nested_schema(d: dict) -> bool:
    """True only for the atmosphere GPU harness nested report
    (run_levante_gpu_scaling.py): a non-ocean mode plus per-result atmosphere
    markers (``grid_type`` / ``physics_level``).  POSITIVE match so an
    unrecognized/typoed nested schema is SKIPPED rather than silently mislabeled
    'atm' (codex review — the prior ``else: 'atm'`` default mislabeled anything
    that was not ocean)."""
    if str(d.get("mode", "")).startswith("ocean"):
        return False
    res = d.get("results")
    if not (isinstance(res, list) and res and isinstance(res[0], dict)):
        return False
    r0 = res[0]
    if str(r0.get("mode", "")).startswith("ocean"):
        return False
    return ("grid_type" in r0) or ("physics_level" in r0)


def _rows_from_nested(d: dict, source: Path, component: str = "ocean") -> list[dict]:
    """Nested ``{backend, results:[TimingResult,...]}`` report → tidy rows.

    Serves BOTH campaigns that emit a nested report:
      * ``component='ocean'`` — the ocean GPU/CPU campaign (mode 'ocean_*'),
        case label 'ocean', grid defaulting to 'latlon'.
      * ``component='atm'`` — the atmosphere GPU harness
        (run_levante_gpu_scaling.py), grid taken from the per-result
        ``grid_type`` field, case derived from physics_level like the flat
        atm path.  Without this branch a nested atm report has no flat
        top-level ``sypd``/``grid_type`` and is silently dropped.
    """
    backend = str(d.get("backend", "")).upper() or backend_from_path(source)
    if backend not in ("CPU", "GPU", "TPU"):
        backend = backend_from_path(source)
    is_ocean = component == "ocean"
    out = []
    for r in d.get("results") or []:
        if not isinstance(r, dict) or r.get("sypd") is None:
            continue
        n_dev = r.get("n_ranks", r.get("n_gpus"))
        if n_dev is None:
            continue
        phys = r.get("physics_level", "none")
        grid = r.get("grid_type", "latlon" if is_ocean else "")
        case = "ocean" if is_ocean else _CASE.get(phys, phys)
        res = r.get("resolution")
        cpt = int(r.get("cpus_per_task") or 1)
        n_cores = int(r.get("n_cores") or (int(n_dev) * cpt))
        out.append({
            "component": component,
            "backend": backend,
            "grid": grid,
            "case": case,
            "precision": r.get("precision", ""),
            "mode": str(r.get("mode", "strong")).replace("ocean_", "") or "strong",
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
            "physics_level": phys,
            "compile_time_s": r.get("compile_time_s"),
            "source": str(source),
        })
    return out


def _key(row: dict) -> tuple:
    # Key on BOTH the resource count (CPU cores / GPU devices) AND n_devices
    # (rank/face count).  n_devices is required for the cube route-B node-fill
    # lane, where every rung fills the node (n_cores ~ 128 for N in {1,2,3,6}
    # via THREADS=128/N) so a cores-only key would collapse the 4-rung curve to
    # ~2 points; it also makes a hybrid 8r x 4c run distinct from a packed
    # 32r x 1c run at the same 32 cores (the stated intent this code previously
    # failed to implement).  component + n_levels + physics_level + fix_mass
    # keep otherwise-identical rows (atm vs ocean latlon, L26 vs L40, ocean
    # baro solver, NO_MASS_FIX ablation) from collapsing (codex review/audit).
    return (row["component"], row["backend"], row["grid"], row["case"],
            row["precision"], row["mode"], row["n_resource"], row["n_devices"],
            row["resolution"], row["n_levels"],
            row.get("physics_level", ""), row.get("fix_mass", True))


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
                # Nested report: ocean campaign OR the atmosphere GPU harness
                # (run_levante_gpu_scaling.py).  Route by POSITIVE schema match;
                # an unrecognized nested schema is SKIPPED with a warning, never
                # silently mislabeled 'atm' (codex review).
                if _is_ocean_schema(d):
                    component = "ocean"
                elif _is_atm_nested_schema(d):
                    component = "atm"
                else:
                    print(f"WARNING: unrecognized nested report schema, skipping "
                          f"{jf}", file=sys.stderr)
                    component = None
                if component is not None:
                    for row in _rows_from_nested(d, jf, component=component):
                        _add(row)
            else:
                _add(_row_from_json(d, jf))
        # SPMD bench lanes (bench_atm_latlon_spmd_scaling,
        # bench_mpas_spmd_scaling, bench_ocean_latlon_spmd_scaling,
        # bench_cube_tiled_step_scaling) append one flat record per line
        # to a .jsonl — same row builder, one dict per line.
        for jf in sorted(rp.rglob("*.jsonl")):
            if any(part.startswith(("val_", "_ab_")) for part in jf.parts):
                continue
            try:
                lines = jf.read_text().splitlines()
            except OSError:
                continue
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(d, dict):
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
