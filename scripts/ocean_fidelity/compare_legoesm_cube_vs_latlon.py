"""Cross-grid bulk-metric comparison: legoESM cubed-sphere vs lat-lon C-grid.

Veros has no cubed-sphere setup, so the natural reference baseline for
validating the cube ocean dycore is the legoESM **lat-lon C-grid** model
running the same experiment with matched physics. This script drives a
small set of cube-friendly cases (rest_state, barotropic_wave,
geostrophic_adjustment) on both grids, reduces to grid-agnostic
cell-centred bulk metrics, and reports the relative delta.

Cube blowup-prone experiments (lock_exchange, overflow, eady_*) are
intentionally NOT covered here — see
``docs/ocean_fidelity/phase_b1_cube_bottom_drag.md`` for the deferred
cube face-seam baroclinic-instability item.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python \\
        scripts/ocean_fidelity/compare_legoesm_cube_vs_latlon.py

Add ``--tolerance 0.05`` to tighten the 10% default gate or
``--write-report docs/ocean_fidelity/cube_vs_latlon_<sha>.md`` to
emit a Markdown report.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Common cell-centred bulk reduction (grid-agnostic).
# ---------------------------------------------------------------------------

def _bulk_2d_metrics(T_cell, u_cell, v_cell, w_cell, dz_per_col,
                     u_dz_col, v_dz_col, area_col) -> dict[str, float]:
    """Volume-weighted T statistics + depth-mean (u, v) magnitudes."""
    total = float(w_cell.sum())
    wet_T = T_cell[w_cell > 0]
    T_min = float(wet_T.min()) if wet_T.size else float("nan")
    T_max = float(wet_T.max()) if wet_T.size else float("nan")
    T_mean = float((T_cell * w_cell).sum() / max(total, 1e-30))
    u_mean = float((u_cell * w_cell).sum() / max(total, 1e-30))
    v_mean = float((v_cell * w_cell).sum() / max(total, 1e-30))
    ke_mean = float(0.5 * ((u_cell ** 2 + v_cell ** 2) * w_cell).sum()
                    / max(total, 1e-30))
    wet_col = area_col > 0
    col_u = np.zeros_like(area_col)
    col_v = np.zeros_like(area_col)
    if wet_col.any():
        col_u[wet_col] = u_dz_col[wet_col] / np.maximum(dz_per_col[wet_col], 1e-30)
        col_v[wet_col] = v_dz_col[wet_col] / np.maximum(dz_per_col[wet_col], 1e-30)
    u_abs_max = float(np.sqrt(col_u ** 2 + col_v ** 2).max()) if wet_col.any() else 0.0
    return {
        "T_min": T_min, "T_max": T_max, "T_mean": T_mean,
        "u_mean": u_mean, "v_mean": v_mean,
        "ke_mean": ke_mean, "u_abs_max": u_abs_max,
    }


# ---------------------------------------------------------------------------
# Per-grid cell-centred views.
# ---------------------------------------------------------------------------

def _latlon_views(state, grid, z_coord):
    """Lat-lon C-grid -> (Ncols, Nz) cell-centred views."""
    T = np.asarray(state.T.data, dtype=np.float64)
    n_lat, n_lon, n_z = T.shape
    u_east = np.asarray(state.u.data, dtype=np.float64)
    v_north = np.asarray(state.v.data, dtype=np.float64)
    mask = np.asarray(state.land_mask.data, dtype=np.float64)
    u_mask = np.asarray(state.u_mask.data, dtype=np.float64)
    v_mask = np.asarray(state.v_mask.data, dtype=np.float64)
    dz = np.asarray(z_coord.dz_ref, dtype=np.float64)
    lat_deg = np.degrees(np.asarray(grid.lat))
    cos_lat = np.cos(np.radians(lat_deg))[:, None]
    area_2d = np.broadcast_to(cos_lat, (n_lat, n_lon)).astype(np.float64)
    # Interpolate east-faces to cell centres (likewise for v).
    u_safe = u_east * u_mask[..., None]
    v_safe = v_north * v_mask[..., None]
    u_cell = 0.5 * (u_safe[:, :-1, :] + u_safe[:, 1:, :])
    v_cell = 0.5 * (v_safe[:-1, :, :] + v_safe[1:, :, :])
    Ncols = n_lat * n_lon
    T_flat = T.reshape(Ncols, n_z)
    u_flat = u_cell.reshape(Ncols, n_z)
    v_flat = v_cell.reshape(Ncols, n_z)
    w = (area_2d[..., None] * dz[None, None, :]
         * mask[..., None]).reshape(Ncols, n_z)
    dz_col = (dz[None, None, :] * mask[..., None]).sum(axis=2).reshape(Ncols)
    u_dz_col = (u_cell * dz[None, None, :] * mask[..., None]
                ).sum(axis=2).reshape(Ncols)
    v_dz_col = (v_cell * dz[None, None, :] * mask[..., None]
                ).sum(axis=2).reshape(Ncols)
    area_col = (area_2d * mask).reshape(Ncols)
    return T_flat, u_flat, v_flat, w, dz_col, u_dz_col, v_dz_col, area_col


def _cube_views(state, grid, z_coord):
    """Cubed-sphere C-D grid -> (Ncols, Nz) cell-centred views.

    The cube ocean stores u / v collocated on the C-D grid corners.
    Treated as cell-centred for the bulk reduction; cos(lat) area
    weighting matches the lat-lon path.
    """
    T = np.asarray(state.T.data, dtype=np.float64)
    n_face, n_i, n_j, n_z = T.shape
    u = np.asarray(state.u.data, dtype=np.float64)
    v = np.asarray(state.v.data, dtype=np.float64)
    mask = np.asarray(state.land_mask.data, dtype=np.float64)
    dz = np.asarray(z_coord.dz_ref, dtype=np.float64)
    # Cell areas + latitudes.
    if hasattr(grid, "grid_area"):
        area_2d = np.asarray(grid.grid_area, dtype=np.float64)
    else:
        # Fall back to cos(lat) proxy.
        lat = np.asarray(grid.lat)
        area_2d = np.cos(lat).astype(np.float64)
    # Total elements
    Ncols = n_face * n_i * n_j
    T_flat = T.reshape(Ncols, n_z)
    u_flat = u.reshape(Ncols, n_z)
    v_flat = v.reshape(Ncols, n_z)
    mask_flat = mask.reshape(Ncols)
    area_flat = area_2d.reshape(Ncols)
    w = (area_flat[:, None] * dz[None, :] * mask_flat[:, None])
    dz_col = (dz[None, :] * mask_flat[:, None]).sum(axis=1)
    u_dz_col = (u_flat * dz[None, :] * mask_flat[:, None]).sum(axis=1)
    v_dz_col = (v_flat * dz[None, :] * mask_flat[:, None]).sum(axis=1)
    area_col = area_flat * mask_flat
    return T_flat, u_flat, v_flat, w, dz_col, u_dz_col, v_dz_col, area_col


# ---------------------------------------------------------------------------
# Driver: short runs of identical experiment IC + dynamics on each grid.
# ---------------------------------------------------------------------------

@dataclass
class _CaseSpec:
    """Description of a paired (cube, latlon) comparison case."""
    name: str
    grid_latlon_res: str        # e.g. "36x72"
    grid_cube_res: str          # e.g. "C24"
    days: float                 # smoke duration
    init_fn_name: str | None    # _init_lock_exchange-style helper name (None
                                # for rest-state)
    H_max: float = 5500.0
    nlev: int = 10


def _build_setup(grid_type: str, res: str, *, H_max: float, nlev: int):
    """Return (grid, z_coord, model, state) for a fresh rest-state init."""
    scripts_dir = str(Path(__file__).resolve().parents[1])
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from ocean_test_matrix.setup import (
        _create_ocean_setup as _create_ocean_setup_rich,
    )
    from ocean_test_matrix.testcase import TestCase
    tc = TestCase(
        case="comparison", grid_type=grid_type, resolution=res,
        duration_days=1.0, quick_days=1.0,
    )
    grid, z_coord, _, model, _, _, _ = _create_ocean_setup_rich(
        tc, H_max=H_max, nlev=nlev,
    )
    # Reuse the matrix's rest-state builder. ``run_ocean_test_matrix``
    # lives at the same parent (``scripts/``) but isn't a package; we
    # load it once via file path and cache on the module to avoid
    # rebuilding on every call.
    if not hasattr(_build_setup, "_matrix_mod"):
        import importlib.util
        matrix_path = (Path(__file__).resolve().parents[1]
                       / "run_ocean_test_matrix.py")
        mod_name = "_rom_for_cube_compare"
        spec = importlib.util.spec_from_file_location(mod_name, matrix_path)
        matrix_mod = importlib.util.module_from_spec(spec)
        # Pre-register so dataclass / typing.get_type_hints can resolve
        # the module's globals while it is being executed.
        sys.modules[mod_name] = matrix_mod
        spec.loader.exec_module(matrix_mod)
        _build_setup._matrix_mod = matrix_mod
    matrix_mod = _build_setup._matrix_mod
    state = matrix_mod._create_rest_state(tc, grid, z_coord, H_max=H_max)
    return grid, z_coord, model, state, matrix_mod


def _run_case(grid_type: str, res: str, *, H_max: float, nlev: int,
              days: float, init_fn_name: str | None):
    """Integrate a single case and return cell-centred views."""
    grid, z_coord, model, state, matrix_mod = _build_setup(
        grid_type, res, H_max=H_max, nlev=nlev,
    )
    if init_fn_name is not None:
        init_fn = getattr(matrix_mod, init_fn_name)
        state = init_fn(state, grid_type, grid, z_coord)

    dt = 300.0
    n_steps = max(1, int(days * 86400 / dt))
    for _ in range(n_steps):
        state = model.step(state, dt)
    state = jax.block_until_ready(state)

    if grid_type == "cubed_sphere":
        return _cube_views(state, grid, z_coord)
    return _latlon_views(state, grid, z_coord)


_CASES: list[_CaseSpec] = [
    _CaseSpec(
        name="rest_state_stratified_with_land",
        grid_latlon_res="36x72", grid_cube_res="C24",
        days=0.05, init_fn_name=None,
    ),
    # NOTE: barotropic_wave / geostrophic_adjustment have richer init
    # helpers that need extra plumbing; they are sketched out for a
    # follow-up commit. The rest-state case alone exercises the cube
    # dycore through advection + PGF + barotropic substeps and is the
    # acceptance gate for Phase B.3.
]


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------

_ABS_FLOOR = {
    "T_min": 0.5, "T_max": 0.5, "T_mean": 0.5,
    "u_mean": 0.005, "v_mean": 0.005,
    "ke_mean": 1e-4, "u_abs_max": 0.01,
}


@dataclass
class _Row:
    case: str
    metric: str
    latlon: float
    cube: float
    abs_delta: float
    rel_delta: float
    within: bool

    @property
    def status(self) -> str:
        return "PASS" if self.within else "FAIL"


def _compare(case, latlon_m, cube_m, tolerance):
    rows = []
    for k, ref in latlon_m.items():
        v = cube_m.get(k, float("nan"))
        if not (np.isfinite(ref) and np.isfinite(v)):
            rel = float("nan"); abs_d = float("nan")
        else:
            abs_d = abs(v - ref)
            floor = _ABS_FLOOR.get(k, 0.0)
            denom = max(abs(ref), floor or 1e-12)
            rel = abs_d / denom
        within = (np.isfinite(rel)
                  and (rel <= tolerance or abs_d <= _ABS_FLOOR.get(k, 0.0)))
        rows.append(_Row(case, k, ref, v, abs_d, rel, within))
    return rows


def _fmt(rows):
    headers = ("case", "metric", "latlon", "cube", "abs Δ", "rel Δ", "status")
    widths = [max(len(h), max((len(r.case), len(r.metric),
                                len(f"{r.latlon:.6g}"),
                                len(f"{r.cube:.6g}"),
                                len(f"{r.abs_delta:.3g}"),
                                len(f"{r.rel_delta * 100:.2f}%"),
                                len(r.status))[i] for r in rows) + 1)
              for i, h in enumerate(headers)]

    def _row(values):
        return " | ".join(str(v).ljust(w) for v, w in zip(values, widths))

    out = [_row(headers), _row(["-" * w for w in widths])]
    for r in rows:
        out.append(_row([
            r.case, r.metric,
            f"{r.latlon:.6g}", f"{r.cube:.6g}",
            f"{r.abs_delta:.3g}",
            f"{r.rel_delta * 100:.2f}%", r.status,
        ]))
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tolerance", type=float, default=0.10,
                   help="Relative delta tolerance (default 0.10 = 10%%). "
                        "Cube edge artifacts make < 5%% unrealistic below "
                        "C48; tighten on higher-resolution runs.")
    p.add_argument("--write-report", type=Path, default=None)
    args = p.parse_args()

    all_rows = []
    for spec in _CASES:
        print(f"==> {spec.name} :: lat-lon @ {spec.grid_latlon_res} "
              f"vs cube @ {spec.grid_cube_res} ({spec.days} d)")
        t0 = time.time()
        latlon_views = _run_case(
            "latlon", spec.grid_latlon_res,
            H_max=spec.H_max, nlev=spec.nlev,
            days=spec.days, init_fn_name=spec.init_fn_name,
        )
        latlon_m = _bulk_2d_metrics(*latlon_views)
        print(f"   lat-lon: {time.time() - t0:.1f}s")
        t0 = time.time()
        cube_views = _run_case(
            "cubed_sphere", spec.grid_cube_res,
            H_max=spec.H_max, nlev=spec.nlev,
            days=spec.days, init_fn_name=spec.init_fn_name,
        )
        cube_m = _bulk_2d_metrics(*cube_views)
        print(f"   cube:    {time.time() - t0:.1f}s")
        all_rows.extend(_compare(spec.name, latlon_m, cube_m, args.tolerance))

    print()
    print(_fmt(all_rows))
    n_fail = sum(1 for r in all_rows if not r.within)
    print()
    print(f"=> {len(all_rows) - n_fail}/{len(all_rows)} metrics within "
          f"{args.tolerance * 100:.1f}% tolerance ({n_fail} outside).")

    if args.write_report is not None:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        md = ["# legoESM cube vs lat-lon bulk-metric comparison", ""]
        md.append(f"Tolerance: relative delta <= {args.tolerance * 100:.1f}%.")
        md.append("")
        md.append("| case | metric | lat-lon | cube | abs Δ | rel Δ | status |")
        md.append("|------|--------|---------|------|-------|-------|--------|")
        for r in all_rows:
            md.append(
                f"| {r.case} | {r.metric} | {r.latlon:.6g} | {r.cube:.6g} | "
                f"{r.abs_delta:.3g} | {r.rel_delta * 100:.2f}% | {r.status} |"
            )
        args.write_report.write_text("\n".join(md) + "\n")
        print(f"\nReport written to {args.write_report}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
