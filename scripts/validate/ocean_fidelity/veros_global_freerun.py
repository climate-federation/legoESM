"""Shared helpers for the global VEROS free-run validators (4°, 1°, flexible).

``run_global_{1deg,4deg,flexible}_freerun.py`` carried byte-identical copies
of the forcing-file reader, the metrics computation and the oracle-ratio
table printer.  This module owns the parts whose logic is identical across
the three drivers; each driver keeps its own ``main()`` / ``run()`` /
``load_and_prepare()`` (their argparse flags, default horizons, stability
fallbacks and forcing-channel composition genuinely differ).

These are pure validator glue (NumPy + stdlib only, no model numerics and no
autodiff), per ``docs/ocean_fidelity/oracle_recipe_strategy.md``.
"""
from __future__ import annotations

import json

import numpy as np

__all__ = ["read_nc", "compute_metrics", "print_comparison"]

# The ratio-table columns (legoESM / Veros oracle), shared by all drivers.
_COMPARE_KEYS = (
    "total_ke_j", "psi_min_sv", "psi_max_sv", "psi_range_sv",
    "vol_mean_T", "vol_mean_S", "max_abs_u", "mean_eke",
)


def read_nc(path, var):
    """Read a Veros forcing variable, transposed to ``(x, y[, …])`` order.

    Mirrors Veros's ``_read_forcing`` / ``_get_data``: read as float and
    transpose so the leading axis is zonal.
    """
    import h5netcdf
    with h5netcdf.File(path, "r") as f:
        return np.array(f.variables[var], dtype="float").T


def compute_metrics(state, recipe, area, dz) -> dict:
    """Oracle-schema metrics for a global free-run state.

    The numerics are identical across the three resolutions; only the
    per-latitude T-cell ``area`` weight ``(NY, 1)`` and the full-cell layer
    thickness ``dz`` ``(NZ,)`` are recipe-specific (each driver builds them
    from its own grid helpers) and so are passed in.

    Parameters
    ----------
    state :
        The integrated ``LatLonCGridOceanState``.
    recipe :
        The recipe object (provides ``z_coord``, ``grid``, ``model_config``).
    area : np.ndarray
        Per-latitude T-cell area weight, shape ``(NY, 1)`` (interior rows).
    dz : np.ndarray
        Full-cell layer thicknesses, shape ``(NZ,)`` (legoESM top-down order).
    """
    ia = np.asarray(recipe.z_coord.is_active)[1:-1, :, :]   # interior
    area = np.asarray(area)
    dz = np.asarray(dz)
    vol = area[:, :, None] * dz[None, None, :] * ia

    u = np.asarray(state.u.data)
    v = np.asarray(state.v.data)
    T = np.asarray(state.T.data)[1:-1]
    S = np.asarray(state.S.data)[1:-1]

    # u/v at the T-cell's east/north face — Veros's u[i,j,k]/v[i,j,k].
    u_cell = u[1:-1, 1:, :]
    v_cell = v[2:-1, :, :]

    ia_e = np.minimum(ia, np.roll(ia, -1, axis=1))           # maskU (min rule)
    rho0 = float(recipe.model_config.rho_0)

    def wmean(x, w):
        sw = w.sum()
        return float((x * w).sum() / max(sw, 1e-30))

    speed2 = u_cell ** 2 + v_cell ** 2

    psi = np.asarray(state.psi) if state.psi is not None else np.zeros((1,))
    psi_min, psi_max = float(psi.min() / 1e6), float(psi.max() / 1e6)

    out_we = {}
    for name in ("tke", "eke"):
        fld = getattr(state, name)
        if fld is None:
            out_we[f"mean_{name}"] = float("nan")
            continue
        e = np.asarray(fld.data)[1:-1, :, :]                 # (NY, NX, NZ-1)
        w_if = vol[:, :, 1:]                                  # cell below
        out_we[f"mean_{name}"] = wmean(e, w_if)

    return dict(
        psi_min_sv=psi_min,
        psi_max_sv=psi_max,
        psi_range_sv=psi_max - psi_min,
        total_ke_j=0.5 * rho0 * float((speed2 * vol).sum()),
        vol_mean_T=wmean(T, vol),
        vol_mean_S=wmean(S, vol),
        max_abs_u=float(np.max(np.abs(u_cell * ia_e))),
        sfc_T_mean=wmean(T[..., :1], vol[..., :1]),
        mean_tke=out_we["mean_tke"],
        mean_eke=out_we["mean_eke"],
        finite=bool(np.isfinite(u).all() and np.isfinite(T).all()
                    and np.isfinite(S).all()),
    )


def print_comparison(yearly, compare_path):
    """Print the per-year (legoESM / Veros oracle) ratio table."""
    with open(compare_path) as f:
        ref = json.load(f)["yearly"]
    keys = _COMPARE_KEYS
    print("\n== per-year ratios (legoESM / Veros oracle) ==")
    print("year " + " ".join(f"{k:>13s}" for k in keys))
    for m in yearly:
        yr = m.get("year")
        if yr is None or float(yr) != int(float(yr)):
            continue
        rv = next((r for r in ref if r.get("year") == int(float(yr))), None)
        if rv is None:
            continue
        cells = []
        for k in keys:
            denom = rv[k]
            cells.append(f"{m[k] / denom:13.3f}" if denom else f"{'n/a':>13s}")
        print(f"{int(float(yr)):4d} " + " ".join(cells))
