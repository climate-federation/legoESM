#!/usr/bin/env python
"""Iter-921 visual diagnostic: A/B production W2 with vs without
`apply_fortran_xppm_boundary` (iter-893's flag).

Iter-893 reduces v_ll_Linf (the production sentinel) by 17 %
(0.159 -> 0.132 m/s) but the h-error Linf norm REGRESSES by 77 %
(4.62 -> 8.18 m).  Total RMS error is unchanged: the cube-vertex
artifact pattern re-localised rather than smoothed out.

This script regenerates the iter-820 baseline and iter-893 snapshots
side by side and writes a Pareto summary plot showing where each
metric lives.  Output:

    diagnostics/fv3_visual/iter921_v_vs_h_pareto.png

Run:

    JAX_ENABLE_X64=1 .venv/bin/python scripts/diag_iter921_w2_v_vs_h_pareto.py
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from legoesm import constants
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
    FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import (
    cell_centre_angles_from_4edge,
    create_cubed_sphere_cdgrid,
)
from legoesm.grids.regridding import (
    apply_cubedsphere_to_latlon,
    get_cubedsphere_to_latlon_weights,
)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2,
)


N = 36
DT = 300.0
NSTEPS = int(86400 / DT)
OUT = os.path.join(os.path.dirname(__file__), "..", "diagnostics", "fv3_visual")


def _div_damp_cube(n: int, ref_n: int = 48, ref_coeff: float = 1.5e7) -> float:
    return ref_coeff * (ref_n / n) ** 2


def _run(apply_xppm: bool) -> dict:
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(N),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
        apply_fortran_xppm_boundary=apply_xppm,
    )
    model = FV3EdgeShallowWaterModel(grid, cfg)
    model.set_initial_mass(state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(NSTEPS):
            state = model.step(state, DT)
    ca, sa = cell_centre_angles_from_4edge(cdgrid)
    u_cc = 0.5 * (
        np.asarray(state.u_d)[:, :, :-1] + np.asarray(state.u_d)[:, :, 1:]
    )
    v_cc = 0.5 * (
        np.asarray(state.v_d)[:, :-1, :] + np.asarray(state.v_d)[:, 1:, :]
    )
    h_err = np.asarray(state.h - sw.h.data)
    v_north = np.asarray(sa) * u_cc + np.asarray(ca) * v_cc
    weights = get_cubedsphere_to_latlon_weights(N, n_lon=360, n_lat=181)
    return {
        "h_err": h_err,
        "v_north": v_north,
        "h_err_ll": apply_cubedsphere_to_latlon(h_err, weights),
        "v_ll": apply_cubedsphere_to_latlon(v_north, weights),
    }


def _summary(label: str, out: dict) -> dict:
    s = {
        "label": label,
        "h_err_max": float(np.abs(out["h_err"]).max()),
        "h_err_l2": float(np.sqrt(np.mean(out["h_err"] ** 2))),
        "v_north_max": float(np.abs(out["v_north"]).max()),
        "v_ll_Linf": float(np.abs(out["v_ll"]).max()),
    }
    print(
        f"  {label:30s} h_err_max={s['h_err_max']:6.3f}  "
        f"h_err_L2={s['h_err_l2']:.4f}  "
        f"v_north_max={s['v_north_max']:.4f}  "
        f"v_ll_Linf={s['v_ll_Linf']:.4f}"
    )
    return s


def _hex_pareto(s_off: dict, s_on: dict, fname: str) -> None:
    """One-panel summary: x = v_ll_Linf (smaller better), y = h_err_max (smaller better)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        [s_off["v_ll_Linf"], s_on["v_ll_Linf"]],
        [s_off["h_err_max"], s_on["h_err_max"]],
        "k--",
        alpha=0.4,
    )
    ax.scatter(
        [s_off["v_ll_Linf"]], [s_off["h_err_max"]],
        s=140, c="C0", label=s_off["label"], zorder=3,
    )
    ax.scatter(
        [s_on["v_ll_Linf"]], [s_on["h_err_max"]],
        s=140, c="C3", label=s_on["label"], zorder=3,
    )
    for s, off in ((s_off, (-0.012, +0.5)), (s_on, (+0.001, -0.6))):
        ax.annotate(
            f"({s['v_ll_Linf']:.4f}, {s['h_err_max']:.2f})",
            (s["v_ll_Linf"], s["h_err_max"]),
            textcoords="offset points",
            xytext=(off[0] * 1000, off[1] * 10),
            fontsize=9,
        )
    ax.set_xlabel("v_ll_Linf  [m s$^{-1}$]   (smaller = better)")
    ax.set_ylabel("h_err Linf [m]            (smaller = better)")
    ax.set_title(
        f"W2 C{N} 1-day Pareto trade-off\n"
        f"`apply_fortran_xppm_boundary` reduces v but worsens h Linf"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    path = os.path.join(OUT, fname)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def main() -> None:
    print(f"=== iter-921 W2 v vs h Pareto at C{N}, 1 day ===")
    out_off = _run(apply_xppm=False)
    out_on = _run(apply_xppm=True)
    s_off = _summary("iter-820 (xppm=False)", out_off)
    s_on = _summary("iter-893 (xppm=True)", out_on)

    print()
    print("Δ relative (iter-893 vs iter-820):")
    for k in ("h_err_max", "h_err_l2", "v_north_max", "v_ll_Linf"):
        delta = s_on[k] - s_off[k]
        pct = 100.0 * delta / s_off[k]
        print(f"  {k:14s}: {pct:+6.2f} %")

    _hex_pareto(s_off, s_on, "iter921_v_vs_h_pareto.png")


if __name__ == "__main__":
    main()
