#!/usr/bin/env python
"""Iter-922 PPM-boundary 5-way Pareto map.

Following iter-921 (which discovered the v_ll_Linf vs h_err_max
trade-off when toggling `apply_fortran_xppm_boundary`), iter-922
extends the audit to the iter-892/iter-900/iter-903 sub-flags
(`fortran_faithful_ppm_left/right`) and shows that the strict-Fortran
variants are PARETO-DOMINATED by the iter-893 default 1-cell-shifted
formula on (v_ll_Linf, h_err_max).

5 production W2 C36 1-day cases:

| label | xppm  | faithful_left | faithful_right |
|-------|-------|---------------|----------------|
| A     | False |               |                |
| B     | True  | False         | False          | (iter-893 default — production)
| C     | True  | True          | False          |
| D     | True  | False         | True           |
| E     | True  | True          | True           |

Two Pareto-non-dominated points emerge:
  A: (v_ll_Linf=0.1593, h_err_max=4.62 m)  best h_err, worst v_ll
  B: (v_ll_Linf=0.1319, h_err_max=8.18 m)  best v_ll, intermediate h_err

C, D, E are all Pareto-dominated by B.

Output: `diagnostics/fv3_visual/iter922_ppm_boundary_pareto.png`
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


def _run(apply_xppm: bool, faithful_left: bool, faithful_right: bool) -> dict:
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
        fortran_faithful_ppm_left=faithful_left,
        fortran_faithful_ppm_right=faithful_right,
    )
    model = FV3EdgeShallowWaterModel(grid, cfg)
    model.set_initial_mass(state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(NSTEPS):
            state = model.step(state, DT)
    h_err = np.asarray(state.h - sw.h.data)
    ca, sa = cell_centre_angles_from_4edge(cdgrid)
    u_cc = 0.5 * (
        np.asarray(state.u_d)[:, :, :-1] + np.asarray(state.u_d)[:, :, 1:]
    )
    v_cc = 0.5 * (
        np.asarray(state.v_d)[:, :-1, :] + np.asarray(state.v_d)[:, 1:, :]
    )
    v_north = np.asarray(sa) * u_cc + np.asarray(ca) * v_cc
    weights = get_cubedsphere_to_latlon_weights(N, n_lon=360, n_lat=181)
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)
    return {
        "h_err_max": float(np.abs(h_err).max()),
        "h_err_l2": float(np.sqrt(np.mean(h_err**2))),
        "v_ll_Linf": float(np.abs(v_ll).max()),
    }


def _is_pareto_dominated(point: dict, others: list[dict]) -> bool:
    """Return True if point is dominated on (v_ll_Linf, h_err_max) by any other."""
    for o in others:
        if (
            o["v_ll_Linf"] <= point["v_ll_Linf"]
            and o["h_err_max"] <= point["h_err_max"]
            and (
                o["v_ll_Linf"] < point["v_ll_Linf"]
                or o["h_err_max"] < point["h_err_max"]
            )
        ):
            return True
    return False


def main() -> None:
    cases = [
        ("A: xppm=False (iter-820)", False, False, False),
        ("B: xppm=True default (iter-893)", True, False, False),
        ("C: xppm=True faithful_left", True, True, False),
        ("D: xppm=True faithful_right", True, False, True),
        ("E: xppm=True faithful_both", True, True, True),
    ]
    results = []
    print(
        f"{'case':40s} {'v_ll_Linf':>10s}  {'h_err_max':>10s}  {'h_err_l2':>10s}"
    )
    for label, axp, fl, fr in cases:
        r = _run(axp, fl, fr)
        r["label"] = label
        results.append(r)
        print(
            f"{label:40s} "
            f"{r['v_ll_Linf']:10.4e}  "
            f"{r['h_err_max']:10.3f}  "
            f"{r['h_err_l2']:10.4f}"
        )

    print()
    print("Pareto analysis on (v_ll_Linf, h_err_max), smaller = better:")
    for r in results:
        others = [o for o in results if o is not r]
        dom = _is_pareto_dominated(r, others)
        tag = "Pareto-DOMINATED" if dom else "Pareto-non-dominated"
        print(f"  {r['label']:40s} -> {tag}")

    fig, ax = plt.subplots(figsize=(9, 6))
    for r in results:
        others = [o for o in results if o is not r]
        dom = _is_pareto_dominated(r, others)
        color = "C7" if dom else "C3"
        marker = "x" if dom else "o"
        ax.scatter(
            r["v_ll_Linf"], r["h_err_max"],
            s=140, c=color, marker=marker, zorder=3,
            label=r["label"],
        )
        ax.annotate(
            f"{r['label'].split(':')[0]}",
            (r["v_ll_Linf"], r["h_err_max"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=10,
            fontweight="bold",
        )
    nondom = [r for r in results
              if not _is_pareto_dominated(
                  r, [o for o in results if o is not r])]
    nondom_sorted = sorted(nondom, key=lambda r: r["v_ll_Linf"])
    if len(nondom_sorted) >= 2:
        ax.plot(
            [r["v_ll_Linf"] for r in nondom_sorted],
            [r["h_err_max"] for r in nondom_sorted],
            "k--", alpha=0.6, label="Pareto frontier",
        )
    ax.set_xlabel(r"$v_{ll}$ Linf  [m s$^{-1}$]")
    ax.set_ylabel("h_err Linf [m]")
    ax.set_title(
        f"W2 C{N} 1-day PPM-boundary 5-way Pareto map\n"
        f"o = Pareto-non-dominated, x = dominated"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUT, "iter922_ppm_boundary_pareto.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
