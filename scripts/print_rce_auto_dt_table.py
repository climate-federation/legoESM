#!/usr/bin/env python
"""Print the auto-dt ladder table for cross-grid RCE runs.

Reads ``legoesm.driver.rce_dt.auto_dt_rce`` + the iter-29 empirical
``dt ∝ dx²`` fit + the gravity-wave CFL formula and shows the chosen
dt + both bounds for every grid+resolution in the production matrix.

Usage::

    .venv/bin/python scripts/print_rce_auto_dt_table.py

Useful for:
* Planning a new resolution before launching a 30-day run (compare
  ladder dt against the dx² fit and gravity-wave CFL).
* Spotting structural drift in the ladder during a refactor.
* Documentation / commit messages (paste the output as a table).

iter-31 (2026-05) addition; no JAX import, runs in <1 s.
"""
from __future__ import annotations

from legoesm.core.cfl import (
    cfl_max_dt, estimate_min_dx_cubed_sphere,
    estimate_min_dx_gaussian, estimate_min_dx_icosahedral,
    estimate_min_dx_latlon,
)
from legoesm.driver.rce_dt import auto_dt_rce, empirical_dt_dx2


CASES = [
    ("cubed_sphere", "C", estimate_min_dx_cubed_sphere, [24, 48, 72, 96]),
    ("latlon", "LL", estimate_min_dx_latlon, [16, 32]),
    ("gaussian", "T", estimate_min_dx_gaussian, [21, 42]),
    ("voronoi", "V", estimate_min_dx_icosahedral, [4, 5]),
]


def main():
    print(
        f"{'grid':>14}  {'res':>5}  {'dx_min[m]':>10}  "
        f"{'ladder dt':>10}  {'CFL [s]':>9}  {'CFL ratio':>10}  "
        f"{'dx² fit':>9}  {'fit ratio':>10}"
    )
    print("-" * 90)
    for grid_type, prefix, dx_fn, resolutions in CASES:
        for N in resolutions:
            dx = dx_fn(N)
            cfl = cfl_max_dt(dx, 300.0, cfl_number=0.8, ndim=2)
            try:
                dt = auto_dt_rce(grid_type, N)
            except ValueError as exc:
                dt_str = "REFUSED"
                cfl_ratio_str = "—"
                fit_str = "—"
                fit_ratio_str = "—"
            else:
                dt_str = f"{dt:>10.1f}"
                cfl_ratio = dt / cfl
                cfl_ratio_str = f"{cfl_ratio:>10.2f}×"
                # dx² fit only meaningful for grids it was fitted on
                # (cubed_sphere); show it anyway for context.
                fit_dt = empirical_dt_dx2(dx)
                fit_ratio = dt / fit_dt
                fit_str = f"{fit_dt:>9.1f}"
                fit_ratio_str = f"{fit_ratio:>10.2f}×"
            print(
                f"{grid_type:>14}  {prefix+str(N):>5}  {dx:>10.0f}  "
                f"{dt_str:>10}  {cfl:>9.1f}  {cfl_ratio_str:>10}  "
                f"{fit_str:>9}  {fit_ratio_str:>10}"
            )
    print()
    print("Empirical lineage (full table CRM_implementation.md iter-12..30):")
    print("  CFL = 0.8 * dx_min / 300 m/s (gravity-wave bound).")
    print("  dx² fit = (600 s / (240753 m)²) * dx² (iter-29 anchor: C24).")
    print("  Ladder values from iter-13..26 30-day stability sweeps;")
    print("  N>96 REFUSES (iter-21 hard refusal — see auto_dt_rce docstring).")


if __name__ == "__main__":
    main()
