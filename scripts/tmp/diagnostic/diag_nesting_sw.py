#!/usr/bin/env python
"""Visual diagnostic for the one-way lat-lon shallow-water NEST.

Runs the C-grid shallow-water solver on three configurations of the SAME
Williamson test case and compares them over the child (nested) footprint:

  (a) uniform-COARSE  : a global lat-lon grid at the PARENT resolution.
  (b) uniform-FINE    : a global lat-lon grid at the CHILD resolution
                        (the gold-standard reference for the nest).
  (c) coarse-parent + fine-NEST : the one-way nest — coarse global parent with a
                        refined child whose lateral boundary is prescribed from
                        the parent each step.

For each of Williamson-2 (steady geostrophic; v-wind should stay ~0 — edge
artifacts show up as v-wind noise at the nest boundary, exactly the W2 v-wind
check in CLAUDE.md) and Williamson-5 (flow over a mountain; wind_speed develops
structure — the W5 wind_speed check), it saves PNGs of:

  * the field on the coarse, fine, and nested runs (over the child window), and
  * the NEST minus UNIFORM-FINE difference (the artifact map).

Human + codex visual sign-off looks for nest-boundary reflection / grid-scale
noise in the difference panels (a clean one-way nest shows a smooth interior with
the imprint confined to the prescribed boundary band).

Run:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
        .venv/bin/python scripts/tmp/diagnostic/diag_nesting_sw.py
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pathlib

import numpy as np
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legoesm import constants
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.nesting import create_nested_latlon_grid
from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
    CGridLatLonShallowWaterModel,
    CGridLatLonShallowWaterConfig,
    williamson_test2_cgrid,
    williamson_test5_cgrid,
)
from legoesm.atmosphere.dynamics.gcm.shallow_water_nesting import (
    initial_nested_state,
    step_child,
    interior_mass,
)

# ---- nest geometry: a mid-latitude box, refinement 3 ----
# The box is placed over the Williamson-5 mountain (centred at lon 270 deg, lat
# 30 deg) so the nest actually resolves the orographic response in the W5 case
# (otherwise the child sees flat topography and the W5 diagnostic is trivial).
PARENT_N_LAT = 48
REFINE = 3
LAT_S, LAT_N = 5.0, 55.0
LON_W, LON_E = 230.0, 310.0
N_HALO = 2
N_RELAX = 8  # Davies relaxation-zone width (cells) — absorbs boundary reflection
DT = 90.0
NSTEPS = 240  # 6 hours

# Polar filter stabilises the global lat-lon SW solver near the converging
# meridians (Williamson-5's orographic gravity waves go unstable at the poles
# without it — a property of the GLOBAL grid, not the nest).
SW_CONFIG = CGridLatLonShallowWaterConfig(
    fix_mass=True, use_ppm_transport=True, use_polar_filter=True,
)


def _cell_winds(state):
    """C-grid face winds -> cell-centre (u, v)."""
    u_c = 0.5 * (state.u[:, :-1] + state.u[:, 1:])
    v_c = 0.5 * (state.v[:-1, :] + state.v[1:, :])
    return np.asarray(u_c), np.asarray(v_c)


def _run_global(grid, ic_fn, label):
    model = CGridLatLonShallowWaterModel(grid, SW_CONFIG, dt=DT)
    state = ic_fn(grid)
    target = model.compute_mass(state)
    for _ in range(NSTEPS):
        state = model.step(state, DT, target_mass=target)
    print(f"  [{label}] done ({grid.n_lat}x{grid.n_lon})")
    return state


def _run_nest(nest, ic_fn, label):
    # One-way nest: integrate the global PARENT with the full standalone model
    # (polar filter included), then force the CHILD boundary from it each step.
    parent_model = CGridLatLonShallowWaterModel(nest.parent, SW_CONFIG, dt=DT)
    parent_ic = ic_fn(nest.parent)
    parent_target = parent_model.compute_mass(parent_ic)
    nstate = initial_nested_state(nest, parent_ic)
    parent = nstate.parent
    child = nstate.child
    target = interior_mass(child, nest)
    m0 = float(target)
    for _ in range(NSTEPS):
        # child forced from the parent at the START of the step (one-way).
        child = step_child(child, parent, nest, DT, target, SW_CONFIG)
        parent = parent_model.step(parent, DT, target_mass=parent_target)
    m1 = float(interior_mass(child, nest))
    drift = abs(m1 - m0) / abs(m0)
    print(f"  [{label}] done; child interior mass rel drift = {drift:.3e}")
    from legoesm.atmosphere.dynamics.gcm.shallow_water_nesting import NestedSWState
    return NestedSWState(parent=parent, child=child), drift


def _child_window(global_state, global_grid, child_grid):
    """Extract the field on the global grid over the child footprint (nearest)."""
    glat = np.asarray(global_grid.lat)
    glon = np.asarray(global_grid.lon)
    clat = np.asarray(child_grid.lat)
    clon = np.asarray(child_grid.lon)
    jj = np.array([np.argmin(np.abs(glat - la)) for la in clat])
    ii = np.array([np.argmin(np.abs(((glon - lo + np.pi) % (2 * np.pi)) - np.pi))
                   for lo in clon])
    u_c, v_c = _cell_winds(global_state)
    spd = np.sqrt(u_c**2 + v_c**2)
    h = np.asarray(global_state.h)
    sel = lambda a: a[np.ix_(jj, ii)]
    return sel(h), sel(u_c), sel(v_c), sel(spd)


def _panel(ax, field, lon, lat, title, cmap, vmin=None, vmax=None, sym=False):
    if sym:
        m = np.nanmax(np.abs(field)) or 1.0
        vmin, vmax = -m, m
    im = ax.pcolormesh(
        np.rad2deg(lon), np.rad2deg(lat), field,
        shading="auto", cmap=cmap, vmin=vmin, vmax=vmax,
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("lon [deg]", fontsize=7)
    ax.set_ylabel("lat [deg]", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _figure(case_name, comp, outdir):
    """comp: dict with coarse/fine/nest field arrays + child lon/lat + diff."""
    clon, clat = comp["clon"], comp["clat"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    _panel(axes[0, 0], comp["coarse"], clon, clat,
           f"{case_name}: {comp['field']} — uniform-COARSE (parent res)",
           comp["cmap"], sym=comp["sym"])
    _panel(axes[0, 1], comp["fine"], clon, clat,
           f"{case_name}: {comp['field']} — uniform-FINE (ref)",
           comp["cmap"], sym=comp["sym"])
    _panel(axes[1, 0], comp["nest"], clon, clat,
           f"{case_name}: {comp['field']} — coarse+NEST",
           comp["cmap"], sym=comp["sym"])
    _panel(axes[1, 1], comp["diff"], clon, clat,
           f"{case_name}: {comp['field']} — NEST minus uniform-FINE",
           "RdBu_r", sym=True)
    fig.tight_layout()
    fname = outdir / f"nesting_{case_name}_{comp['field']}.png"
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    return fname


def main():
    repo = pathlib.Path(__file__).resolve().parents[2]
    outdir = repo / "results" / "diagnostic"
    outdir.mkdir(parents=True, exist_ok=True)

    nest = create_nested_latlon_grid(
        parent_n_lat=PARENT_N_LAT, refinement_ratio=REFINE,
        lat_south_deg=LAT_S, lat_north_deg=LAT_N,
        lon_west_deg=LON_W, lon_east_deg=LON_E,
        n_halo=N_HALO, n_relax=N_RELAX,
    )
    child = nest.child
    coarse = nest.parent
    fine = create_latlon_grid(
        n_lat=PARENT_N_LAT * REFINE, radius=constants.R_earth,
        omega=constants.Omega,
    )
    print(f"parent {coarse.n_lat}x{coarse.n_lon}  child {child.n_lat}x{child.n_lon} "
          f"(r={REFINE})  fine {fine.n_lat}x{fine.n_lon}")

    cases = {
        "W2": (williamson_test2_cgrid, "v_wind"),
        "W5": (williamson_test5_cgrid, "wind_speed"),
    }
    saved = []
    drifts = {}
    for case_name, (ic_fn, field_name) in cases.items():
        print(f"== {case_name} ==")
        gc = _run_global(coarse, ic_fn, f"{case_name} coarse")
        gf = _run_global(fine, ic_fn, f"{case_name} fine")
        ns, drift = _run_nest(nest, ic_fn, f"{case_name} nest")
        drifts[case_name] = drift

        hc, uc, vc, sc = _child_window(gc, coarse, child)
        hf, uf, vf, sf = _child_window(gf, fine, child)
        u_n, v_n = _cell_winds(ns.child)
        s_n = np.sqrt(u_n**2 + v_n**2)

        if field_name == "v_wind":
            coarse_f, fine_f, nest_f = vc, vf, v_n
            cmap, sym = "RdBu_r", True
        else:  # wind_speed
            coarse_f, fine_f, nest_f = sc, sf, s_n
            cmap, sym = "viridis", False

        comp = dict(
            field=field_name, coarse=coarse_f, fine=fine_f, nest=nest_f,
            diff=nest_f - fine_f, clon=child.lon, clat=child.lat,
            cmap=cmap, sym=sym,
        )
        saved.append(_figure(case_name, comp, outdir))

    print("\nSaved diagnostic PNGs:")
    for f in saved:
        print(f"  {f}")
    print("\nChild interior mass rel drift (fix_mass=True):")
    for k, v in drifts.items():
        print(f"  {k}: {v:.3e}")


if __name__ == "__main__":
    main()
