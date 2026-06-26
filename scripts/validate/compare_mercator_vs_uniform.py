#!/usr/bin/env python
"""Side-by-side comparison: uniform lat-lon vs Mercator ocean evolution.

For each of three test cases (``rest_state``, ``barotropic_wave``,
``global_barotropic_wind``) we run *identical* physical setups on:

* a uniform global lat-lon grid (36 × 72 cells, 5° resolution); and
* a Mercator grid with the same ``n_lon = 72`` (so equal zonal spacing
  at the equator), ``lat_max_deg = 80``.

Both grids use the same ``LatLonCGridOceanModel`` and the same time
step. The grids differ in cell shape: uniform is anisotropic (dy/dx
varies as 1/cos φ), Mercator is isotropic by construction. They cover
slightly different latitude bands (uniform: ±90°, Mercator: ±80°).

We track integrated diagnostics over time:

* ``mean_eta`` — global mean SSH (drift indicator)
* ``max_|eta|`` — peak SSH (wave amplitude / forcing response)
* ``max_|u|``, ``max_|v|`` — peak face velocities
* ``total_mass`` — ``sum(eta · cell_area)`` (mass conservation)

After each case the script prints a comparison table and dumps the
timeseries to ``results/mercator_vs_uniform/<case>.png``.

Usage::

    PYTHONPATH=src JAX_ENABLE_X64=1 python scripts/compare_mercator_vs_uniform.py
    # or just one case
    PYTHONPATH=src JAX_ENABLE_X64=1 python scripts/compare_mercator_vs_uniform.py --only rest_state
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from legoesm import constants
from legoesm.core.field import Field
from legoesm.grids.latlon import (
    create_latlon_grid,
    create_mercator_grid,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class Diagnostics:
    t_days: list[float] = field(default_factory=list)
    mean_eta: list[float] = field(default_factory=list)
    max_abs_eta: list[float] = field(default_factory=list)
    max_abs_u: list[float] = field(default_factory=list)
    max_abs_v: list[float] = field(default_factory=list)
    total_mass: list[float] = field(default_factory=list)


def _diagnose(state, grid, t_days: float, diag: Diagnostics) -> None:
    eta = np.asarray(state.eta.data, dtype=np.float64)
    u = np.asarray(state.u.data, dtype=np.float64)
    v = np.asarray(state.v.data, dtype=np.float64)
    area = np.asarray(grid.area, dtype=np.float64)
    total_area = float(np.sum(area))

    diag.t_days.append(t_days)
    diag.mean_eta.append(float(np.sum(eta * area) / total_area))
    diag.max_abs_eta.append(float(np.max(np.abs(eta))))
    diag.max_abs_u.append(float(np.max(np.abs(u))))
    diag.max_abs_v.append(float(np.max(np.abs(v))))
    diag.total_mass.append(float(np.sum(eta * area)))


# ---------------------------------------------------------------------------
# Common setup helpers
# ---------------------------------------------------------------------------

DEFAULT_DT = 300.0           # seconds
DEFAULT_H_MAX = 4000.0       # m
DEFAULT_NLEV = 4             # cheap stratification for the comparison


def _build_grid(kind: str):
    """Return (label, grid) for the named comparison grid."""
    if kind == "uniform":
        # 5° resolution, 36 × 72.
        return "uniform 36x72", create_latlon_grid(n_lat=36, n_lon=72)
    if kind == "mercator":
        # n_lon=72 → Δλ = 5°. lat_max_deg=80 → n_lat ≈ 56.
        return ("mercator 72/lat_max=80", create_mercator_grid(
            n_lon=72, lat_max_deg=80.0))
    raise ValueError(f"Unknown grid kind: {kind}")


def _build_model(grid, *, A_h=None, bottom_drag_r=None, physics=None):
    kw = dict(n_barotropic_substeps=30)
    if A_h is not None:
        kw["A_h"] = A_h
    if bottom_drag_r is not None:
        kw["bottom_drag_r"] = bottom_drag_r
    if physics is not None:
        kw["physics"] = physics
    config = LatLonCGridOceanConfig.from_flat(**kw)
    return LatLonCGridOceanModel(grid, _z_coord(), config)


def _z_coord():
    """Cheap z-star vertical coordinate used by every comparison run."""
    return create_ocean_z_star(n_levels=DEFAULT_NLEV, H_max=DEFAULT_H_MAX)


# ---------------------------------------------------------------------------
# Time loop
# ---------------------------------------------------------------------------

def _run(grid, model, state, *, n_steps: int, dt: float, label: str,
         diag_every: int) -> Diagnostics:
    diag = Diagnostics()
    _diagnose(state, grid, 0.0, diag)

    @jax.jit
    def _step(s, dt_):
        return model.step(s, dt_)

    t0 = time.time()
    for step in range(1, n_steps + 1):
        state = _step(state, dt)
        if step % diag_every == 0 or step == n_steps:
            _diagnose(state, grid, step * dt / 86400.0, diag)
    wall = time.time() - t0
    print(f"    {label}: {n_steps} steps in {wall:.1f}s "
          f"({wall / max(n_steps, 1) * 1000:.1f} ms/step)")
    return diag


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def _make_gaussian_ssh(grid):
    """A 1 m SSH bump at (180°E, 0°N), σ = 10° great-circle distance."""
    lat_1d = np.asarray(grid.lat, dtype=np.float64)
    lon_1d = np.asarray(grid.lon, dtype=np.float64)
    lon0 = np.pi
    lat0 = 0.0
    sigma_rad = 10.0 * np.pi / 180.0

    lat_2d, lon_2d = np.meshgrid(lat_1d, lon_1d, indexing="ij")
    dlon = lon_2d - lon0
    dist = np.arccos(np.clip(
        np.sin(lat_2d) * np.sin(lat0)
        + np.cos(lat_2d) * np.cos(lat0) * np.cos(dlon),
        -1.0, 1.0,
    ))
    perturb = np.exp(-0.5 * (dist / sigma_rad) ** 2)
    return perturb


def _simplified_continent_mask(grid):
    """A north-south continent at 30-90°E from +80° to -55° lat, with
    open Drake Passage south of -55°. Same as run_ocean_test_matrix."""
    lat_1d = np.asarray(grid.lat, dtype=np.float64) * 180.0 / np.pi
    lon_1d = np.asarray(grid.lon, dtype=np.float64) * 180.0 / np.pi
    lat_2d, lon_2d = np.meshgrid(lat_1d, lon_1d, indexing="ij")
    mask = np.ones_like(lat_2d, dtype=np.float64)
    continent = (lon_2d >= 30.0) & (lon_2d <= 90.0) & (lat_2d >= -55.0)
    mask[continent] = 0.0
    polar = np.abs(lat_2d) >= 80.0
    mask[polar] = 0.0
    return mask


def case_rest_state(kind: str, days: float = 0.5) -> Diagnostics:
    label, grid = _build_grid(kind)
    print(f"  Building {label}:  n_lat={grid.n_lat}  n_lon={grid.n_lon}")
    model = _build_model(grid)
    state = rest_state_latlon_cgrid_ocean(
        grid, _z_coord(), H_max=DEFAULT_H_MAX, land_lat_threshold=90.0,
    )
    n_steps = int(days * 86400 / DEFAULT_DT)
    return _run(grid, model, state, n_steps=n_steps, dt=DEFAULT_DT,
                label=label, diag_every=max(1, n_steps // 20))


def case_barotropic_wave(kind: str, days: float = 2.0) -> Diagnostics:
    label, grid = _build_grid(kind)
    print(f"  Building {label}:  n_lat={grid.n_lat}  n_lon={grid.n_lon}")
    model = _build_model(grid)
    state = rest_state_latlon_cgrid_ocean(
        grid, _z_coord(), H_max=DEFAULT_H_MAX, land_lat_threshold=90.0,
    )
    # Add the 1 m equatorial Gaussian SSH bump.
    perturb = _make_gaussian_ssh(grid)
    new_eta = state.eta.data + jnp.asarray(perturb, dtype=state.eta.data.dtype)
    state = state._replace(eta=Field(data=new_eta))

    n_steps = int(days * 86400 / DEFAULT_DT)
    return _run(grid, model, state, n_steps=n_steps, dt=DEFAULT_DT,
                label=label, diag_every=max(1, n_steps // 30))


def case_global_wind(kind: str, days: float = 5.0) -> Diagnostics:
    """Wind-driven global circulation, 3-belt zonal stress."""
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.surface_forcing.config import (
        PrescribedForcingConfig, SurfaceForcingConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig

    physics = OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(
            scheme="prescribed",
            prescribed=PrescribedForcingConfig(
                wind_profile="global_wind",
                tau_max=0.1,
            ),
        ),
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )

    label, grid = _build_grid(kind)
    print(f"  Building {label}:  n_lat={grid.n_lat}  n_lon={grid.n_lon}")
    model = _build_model(
        grid, A_h=5e5, bottom_drag_r=1e-4, physics=physics,
    )

    # Uniform-T baroclinic-free initial state with simplified continent.
    state = rest_state_latlon_cgrid_ocean(
        grid, _z_coord(),
        H_max=DEFAULT_H_MAX,
        T_water_init_C=10.0, T_deep=10.0, S_uniform=35.0,
        land_lat_threshold=90.0,
    )
    mask = _simplified_continent_mask(grid)
    mask_typed = jnp.asarray(mask, dtype=state.eta.data.dtype)
    u_mask_new, v_mask_new = compute_face_masks(mask_typed)
    state = state._replace(
        land_mask=Field(data=mask_typed),
        u_mask=Field(data=u_mask_new),
        v_mask=Field(data=v_mask_new),
    )

    n_steps = int(days * 86400 / DEFAULT_DT)
    return _run(grid, model, state, n_steps=n_steps, dt=DEFAULT_DT,
                label=label, diag_every=max(1, n_steps // 30))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _plot_pair(case_name: str, uni: Diagnostics, merc: Diagnostics,
               out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("mean_eta", "mean SSH [m]"),
        ("max_abs_eta", "max |SSH| [m]"),
        ("max_abs_u", "max |u| [m/s]"),
        ("max_abs_v", "max |v| [m/s]"),
        ("total_mass", "∫η dA [m³]"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 12), sharex=True)
    for ax, (key, ylab) in zip(axes, metrics):
        ax.plot(uni.t_days, getattr(uni, key), "o-", label="uniform", lw=1.5)
        ax.plot(merc.t_days, getattr(merc, key), "s--", label="mercator", lw=1.5)
        ax.set_ylabel(ylab)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [days]")
    fig.suptitle(f"Mercator vs uniform: {case_name}")
    path = out_dir / f"{case_name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _diff_table(case_name: str, uni: Diagnostics, merc: Diagnostics) -> str:
    """Print a final-time comparison table."""
    lines = [f"  === {case_name} comparison @ final time ==="]
    for key in ("mean_eta", "max_abs_eta", "max_abs_u",
                "max_abs_v", "total_mass"):
        u = getattr(uni, key)[-1]
        m = getattr(merc, key)[-1]
        if abs(u) > 1e-15:
            relerr = (m - u) / u
            lines.append(
                f"    {key:14s}  uniform={u: .4e}  mercator={m: .4e}  "
                f"Δrel={relerr: .3e}"
            )
        else:
            lines.append(
                f"    {key:14s}  uniform={u: .4e}  mercator={m: .4e}  "
                f"(absolute Δ={m - u: .3e})"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

CASES = {
    "rest_state": (case_rest_state, 0.5),
    "barotropic_wave": (case_barotropic_wave, 2.0),
    "global_wind": (case_global_wind, 5.0),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None,
                        help="Run only this case (one of: "
                             f"{', '.join(CASES)}). Default: all.")
    parser.add_argument("--days", type=float, default=None,
                        help="Override duration for the selected case.")
    parser.add_argument("--output", default="results/mercator_vs_uniform",
                        help="Output directory for plots.")
    args = parser.parse_args()

    out_dir = Path(args.output)
    cases = list(CASES.items()) if args.only is None else [
        (args.only, CASES[args.only])]

    summary: list[str] = []
    for name, (fn, default_days) in cases:
        days = args.days if args.days is not None else default_days
        print(f"\n--- {name}  (days = {days})  ---")
        uni = fn("uniform", days=days)
        merc = fn("mercator", days=days)
        png = _plot_pair(name, uni, merc, out_dir)
        diff = _diff_table(name, uni, merc)
        print(diff)
        print(f"    plot: {png}")
        summary.append(diff)

    print("\n=== summary ===")
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
