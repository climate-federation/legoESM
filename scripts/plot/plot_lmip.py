#!/usr/bin/env python
"""Visualise output from scripts/run/run_lmip.py.

Reads ``land_spinup.nc`` (or npz chunks via ``land_spinup_index.json``) from
one or more LMIP output directories and produces a multi-panel diagnostic
figure.

Usage::

    # Single run
    python scripts/plot/plot_lmip.py /tmp/lmip_1yr

    # Compare two runs side by side
    python scripts/plot/plot_lmip.py /tmp/lmip_loam /tmp/lmip_clay --labels loam clay

    # Save to file (default: show interactively)
    python scripts/plot/plot_lmip.py /tmp/lmip_1yr --out lmip_diag.png

Panels produced
---------------
Row 1 – Surface energy (sensible + latent heat flux, snow depth)
Row 2 – Soil temperature depth–time heatmap (first run)
Row 3 – Soil moisture depth–time heatmap (first run)
Row 4 – Surface-layer T_soil and theta_soil time series (all runs overlaid)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from legoesm import constants


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_nc(nc_path: Path) -> dict:
    """Load from a netCDF4 file.  Returns dict of numpy arrays."""
    try:
        import netCDF4 as nc  # type: ignore
    except ImportError:
        raise ImportError("netCDF4 not installed; use npz fallback path")

    with nc.Dataset(str(nc_path), "r") as ds:
        time = np.array(ds.variables["time"][:])
        T_soil = np.array(ds.variables["T_soil"][:])          # (T, L)
        theta_soil = np.array(ds.variables["theta_soil"][:])  # (T, L)
        psi_soil = np.array(ds.variables["psi_soil"][:])      # (T, L)
        snow_depth = np.array(ds.variables["snow_depth"][:])  # (T,)
        shflx = np.array(ds.variables["shflx"][:])            # (T,)
        lhflx = np.array(ds.variables["lhflx"][:])            # (T,)
        runoff_surface = np.array(ds.variables["runoff_surface"][:])
        runoff_subsurface = np.array(ds.variables["runoff_subsurface"][:])
        attrs = {k: getattr(ds, k) for k in ds.ncattrs()}

    return dict(
        time=time,
        T_soil=T_soil,
        theta_soil=theta_soil,
        psi_soil=psi_soil,
        snow_depth=snow_depth,
        shflx=shflx,
        lhflx=lhflx,
        runoff_surface=runoff_surface,
        runoff_subsurface=runoff_subsurface,
        attrs=attrs,
    )


def _load_npz_chunks(index_path: Path) -> dict:
    """Load npz chunks described by land_spinup_index.json."""
    with open(index_path) as f:
        index = json.load(f)

    chunks = sorted(index["chunks"])
    parts = [np.load(str(index_path.parent / c)) for c in chunks]

    time = np.concatenate([p["time"] for p in parts])
    T_soil = np.concatenate([p["T_soil"] for p in parts], axis=0)
    theta_soil = np.concatenate([p["theta_soil"] for p in parts], axis=0)
    psi_soil = np.concatenate([p["psi_soil"] for p in parts], axis=0)
    snow_depth = np.concatenate([p["snow_depth"] for p in parts])
    shflx = np.concatenate([p["shflx"] for p in parts])
    lhflx = np.concatenate([p["lhflx"] for p in parts])
    runoff_surface = np.concatenate([p["runoff_surface"] for p in parts])
    runoff_subsurface = np.concatenate([p["runoff_subsurface"] for p in parts])

    return dict(
        time=time,
        T_soil=T_soil,
        theta_soil=theta_soil,
        psi_soil=psi_soil,
        snow_depth=snow_depth,
        shflx=shflx,
        lhflx=lhflx,
        runoff_surface=runoff_surface,
        runoff_subsurface=runoff_subsurface,
        attrs=index.get("attrs", {}),
    )


def load_run(run_dir: str | Path) -> dict:
    """Load LMIP output from *run_dir*.

    Tries ``land_spinup.nc`` first, then npz chunks.
    """
    run_dir = Path(run_dir)
    nc_path = run_dir / "land_spinup.nc"
    idx_path = run_dir / "land_spinup_index.json"

    if nc_path.exists():
        try:
            return _load_nc(nc_path)
        except ImportError:
            pass  # fall through to npz

    if idx_path.exists():
        return _load_npz_chunks(idx_path)

    raise FileNotFoundError(
        f"No land_spinup.nc or land_spinup_index.json found in {run_dir}"
    )


# ---------------------------------------------------------------------------
# Soil depth axis helper
# ---------------------------------------------------------------------------

def _soil_layer_depths(n_layers: int, total_depth: float = 3.0,
                        growth_factor: float = 1.5) -> np.ndarray:
    """Return mid-point depths [m] for each soil layer (geometric grid)."""
    r = growth_factor
    n = n_layers
    # dz_top chosen so total depth = total_depth
    dz_top = total_depth * (r - 1.0) / (r**n - 1.0)
    dz = dz_top * r ** np.arange(n)
    # cumulative interface depths
    interfaces = np.concatenate([[0.0], np.cumsum(dz)])
    midpoints = 0.5 * (interfaces[:-1] + interfaces[1:])
    return midpoints


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _celsius(T_K: np.ndarray) -> np.ndarray:
    return T_K - constants.T_freeze


def plot_lmip(
    datasets: list[dict],
    labels: list[str],
    title: str = "LMIP land spin-up diagnostics",
) -> plt.Figure:
    """Create diagnostic figure for one or more LMIP runs.

    Parameters
    ----------
    datasets : list of data dicts from load_run()
    labels   : display label for each run
    title    : figure suptitle
    """
    colors = ["C0", "C1", "C2", "C3"]

    # Use first dataset for heatmaps; layer depths from its attrs
    d0 = datasets[0]
    n_layers = d0["T_soil"].shape[1]
    total_depth = float(d0["attrs"].get("soil_depth_m", 3.0))
    depths = _soil_layer_depths(n_layers, total_depth)  # (n_layers,)
    time0 = d0["time"]  # days

    fig, axes = plt.subplots(
        4, 2, figsize=(14, 16),
        gridspec_kw={"height_ratios": [1.2, 1.5, 1.5, 1.2]},
    )
    fig.suptitle(title, fontsize=13, y=0.995)

    # ------------------------------------------------------------------
    # Row 0, col 0: Sensible + latent heat fluxes
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    for i, (d, lbl) in enumerate(zip(datasets, labels)):
        c = colors[i % len(colors)]
        ax.plot(d["time"] / 365.25, d["shflx"], color=c, alpha=0.7,
                lw=0.8, label=f"{lbl} SH")
        ax.plot(d["time"] / 365.25, d["lhflx"], color=c, alpha=0.4,
                lw=0.8, linestyle="--", label=f"{lbl} LH")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_ylabel("Flux [W m⁻²]")
    ax.set_title("Sensible (solid) & latent (dashed) heat flux")
    ax.legend(fontsize=7, ncol=2)

    # ------------------------------------------------------------------
    # Row 0, col 1: Snow depth
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    for i, (d, lbl) in enumerate(zip(datasets, labels)):
        ax.plot(d["time"] / 365.25, d["snow_depth"],
                color=colors[i % len(colors)], lw=0.9, label=lbl)
    ax.set_ylabel("SWE [kg m⁻²]")
    ax.set_title("Snow water equivalent")
    if len(datasets) > 1:
        ax.legend(fontsize=8)

    # ------------------------------------------------------------------
    # Row 1, col 0: T_soil heatmap (first run)
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    T_C = _celsius(d0["T_soil"])  # (time, layer)
    t_years = time0 / 365.25
    vmin, vmax = np.nanpercentile(T_C, [2, 98])
    im = ax.pcolormesh(
        t_years, depths, T_C.T,
        cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="nearest",
    )
    ax.set_ylabel("Depth [m]")
    ax.set_ylim([depths[-1], 0])  # surface at top
    ax.set_title(f"Soil temperature [°C] — {labels[0]}")
    plt.colorbar(im, ax=ax, label="°C", pad=0.02)

    # ------------------------------------------------------------------
    # Row 1, col 1: theta_soil heatmap (first run)
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    th = d0["theta_soil"]
    vmin_th = max(0, np.nanpercentile(th, 1))
    vmax_th = np.nanpercentile(th, 99)
    im2 = ax.pcolormesh(
        t_years, depths, th.T,
        cmap="YlGnBu", vmin=vmin_th, vmax=vmax_th, shading="nearest",
    )
    ax.set_ylabel("Depth [m]")
    ax.set_ylim([depths[-1], 0])
    ax.set_title(f"Volumetric water content [m³ m⁻³] — {labels[0]}")
    plt.colorbar(im2, ax=ax, label="m³ m⁻³", pad=0.02)

    # ------------------------------------------------------------------
    # Row 2, col 0: log10(|psi_soil|) heatmap (first run)
    # ------------------------------------------------------------------
    ax = axes[2, 0]
    psi = d0["psi_soil"]
    # log10(|psi|): matric potential spans many orders of magnitude
    log_psi = np.log10(np.abs(psi) + 1e-3)
    im3 = ax.pcolormesh(
        t_years, depths, log_psi.T,
        cmap="plasma", shading="nearest",
    )
    ax.set_ylabel("Depth [m]")
    ax.set_ylim([depths[-1], 0])
    ax.set_title(f"log₁₀|ψ| [log₁₀(m)] — {labels[0]}")
    plt.colorbar(im3, ax=ax, label="log₁₀(m)", pad=0.02)

    # ------------------------------------------------------------------
    # Row 2, col 1: Surface + deep layer T_soil time series (all runs)
    # ------------------------------------------------------------------
    ax = axes[2, 1]
    for i, (d, lbl) in enumerate(zip(datasets, labels)):
        c = colors[i % len(colors)]
        # surface layer (index 0) and deepest layer
        ax.plot(d["time"] / 365.25, _celsius(d["T_soil"][:, 0]),
                color=c, lw=0.8, label=f"{lbl} surf")
        ax.plot(d["time"] / 365.25, _celsius(d["T_soil"][:, -1]),
                color=c, lw=0.8, ls="--", alpha=0.6, label=f"{lbl} deep")
    ax.set_ylabel("T [°C]")
    ax.set_title("Surface (solid) & deep (dashed) soil temperature")
    ax.legend(fontsize=7, ncol=2)

    # ------------------------------------------------------------------
    # Row 3, col 0: Surface theta_soil (all runs)
    # ------------------------------------------------------------------
    ax = axes[3, 0]
    for i, (d, lbl) in enumerate(zip(datasets, labels)):
        ax.plot(d["time"] / 365.25, d["theta_soil"][:, 0],
                color=colors[i % len(colors)], lw=0.8, label=lbl)
    ax.set_ylabel("θ [m³ m⁻³]")
    ax.set_xlabel("Year")
    ax.set_title("Surface-layer soil moisture")
    if len(datasets) > 1:
        ax.legend(fontsize=8)

    # ------------------------------------------------------------------
    # Row 3, col 1: Runoff (all runs)
    # ------------------------------------------------------------------
    ax = axes[3, 1]
    for i, (d, lbl) in enumerate(zip(datasets, labels)):
        c = colors[i % len(colors)]
        ax.plot(d["time"] / 365.25, d["runoff_surface"] * 86400,
                color=c, lw=0.8, label=f"{lbl} sfc")
        ax.plot(d["time"] / 365.25, d["runoff_subsurface"] * 86400,
                color=c, lw=0.8, ls="--", alpha=0.6, label=f"{lbl} sub")
    ax.set_ylabel("Runoff [kg m⁻² d⁻¹]")
    ax.set_xlabel("Year")
    ax.set_title("Surface (solid) & subsurface (dashed) runoff")
    if len(datasets) > 1:
        ax.legend(fontsize=7, ncol=2)

    # Common x-label for rows 0–2
    for ax_row in axes[:3, :].flat:
        ax_row.set_xlabel("Year")

    fig.tight_layout(rect=[0, 0, 1, 0.993])
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise run_lmip.py output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("run_dirs", nargs="+",
                   help="One or more LMIP output directories")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Display label for each run directory (default: dir name)")
    p.add_argument("--out", default=None,
                   help="Output PNG path. If omitted, displays interactively.")
    p.add_argument("--title", default="LMIP land spin-up diagnostics",
                   help="Figure suptitle")
    p.add_argument("--dpi", type=int, default=150,
                   help="Output DPI")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    labels = args.labels
    if labels is None:
        labels = [Path(d).name for d in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        print("ERROR: --labels count must match number of run directories",
              file=sys.stderr)
        sys.exit(1)

    datasets = []
    for d, lbl in zip(args.run_dirs, labels):
        print(f"Loading {lbl} from {d} …")
        try:
            datasets.append(load_run(d))
        except FileNotFoundError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"  {datasets[-1]['T_soil'].shape[0]} days loaded, "
              f"{datasets[-1]['T_soil'].shape[1]} layers")

    fig = plot_lmip(datasets, labels, title=args.title)

    if args.out is not None:
        out = Path(args.out)
        fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {out}")
    else:
        matplotlib.use("TkAgg")
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
