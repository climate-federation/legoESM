"""Global map visualization for legoESM using Cartopy.

Provides plotting functions for fields on the cubed-sphere grid,
including global Mollweide projections, orthographic views, and
conservation time series.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from legoesm.core.field import Field
from legoesm.grids.cubed_sphere import CubedSphereGrid


def plot_global_field(
    field: Field,
    grid: CubedSphereGrid,
    title: str = "",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    projection: str = "mollweide",
    figsize: tuple[float, float] = (12, 6),
    colorbar_label: str = "",
    save_path: Optional[str] = None,
):
    """Plot a scalar field on the cubed-sphere as a global map.

    Parameters
    ----------
    field : Field
        Scalar field with shape (6, n, n).
    grid : CubedSphereGrid
        The cubed-sphere grid.
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Color scale limits.
    projection : str
        Map projection: "mollweide", "orthographic", "robinson", "platecarree".
    figsize : tuple
        Figure size.
    colorbar_label : str
        Label for the colorbar (defaults to field.units).
    save_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    fig, ax : matplotlib figure and axes.
    """
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
    except ImportError:
        raise ImportError("matplotlib and cartopy are required for visualization. "
                          "Install with: pip install matplotlib cartopy")

    # Select projection
    proj_map = {
        "mollweide": ccrs.Mollweide(),
        "orthographic": ccrs.Orthographic(central_longitude=0, central_latitude=30),
        "robinson": ccrs.Robinson(),
        "platecarree": ccrs.PlateCarree(),
    }
    proj = proj_map.get(projection, ccrs.Mollweide())

    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={"projection": proj})

    # Convert to numpy
    lon_np = np.asarray(grid.lon) * 180.0 / np.pi  # Convert to degrees
    lat_np = np.asarray(grid.lat) * 180.0 / np.pi
    field_new = field.data if hasattr(field, 'data') else field
    data_np = np.asarray(field_new)

    # Plot each face as a scatter plot
    if vmin is None:
        vmin = float(np.nanmin(data_np))
    if vmax is None:
        vmax = float(np.nanmax(data_np))

    for face in range(6):
        sc = ax.scatter(
            lon_np[face].ravel(),
            lat_np[face].ravel(),
            c=data_np[face].ravel(),
            s=max(1, 200 / grid.n),  # Point size scales with resolution
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            edgecolors="none",
            alpha=0.8,
        )

    ax.set_global()
    ax.coastlines(linewidth=0.5, color="gray")
    ax.gridlines(linewidth=0.3, color="gray", alpha=0.5)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label(colorbar_label or getattr(field, "units", ""))

    if title:
        ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_conservation_timeseries(
    diagnostics: list[dict],
    dt: float,
    title: str = "Conservation Diagnostics",
    figsize: tuple[float, float] = (12, 8),
    save_path: Optional[str] = None,
):
    """Plot conservation diagnostics over time.

    Shows relative changes in mass and energy vs. initial values.

    Parameters
    ----------
    diagnostics : list of dict
        List of diagnostics dicts from compute_conservation_diagnostics().
    dt : float
        Time step [seconds] between diagnostics entries.
    title : str
        Plot title.
    save_path : str, optional
        If provided, save figure to this path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization.")

    n = len(diagnostics)
    times = np.arange(n) * dt / 86400.0  # Convert to days

    mass = np.array([d['total_mass'] for d in diagnostics])
    energy = np.array([d['total_energy'] for d in diagnostics])

    # Relative changes
    mass_rel = (mass - mass[0]) / mass[0]
    energy_rel = (energy - energy[0]) / energy[0]

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Mass conservation
    axes[0].plot(times, mass_rel, 'b-', linewidth=1.5)
    axes[0].set_ylabel("Relative mass change")
    axes[0].set_title("Mass Conservation")
    axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[0].ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
    axes[0].grid(True, alpha=0.3)

    # Energy conservation
    axes[1].plot(times, energy_rel, 'r-', linewidth=1.5)
    axes[1].set_ylabel("Relative energy change")
    axes[1].set_xlabel("Time [days]")
    axes[1].set_title("Energy Conservation")
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[1].ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, axes


def plot_dgrid_winds_per_tile(
    u_d,
    v_d,
    title: str = "Native D-grid winds",
    cmap: str = "RdBu_r",
    figsize: tuple[float, float] = (16, 18),
    save_path: Optional[str] = None,
):
    """Plot native D-grid wind components on each cubed-sphere tile (issue #274).

    Two D-grid layouts are accepted:

    - **Corner-staggered** (``CDGridShallowWaterState``):
      ``u_d.shape == v_d.shape == (6, n+1, n+1)``.  Both components are
      sampled at cell corners.
    - **FV3 edge-staggered**: ``u_d.shape == (6, n, n+1)`` (south/north
      cell edges) and ``v_d.shape == (6, n+1, n)`` (west/east cell
      edges).  ``imshow`` of the raw arrays plots the staggered sample
      grid faithfully — cell vs. edge counts differ between the two
      panels, which is the whole point of inspecting the *native*
      grid for tile-edge artifacts.

    No global PlateCarree projection is drawn because the D-grid edge
    samples do not have a canonical (lon, lat); use
    ``dgrid_to_center_geographic`` first if a global geographic plot is
    needed.

    Parameters
    ----------
    u_d, v_d : array-like
        D-grid wind components.  See accepted shapes above.  Either
        ``jax.Array``, ``np.ndarray``, or a ``Field`` carrying ``.data``.
    title : str
        Overall figure title.
    cmap : str
        Matplotlib colormap (default diverging so sign reads at a glance).
    figsize : tuple
        Figure size (default fits a 6×2 grid of panels).
    save_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    fig, axes : matplotlib figure and (6, 2) axes array.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization.")

    u_arr = np.asarray(u_d.data if hasattr(u_d, "data") else u_d)
    v_arr = np.asarray(v_d.data if hasattr(v_d, "data") else v_d)
    if u_arr.ndim != 3 or v_arr.ndim != 3:
        raise ValueError(
            f"Expected 3-D arrays (6, ny, nx); got u_d.ndim={u_arr.ndim}, "
            f"v_d.ndim={v_arr.ndim}"
        )
    if u_arr.shape[0] != 6 or v_arr.shape[0] != 6:
        raise ValueError(
            f"Expected leading axis size 6 (cubed-sphere faces); got "
            f"u_d.shape={u_arr.shape}, v_d.shape={v_arr.shape}"
        )

    # Validate staggering: either same-shape corner-D or FV3 edge-stag.
    same_shape = u_arr.shape == v_arr.shape
    if not same_shape:
        u_ny, u_nx = u_arr.shape[1], u_arr.shape[2]
        v_ny, v_nx = v_arr.shape[1], v_arr.shape[2]
        # FV3 layout: u_d on south/north edges → (n, n+1); v_d on
        # west/east edges → (n+1, n).  Both share the same ``n``.
        if not (u_nx == v_ny and u_ny + 1 == u_nx and v_nx + 1 == v_ny):
            raise ValueError(
                f"u_d/v_d shape mismatch: {u_arr.shape} vs {v_arr.shape}. "
                f"Expected matching corner-D shapes (6, n+1, n+1) or FV3 "
                f"edge stagger (6, n, n+1) / (6, n+1, n)."
            )

    face_names = ["+X (Front)", "+Y (Right)", "-X (Back)",
                  "-Y (Left)", "+Z (Top)", "-Z (Bottom)"]

    # Symmetric colour limits per component so sign reads consistently.
    u_lim = float(np.nanmax(np.abs(u_arr)))
    v_lim = float(np.nanmax(np.abs(v_arr)))
    u_lim = u_lim if u_lim > 0 else 1.0
    v_lim = v_lim if v_lim > 0 else 1.0

    fig, axes = plt.subplots(6, 2, figsize=figsize)
    for face in range(6):
        for col, (arr, lim, label) in enumerate(
            ((u_arr, u_lim, "u_d"), (v_arr, v_lim, "v_d"))
        ):
            ax = axes[face, col]
            im = ax.imshow(
                arr[face].T,
                origin="lower",
                cmap=cmap,
                vmin=-lim,
                vmax=lim,
            )
            ax.set_title(f"Face {face} ({face_names[face]}) — {label}")
            ax.set_xlabel("i")
            ax.set_ylabel("j")
            plt.colorbar(im, ax=ax, shrink=0.85, label="m/s")

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, axes
