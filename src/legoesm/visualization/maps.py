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
    #data_np = np.asarray(field.data)
    field_new=field.data if hasattr(field, 'data') else field
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
    cbar.set_label(colorbar_label or field.units)

    if title:
        ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_cubed_sphere_field(
    field: Field,
    grid: CubedSphereGrid,
    title: str = "",
    cmap: str = "viridis",
    figsize: tuple[float, float] = (16, 8),
    save_path: Optional[str] = None,
):
    """Plot a field showing all 6 cubed-sphere faces individually.

    Useful for debugging grid structure and inter-face boundaries.

    Parameters
    ----------
    field : Field
        Scalar field with shape (6, n, n).
    grid : CubedSphereGrid
        The cubed-sphere grid.
    title : str
        Overall title.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization.")

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()

    data_np = np.asarray(field.data)
    vmin = float(np.nanmin(data_np))
    vmax = float(np.nanmax(data_np))

    face_names = ["+X (Front)", "+Y (Right)", "-X (Back)",
                  "-Y (Left)", "+Z (Top)", "-Z (Bottom)"]

    for face in range(6):
        im = axes[face].imshow(
            data_np[face].T,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axes[face].set_title(f"Face {face}: {face_names[face]}")
        axes[face].set_xlabel("i")
        axes[face].set_ylabel("j")
        plt.colorbar(im, ax=axes[face], shrink=0.8)

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, axes


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


def plot_wind_field(
    u_field: Field,
    v_field: Field,
    grid: CubedSphereGrid,
    title: str = "Wind Field",
    skip: int = 4,
    figsize: tuple[float, float] = (12, 6),
    save_path: Optional[str] = None,
):
    """Plot wind vectors on a global map.

    Parameters
    ----------
    u_field, v_field : Field
        Wind components (grid-aligned), shape (6, n, n).
    grid : CubedSphereGrid
        The grid.
    title : str
        Plot title.
    skip : int
        Plot every Nth vector for clarity.
    """
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
    except ImportError:
        raise ImportError("matplotlib and cartopy required.")

    fig, ax = plt.subplots(1, 1, figsize=figsize,
                           subplot_kw={"projection": ccrs.PlateCarree()})

    lon_np = np.asarray(grid.lon) * 180.0 / np.pi
    lat_np = np.asarray(grid.lat) * 180.0 / np.pi

    u_np = np.asarray(u_field.data)
    v_np = np.asarray(v_field.data)

    # Speed for coloring
    speed = np.sqrt(u_np**2 + v_np**2)

    for face in range(6):
        ax.quiver(
            lon_np[face, ::skip, ::skip].ravel(),
            lat_np[face, ::skip, ::skip].ravel(),
            u_np[face, ::skip, ::skip].ravel(),
            v_np[face, ::skip, ::skip].ravel(),
            speed[face, ::skip, ::skip].ravel(),
            transform=ccrs.PlateCarree(),
            scale=800,
            width=0.002,
            cmap="plasma",
        )

    ax.set_global()
    ax.coastlines(linewidth=0.5)
    ax.gridlines(linewidth=0.3, alpha=0.5)
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax
