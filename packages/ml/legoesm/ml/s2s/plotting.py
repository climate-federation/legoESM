"""Shared plotting helpers for slab subseasonal workflows."""

from __future__ import annotations

from io import BytesIO

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

EXPERIMENT_COLORS = {"coupled": "#1f77b4", "uncoupled": "#ff7f0e"}


def plot_latlon_map(
    ax: plt.Axes,
    field: xr.DataArray,
    title: str,
    *,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
) -> None:
    plot_field = field.transpose("latitude", "longitude")
    plot_field.plot(
        ax=ax,
        x="longitude",
        y="latitude",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def frame_to_image(
    fig: plt.Figure,
    *,
    dpi: int = 110,
    error_message: str | None = None,
) -> np.ndarray:
    if imageio is None:
        raise ImportError(
            error_message
            or "imageio is required for GIF postprocessing. Install it in the active environment."
        )
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    plt.close(fig)
    buffer.seek(0)
    return imageio.imread(buffer)


__all__ = ["EXPERIMENT_COLORS", "frame_to_image", "plot_latlon_map"]
