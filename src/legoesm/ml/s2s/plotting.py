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


def normalize_longitude_data(
    data: xr.DataArray | xr.Dataset,
) -> xr.DataArray | xr.Dataset:
    """Return data with longitudes normalized to [-180, 180) and sorted."""
    if "longitude" not in data.coords:
        return data
    longitude = np.asarray(data["longitude"], dtype=float)
    normalized = ((longitude + 180.0) % 360.0) - 180.0
    return data.assign_coords(longitude=normalized).sortby("longitude")


def _normalize_longitude_field(field: xr.DataArray) -> xr.DataArray:
    """Backward-compatible field wrapper around `normalize_longitude_data`."""
    return normalize_longitude_data(field)


def plot_latlon_map(
    ax: plt.Axes,
    field: xr.DataArray,
    title: str,
    *,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
) -> None:
    plot_field = normalize_longitude_data(field).transpose("latitude", "longitude")
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


__all__ = [
    "EXPERIMENT_COLORS",
    "frame_to_image",
    "normalize_longitude_data",
    "plot_latlon_map",
]
