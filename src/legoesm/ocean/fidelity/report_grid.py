"""Canonical lat-lon report grid for cross-model and obs comparison.

The report grid is the regridding target for every comparison that crosses
grid conventions (Veros B-grid, NEMO ORCA, observational climatology).
Edges and centers are stored in radians to avoid degree/radian confusion at
call sites; ``cell_area`` uses :data:`legoesm.constants.R_earth`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np

from legoesm import constants

Resolution = Literal["0p5deg", "1deg", "2deg"]

_RESOLUTION_SHAPES: dict[str, tuple[int, int]] = {
    "0p5deg": (360, 720),
    "1deg": (180, 360),
    "2deg": (90, 180),
}


@dataclass(frozen=True)
class ReportGrid:
    """A regular lat-lon grid on the sphere with cached derived arrays."""

    n_lat: int
    n_lon: int

    @cached_property
    def lat_edges(self) -> np.ndarray:
        """Latitude edges (rad), length ``n_lat + 1``, south to north."""
        return np.linspace(-math.pi / 2.0, math.pi / 2.0, self.n_lat + 1)

    @cached_property
    def lon_edges(self) -> np.ndarray:
        """Longitude edges (rad), length ``n_lon + 1``, ``[0, 2π]``."""
        return np.linspace(0.0, 2.0 * math.pi, self.n_lon + 1)

    @cached_property
    def lat_centers(self) -> np.ndarray:
        e = self.lat_edges
        return 0.5 * (e[:-1] + e[1:])

    @cached_property
    def lon_centers(self) -> np.ndarray:
        e = self.lon_edges
        return 0.5 * (e[:-1] + e[1:])

    @cached_property
    def cell_area(self) -> np.ndarray:
        """Spherical cell area (m^2), shape ``(n_lat, n_lon)``.

        ``A = R^2 · Δλ · (sin φ_n − sin φ_s)`` with ``R = constants.R_earth``.
        Summing over all cells reproduces ``4 π R^2`` to machine precision.
        """
        R = constants.R_earth
        dlon = np.diff(self.lon_edges)
        sin_lat_s = np.sin(self.lat_edges[:-1])
        sin_lat_n = np.sin(self.lat_edges[1:])
        per_lat = (R * R) * (sin_lat_n - sin_lat_s)
        return per_lat[:, None] * dlon[None, :]


def get_report_grid(resolution: Resolution = "1deg") -> ReportGrid:
    """Return the canonical report grid for the requested resolution.

    Supported values: ``"0p5deg"``, ``"1deg"`` (default), ``"2deg"``. Any
    other string raises :class:`ValueError` rather than silently degrading.
    """
    if resolution not in _RESOLUTION_SHAPES:
        raise ValueError(
            f"Unknown report-grid resolution {resolution!r}; "
            f"expected one of {tuple(_RESOLUTION_SHAPES)}"
        )
    n_lat, n_lon = _RESOLUTION_SHAPES[resolution]
    return ReportGrid(n_lat=n_lat, n_lon=n_lon)
