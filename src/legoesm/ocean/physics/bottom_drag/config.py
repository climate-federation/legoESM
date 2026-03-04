"""Configuration for ocean bottom drag schemes."""

from __future__ import annotations

from typing import NamedTuple


class LinearDragConfig(NamedTuple):
    """Linear bottom drag: tau = -r * u."""
    r: float = 1e-4   # Linear drag coefficient [1/s]


class QuadraticDragConfig(NamedTuple):
    """Quadratic bottom drag: tau = -C_d * |u| * u."""
    C_d: float = 2.5e-3   # Quadratic drag coefficient [dimensionless]


class BottomDragConfig(NamedTuple):
    """Top-level bottom drag configuration."""
    scheme: str = "none"  # "linear", "quadratic", "none"
    linear: LinearDragConfig = LinearDragConfig()
    quadratic: QuadraticDragConfig = QuadraticDragConfig()
