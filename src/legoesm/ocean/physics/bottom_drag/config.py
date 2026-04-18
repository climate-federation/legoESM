"""Configuration for ocean bottom drag schemes."""

from __future__ import annotations

from typing import NamedTuple


class LinearDragConfig(NamedTuple):
    """Linear bottom drag: du/dt = -r * u / dz_bottom.

    The coefficient r has units [m/s] so that the bottom stress
    tau = rho_0 * r * u [N/m^2] is independent of vertical resolution.
    Thinner bottom layers feel more deceleration from the same stress.
    This matches MITgcm's ``bottomDragLinear`` convention.
    """
    r: float = 1.1e-3   # Linear drag coefficient [m/s]


class QuadraticDragConfig(NamedTuple):
    """Quadratic bottom drag: tau = -C_d * |u| * u."""
    C_d: float = 2.5e-3   # Quadratic drag coefficient [dimensionless]


class BottomDragConfig(NamedTuple):
    """Top-level bottom drag configuration."""
    scheme: str = "none"  # "linear", "quadratic", "none"
    linear: LinearDragConfig = LinearDragConfig()
    quadratic: QuadraticDragConfig = QuadraticDragConfig()
