"""Configuration for ocean bottom drag schemes."""

from __future__ import annotations

from typing import NamedTuple

__param_spec__ = {
    "LinearDragConfig": {
        "scheme_key": "ocean.bottom_drag.linear",
        "excluded": {},
        "params": {
            "r": {
                "units": "m/s", "bounds": (1.0e-4, 5.0e-3), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "MITgcm bottomDragLinear (default 1.1e-3 m/s)",
                "shape": None,
            },
        },
    },
    "QuadraticDragConfig": {
        "scheme_key": "ocean.bottom_drag.quadratic",
        "excluded": {},
        "params": {
            "C_d": {
                "units": "1", "bounds": (1.0e-3, 5.0e-3), "tunable_tier": 1,
                "transform": "sigmoid", "category": "closure",
                "reference": "quadratic bottom-drag coefficient (default 2.5e-3)",
                "shape": None,
            },
        },
    },
}


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
