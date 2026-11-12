"""Numerical kernels of the wave component (NeuralSWM port)."""

from legoesm.wave.physics.dispersion import (
    calculate_depth_dispersion_derivative,
    compute_group_velocity,
    solve_wavenumber,
)

__all__ = [
    "calculate_depth_dispersion_derivative",
    "compute_group_velocity",
    "solve_wavenumber",
]
