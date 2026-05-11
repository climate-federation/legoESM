"""Configuration for ocean convection schemes."""

from __future__ import annotations

from typing import NamedTuple


class EnhancedDiffusionConfig(NamedTuple):
    """Enhanced diffusion where N^2 < 0."""
    K_conv: float = 1.0        # Convective diffusivity [m^2/s]
    K_bg: float = 1e-5         # Background diffusivity [m^2/s]
    smooth_transition: bool = True
    sigmoid_sharpness: float = 1e6


class PlumeConfig(NamedTuple):
    """Entraining mass-flux convective plume."""
    epsilon: float = 1e-3       # Entrainment rate [1/m]
    alpha_plume: float = 0.1    # Detrainment tendency scaling
    # Plume vertical velocity [m/s].  Used directly as ``w_p`` in
    # dT/dt = w_p · α · ε · (T_p − T_env).  Despite the legacy
    # ``_min`` suffix, this is the actual plume speed for the
    # unresolved-plume detrainment closure (see plume.py:75-88).
    w_plume_min: float = 0.01
    T_excess: float = 0.05      # Initial plume temperature excess [K]


class OceanConvectionConfig(NamedTuple):
    """Top-level ocean convection configuration."""
    scheme: str = "none"  # "enhanced_diffusion", "plume", "none"
    enhanced_diffusion: EnhancedDiffusionConfig = EnhancedDiffusionConfig()
    plume: PlumeConfig = PlumeConfig()
