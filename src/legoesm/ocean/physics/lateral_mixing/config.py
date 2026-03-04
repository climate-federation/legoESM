"""Configuration for ocean lateral mixing schemes."""

from __future__ import annotations

from typing import NamedTuple


class HarmonicConfig(NamedTuple):
    """Laplacian (harmonic) lateral mixing."""
    A_h: float = 1e4   # Horizontal viscosity [m^2/s]
    K_h: float = 1e3   # Horizontal tracer diffusivity [m^2/s]


class BiharmonicConfig(NamedTuple):
    """Biharmonic lateral mixing."""
    B_h_momentum: float = 0.0   # Biharmonic viscosity [m^4/s]
    B_h_tracer: float = 0.0     # Biharmonic tracer diffusivity [m^4/s]


class GMRediConfig(NamedTuple):
    """Gent-McWilliams (1990) / Redi isopycnal diffusion."""
    kappa_GM: float = 1e3       # GM bolus transport coefficient [m^2/s]
    kappa_Redi: float = 1e3     # Redi isopycnal diffusivity [m^2/s]
    S_max: float = 0.01         # Maximum isopycnal slope
    taper_scheme: str = "dm95"  # Tapering scheme


class LateralMixingConfig(NamedTuple):
    """Top-level lateral mixing configuration."""
    scheme: str = "harmonic"  # "harmonic", "biharmonic", "gm_redi", "none"
    harmonic: HarmonicConfig = HarmonicConfig()
    biharmonic: BiharmonicConfig = BiharmonicConfig()
    gm_redi: GMRediConfig = GMRediConfig()
