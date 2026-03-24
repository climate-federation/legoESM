"""Configuration for ocean vertical mixing schemes."""

from __future__ import annotations

from typing import NamedTuple


class ConstantVerticalMixingConfig(NamedTuple):
    """Constant-coefficient vertical mixing."""
    A_v: float = 1e-3   # Vertical viscosity [m^2/s]
    K_v: float = 1e-4   # Vertical diffusivity [m^2/s]


class RichardsonVerticalMixingConfig(NamedTuple):
    """Pacanowski & Philander (1981) Richardson-number dependent mixing."""
    K_0: float = 5e-3    # Maximum diffusivity [m^2/s]
    alpha: float = 5.0   # Stability parameter
    n: int = 2           # Exponent
    K_bg: float = 1e-5   # Background diffusivity [m^2/s]
    A_bg: float = 1e-4   # Background viscosity [m^2/s]
    Pr_t: float = 10.0   # Turbulent Prandtl number


class KPPConfig(NamedTuple):
    """LMD94-style K-Profile Parameterization.

    Extends the Large, McWilliams & Doney (1994) KPP boundary-layer
    parameterization with linear interpolation for h_bl, proper
    turbulent velocity scales, and interior convective instability
    handling.

    The caller should supply surface wind stress (tau_x, tau_y) and
    surface buoyancy flux (B_f) so that friction velocity and turbulent
    velocity scales can be computed correctly.  If these are not
    provided, simplified proxies from the ocean state are used.
    """
    Ri_crit: float = 0.25   # Critical bulk Richardson number (LMD94)
    Cv: float = 1.6          # Unresolved shear coefficient
    kappa_vk: float = 0.4    # von Karman constant
    K_max: float = 1.0       # Maximum diffusivity [m^2/s]
    K_bg: float = 1e-5       # Background diffusivity [m^2/s]
    A_bg: float = 1e-4       # Background viscosity [m^2/s]
    gamma_T: float = 6.33    # Non-local transport coefficient for T
    gamma_S: float = 6.33    # Non-local transport coefficient for S
    K_conv: float = 1.0      # Convective mixing diffusivity [m^2/s]
    Ri_conv: float = 0.0     # Ri threshold for convective instability


class VerticalMixingConfig(NamedTuple):
    """Top-level vertical mixing configuration."""
    scheme: str = "constant"  # "constant", "richardson", "kpp", "none"
    constant: ConstantVerticalMixingConfig = ConstantVerticalMixingConfig()
    richardson: RichardsonVerticalMixingConfig = RichardsonVerticalMixingConfig()
    kpp: KPPConfig = KPPConfig()
