"""Configuration for ocean vertical mixing schemes."""

from __future__ import annotations

from typing import NamedTuple

from legoesm import constants


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
    Ri_crit: float = 0.3    # Critical bulk Richardson number (Large, McWilliams & Doney 1994; matches NCAR POP2 / MOM6 default)
    Cv: float = 1.6          # Unresolved shear coefficient
    kappa_vk: float = constants.kappa_vk
    K_max: float = 1.0       # Maximum diffusivity [m^2/s]
    K_bg: float = 1e-5       # Background diffusivity [m^2/s]
    A_bg: float = 1e-4       # Background viscosity [m^2/s]
    gamma_T: float = 6.33    # Non-local transport coefficient for T
    gamma_S: float = 6.33    # Non-local transport coefficient for S
    K_conv: float = 1.0      # Convective mixing diffusivity [m^2/s]
    Ri_conv: float = 0.0     # Ri threshold for convective instability
    K_0_shear: float = 5e-3  # LMD94 interior shear instability peak K [m^2/s]
    Ri_0: float = 0.7        # LMD94 critical Ri for interior shear mixing
    c_s: float = 98.96       # LMD94 parameter for V_t^2 (Appendix B)
    c_b: float = 0.599       # LMD94 convective velocity scale parameter
    epsilon_lmd: float = 0.1  # LMD94 surface-layer fraction (App. A/B)
    crossing_sharpness: float = 20.0  # Sigmoid sharpness for h_bl crossing-depth selector
    crossing_threshold: float = 0.1   # Crossing-strength threshold for h_bl blend


class VerticalMixingConfig(NamedTuple):
    """Top-level vertical mixing configuration."""
    scheme: str = "constant"  # "constant", "richardson", "kpp", "none"
    constant: ConstantVerticalMixingConfig = ConstantVerticalMixingConfig()
    richardson: RichardsonVerticalMixingConfig = RichardsonVerticalMixingConfig()
    kpp: KPPConfig = KPPConfig()
