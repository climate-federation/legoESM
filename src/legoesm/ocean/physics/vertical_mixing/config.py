"""Configuration for ocean vertical mixing schemes."""

from __future__ import annotations

from typing import NamedTuple

from legoesm import constants
from legoesm.ocean.physics.vertical_mixing.tidal import TidalMixingConfig


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


class TKEConfig(NamedTuple):
    """Gaspar (1990) / Burchard (2002) prognostic TKE closure.

    Veros's canonical vertical-mixing scheme (``enable_tke=True``).
    Solves a prognostic budget for turbulent kinetic energy per unit
    mass, then derives the eddy diffusivities ``K_M`` / ``K_H`` from
    TKE and a mixing-length closure.

    Closure equations (per column, at interfaces unless noted):

        dTKE/dt = P_s + P_b - eps + d/dz(alpha_tke * K_M * dTKE/dz)

    with

        P_s   = K_M * (du/dz)^2 + K_M * (dv/dz)^2     (shear production)
        P_b   = -K_H * N^2                            (buoyancy work)
        eps   = c_eps * TKE^(3/2) / l_eps             (dissipation)
        K_M   = c_k * l_k * sqrt(2 * TKE)
        K_H   = K_M

    The mixing lengths ``l_k`` (for K_M) and ``l_eps`` (for dissipation)
    follow the Bougeault-Lacarrere asymmetric construction
    (``tke_mxl_choice=2``): an upward and downward integration of TKE
    against the local Brunt-Vaisala frequency gives ``l_up`` and
    ``l_dn``; ``l_k = sqrt(l_up * l_dn)`` and
    ``l_eps = max(l_up, l_dn)``. ``tke_mxl_choice=1`` selects a simple
    parabolic-bounded length scale; choice ``2`` is the production
    default.

    Surface flux into TKE is ``(|tau|/rho_0)^(3/2)`` (Wallace surface
    parameterisation).

    Parameter naming mirrors Veros's settings 1:1 so the legoESM-Veros
    recipe maps each knob directly. Defaults are Veros's DINO / ACC
    canonical values.
    """
    c_k: float = 0.1
    c_eps: float = 0.7
    alpha_tke: float = 30.0
    mxl_min: float = 1.0e-8
    tke_mxl_choice: int = 2
    kappaM_min: float = 2.0e-4
    kappaH_min: float = 2.0e-5
    enable_kappaH_profile: bool = True
    tke_surface_min: float = 1.0e-4      # surface TKE floor [m^2/s^2]
    tke_background: float = 1.0e-6       # interior TKE floor [m^2/s^2]


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
    scheme: str = "constant"  # "constant", "richardson", "kpp", "tke", "none"
    constant: ConstantVerticalMixingConfig = ConstantVerticalMixingConfig()
    richardson: RichardsonVerticalMixingConfig = RichardsonVerticalMixingConfig()
    kpp: KPPConfig = KPPConfig()
    tke: TKEConfig = TKEConfig()
    # Tidal mixing is ADDITIVE: when ``tidal.enabled=True`` the
    # caller computes a ``K_tidal(x, y, z)`` field via
    # :func:`legoesm.ocean.physics.vertical_mixing.tidal.compute_tidal_diffusivity`
    # and adds it on top of the diffusivity field from ``scheme``
    # before applying the tracer mixing step.  Default off
    # (``enabled=False``) preserves bit-exact regression on legacy
    # configs.
    tidal: TidalMixingConfig = TidalMixingConfig()
