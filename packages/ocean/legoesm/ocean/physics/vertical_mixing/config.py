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
    c_s: float = 98.96       # LMD94 scalar stability constant (App. B; V_t^2 + scalar convective scale)
    c_b: float = 0.599       # LMD94 convective velocity scale parameter (legacy single-scale form)
    epsilon_lmd: float = 0.1  # LMD94 surface-layer fraction (App. A/B)
    # LMD94 Appendix B separate momentum/scalar velocity scales w_m, w_s.
    # In the code's sign convention zeta = d/L_MO ≥ 0 for unstable, so the
    # weakly-unstable→convective transition is at |zeta| = zeta_{m,s}_abs
    # (= −zeta_{m,s} of LMD94).  a_*/c_* are chosen by LMD94 so the
    # convective scale w = kappa·(a·u*³ + c·kappa·B_f·d)^{1/3} joins the
    # weakly-unstable kappa·u*·(1+16|zeta|)^{1/4 (m), 1/2 (s)} continuously.
    zeta_m_abs: float = 0.2   # |zeta| momentum weakly→convective join (LMD94 zeta_m=−0.2)
    zeta_s_abs: float = 1.0   # |zeta| scalar   weakly→convective join (LMD94 zeta_s=−1.0)
    a_m: float = 1.26         # LMD94 momentum convective constant
    c_m: float = 8.38         # LMD94 momentum convective constant
    a_s: float = -28.86       # LMD94 scalar convective constant (slope = c_s)
    crossing_sharpness: float = 20.0  # Sigmoid sharpness for h_bl crossing-depth selector
    crossing_threshold: float = 0.1   # Crossing-strength threshold for h_bl blend


class VerticalMixingConfig(NamedTuple):
    """Top-level vertical mixing configuration."""
    scheme: str = "constant"  # "constant", "richardson", "kpp", "none"
    constant: ConstantVerticalMixingConfig = ConstantVerticalMixingConfig()
    richardson: RichardsonVerticalMixingConfig = RichardsonVerticalMixingConfig()
    kpp: KPPConfig = KPPConfig()
    # Tidal mixing is ADDITIVE: when ``tidal.enabled=True`` the
    # caller computes a ``K_tidal(x, y, z)`` field via
    # :func:`legoesm.ocean.physics.vertical_mixing.tidal.compute_tidal_diffusivity`
    # and adds it on top of the diffusivity field from ``scheme``
    # before applying the tracer mixing step.  Default off
    # (``enabled=False``) preserves bit-exact regression on legacy
    # configs.  NOTE: ``make_vertical_mixing_physics`` / ``make_ocean_physics``
    # do NOT apply tidal mixing and RAISE if ``enabled=True`` — apply it via
    # ``ocean.coupler.tidal_mixing_apply.apply_tidal_mixing_step`` so it cannot
    # silently no-op inside the physics composition.
    tidal: TidalMixingConfig = TidalMixingConfig()
