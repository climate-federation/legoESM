"""Configuration for atmospheric convection schemes.

Provides configuration NamedTuples for:
1. Simplified Betts-Miller (SBM) — relaxation-based convection (Frierson 2007)
2. Deep Convective Adjustment (DCA) — simplest baseline adjustment
3. Kuo — column moisture-excess (Kuo 1965/1974)
4. Prognostic Mass-Flux — Arakawa-Wu type (1 prognostic var: M_c)
5. Simplified EDMF — eddy-diffusivity mass-flux (1 prognostic var: a_u)
6. Top-level ConvectionConfig that selects the active scheme.

References
----------
- Frierson, D. M. W. (2007). The Dynamics of Idealized Convection
  Schemes and Their Effect on the Zonally Averaged Tropical Circulation.
  J. Atmos. Sci., 64, 1959-1976.
- Kuo, H. L. (1974). Further studies of the parameterization of the
  influence of cumulus convection on large-scale flow. J. Atmos. Sci.,
  31, 1232-1240.
- Arakawa, A., & Wu, C.-M. (2013). A unified representation of deep
  moist convection in numerical modeling of the atmosphere. Part I.
  J. Atmos. Sci., 70, 1977-1992.
"""

from __future__ import annotations

from typing import NamedTuple


class SBMConfig(NamedTuple):
    """Configuration for Simplified Betts-Miller convection.

    Fields
    ------
    tau_c : float
        Relaxation timescale [s] (default 7200 = 2 hours).
    RH_ref : float
        Reference relative humidity for moisture profile (default 0.7).
    CAPE_threshold : float
        Minimum CAPE [J/kg] to trigger convection (default 70.0).
    T_min_convect : float
        Minimum temperature [K] for convection (default 200.0).
    smooth_trigger_sharpness : float
        Sigmoid sharpness for smooth trigger [1/(J/kg)] (default 0.01).
    """
    tau_c: float = 7200.0
    RH_ref: float = 0.7
    CAPE_threshold: float = 70.0
    T_min_convect: float = 200.0
    smooth_trigger_sharpness: float = 0.01


class DCAConfig(NamedTuple):
    """Configuration for Deep Convective Adjustment.

    Fields
    ------
    n_iterations : int
        Number of adjustment iterations per call (default 1).
    mixing_fraction : float
        Fraction of adjustment applied per iteration (default 1.0).
    cape_threshold : float
        Minimum CAPE [J/kg] to trigger convection (default 100.0).
        Columns with CAPE below this are not adjusted.
    cape_sharpness : float
        Sigmoid sharpness [1/(J/kg)] for smooth CAPE gating (default 0.02).
    """
    n_iterations: int = 1
    mixing_fraction: float = 1.0
    cape_threshold: float = 100.0
    cape_sharpness: float = 0.1   # sigmoid(-10)≈5e-5 at CAPE=0; 0.5 at threshold


class KuoConfig(NamedTuple):
    """Configuration for Kuo column moisture-excess convection.

    Fields
    ------
    alpha_heat : float
        Fraction of column moisture excess going to heating vs moistening.
    me_threshold : float
        Minimum column moisture excess to trigger convection [kg/m^2].
    smooth_trigger_sharpness : float
        Sigmoid sharpness on column moisture excess trigger [1/(kg/m^2)].
    tau_relax : float
        Relaxation timescale [s].
    """
    alpha_heat: float = 0.75
    me_threshold: float = 1e-5
    smooth_trigger_sharpness: float = 1e4
    tau_relax: float = 3600.0


class MassFluxConfig(NamedTuple):
    """Configuration for Prognostic Mass-Flux convection (Arakawa-Wu type).

    Fields
    ------
    tau_adj : float
        Mass flux relaxation timescale [s].
    epsilon_0 : float
        Entrainment rate [1/m].
    delta_0 : float
        Detrainment rate [1/m].
    M_scale : float
        Equilibrium mass flux scale [kg/m^2/s].
    cape_activation_scale : float
        Sigmoid scale for CAPE trigger [J/kg].
    cape_threshold : float
        CAPE threshold [J/kg].
    M_c_init : float
        Initial base mass flux [kg/m^2/s].
    """
    tau_adj: float = 3600.0
    epsilon_0: float = 1e-3
    delta_0: float = 1e-3
    M_scale: float = 0.01
    cape_activation_scale: float = 100.0
    cape_threshold: float = 0.0
    M_c_init: float = 0.0


class EDMFConfig(NamedTuple):
    """Configuration for simplified EDMF convection (mass-flux part only).

    Fields
    ------
    epsilon_0 : float
        Entrainment rate [1/m].
    delta_0 : float
        Detrainment rate [1/m].
    a_u_init : float
        Initial updraft area fraction.
    tau_a : float
        Relaxation timescale for a_u [s].
    w_u_min : float
        Minimum updraft velocity [m/s].
    cape_activation_scale : float
        Sigmoid scale for CAPE trigger [J/kg].
    cape_threshold : float
        CAPE threshold [J/kg].
    """
    epsilon_0: float = 2e-3
    delta_0: float = 2e-3
    a_u_init: float = 0.1
    tau_a: float = 1800.0
    w_u_min: float = 0.1
    cape_activation_scale: float = 100.0
    cape_threshold: float = 0.0


class ConvectionConfig(NamedTuple):
    """Top-level convection configuration.

    Selects the active scheme and holds sub-configurations.

    Fields
    ------
    scheme : str
        Active convection scheme: "sbm", "dca", "kuo", "mass_flux",
        "edmf", or "none".
    sbm : SBMConfig
        Configuration for Simplified Betts-Miller.
    dca : DCAConfig
        Configuration for Deep Convective Adjustment.
    kuo : KuoConfig
        Configuration for Kuo column moisture excess.
    mass_flux : MassFluxConfig
        Configuration for Prognostic Mass-Flux.
    edmf : EDMFConfig
        Configuration for simplified EDMF.
    update_interval_steps : int
        Recompute convection every N time steps (1 = every step).
    """
    scheme: str = "sbm"
    sbm: SBMConfig = SBMConfig()
    dca: DCAConfig = DCAConfig()
    kuo: KuoConfig = KuoConfig()
    mass_flux: MassFluxConfig = MassFluxConfig()
    edmf: EDMFConfig = EDMFConfig()
    update_interval_steps: int = 1
