"""Configuration for atmospheric turbulence / boundary layer schemes.

Provides configuration NamedTuples for:
1. Surface layer: bulk aerodynamic surface fluxes
2. Smagorinsky: constant eddy diffusivity (simplest baseline)
3. Louis (1979): stability-dependent diffusion
4. TKE / Mellor-Yamada 2.5: prognostic TKE closure
5. CLUBB-lite: higher-order closure skeleton
6. Holtslag-Boville: nonlocal K-profile with counter-gradient
7. YSU: nonlocal K-profile with entrainment flux
8. EDMF: eddy-diffusivity mass-flux unified framework
9. ML Turbulence Emulator: Equinox MLP surrogate
10. Top-level TurbulenceConfig that selects the active scheme.

References
----------
- Louis, J.-F. (1979). A parametric model of vertical eddy fluxes in the
  atmosphere. Boundary-Layer Meteorol., 17, 187-202.
- Mellor, G. L., & Yamada, T. (1982). Development of a turbulence closure
  model for geophysical fluid problems. Rev. Geophys., 20, 851-875.
- Holtslag, A. A. M., & Boville, B. A. (1993). Local versus nonlocal
  boundary-layer diffusion in a global climate model. J. Climate, 6,
  1825-1842.
- Hong, S.-Y., Noh, Y., & Dudhia, J. (2006). A new vertical diffusion
  package with an explicit treatment of entrainment processes. Mon. Wea.
  Rev., 134, 2318-2341.
- Siebesma, A. P., et al. (2007). A combined eddy-diffusivity mass-flux
  approach for the convective boundary layer. J. Atmos. Sci., 64, 1230-1248.
"""

from __future__ import annotations

from typing import NamedTuple


class SurfaceLayerConfig(NamedTuple):
    """Configuration for bulk aerodynamic surface fluxes.

    Fields
    ------
    z0 : float
        Roughness length [m] (default 1e-4).
    Cd_neutral : float
        Neutral drag coefficient (default 1.5e-3).
    Ch_neutral : float
        Neutral heat transfer coefficient (default 1.5e-3).
    bulk_scheme : str
        Bulk flux algorithm: "constant", "coare3", "large_yeager"
        (default "constant").
    z_ref : float
        Reference height for MOST bulk formulas [m] (default 10.0).
    bulk_n_iter : int
        Number of MOST iterations (default 5).
    """
    z0: float = 1e-4
    Cd_neutral: float = 1.5e-3
    Ch_neutral: float = 1.5e-3
    bulk_scheme: str = "constant"
    z_ref: float = 10.0
    bulk_n_iter: int = 5


class SmagorinskyConfig(NamedTuple):
    """Configuration for constant-Km Smagorinsky turbulence.

    Fields
    ------
    Km : float
        Constant eddy diffusivity for momentum [m^2/s] (default 10.0).
    Pr_t : float
        Turbulent Prandtl number; Kh = Km / Pr_t (default 1.0).
    surface : SurfaceLayerConfig
        Surface layer parameters.
    """
    Km: float = 10.0
    Pr_t: float = 1.0
    surface: SurfaceLayerConfig = SurfaceLayerConfig()


class LouisConfig(NamedTuple):
    """Configuration for Louis (1979) stability-dependent turbulence.

    Fields
    ------
    l_mix_max : float
        Maximum mixing length [m] (default 100.0).
    Ck : float
        Mixing length coefficient (default 0.4).
    Ri_crit : float
        Critical Richardson number (default 0.25).
    surface : SurfaceLayerConfig
        Surface layer parameters.
    """
    l_mix_max: float = 100.0
    Ck: float = 0.4
    Ri_crit: float = 0.25
    surface: SurfaceLayerConfig = SurfaceLayerConfig()


class TKEConfig(NamedTuple):
    """Configuration for prognostic TKE / Mellor-Yamada 2.5 turbulence.

    Fields
    ------
    l_mix_max : float
        Maximum mixing length [m] (default 100.0).
    Ck : float
        TKE -> Km coefficient (default 0.1).
    Ce : float
        TKE dissipation coefficient (default 0.19).
    tke_min : float
        Minimum TKE [m^2/s^2] (default 1e-6).
    Pr_t : float
        Turbulent Prandtl number (default 0.33).
    surface : SurfaceLayerConfig
        Surface layer parameters.
    """
    l_mix_max: float = 100.0
    Ck: float = 0.1
    Ce: float = 0.19
    tke_min: float = 1e-6
    Pr_t: float = 0.33
    surface: SurfaceLayerConfig = SurfaceLayerConfig()


class CLUBBLiteConfig(NamedTuple):
    """Configuration for CLUBB-lite higher-order closure skeleton.

    Fields
    ------
    C1 : float
        Placeholder tuning parameter (default 1.0).
    tke_min : float
        Minimum TKE [m^2/s^2] (default 1e-6).
    surface : SurfaceLayerConfig
        Surface layer parameters.
    """
    C1: float = 1.0
    tke_min: float = 1e-6
    surface: SurfaceLayerConfig = SurfaceLayerConfig()


class HoltslagBovilleConfig(NamedTuple):
    """Configuration for Holtslag-Boville nonlocal K-profile turbulence.

    Fields
    ------
    l_mix_max : float
        Maximum mixing length [m] (default 100.0).
    Pr_t : float
        Turbulent Prandtl number (default 1.0).
    gamma_h : float
        Counter-gradient heat coefficient (default 10.0).
    gamma_m : float
        Counter-gradient momentum coefficient (default 0.0).
    Ri_crit : float
        Critical Richardson number (default 0.25).
    surface : SurfaceLayerConfig
        Surface layer parameters.
    """
    l_mix_max: float = 100.0
    Pr_t: float = 1.0
    gamma_h: float = 10.0
    gamma_m: float = 0.0
    Ri_crit: float = 0.25
    surface: SurfaceLayerConfig = SurfaceLayerConfig()


class YSUConfig(NamedTuple):
    """Configuration for YSU (Yonsei University) PBL turbulence scheme.

    Fields
    ------
    l_mix_max : float
        Maximum mixing length [m] (default 100.0).
    Pr_t : float
        Turbulent Prandtl number (default 1.0).
    entrainment_coeff : float
        Entrainment coefficient at PBL top (default 0.2).
    Ri_crit : float
        Critical Richardson number (default 0.25).
    pbl_smooth_sharpness : float
        Sigmoid sharpness for smooth PBL-top detection (default 20.0).
    surface : SurfaceLayerConfig
        Surface layer parameters.
    """
    l_mix_max: float = 100.0
    Pr_t: float = 1.0
    entrainment_coeff: float = 0.2
    Ri_crit: float = 0.25
    pbl_smooth_sharpness: float = 20.0
    surface: SurfaceLayerConfig = SurfaceLayerConfig()


class EDMFConfig(NamedTuple):
    """Configuration for EDMF eddy-diffusivity mass-flux turbulence.

    Fields
    ------
    l_mix_max : float
        Maximum mixing length [m] (default 100.0).
    Ck : float
        TKE -> Km coefficient (default 0.1).
    Ce : float
        TKE dissipation coefficient (default 0.19).
    tke_min : float
        Minimum TKE [m^2/s^2] (default 1e-6).
    Pr_t : float
        Turbulent Prandtl number (default 0.33).
    n_updrafts : int
        Number of updraft plumes (default 1).
    a_updraft : float
        Updraft area fraction (default 0.1).
    w_updraft_min : float
        Minimum updraft velocity [m/s] (default 0.1).
    entrainment_rate : float
        Lateral entrainment rate [1/m] (default 1e-3).
    detrainment_rate : float
        Lateral detrainment rate [1/m] (default 2e-3).
    surface : SurfaceLayerConfig
        Surface layer parameters.
    """
    l_mix_max: float = 100.0
    Ck: float = 0.1
    Ce: float = 0.19
    tke_min: float = 1e-6
    Pr_t: float = 0.33
    n_updrafts: int = 1
    a_updraft: float = 0.1
    w_updraft_min: float = 0.1
    entrainment_rate: float = 1e-3
    detrainment_rate: float = 2e-3
    surface: SurfaceLayerConfig = SurfaceLayerConfig()


class MLTurbulenceEmulatorConfig(NamedTuple):
    """Configuration for ML turbulence emulator (Equinox MLP).

    Fields
    ------
    n_input : int
        Number of input features per level (default 8).
    n_hidden : int
        Hidden layer width (default 128).
    n_layers : int
        Number of MLP layers (default 3).
    n_output : int
        Number of output features per level (default 9).
    seed : int
        Random seed for model initialization (default 0).
    use_residual : bool
        Apply residual scaling for near-zero untrained output (default True).
    surface : SurfaceLayerConfig
        Surface layer parameters.
    """
    n_input: int = 8
    n_hidden: int = 128
    n_layers: int = 3
    n_output: int = 9
    seed: int = 0
    use_residual: bool = True
    surface: SurfaceLayerConfig = SurfaceLayerConfig()


class TurbulenceConfig(NamedTuple):
    """Top-level turbulence configuration.

    Selects the active scheme and holds sub-configurations.

    Fields
    ------
    scheme : str
        Active turbulence scheme: "smagorinsky", "louis", "tke",
        "clubb_lite", "holtslag_boville", "ysu", "edmf",
        "ml_emulator", or "none".
    smagorinsky : SmagorinskyConfig
        Configuration for Smagorinsky scheme.
    louis : LouisConfig
        Configuration for Louis scheme.
    tke : TKEConfig
        Configuration for TKE scheme.
    clubb_lite : CLUBBLiteConfig
        Configuration for CLUBB-lite scheme.
    holtslag_boville : HoltslagBovilleConfig
        Configuration for Holtslag-Boville scheme.
    ysu : YSUConfig
        Configuration for YSU scheme.
    edmf : EDMFConfig
        Configuration for EDMF scheme.
    ml_emulator : MLTurbulenceEmulatorConfig
        Configuration for ML emulator scheme.
    update_interval_steps : int
        Recompute turbulence every N time steps (1 = every step).
    """
    scheme: str = "smagorinsky"
    smagorinsky: SmagorinskyConfig = SmagorinskyConfig()
    louis: LouisConfig = LouisConfig()
    tke: TKEConfig = TKEConfig()
    clubb_lite: CLUBBLiteConfig = CLUBBLiteConfig()
    holtslag_boville: HoltslagBovilleConfig = HoltslagBovilleConfig()
    ysu: YSUConfig = YSUConfig()
    edmf: EDMFConfig = EDMFConfig()
    ml_emulator: MLTurbulenceEmulatorConfig = MLTurbulenceEmulatorConfig()
    update_interval_steps: int = 1
