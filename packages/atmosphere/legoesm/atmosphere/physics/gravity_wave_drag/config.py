"""Configuration for gravity wave drag schemes.

Provides configuration NamedTuples for:
1. Rayleigh: simple Rayleigh friction drag
2. Lindzen: smoothed Lindzen (1981) orographic GWD
3. McFarlane: smoothed McFarlane (1987) orographic GWD
4. Hines: Hines (1997) Doppler-spread parameterization
5. PrognosticSpectral: prognostic spectral GWD
6. MLEmulator: ML-based GWD emulator (Equinox MLP)
7. GravityWaveDragConfig: top-level selector

References
----------
- Lindzen, R. S. (1981). Turbulence and stress owing to gravity wave and
  tidal breakdown. J. Geophys. Res., 86, 9707-9714.
- McFarlane, N. A. (1987). The effect of orographically excited gravity
  wave drag on the general circulation of the lower stratosphere and
  troposphere. J. Atmos. Sci., 44, 1775-1800.
- Hines, C. O. (1997). Doppler-spread parameterization of gravity-wave
  momentum deposition in the middle atmosphere. 1. Basic formulation.
  J. Atmos. Solar-Terr. Phys., 59, 371-386.
"""

from __future__ import annotations

import math
from typing import NamedTuple


class RayleighConfig(NamedTuple):
    """Configuration for Rayleigh friction drag.

    Fields
    ------
    k_max : float
        Maximum drag coefficient [1/s] (default 1/(1*86400)).
    sigma_b : float
        Boundary layer top sigma level (default 0.7).
    sponge_top : float
        Upper sponge sigma level (default 0.02).
    sponge_k : float
        Upper sponge drag coefficient [1/s] (default 1/(0.5*86400)).
    """
    k_max: float = 1.0 / 86400.0
    sigma_b: float = 0.7
    sponge_top: float = 0.02
    sponge_k: float = 1.0 / (0.5 * 86400.0)


class LindzenConfig(NamedTuple):
    """Configuration for smoothed Lindzen (1981) orographic GWD.

    Fields
    ------
    h_topo : float
        Sub-grid topographic height [m] (default 500).
    k_wave : float
        Horizontal wavenumber [1/m] (default 2*pi/100e3).
    N_ref : float
        Reference Brunt-Väisälä frequency [1/s] (default 0.01).
    critical_Fr : float
        Critical Froude number threshold (default 1.0).
    Fr_sharpness : float
        Sigmoid sharpness for Froude number transition (default 20.0).
    """
    h_topo: float = 500.0
    k_wave: float = 2.0 * math.pi / 100e3
    N_ref: float = 0.01
    critical_Fr: float = 1.0
    Fr_sharpness: float = 20.0


class McFarlaneConfig(NamedTuple):
    """Configuration for smoothed McFarlane (1987) orographic GWD.

    Extends the Lindzen approach with explicit launch flux control
    and directional spreading.

    Fields
    ------
    h_topo : float
        Sub-grid topographic height [m] (default 500).
    k_wave : float
        Horizontal wavenumber [1/m] (default 2*pi/100e3).
    N_ref : float
        Reference Brunt-Väisälä frequency [1/s] (default 0.01).
    G_0 : float
        Dimensionless launch-flux efficiency factor (default 0.5).
        The orographic launch stress is
        ``tau_0 = G_0 * rho * N * k * h^2 * U`` [Pa] — ``G_0`` is the
        dimensionless prefactor; the dimensional content comes from the
        thermodynamics / wind / wavenumber.  (Earlier docstring labeled
        ``G_0`` as Pa, which combined with the missing ``k_wave`` factor
        in the formula produced stress with the wrong units.)
    efficiency : float
        Breaking efficiency (default 0.5).
    min_wind : float
        Minimum wind for wave activity [m/s] (default 2.0).
    envelope_scale : float
        Vertical envelope scale (default 1.0).
    directional_spread : float
        Multi-directional spreading factor (default 1.0).
    min_wind_sharpness : float
        Sigmoid sharpness for the smooth ``U > min_wind`` activation
        (default 20.0).  Higher values approach a hard step.
    softmin_sharpness : float
        Log-sum-exp softmin sharpness used by the saturation cap
        ``min(tau_carry, tau_sat)`` (default 50.0).  Higher values give
        a sharper cap at the cost of larger gradients near the kink.
    tau_max : float
        Upper clip on launch stress [Pa] (default 10.0).  Operationally
        protects against runaway stress in pathological columns.
    """
    h_topo: float = 500.0
    k_wave: float = 2.0 * math.pi / 100e3
    N_ref: float = 0.01
    G_0: float = 0.5
    efficiency: float = 0.5
    min_wind: float = 2.0
    envelope_scale: float = 1.0
    directional_spread: float = 1.0
    min_wind_sharpness: float = 20.0
    softmin_sharpness: float = 50.0
    tau_max: float = 10.0


class HinesConfig(NamedTuple):
    """Configuration for Hines (1997) Doppler-spread parameterization.

    Fields
    ------
    rms_gw_speed : float
        RMS gravity wave speed [m/s] (default 1.0).
    m_star : float
        Characteristic vertical wavenumber [1/m] (default 2*pi/2e3).
    total_rms_wind : float
        Total RMS gravity wave wind [m/s] (default 2.0).
    cutoff_wn : float
        Maximum vertical wavenumber [1/m] (default 2*pi/500).
    Fmax : float
        Saturation momentum flux cap [Pa] (default 0.1).
    doppler_sharpness : float
        Sigmoid sharpness for Doppler saturation (default 50.0).
    """
    rms_gw_speed: float = 1.0
    m_star: float = 2.0 * math.pi / 2e3
    total_rms_wind: float = 2.0
    cutoff_wn: float = 2.0 * math.pi / 500.0
    Fmax: float = 0.1
    doppler_sharpness: float = 50.0
    U_mag_floor: float = 0.1  # Wind-magnitude floor for projection [m/s]


class PrognosticSpectralConfig(NamedTuple):
    """Configuration for prognostic spectral GWD.

    Fields
    ------
    n_azimuths : int
        Number of azimuthal directions (default 4).
    n_wavenumbers : int
        Number of spectral bins (default 20).
    k_min : float
        Minimum horizontal wavenumber [1/m] (default 2*pi/100e3).
    k_max : float
        Maximum horizontal wavenumber [1/m] (default 2*pi/1e3).
    launch_flux : float
        Source momentum flux [Pa] (default 1e-3).
    breaking_threshold : float
        Froude threshold for wave breaking (default 1.0).
    breaking_sharpness : float
        Sigmoid sharpness for breaking transition (default 10.0).
    tau_decay : float
        Relaxation timescale for prognostic spectrum [s] (default 86400).
    """
    n_azimuths: int = 4
    n_wavenumbers: int = 20
    k_min: float = 2.0 * math.pi / 100e3
    k_max: float = 2.0 * math.pi / 1e3
    launch_flux: float = 1e-3
    breaking_threshold: float = 1.0
    breaking_sharpness: float = 10.0
    tau_decay: float = 86400.0


class GWDMLEmulatorConfig(NamedTuple):
    """Configuration for ML-based GWD emulator.

    Fields
    ------
    n_input : int
        Number of input features per level (default 7).
    n_hidden : int
        Hidden layer width (default 128).
    n_layers : int
        Number of MLP layers (default 3).
    n_output : int
        Number of output tendencies per level (default 3).
    seed : int
        Random seed for initialization (default 0).
    use_residual : bool
        Scale outputs for residual learning (default True).
    """
    n_input: int = 7
    n_hidden: int = 128
    n_layers: int = 3
    n_output: int = 3
    seed: int = 0
    use_residual: bool = True
    norm_u: float = 30.0     # Wind scale [m/s] for u, v normalization
    norm_T: float = 300.0    # Temperature scale [K]
    norm_z: float = 30000.0  # Height scale [m]


class GravityWaveDragConfig(NamedTuple):
    """Top-level gravity wave drag configuration.

    Selects the active scheme and holds sub-configurations.

    Fields
    ------
    scheme : str
        Active GWD scheme: "rayleigh", "lindzen", "mcfarlane",
        "hines", "prognostic_spectral", "ml_emulator", or "none".
    rayleigh : RayleighConfig
        Configuration for Rayleigh friction scheme.
    lindzen : LindzenConfig
        Configuration for Lindzen orographic scheme.
    mcfarlane : McFarlaneConfig
        Configuration for McFarlane orographic scheme.
    hines : HinesConfig
        Configuration for Hines Doppler-spread scheme.
    prognostic_spectral : PrognosticSpectralConfig
        Configuration for prognostic spectral scheme.
    ml_emulator : GWDMLEmulatorConfig
        Configuration for ML emulator scheme.
    """
    scheme: str = "none"
    rayleigh: RayleighConfig = RayleighConfig()
    lindzen: LindzenConfig = LindzenConfig()
    mcfarlane: McFarlaneConfig = McFarlaneConfig()
    hines: HinesConfig = HinesConfig()
    prognostic_spectral: PrognosticSpectralConfig = PrognosticSpectralConfig()
    ml_emulator: GWDMLEmulatorConfig = GWDMLEmulatorConfig()
