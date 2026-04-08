"""Configuration for MPAS Voronoi-mesh ocean model.

Provides config NamedTuples for the full 3D MPAS ocean dynamics
and for simplified (slab/fixed) ocean modes on Voronoi meshes.
"""

from __future__ import annotations

from typing import NamedTuple


class MPASOceanConfig(NamedTuple):
    """Configuration for MPAS ocean primitive equation solver.

    Fields
    ------
    g : float
        Gravitational acceleration [m/s²].
    rho_0 : float
        Reference seawater density [kg/m³].
    A_h : float
        Horizontal viscosity [m²/s].
    K_h : float
        Horizontal tracer diffusivity [m²/s].
    A_v : float
        Vertical viscosity [m²/s].
    K_v : float
        Vertical tracer diffusivity [m²/s].
    n_barotropic_substeps : int
        Number of barotropic (free-surface) substeps per baroclinic step.
    pv_scheme : str
        PV flux scheme: "energy" or "enstrophy".
    use_conservation_fixer : bool
        Apply conservation fixers after each step.
    fix_volume : bool
        Fix volume (eta) conservation.
    fix_heat : bool
        Fix heat (T) conservation.
    fix_salt : bool
        Fix salt (S) conservation.
    min_water_column_m : float
        Minimum allowed water column depth [m].
    barotropic_damping : float
        Rayleigh damping coefficient for barotropic mode [1/s].
    freshwater_closure : str
        Freshwater closure: "none", "virtual_salt_flux", or "real_freshwater".
        "virtual_salt_flux": apply virtual salt flux to S, real mass flux to eta.
        "real_freshwater": apply real freshwater mass to eta and dilution to S.
        "none": ignore freshwater forcing.
    S_ref : float
        Reference salinity [PSU] for virtual salt flux.
    """
    g: float = 9.80616           # = constants.g
    rho_0: float = 1025.0        # = eos.rho_0
    A_h: float = 1.0e4
    K_h: float = 1.0e3
    A_v: float = 1.0e-3
    K_v: float = 1.0e-4
    n_barotropic_substeps: int = 30
    pv_scheme: str = "energy"
    use_conservation_fixer: bool = False
    fix_volume: bool = True
    fix_heat: bool = True
    fix_salt: bool = True
    min_water_column_m: float = 0.5
    barotropic_damping: float = 0.0
    barotropic_diffusion_alpha: float = 0.01
    barotropic_diffusion_dt_ref: float = 60.0
    semi_implicit_coriolis: bool = True
    freshwater_closure: str = "virtual_salt_flux"
    S_ref: float = 35.0
    physics: object = None  # OceanPhysicsConfig or None
    eos: str = "wright"    # "wright" or "linear"
    eos_linear: object = None  # LinearEOSConfig when eos="linear"
    # Runtime bounds checks (matching cubed-sphere ocean)
    enable_runtime_checks: bool = False
    temperature_min_c: float = -5.0
    temperature_max_c: float = 45.0
    salinity_min_psu: float = 0.0
    salinity_max_psu: float = 50.0


class MPASSimpleOceanConfig(NamedTuple):
    """Configuration for simplified ocean on Voronoi mesh.

    Mirrors SimpleOceanConfig but operates on (nCells,) arrays.

    Fields
    ------
    mode : str
        Ocean mode: "fixed", "slab", or "two_layer".
    sst_constant : float
        Fixed SST [K] (for "fixed" mode).
    h_mix : float
        Mixed layer depth [m].
    rho_ocean : float
        Seawater density [kg/m³].
    c_ocean : float
        Specific heat of seawater [J/(kg·K)].
    Q_flux : float
        Prescribed ocean heat flux [W/m²].
    albedo_ocean : float
        Ocean albedo.
    emissivity_ocean : float
        Ocean emissivity.
    Cd_ocean : float
        Drag coefficient over ocean.
    Ch_ocean : float
        Heat transfer coefficient over ocean.
    U_min : float
        Minimum wind speed [m/s].
    T_freeze : float
        Freezing temperature [K].
    h_deep : float
        Deep layer depth [m] (for "two_layer" mode).
    k_mix : float
        Vertical mixing coefficient [m²/s] (for "two_layer" mode).
    restore_deep : bool
        Restore deep layer toward reference (for "two_layer" mode).
    T_deep_ref : float
        Deep layer reference temperature [K].
    tau_deep : float
        Deep layer restoring timescale [s].
    """
    mode: str = "fixed"
    sst_constant: float = 300.0
    h_mix: float = 50.0
    rho_ocean: float = 1025.0
    c_ocean: float = 3994.0
    Q_flux: float = 0.0
    albedo_ocean: float = 0.06
    emissivity_ocean: float = 0.97
    Cd_ocean: float = 1.5e-3
    Ch_ocean: float = 1.5e-3
    U_min: float = 1.0
    T_freeze: float = 271.35
    h_deep: float = 200.0
    k_mix: float = 1.0e-4
    restore_deep: bool = False
    T_deep_ref: float = 278.0
    tau_deep: float = 365.25 * 86400.0
