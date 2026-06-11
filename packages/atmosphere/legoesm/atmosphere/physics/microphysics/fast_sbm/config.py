"""Configuration for the fast-SBM bin microphysics scheme.

Defaults are the WRF ``module_mp_fast_sbm.F`` oracle values (CGS → SI).
Where the oracle's reference coefficient differs from the repo-wide
``legoesm.constants`` value (different reference temperature), the oracle
value is kept here as an explicit scheme parameter so the port stays
faithful without monkey-patching shared constants.
"""

from __future__ import annotations

from typing import NamedTuple


class FastSBMConfig(NamedTuple):
    """Tunable parameters of the fast-SBM scheme (oracle defaults).

    Attributes:
        d_vapor_ref_m2s: water-vapor diffusivity at the reference state
            [m²/s] (oracle ``D_MYIN = 0.211`` cm²/s; ``legoesm.constants.
            D_vapor = 2.21e-5`` uses a different reference temperature).
        nu_air_ref_m2s: kinematic viscosity of air used in the ventilation
            Reynolds number [m²/s] (oracle ``COEFF_VISCOUS = 0.13`` cm²/s).
        diffusivity_T_exponent: temperature exponent of the diffusivity law
            ``D = D_ref (p₀/p)(T/T₀)^a`` (oracle ``1.94``).
        ventilation_max: cap on the ventilation factor (oracle
            ``VENTPL_MAX = 5.0``).
    """

    d_vapor_ref_m2s: float = 2.11e-5
    nu_air_ref_m2s: float = 1.3e-5
    diffusivity_T_exponent: float = 1.94
    ventilation_max: float = 5.0
    # --- column-adapter reconstruction (stateless interface; see
    # column.py docstring) ---
    # Prescribed cloud-droplet number for spectrum reconstruction [1/m^3]
    # (same closure role and default as SDMConfig.cdnc / Seifert-Beheng
    # maritime Nc_0).
    cdnc: float = 1.0e8
    # Geometric std of the reconstructed cloud-droplet lognormal [-].
    cloud_geom_std: float = 1.4
    # Rain-drop number floor for reconstruction when the dycore carries no
    # N_r [1/m^3].
    n_rain_floor: float = 1.0e3
    # Collision kernel for the in-step Bott coalescence: "hall", "long",
    # or "golovin" (analytic test kernel, uses golovin_b).
    collision_kernel: str = "hall"
    golovin_b: float = 1.5e3
    # Static per-step count for per-bin sedimentation (oracle FALFLUXHUCM_Z
    # adapts NSUB at runtime; a traced trip count is not reverse-mode
    # differentiable, so the port fixes it).
    n_fall_substeps: int = 4
    # --- CCN activation (Köhler; oracle JERNUCL01_KS aerosol mode) ---
    # Total CCN number concentration available to activate [1/m^3].
    ccn_number: float = 1.0e8
    # Log-normal dry-aerosol mode: median radius [m] and geometric std [-]
    # (continental accumulation mode defaults).
    aerosol_dry_median: float = 5.0e-8
    aerosol_geom_std: float = 2.0
    # Solute van't Hoff ions, molar mass [kg/mol], density [kg/m^3]
    # (ammonium sulfate defaults: 3 ions, 0.132 kg/mol, 1770 kg/m^3).
    aerosol_ions: float = 3.0
    aerosol_molar_mass: float = 0.132
    aerosol_solute_density: float = 1770.0
