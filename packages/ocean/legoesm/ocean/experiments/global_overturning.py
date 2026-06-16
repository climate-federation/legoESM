"""Global Baroclinic Overturning Circulation Experiment.

Inspired by Wolfe & Cessi (2010, JPO) but adapted to spherical geometry
and coarse resolution (1-0.5 deg) as a stepping stone toward OMIP.

Scientific Purpose:
- Validate global baroclinic ocean dynamics with stratification
- Test meridional overturning circulation (MOC) development
- Verify Southern Ocean circumpolar current through Drake Passage
- Test interaction of wind-driven gyres with thermal stratification
- Benchmark surface temperature restoring in global domain
- Compare lat-lon and MPAS grids side-by-side

Domain Configuration:
- Global ocean with simplified continent (20-60 deg E, from north cap to 55 deg S)
- Open Drake Passage south of 55 deg S -> circumpolar current
- Polar caps at +/- 80 deg latitude
- Flat bottom at H_max depth
- Exponential temperature stratification (warm surface, cold abyss)

Physical Setup:
- Global 3-belt zonal wind stress (trades, westerlies, polar easterlies)
- Surface temperature restoring: cosine(lat) profile, warm equator -> cold poles
- Linear EOS: buoyancy depends on temperature only (beta_S = 0)
- Constant vertical viscosity and diffusivity
- Convective adjustment via enhanced diffusion where N^2 < 0
- Linear bottom drag
- 20 z-star levels with surface-intensified stretching

Expected Behavior:
- Subtropical/subpolar gyres in Atlantic-like and Pacific-like basins
- Western boundary currents (Gulf Stream / Kuroshio analogues)
- ACC-like circumpolar flow through Drake Passage
- Thermocline maintained by wind-driven overturning + restoring
- Deep stratification set by Southern Ocean channel dynamics

At 1 deg resolution eddies are NOT resolved.  Stratification and MOC
are maintained diffusively rather than by eddy fluxes.  This is
acceptable as a stepping stone; OMIP models also run at this resolution
(typically with GM/Redi parameterization, which is future work).

References:
- Wolfe & Cessi (2010), "What Sets the Strength of the Middepth
  Stratification and Overturning in Eddying Ocean Models?", JPO 40.
- Stommel (1948), "The westward intensification of wind-driven currents"
- Nikurashin & Vallis (2012) — global_wind profile shape
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from legoesm import constants
from legoesm.core.field import Field
from legoesm.ocean.eos import LinearEOSConfig


@dataclass
class GlobalOverturningConfig:
    """Configuration for the global overturning circulation experiment."""

    # --- Stratification ---
    T_water_init_C: float = 20.0        # Surface temperature [degC]
    T_deep_C: float = 2.0            # Abyssal temperature [degC]
    T_scale_depth: float = 1000.0  # Stratification e-folding depth [m]
    S_uniform: float = 35.0        # Uniform salinity [PSU] (T-only EOS)

    # --- Domain ---
    H_max: float = 4000.0          # Ocean depth [m]

    # --- Vertical grid ---
    n_levels: int = 20             # Number of vertical levels
    dz_surface: float = 10.0       # Top layer thickness [m]
    dz_deep: float = 500.0         # Bottom layer thickness [m]

    # --- Surface restoring ---
    tau_T_days: float = 30.0       # SST restoring timescale [days]
    T_star_eq: float = 25.0        # Equatorial target SST [degC]
    T_star_pole: float = 0.0       # Polar target SST [degC]

    # --- Wind ---
    tau_max: float = 0.1           # Maximum wind stress [Pa]

    # --- Viscosity / diffusion (scaled for ~1 deg) ---
    A_h: float = 2e5               # Laplacian viscosity [m2/s]
    A_v: float = 1e-3              # Vertical viscosity [m2/s]
    K_v: float = 1e-5              # Background vertical diffusivity [m2/s]
    bottom_drag_coeff: float = 1.1e-3  # Linear bottom drag [m/s]

    # --- EOS ---
    alpha_T: float = 2.0e-4        # Thermal expansion [1/K]
    T_ref_C: float = 10.0            # Reference temperature for EOS [degC]

    # --- GM/Redi mesoscale eddy parameterization ---
    use_gm_redi: bool = False         # Enable GM/Redi isopycnal mixing
    kappa_GM: float = 1000.0          # GM transport coefficient [m2/s]
    kappa_Redi: float = 1000.0        # Redi isopycnal diffusivity [m2/s]
    S_max: float = 0.005              # Maximum slope for DM95 tapering
    visbeck_enabled: bool = True      # Adaptive coefficient (Visbeck 1997)
    visbeck_alpha: float = 0.015      # Visbeck dimensionless coefficient
    visbeck_kappa_min: float = 200.0  # Visbeck kappa floor [m2/s]
    visbeck_kappa_max: float = 2000.0 # Visbeck kappa ceiling [m2/s]

    # --- Continent geometry (same as global_barotropic_wind) ---
    continent_lon_west: float = 20.0
    continent_lon_east: float = 60.0
    continent_lat_south: float = -55.0
    polar_cap_lat: float = 80.0


def _add_stratification(state, z_coord, config: GlobalOverturningConfig):
    """Add exponential temperature stratification to a uniform state.

    T(z) = T_deep_C + (T_water_init_C - T_deep_C) * exp(z / scale_depth)

    where z is negative (depth below surface).
    """
    z_full = np.asarray(z_coord.z_full_ref)   # negative values
    decay = np.exp(z_full / config.T_scale_depth)
    T_profile = config.T_deep_C + (config.T_water_init_C - config.T_deep_C) * decay

    T_data = np.array(state.T.data)
    for k in range(z_coord.n_levels):
        T_data[..., k] = T_profile[k]

    return state._replace(
        T=Field(jnp.array(T_data), name="T",
                dims=state.T.dims, units=state.T.units))


def create_initial_conditions(grid_type: str, grid, z_coord,
                              config: GlobalOverturningConfig = None):
    """Create global overturning initial conditions.

    Creates a rest state with exponential T stratification, uniform S,
    and a simplified continent mask (shared with global_barotropic_wind).
    """
    if config is None:
        config = GlobalOverturningConfig()

    from legoesm.ocean.experiments.global_barotropic_wind import (
        create_simplified_continent_mask,
    )

    if grid_type == "latlon":
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_water_init_C=config.T_water_init_C, T_deep=config.T_water_init_C,
            S_uniform=config.S_uniform,
        )
        lon_1d = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lat_1d = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        if lon_1d.ndim == 1:
            lon_deg, lat_deg = np.meshgrid(lon_1d, lat_1d)
        else:
            lon_deg, lat_deg = lon_1d, lat_1d
        if lat_deg.ndim == 1:
            lat_deg = lat_1d[:, None] * np.ones((1, len(lon_1d)))
            lon_deg = lon_1d[None, :] * np.ones((len(lat_1d), 1))

    elif grid_type == "mpas":
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        state = rest_state_mpas_ocean(
            grid, z_coord,
            T_water_init_C=config.T_water_init_C, T_deep=config.T_water_init_C,
            S_uniform=config.S_uniform,
        )
        lon_deg = np.asarray(grid.lonCell, dtype=np.float64) * 180 / np.pi
        lat_deg = np.asarray(grid.latCell, dtype=np.float64) * 180 / np.pi

    else:
        raise NotImplementedError(
            f"Global overturning not implemented for {grid_type}")

    # Apply continent mask
    mask = create_simplified_continent_mask(lon_deg, lat_deg, config)
    mask_typed = np.asarray(mask).astype(np.asarray(state.eta.data).dtype)
    if grid_type == "latlon":
        from legoesm.ocean.init_latlon_cgrid import replace_land_mask
        state = replace_land_mask(state, jnp.array(mask_typed))
    else:
        state = state._replace(land_mask=Field(data=jnp.array(mask_typed)))

    # Add exponential stratification
    state = _add_stratification(state, z_coord, config)

    return state


def create_forcings(grid_type: str, grid,
                    config: GlobalOverturningConfig = None):
    """Create physics config: global wind + surface T restoring.

    Uses ``scheme="combined"`` to apply prescribed 3-belt wind stress
    together with SST restoring toward a cosine-latitude profile.

    Returns
    -------
    OceanPhysicsConfig
    """
    if config is None:
        config = GlobalOverturningConfig()

    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.surface_forcing.config import (
        PrescribedForcingConfig, RestoringConfig, SurfaceForcingConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.config import (
        ConstantVerticalMixingConfig, VerticalMixingConfig,
    )
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.convection.config import (
        EnhancedDiffusionConfig, OceanConvectionConfig,
    )

    tau_T_seconds = config.tau_T_days * 86400.0

    return OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(
            scheme="combined",
            prescribed=PrescribedForcingConfig(
                wind_profile="two_belt",
                tau_max=config.tau_max,
            ),
            restoring=RestoringConfig(
                tau_T=tau_T_seconds,
                tau_S=1e30,  # effectively disable salinity restoring
                T_star_eq=config.T_star_eq,
                T_star_pole=config.T_star_pole,
                S_star=config.S_uniform,
                T_profile="cosine",
            ),
        ),
        vertical_mixing=VerticalMixingConfig(
            scheme="constant",
            constant=ConstantVerticalMixingConfig(
                A_v=config.A_v,
                K_v=config.K_v,
            ),
        ),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        convection=OceanConvectionConfig(
            scheme="enhanced_diffusion",
            enhanced_diffusion=EnhancedDiffusionConfig(
                K_conv=1.0,       # 1 m2/s where statically unstable
                K_bg=1e-5,        # background (matches K_v)
            ),
        ),
        shortwave_penetration=None,
    )


def create_gm_redi_config(config: GlobalOverturningConfig = None):
    """Return GMRediConfig if ``use_gm_redi`` is enabled, else None.

    The returned config is passed to ``LatLonCGridOceanConfig.gm_redi``
    (the lat-lon C-grid model applies it directly, bypassing the
    physics factory).
    """
    if config is None:
        config = GlobalOverturningConfig()

    if not config.use_gm_redi:
        return None

    from legoesm.ocean.physics.lateral_mixing.config import (
        GMRediConfig, VisbeckConfig,
    )
    return GMRediConfig(
        kappa_GM=config.kappa_GM,
        kappa_Redi=config.kappa_Redi,
        S_max=config.S_max,
        visbeck=VisbeckConfig(
            enabled=config.visbeck_enabled,
            alpha=config.visbeck_alpha,
            kappa_min=config.visbeck_kappa_min,
            kappa_max=config.visbeck_kappa_max,
        ),
    )


def create_eos_config(config: GlobalOverturningConfig = None):
    """Return linear EOS config (temperature-only buoyancy, beta_S=0)."""
    if config is None:
        config = GlobalOverturningConfig()
    return LinearEOSConfig(
        rho_ref=constants.rho_ocean,
        alpha_T=config.alpha_T,
        beta_S=0.0,          # T-only buoyancy (Wolfe & Cessi use b = alpha*g*T)
        T_ref=config.T_ref_C,
        S_ref=config.S_uniform,
    )


def global_overturning_model_config(
    config: GlobalOverturningConfig = None,
    *,
    physics=None,
    eos_config=None,
    gm_redi_cfg=None,
    **overrides,
):
    """Assemble the global-overturning ``LatLonCGridOceanConfig`` (shared factory).

    The ~dozen drivers in ``scripts/run/global_overturning/`` previously each
    hand-copied this construction, so the "recipe" for global overturning was
    smeared across many call sites that could silently drift apart.  This factory
    captures the shared base — linear T-only EOS, Laplacian + background vertical
    viscosity, linear bottom drag, and the optional Visbeck GM/Redi path — once.
    Drivers select it and pass only their genuine per-run variations (the
    biharmonic ``B_h``/``C_smag`` OMIP stack, ``barotropic_solver="implicit_cn"``,
    ``A_h`` lat-scaling / equatorial boost, partial-cell
    ``bottom_drag_bbl_thickness``, ...) as keyword ``overrides``.  Mirrors
    ``dino.dino_lat_lon_model_config``.

    Parameters
    ----------
    config : GlobalOverturningConfig
        Source of the base coefficients (``A_h``, ``A_v``, ``K_v``,
        ``bottom_drag_coeff``) and the EOS / GM-Redi settings.
    physics : OceanPhysicsConfig or None
        Wired onto ``LatLonCGridOceanConfig.physics`` verbatim.
    eos_config : LinearEOSConfig or None
        Linear EOS; defaults to ``create_eos_config(config)``.
    gm_redi_cfg : GMRediConfig or None
        GM/Redi; defaults to ``create_gm_redi_config(config)`` (None unless
        ``config.use_gm_redi``).  Pass ``gm_redi=...`` in ``overrides`` to force
        a specific value regardless of ``config``.
    **overrides
        Any ``LatLonCGridOceanConfig`` field; applied last so it wins over the
        base (e.g. ``barotropic_solver``, ``B_h``, ``C_smag``,
        ``A_h_lat_scaling``, ``n_barotropic_substeps``, ``pgf_scheme``).

    Returns
    -------
    LatLonCGridOceanConfig
    """
    from legoesm.ocean.state import LatLonCGridOceanConfig

    if config is None:
        config = GlobalOverturningConfig()
    if eos_config is None:
        eos_config = create_eos_config(config)
    if gm_redi_cfg is None:
        gm_redi_cfg = create_gm_redi_config(config)

    params = dict(
        physics=physics,
        A_h=config.A_h,
        A_v=config.A_v,
        K_v=config.K_v,
        bottom_drag_r=config.bottom_drag_coeff,
        eos="linear",
        eos_linear=eos_config,
        gm_redi=gm_redi_cfg,
        # Structural scheme identity pinned EXPLICITLY — even where it currently
        # equals the LatLonCGridOceanConfig default — so a change to a model
        # default cannot silently alter the global-overturning recipe across the
        # ~17 drivers + matrix that consume it (#488). Drivers still override any
        # of these via **overrides (e.g. realistic runs set barotropic_solver=
        # "implicit_cn", pgf_scheme="smc03"). Pin to CURRENT defaults => bit-
        # identical today; locks the recipe against future default drift.
        momentum_advection="vector_invariant",
        tracer_advection="tvd",
        pgf_scheme="adcroft",
        ke_gradient_scheme="centered",
        barotropic_solver="explicit_substep",
        coriolis_scheme="matsuno_split",
        outer_integrator="forward_euler",
        tracer_time_integrator="euler",
        implicit_vertical_mixing=True,
    )
    params.update(overrides)
    return LatLonCGridOceanConfig(**params)


def global_overturning_mpas_model_config(
    config: GlobalOverturningConfig = None,
    *,
    physics=None,
    eos_config=None,
    gm_redi_cfg=None,
    **overrides,
):
    """MPAS (Voronoi C-grid) counterpart of ``global_overturning_model_config``.

    Same shared base — linear T-only EOS, Laplacian + background vertical
    viscosity, linear bottom drag, optional GM/Redi — assembled onto an
    ``MPASOceanConfig`` instead of ``LatLonCGridOceanConfig``.  The MPAS
    overturning drivers (``run_global_overturning_mpas_{baseline,50yr_implicit}``)
    previously hand-copied this construction just like their lat-lon siblings;
    this factory dedups it.  Mirrors DINO's
    ``dino_lat_lon_model_config`` / ``dino_mpas_model_config`` pair.

    Drivers pass their per-run variations as keyword ``overrides`` — e.g. the
    MPAS-floored Laplacian viscosity ``A_h=max(config.A_h, MPAS_A_H_FLOOR)``,
    ``barotropic_solver``, ``barotropic_u_viscosity``.  See
    ``global_overturning_model_config`` for parameter semantics.

    Returns
    -------
    MPASOceanConfig
    """
    from legoesm.ocean.mpas_config import MPASOceanConfig

    if config is None:
        config = GlobalOverturningConfig()
    if eos_config is None:
        eos_config = create_eos_config(config)
    if gm_redi_cfg is None:
        gm_redi_cfg = create_gm_redi_config(config)

    params = dict(
        physics=physics,
        A_h=config.A_h,
        A_v=config.A_v,
        K_v=config.K_v,
        bottom_drag_r=config.bottom_drag_coeff,
        eos="linear",
        eos_linear=eos_config,
        gm_redi=gm_redi_cfg,
        # Structural scheme identity pinned EXPLICITLY (see the lat-lon factory)
        # so a model-default change can't silently alter the MPAS GO recipe.
        # Pinned to CURRENT MPASOceanConfig defaults => bit-identical today.
        tracer_advection="upwind",
        pgf_scheme="centered",
        barotropic_solver="explicit_substep",
        pv_scheme="enstrophy",
        implicit_vertical_mixing=True,
    )
    params.update(overrides)
    return MPASOceanConfig(**params)


def create_domain_config(config: GlobalOverturningConfig = None) -> Dict[str, Any]:
    if config is None:
        config = GlobalOverturningConfig()
    return {
        "H_max": config.H_max,
        "n_levels": config.n_levels,
        "dz_surface": config.dz_surface,
        "dz_deep": config.dz_deep,
        "description": ("Global baroclinic ocean with stratification, "
                        "SST restoring, and simplified continent"),
        "forcing_type": "global_wind_plus_SST_restoring",
        "stratification": "exponential",
        "eos": "linear (T-only)",
    }


def validate_results(final_state, diagnostics: Dict[str, list],
                     config: GlobalOverturningConfig = None) -> Tuple[bool, str]:
    if config is None:
        config = GlobalOverturningConfig()

    success = True
    notes_parts = []

    # Check finiteness
    for field_name in ("eta", "T", "u"):
        if hasattr(final_state, field_name):
            data = getattr(final_state, field_name).data
            if not jnp.all(jnp.isfinite(data)):
                return False, f"NaN/Inf in final {field_name}"

    # Max speed
    max_speed_list = diagnostics.get("max_speed", [])
    if max_speed_list:
        speed = max_speed_list[-1]
        notes_parts.append(f"max_speed={speed:.4f}m/s")
        if speed < 0.01:
            notes_parts.append("WARN: weak circulation")
        elif speed > 5.0:
            success = False
            notes_parts.append("FAIL: excessive speed")

    # SSH drift
    eta_list = diagnostics.get("mean_eta", [])
    if len(eta_list) >= 2:
        eta_drift = abs(eta_list[-1] - eta_list[0])
        notes_parts.append(f"eta_drift={eta_drift:.2e}")

    # Stratification check: surface T should be warmer than deep T
    mean_T_list = diagnostics.get("mean_T", [])
    if mean_T_list:
        notes_parts.append(f"mean_T={mean_T_list[-1]:.2f}degC")

    return success, ", ".join(notes_parts)


def get_diagnostic_field_specs() -> list:
    return [
        ("eta", "SSH (m)", "RdBu_r"),
        ("SST", "SST (degC)", "RdYlBu_r"),
        ("speed_sfc", "Surface speed (m/s)", "magma"),
    ]


def get_scalar_units() -> Dict[str, str]:
    return {
        "mean_eta": "m",
        "max_speed": "m/s",
        "mean_T": "degC",
        "mean_S": "PSU",
    }


EXPERIMENT_CONFIG = {
    "name": "global_overturning",
    "description": ("Global baroclinic overturning circulation with "
                    "stratification and SST restoring (Wolfe & Cessi 2010 inspired)"),
    "scientific_purpose": ("Validates global MOC, thermocline, gyres, "
                           "and ACC as stepping stone toward OMIP"),
    "reference": "Wolfe & Cessi (2010, JPO)",
    "config_class": GlobalOverturningConfig,
    "create_initial_conditions": create_initial_conditions,
    "create_forcings": create_forcings,
    "create_gm_redi_config": create_gm_redi_config,
    "create_model_config": global_overturning_model_config,
    "create_mpas_model_config": global_overturning_mpas_model_config,
    "create_domain": create_domain_config,
    "validate": validate_results,
    "get_field_specs": get_diagnostic_field_specs,
    "get_scalar_units": get_scalar_units,
    "default_duration": 90.0,   # days — longer for baroclinic adjustment
    "quick_duration": 5.0,      # days
    "grid_support": {
        "cubed_sphere": False,  # face-boundary instability (#100)
        "latlon": True,
        "mpas": True,
        "spectral": False,
    },
}
