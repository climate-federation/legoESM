"""Ocean biogeochemistry integration: step function and tendency computation.

Provides ``step_ocean_biogeochemistry()`` which advances the
biogeochemistry state by one timestep, computing:
1. Air-sea CO2 gas exchange (surface DIC source/sink)
2. NPZD source/sink terms (if enabled)
3. Forward-Euler update with non-negativity clipping

Advection and diffusion of the biogeochemistry tracers is handled by the
calling ocean model using the same operators as for T and S.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.biogeochemistry.config import (
    BiogeoConfig,
    OceanBiogeoState,
    BiogeoTendencies,
    AirSeaCO2Diagnostics,
)
from legoesm.ocean.biogeochemistry.gas_exchange import air_sea_co2_flux
from legoesm.ocean.biogeochemistry.npzd import (
    npzd_source_sink, par_profile,
    npzd_v2_source_sink, par_profile_v2)


def compute_biogeo_tendencies(
    state: OceanBiogeoState,
    T_degC: jnp.ndarray,
    S_psu: jnp.ndarray,
    dz_ref: jnp.ndarray,
    z_full_ref: jnp.ndarray,
    ocean_mask: jnp.ndarray,
    cfg: BiogeoConfig,
    U10: jnp.ndarray | None = None,
    PAR_surf: jnp.ndarray | None = None,
) -> tuple[BiogeoTendencies, AirSeaCO2Diagnostics]:
    """Compute biogeochemistry source/sink tendencies.

    Parameters
    ----------
    state : OceanBiogeoState
        Current biogeochemistry tracers.
    T_degC : array (..., nlev)
        Ocean temperature [degC].
    S_psu : array (..., nlev)
        Ocean salinity [PSU].
    dz_ref : array (nlev,)
        Reference layer thicknesses [m].
    z_full_ref : array (nlev,)
        Reference depths [m], negative.
    ocean_mask : array (...)
        1=ocean, 0=land.
    cfg : BiogeoConfig
    U10 : array (...), optional
        10-m wind speed [m/s]. If None, uses cfg.wind_speed.
    PAR_surf : array (...), optional
        Surface PAR [W/m^2]. If None, uses 200 W/m^2 default.

    Returns
    -------
    tendencies : BiogeoTendencies
        Source/sink tendencies for all tracers.
    co2_diag : AirSeaCO2Diagnostics
        Air-sea CO2 flux diagnostics.
    """
    shape = state.DIC.shape
    mask_3d = ocean_mask[..., jnp.newaxis] if ocean_mask.ndim < state.DIC.ndim else ocean_mask
    # Pin default forcing dtype to match the BGC state precision so a
    # missing forcing input does not silently widen the column path.
    _state_dtype = state.DIC.dtype

    # Default forcing if not provided
    if U10 is None:
        U10 = jnp.full(ocean_mask.shape, cfg.wind_speed, dtype=_state_dtype)
    if PAR_surf is None:
        PAR_surf = jnp.full(ocean_mask.shape, 200.0, dtype=_state_dtype)

    # ---- 1. Air-sea CO2 flux (surface layer only) ----
    DIC_surf = state.DIC[..., 0]
    ALK_surf = state.ALK[..., 0]
    T_surf = T_degC[..., 0]
    S_surf = S_psu[..., 0]

    co2_diag = air_sea_co2_flux(
        DIC_surf, ALK_surf, T_surf, S_surf, U10,
        pCO2_atm=cfg.pCO2_atm,
    )

    # Apply flux to surface layer: dDIC/dt = F_CO2 / dz[0]
    dz_surface = dz_ref[0]
    dDIC_gas = co2_diag.flux_co2 / jnp.clip(dz_surface, 1.0, None)

    # Initialize tendencies — pin to state precision so the BGC tendency
    # struct does not silently widen to f64 under x64 mode.
    dDIC_dt = jnp.zeros(shape, dtype=_state_dtype)
    dALK_dt = jnp.zeros(shape, dtype=_state_dtype)

    # Add gas exchange to surface layer
    dDIC_dt = dDIC_dt.at[..., 0].add(dDIC_gas * ocean_mask)

    # ---- 2. NPZD source/sink (if scheme == "npzd") ----
    dNO3_dt = None
    dPhyto_dt = None
    dZoo_dt = None
    dDet_dt = None

    if cfg.scheme == "npzd" and state.NO3 is not None:
        # Compute PAR profile with self-shading
        PAR = par_profile(PAR_surf, z_full_ref, state.Phyto, dz_ref, cfg)

        (dNO3_bio, dPhyto_bio, dZoo_bio, dDet_bio,
         dDIC_bio, dALK_bio) = npzd_source_sink(
            state.NO3, state.Phyto, state.Zoo, state.Det,
            state.DIC, state.ALK, T_degC, PAR, dz_ref, cfg,
        )

        dDIC_dt = dDIC_dt + dDIC_bio * mask_3d
        dALK_dt = dALK_dt + dALK_bio * mask_3d
        dNO3_dt = dNO3_bio * mask_3d
        dPhyto_dt = dPhyto_bio * mask_3d
        dZoo_dt = dZoo_bio * mask_3d
        dDet_dt = dDet_bio * mask_3d

    # ---- 3. NPZDv2 source/sink ----
    dNH4_dt = dPO4_dt = dSi_dt = dFe_dt = None
    dPd_dt = dPn_dt = dChl_d_dt = dChl_n_dt = None
    dDON_dt = dDOP_dt = dPOC_dt = dDOC_dt = None

    if cfg.scheme == "npzd_v2" and state.Pd is not None:
        PAR = par_profile_v2(PAR_surf, state.Chl_d, state.Chl_n, dz_ref, cfg)

        (dNO3_v2, dNH4_v2, dPO4_v2, dSi_v2, dFe_v2,
         dPd_v2, dPn_v2, dChl_d_v2, dChl_n_v2,
         dZoo_v2, dDet_v2, dDON_v2, dDOP_v2,
         dDIC_v2, dPOC_v2, dDOC_v2, dALK_v2) = npzd_v2_source_sink(
            state.NO3, state.NH4, state.PO4, state.Si, state.Fe,
            state.Pd, state.Pn, state.Chl_d, state.Chl_n,
            state.Zoo, state.Det, state.DON, state.DOP,
            state.DIC, state.POC, state.DOC, state.ALK,
            T_degC, PAR, dz_ref, cfg,
        )

        dDIC_dt   = dDIC_dt + dDIC_v2  * mask_3d
        dALK_dt   = dALK_dt + dALK_v2  * mask_3d
        dNO3_dt   = dNO3_v2  * mask_3d
        dNH4_dt   = dNH4_v2  * mask_3d
        dPO4_dt   = dPO4_v2  * mask_3d
        dSi_dt    = dSi_v2   * mask_3d
        dFe_dt    = dFe_v2   * mask_3d
        dPd_dt    = dPd_v2   * mask_3d
        dPn_dt    = dPn_v2   * mask_3d
        dChl_d_dt = dChl_d_v2 * mask_3d
        dChl_n_dt = dChl_n_v2 * mask_3d
        dZoo_dt   = dZoo_v2  * mask_3d
        dDet_dt   = dDet_v2  * mask_3d
        dDON_dt   = dDON_v2  * mask_3d
        dDOP_dt   = dDOP_v2  * mask_3d
        dPOC_dt   = dPOC_v2  * mask_3d
        dDOC_dt   = dDOC_v2  * mask_3d

    # Mask land
    dDIC_dt = dDIC_dt * mask_3d
    dALK_dt = dALK_dt * mask_3d

    tendencies = BiogeoTendencies(
        dDIC_dt=dDIC_dt,
        dALK_dt=dALK_dt,
        dNO3_dt=dNO3_dt,
        dPhyto_dt=dPhyto_dt,
        dZoo_dt=dZoo_dt,
        dDet_dt=dDet_dt,
        dNH4_dt=dNH4_dt,
        dPO4_dt=dPO4_dt,
        dSi_dt=dSi_dt,
        dFe_dt=dFe_dt,
        dPd_dt=dPd_dt,
        dPn_dt=dPn_dt,
        dChl_d_dt=dChl_d_dt,
        dChl_n_dt=dChl_n_dt,
        dDON_dt=dDON_dt,
        dDOP_dt=dDOP_dt,
        dPOC_dt=dPOC_dt,
        dDOC_dt=dDOC_dt,
    )

    return tendencies, co2_diag


def step_ocean_biogeochemistry(
    state: OceanBiogeoState,
    T_degC: jnp.ndarray,
    S_psu: jnp.ndarray,
    dz_ref: jnp.ndarray,
    z_full_ref: jnp.ndarray,
    ocean_mask: jnp.ndarray,
    dt: float,
    cfg: BiogeoConfig,
    U10: jnp.ndarray | None = None,
    PAR_surf: jnp.ndarray | None = None,
) -> tuple[OceanBiogeoState, AirSeaCO2Diagnostics]:
    """Advance biogeochemistry state by one timestep (source/sink only).

    This function handles the biogeochemistry-specific physics
    (gas exchange, NPZD biology). Advection and diffusion of the
    biogeo tracers should be handled separately by the ocean dynamics
    using the same operators as for T and S.

    Parameters
    ----------
    state : OceanBiogeoState
        Current state.
    T_degC : array (..., nlev)
        Temperature [degC].
    S_psu : array (..., nlev)
        Salinity [PSU].
    dz_ref : array (nlev,)
        Reference layer thickness [m].
    z_full_ref : array (nlev,)
        Reference depths [m], negative.
    ocean_mask : array (...)
        1=ocean, 0=land.
    dt : float
        Timestep [s].
    cfg : BiogeoConfig
    U10 : array, optional
        Wind speed [m/s].
    PAR_surf : array, optional
        Surface PAR [W/m^2].

    Returns
    -------
    state_new : OceanBiogeoState
        Updated state after source/sink physics.
    co2_diag : AirSeaCO2Diagnostics
        Diagnostics.
    """
    tend, co2_diag = compute_biogeo_tendencies(
        state, T_degC, S_psu, dz_ref, z_full_ref,
        ocean_mask, cfg, U10, PAR_surf,
    )

    # Forward-Euler update with smooth non-negativity (softplus).
    # softplus(x, alpha) ≈ max(x, 0) but preserves AD gradients and
    # does not silently zero-out overshoots the way hard clip does.
    _alpha = 1.0e-6  # smoothing scale [mol/m^3]; tight but differentiable
    def _soft_pos(x):
        return _alpha * jnp.logaddexp(x / _alpha, 0.0)

    DIC_new = _soft_pos(state.DIC + dt * tend.dDIC_dt)
    ALK_new = _soft_pos(state.ALK + dt * tend.dALK_dt)

    NO3_new = None
    Phyto_new = None
    Zoo_new = None
    Det_new = None

    if cfg.scheme == "npzd" and state.NO3 is not None:
        NO3_new = _soft_pos(state.NO3 + dt * tend.dNO3_dt)
        Phyto_new = _soft_pos(state.Phyto + dt * tend.dPhyto_dt)
        Zoo_new = _soft_pos(state.Zoo + dt * tend.dZoo_dt)
        Det_new = _soft_pos(state.Det + dt * tend.dDet_dt)

    # npzd_v2 tracer updates
    NH4_new = Pd_new = Pn_new = None
    Chl_d_new = Chl_n_new = None
    PO4_new = Si_new = Fe_new = None
    DON_new = DOP_new = POC_new = DOC_new = None

    if cfg.scheme == "npzd_v2" and state.Pd is not None:
        NO3_new   = _soft_pos(state.NO3   + dt * tend.dNO3_dt)
        NH4_new   = _soft_pos(state.NH4   + dt * tend.dNH4_dt)
        PO4_new   = _soft_pos(state.PO4   + dt * tend.dPO4_dt)
        Si_new    = _soft_pos(state.Si    + dt * tend.dSi_dt)
        # Fe: diagnostic — prescribe depth profile rather than prognose
        # Parekh et al. (2005): ~0.1 nM surface, ~0.6 nM deep
        _fe_surf = 0.1e-9   # mol/m³
        _fe_deep = 0.6e-9   # mol/m³
        _nlev = state.Pd.shape[-1]
        _z_norm = jnp.linspace(0.0, 1.0, _nlev)
        _fe_prof = _fe_surf + (_fe_deep - _fe_surf) * _z_norm
        Fe_new = jnp.broadcast_to(
            _fe_prof.reshape((1,) * (state.Pd.ndim - 1) + (_nlev,)),
            state.Pd.shape,
        )
        Pd_new    = _soft_pos(state.Pd    + dt * tend.dPd_dt)
        Pn_new    = _soft_pos(state.Pn    + dt * tend.dPn_dt)
        Chl_d_new = _soft_pos(state.Chl_d + dt * tend.dChl_d_dt)
        Chl_n_new = _soft_pos(state.Chl_n + dt * tend.dChl_n_dt)
        # Safety clip: theta must stay in [theta_min, theta_max]
        Chl_d_new = jnp.clip(Chl_d_new,
                             Pd_new * cfg.theta_min_d,
                             Pd_new * cfg.theta_max_d)
        Chl_n_new = jnp.clip(Chl_n_new,
                             Pn_new * cfg.theta_min_n,
                             Pn_new * cfg.theta_max_n)
        Zoo_new   = _soft_pos(state.Zoo   + dt * tend.dZoo_dt)
        Det_new   = _soft_pos(state.Det   + dt * tend.dDet_dt)
        DON_new   = _soft_pos(state.DON   + dt * tend.dDON_dt)
        DOP_new   = _soft_pos(state.DOP   + dt * tend.dDOP_dt)
        POC_new   = _soft_pos(state.POC   + dt * tend.dPOC_dt)
        DOC_new   = _soft_pos(state.DOC   + dt * tend.dDOC_dt)

    # ---- Nutrient nudging toward WOA climatology (tau=365 days) ----
    # Prevents surface nutrient collapse from insufficient upwelling
    if cfg.scheme == 'npzd_v2' and getattr(cfg, 'nudge_nutrients', False):
        tau_nut = 365.0 * 86400.0   # 1-year relaxation
        if NO3_new is not None and cfg.NO3_target is not None:
            NO3_new = NO3_new + dt * (cfg.NO3_target - NO3_new) / tau_nut
        if PO4_new is not None and cfg.PO4_target is not None:
            PO4_new = PO4_new + dt * (cfg.PO4_target - PO4_new) / tau_nut
        if Si_new is not None and cfg.Si_target is not None:
            Si_new  = Si_new  + dt * (cfg.Si_target  - Si_new)  / tau_nut

    # ---- Nutrient nudging toward WOA climatology ----
    if (cfg.scheme == 'npzd_v2' and cfg.nudge_nutrients
            and cfg.NO3_target is not None):
        import jax.numpy as _jnp
        _tau = cfg.tau_nudge_days * 86400.0
        if NO3_new is not None:
            NO3_new = NO3_new + dt * (_jnp.array(cfg.NO3_target) - NO3_new) / _tau
        if PO4_new is not None and cfg.PO4_target is not None:
            PO4_new = PO4_new + dt * (_jnp.array(cfg.PO4_target) - PO4_new) / _tau
        if Si_new is not None and cfg.Si_target is not None:
            Si_new  = Si_new  + dt * (_jnp.array(cfg.Si_target)  - Si_new)  / _tau

    return OceanBiogeoState(
        DIC=DIC_new,
        ALK=ALK_new,
        NO3=NO3_new,
        Phyto=Phyto_new,
        Zoo=Zoo_new,
        Det=Det_new,
        NH4=NH4_new,
        PO4=PO4_new,
        Si=Si_new,
        Fe=Fe_new,
        Pd=Pd_new,
        Pn=Pn_new,
        Chl_d=Chl_d_new,
        Chl_n=Chl_n_new,
        DON=DON_new,
        DOP=DOP_new,
        POC=POC_new,
        DOC=DOC_new,
    ), co2_diag
