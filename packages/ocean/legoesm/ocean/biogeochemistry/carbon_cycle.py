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
from legoesm.ocean.biogeochemistry.npzd import npzd_source_sink, par_profile
from legoesm.ocean.biogeochemistry.npzd import npzd_v2_source_sink, par_profile_v2
from legoesm.ocean.biogeochemistry.gas_exchange import air_sea_o2_flux


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

    # ---- 3. npzd_v2 source/sink ----
    _v2 = {}
    if cfg.scheme == "npzd_v2" and state.Pd is not None:
        PAR = par_profile_v2(PAR_surf, state.Chl_d, state.Chl_n, dz_ref, cfg)
        out = npzd_v2_source_sink(
            state.NO3, state.NH4, state.PO4, state.Si, state.Fe,
            state.Pd, state.Pn, state.Chl_d, state.Chl_n,
            state.Zoo, state.Det, state.DON, state.DOP,
            state.DIC, state.POC, state.DOC, state.ALK, state.O2,
            T_degC, PAR, dz_ref, cfg,
        )
        names = ("dNO3_dt", "dNH4_dt", "dPO4_dt", "dSi_dt", "dFe_dt",
                 "dPd_dt", "dPn_dt", "dChl_d_dt", "dChl_n_dt",
                 "dZoo_dt", "dDet_dt", "dDON_dt", "dDOP_dt",
                 "dDIC_bio", "dPOC_dt", "dDOC_dt", "dALK_bio", "dO2_dt")
        _v2 = {k: v * mask_3d for k, v in zip(names, out)}
        dDIC_dt = dDIC_dt + _v2.pop("dDIC_bio")
        dALK_dt = dALK_dt + _v2.pop("dALK_bio")
        dNO3_dt = _v2.pop("dNO3_dt")
        dZoo_dt = _v2.pop("dZoo_dt")
        dDet_dt = _v2.pop("dDet_dt")
        # aeolian iron deposition into the surface level [mol Fe/m^3/s]
        _v2["dFe_dt"] = _v2["dFe_dt"].at[..., 0].add(
            (cfg.Fe_dust / 86400.0) / jnp.clip(dz_ref[0], 1.0, None) * ocean_mask)
        # air-sea O2 exchange at the surface; U10 defaults to cfg.wind_speed
        _U10 = U10 if U10 is not None else jnp.full_like(T_degC[..., 0], cfg.wind_speed)
        _fo2 = air_sea_o2_flux(state.O2[..., 0], T_degC[..., 0], S_psu[..., 0], _U10,
                               P_atm=cfg.O2_atm / 0.2095)
        _v2["dO2_dt"] = _v2["dO2_dt"].at[..., 0].add(
            _fo2 / jnp.clip(dz_ref[0], 1.0, None) * ocean_mask)

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
        **_v2,
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

    _v2_new = {}
    if cfg.scheme == "npzd_v2" and state.Pd is not None:
        # Positivity for the v2 tracers: a plain clip, NOT _soft_pos.  The
        # softplus above has a fixed scale alpha = 1e-6 mol/m^3, so for any
        # tracer at or below that concentration (Fe ~1e-10, Chl ~1e-6 kg/m^3,
        # NH4 ~1e-6) it injects ~0.7*alpha EVERY step and the tracer runs away
        # (measured: Fe 5e-11 -> 1.8e-6 in 48 steps).  The clip is exact and
        # differentiable almost everywhere; v1 tracers keep _soft_pos.
        def _pos(x):
            return jnp.maximum(x, 0.0)
        NO3_new = _pos(state.NO3 + dt * tend.dNO3_dt)
        Zoo_new = _pos(state.Zoo + dt * tend.dZoo_dt)
        Det_new = _pos(state.Det + dt * tend.dDet_dt)
        for f in ("NH4", "PO4", "Si", "Fe", "Pd", "Pn", "Chl_d", "Chl_n",
                  "DON", "DOP", "POC", "DOC", "O2"):
            _v2_new[f] = _pos(getattr(state, f) + dt * getattr(tend, f"d{f}_dt"))
        # Chl:N kept inside [theta_min, theta_max]
        _v2_new["Chl_d"] = jnp.clip(_v2_new["Chl_d"], _v2_new["Pd"] * cfg.theta_min_d,
                                    _v2_new["Pd"] * cfg.theta_max_d)
        _v2_new["Chl_n"] = jnp.clip(_v2_new["Chl_n"], _v2_new["Pn"] * cfg.theta_min_n,
                                    _v2_new["Pn"] * cfg.theta_max_n)
        # optional nudging of macronutrients toward a climatology
        if cfg.nudge_nutrients and cfg.NO3_target is not None:
            _r = dt / (cfg.tau_nudge_days * 86400.0)
            NO3_new = NO3_new + _r * (jnp.asarray(cfg.NO3_target) - NO3_new)
            if cfg.PO4_target is not None:
                _v2_new["PO4"] = _v2_new["PO4"] + _r * (jnp.asarray(cfg.PO4_target) - _v2_new["PO4"])
            if cfg.Si_target is not None:
                _v2_new["Si"] = _v2_new["Si"] + _r * (jnp.asarray(cfg.Si_target) - _v2_new["Si"])

    return OceanBiogeoState(
        DIC=DIC_new,
        ALK=ALK_new,
        NO3=NO3_new,
        Phyto=Phyto_new,
        Zoo=Zoo_new,
        Det=Det_new,
        **_v2_new,
    ), co2_diag
