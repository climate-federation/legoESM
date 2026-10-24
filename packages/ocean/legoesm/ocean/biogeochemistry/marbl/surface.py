"""MARBL surface fluxes (marbl_surface_flux_mod, marbl_nhx_surface_emis_mod).

Fluxes are in mmol/m^2/s (meq/m^2/s for ALK), positive into the ocean.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.biogeochemistry.carbonate import (
    carbonate_coefficients,
    carbonate_speciation,
)
from legoesm.ocean.biogeochemistry.gas_exchange import (
    o2_saturation,
    piston_velocity,
    schmidt_number_co2,
    schmidt_number_o2,
)
from legoesm.ocean.biogeochemistry.marbl.config import (
    MarblConfig,
    MarblSurfaceDiagnostics,
    MarblSurfaceForcing,
    MarblTracers,
)

# mol/kg -> mmol/m^3 with MARBL's fixed seawater density
MASS_TO_VOL = constants.rho_sw_marbl * 1.0e3
# kg dust -> g, mol -> mmol (MARBL unit_system mks: mass2g * mol_prefix)
_KG_TO_G = 1.0e3
_MOL_TO_MMOL = 1.0e3


def schmidt_number_nh3_sea(T_degC: jnp.ndarray) -> jnp.ndarray:
    """Schmidt number of NH3 in seawater (Johnson 2010 values, 4th-order fit)."""
    T = jnp.clip(T_degC, -2.0, 40.0)
    return 1.809018e3 + T * (-1.199578e2 + T * (4.369922 + T * (-8.824558e-2 + T * 7.377934e-4)))


def schmidt_number_nh3_air(T_degC: jnp.ndarray, P_atm: jnp.ndarray) -> jnp.ndarray:
    """Schmidt number of NH3 in air (Johnson 2010 values, 4th-order fit), scales as P^2."""
    T = jnp.clip(T_degC, -2.0, 40.0)
    sc = 7.537648e-1 + T * (4.728550e-4 + T * (-2.546799e-5 + T * (3.187285e-7 + T * (-1.985024e-9))))
    return P_atm * P_atm * sc


def nhx_surface_emission(
    NH4: jnp.ndarray, pH: jnp.ndarray, sst: jnp.ndarray, sss: jnp.ndarray,
    u10_sqr: jnp.ndarray, P_atm: jnp.ndarray, ifrac: jnp.ndarray,
) -> jnp.ndarray:
    """Sea-to-air NHx emission [mmol N/m^2/s] (Paulot et al. 2015)."""
    u10 = jnp.sqrt(u10_sqr)
    # water-side transfer velocity (Nightingale et al. 2000), cm/h -> m/s
    kw = (1.0e-2 / 3600.0) * u10 * (0.061 + 0.24 * u10) * jnp.sqrt(600.0 / schmidt_number_nh3_sea(sst))
    # air-side transfer velocity (Johnson 2010, eqs. 10, 11, 15), m/s
    sc_air = schmidt_number_nh3_air(sst, P_atm)
    ustar_div_u10 = jnp.sqrt(6.1e-4 + 6.3e-5 * u10)
    ustar = ustar_div_u10 * u10
    kg = 1.0e-3 + ustar / (13.3 * jnp.sqrt(sc_air) + 1.0 / ustar_div_u10 - 5.0
                           + jnp.log(sc_air) / (2.0 * constants.kappa_vk))
    # effective Henry constant of NHx (Bell et al. 2008 pKa)
    pKa = 10.0423 - 0.0315536 * sst + 0.003071 * sss
    inv_T = 1.0 / (sst + constants.T_freeze)
    H_nhx = (constants.T_freeze / 17.93) * inv_T * jnp.exp(9.7 - 4092.0 * inv_T)
    Hstar = H_nhx / (1.0 + 10.0 ** (pKa - pH))
    K = kg * kw / (kw + Hstar * kg)
    return (1.0 - ifrac) * K * Hstar * jnp.maximum(NH4, 0.0)


def _co2_terms(DIC, ALK, PO4, SiO3, coeffs, xco2, P_atm, n_iter):
    spec = carbonate_speciation(DIC / MASS_TO_VOL, ALK / MASS_TO_VOL,
                                PO4 / MASS_TO_VOL, SiO3 / MASS_TO_VOL, coeffs, n_iter=n_iter)
    co2star_air = xco2 * 1.0e-6 * coeffs.ff * P_atm
    dco2star = (co2star_air - spec.H2CO3) * MASS_TO_VOL
    pco2surf = spec.H2CO3 / coeffs.ff * 1.0e6
    return spec.pH, dco2star, pco2surf


def marbl_surface_fluxes(
    surf: MarblTracers,
    forcing: MarblSurfaceForcing,
    cfg: MarblConfig,
    n_iter: int = 50,
) -> tuple[MarblTracers, MarblSurfaceDiagnostics]:
    """Surface fluxes of all 32 tracers from top-layer values ``surf`` (each (...,)).

    Tracers without a surface flux get zeros.
    """
    f = forcing
    zeros = jnp.zeros_like(surf.DIC)

    sc_co2 = schmidt_number_co2(f.sst)
    sc_o2 = schmidt_number_o2(f.sst)
    pv_co2 = piston_velocity(f.u10_sqr, sc_co2, f.ifrac)
    pv_o2 = piston_velocity(f.u10_sqr, sc_o2, f.ifrac)

    o2sat = f.atm_press * o2_saturation(f.sst, f.sss) * 1.0e3   # mol/m^3 -> mmol/m^3
    flux_o2 = pv_o2 * (o2sat - surf.O2)

    coeffs = carbonate_coefficients(f.sst, f.sss)
    pH, dco2star, pco2surf = _co2_terms(surf.DIC, surf.ALK, surf.PO4, surf.SiO3,
                                        coeffs, f.xco2, f.atm_press, n_iter)
    _, dco2star_alt, _ = _co2_terms(surf.DIC_ALT_CO2, surf.ALK_ALT_CO2, surf.PO4, surf.SiO3,
                                    coeffs, f.xco2_alt_co2, f.atm_press, n_iter)
    flux_co2 = pv_co2 * dco2star
    flux_alt_co2 = pv_co2 * dco2star_alt

    nhx = nhx_surface_emission(surf.NH4, pH, f.sst, f.sss, f.u10_sqr, f.atm_press, f.ifrac)

    dust_to_P = (cfg.dust_P_frac * cfg.dust_P_solubility / constants.M_P) * _KG_TO_G * _MOL_TO_MMOL
    dust_to_Si = (cfg.dust_Si_frac * cfg.dust_Si_solubility / constants.M_Si) * _KG_TO_G * _MOL_TO_MMOL
    F_NO3 = f.nox_flux + zeros
    F_NH4 = f.nhy_flux - nhx
    F_ALK = F_NH4 - F_NO3

    fluxes = MarblTracers(**{name: zeros for name in MarblTracers._fields})._replace(
        O2=flux_o2,
        DIC=flux_co2,
        DIC_ALT_CO2=flux_alt_co2,
        Fe=f.iron_flux + zeros,
        PO4=dust_to_P * f.dust_flux + zeros,
        SiO3=dust_to_Si * f.dust_flux + zeros,
        NO3=F_NO3,
        NH4=F_NH4,
        ALK=F_ALK,
        ALK_ALT_CO2=F_ALK,
    )
    diags = MarblSurfaceDiagnostics(
        pH_surf=pH, pCO2surf=pco2surf, dpCO2=pco2surf - f.xco2 * f.atm_press,
        flux_co2=flux_co2, flux_o2=flux_o2, o2sat=o2sat, nhx_surface_emis=nhx,
        pv_co2=pv_co2, pv_o2=pv_o2,
    )
    return fluxes, diags
