"""NPZD ecosystem model coupled to the inorganic carbon cycle.

A Nutrient-Phytoplankton-Zooplankton-Detritus model following the
Fasham et al. (1990) / Oschlies & Garçon (1999) framework with
coupling to DIC, alkalinity, and air-sea CO2 exchange.

Processes:
1. Light-limited, nutrient-limited phytoplankton growth
2. Zooplankton grazing (Holling type II)
3. Phytoplankton and zooplankton mortality
4. Detritus sinking and remineralization
5. Stoichiometric coupling to DIC/ALK via Redfield ratios
6. Implicit CaCO3: production ∝ particulate organic (detritus) formation,
   dissolved lower in the same column (OCMIP-2 / MITgcm ``car_flux.F``)
7. Sea floor: detritus and CaCO3 reaching the bottom wet cell are returned
   to the water there (3-D wet mask)

All functions are JAX-compatible (differentiable, JIT-friendly).

References
----------
- Fasham, M. J. R., et al. (1990). A nitrogen-based model of plankton
  dynamics in the oceanic mixed layer. J. Mar. Res., 48, 591-639.
- Oschlies, A. & Garçon, V. (1999). An eddy-permitting coupled physical-
  biological model of the North Atlantic. Global Biogeochem. Cycles, 13.
- Najjar, R. & Orr, J. (1999). Biotic-HOWTO, OCMIP-2. CaCO3 port follows
  MITgcm ``pkg/dic/car_flux.F`` + ``phos_flux.F`` + ``dic_biotic_forcing.F``
  (github.com/MITgcm/MITgcm, commit c3f71ac8).
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.biogeochemistry.config import BiogeoConfig

__physics_contract__ = {
    "summary": (
        "NPZD ecosystem source/sink terms with Redfield carbon/alkalinity "
        "coupling. CaCO3 is implicit (no tracer): it forms at rain ratio x "
        "detritus formation in the production levels and dissolves lower in "
        "the same column with an exponential profile; whatever reaches the "
        "bottom wet cell dissolves there (MITgcm car_flux.F). Detritus that "
        "sinks out of the bottom wet cell is remineralised in that cell."
    ),
    "inputs": {
        "NO3": "mol N/m^3", "Phyto": "mol N/m^3", "Zoo": "mol N/m^3",
        "Det": "mol N/m^3", "DIC": "mol C/m^3", "ALK": "mol eq/m^3",
        "T_degC": "degC", "PAR": "W/m^2", "dz_ref": "m", "wet_mask": "1",
    },
    "outputs": {
        "dNO3_dt": "mol N/m^3/s", "dPhyto_dt": "mol N/m^3/s",
        "dZoo_dt": "mol N/m^3/s", "dDet_dt": "mol N/m^3/s",
        "dDIC_dt": "mol C/m^3/s", "dALK_dt": "mol eq/m^3/s",
    },
    "sign_convention": (
        "Depth positive DOWN from the surface; sinking fluxes positive "
        "downward. CaCO3 formation removes 1 DIC and 2 ALK per mol; "
        "dissolution returns them. Organic N uptake raises ALK by -dNO3."
    ),
    # Column integrals sum_k dz*(.) of these tendencies vanish exactly (no
    # air-sea exchange inside this function): carbon DIC + R_CN*(P+Z+D),
    # nitrogen NO3+P+Z+D, alkalinity ALK + NO3.
    "conserves": ["carbon", "nitrogen", "alkalinity"],
    "differentiable": True,
    "reference": (
        "Fasham+ (1990) J. Mar. Res. 48; Najjar & Orr (1999) OCMIP-2 "
        "Biotic-HOWTO; MITgcm pkg/dic car_flux.F, phos_flux.F, "
        "dic_biotic_forcing.F, dic_init_fixed.F (commit c3f71ac8)."
    ),
    "idealized_test": (
        "tests/ocean/unit/test_npzd_caco3_pool.py: x64 column C/N/ALK "
        "closure on a 3-D masked fixture, term-by-term match to a literal "
        "transcription of car_flux.F, sign test, check_grads order 2."
    ),
}


def _interface_depths(dz: jnp.ndarray) -> jnp.ndarray:
    """Interface depths [m], positive down, shape (nlev+1,), 0 at surface."""
    return jnp.concatenate([jnp.zeros((1,), dz.dtype), jnp.cumsum(dz)])


def _production_levels(dz: jnp.ndarray, zcrit: float) -> jnp.ndarray:
    """Levels where CaCO3 forms (MITgcm dic_init_fixed.F ``nlev``).

    Oracle: ``k=1; DO WHILE (k<Nr .AND. rC(k)>=-zcrit) k=k+1; nlev=k`` —
    levels 1..nlev, i.e. every level down to and INCLUDING the first one
    whose centre is deeper than ``zcrit``. Static depth selection (not
    traced state), so ported exactly. Returns a 0/1 array, shape (nlev,).
    """
    z_i = _interface_depths(dz)
    z_c = 0.5 * (z_i[:-1] + z_i[1:])
    above = jnp.concatenate([jnp.ones((1,), dz.dtype),
                             jnp.cumprod(z_c[:-1] <= zcrit).astype(dz.dtype)])
    return above


def implicit_caco3_dissolution(
    production: jnp.ndarray,
    dz: jnp.ndarray,
    wet_mask: jnp.ndarray,
    remin_depth: float,
) -> jnp.ndarray:
    """Implicit CaCO3 dissolution [mol C/m^3/s] (MITgcm ``car_flux.F``).

    The CaCO3 formed in layer k leaves through its bottom interface as a
    flux ``E_k = production_k * dz_k`` that decays as
    ``exp(-(z - z_bot,k)/remin_depth)``; each lower layer dissolves the flux
    divergence across it. The flux is cut at the first dry cell (or the
    array bottom), so everything left dissolves in the bottom wet cell —
    the column integral of the returned rate times ``dz`` equals that of
    ``production`` over the wet cells contiguous from the surface
    (production below the first dry cell is ignored). ``dz`` must be > 0.

    Parameters
    ----------
    production : array (..., nlev)  CaCO3 formation [mol C/m^3/s], >= 0.
    dz : array (nlev,)  Layer thickness [m].
    wet_mask : array (..., nlev)  1 = ocean cell, 0 = below sea floor/land.
    remin_depth : float  e-folding depth [m] (oracle ``zca``).

    Oracle (``tests/ocean/unit/test_npzd_caco3_mitgcm_oracle.py``): matches
    compiled MITgcm ``CAR_FLUX`` (c3f71ac8) to 1e-12, and
    ``_production_levels`` matches its ``nlev``.  KNOWN DIFFERENCE: ``dz`` is
    one profile for every column, with no partial cells.  In MITgcm only the
    bottom wet cell is partial (z-level partial steps).  The flux arriving
    from above dissolves there over ``drF*hFacC`` in MITgcm and over the full
    thickness here, so that part of the rate is ``hFacC`` times MITgcm's;
    CaCO3 formed in that cell dissolves in place in both (pinned at
    ``hFacC = 0.5``, with and without production in the partial cell).
    """
    z_i = _interface_depths(dz)
    z_bot = z_i[1:]
    # decay[k, j] = exp(-(z_bot[j] - z_bot[k]) / L) for j >= k, else 0.
    dzb = z_bot[None, :] - z_bot[:, None]
    lev = jnp.arange(dz.shape[-1])
    upper = lev[:, None] <= lev[None, :]
    decay = jnp.where(upper, jnp.exp(-jnp.where(upper, dzb, 0.0) / remin_depth), 0.0)
    # Wet cells contiguous from the surface; the sea floor is the first dry
    # cell (or the array bottom).
    wet_contig = jnp.cumprod(wet_mask.astype(production.dtype), axis=-1)
    export = production * dz * wet_contig
    # Flux at the bottom interface of layer j before the sea-floor cut.
    s_bot = jnp.einsum("...k,kj->...j", export, decay)
    _pad = ((0, 0),) * (wet_contig.ndim - 1)
    wet_next = jnp.pad(wet_contig[..., 1:], (*_pad, (0, 1)))
    f_bot = wet_next * s_bot
    f_top = jnp.pad(f_bot[..., :-1], (*_pad, (1, 0)))
    return (f_top + export - f_bot) / dz


def par_profile(
    PAR_surf: jnp.ndarray,
    z_full_ref: jnp.ndarray,
    Phyto: jnp.ndarray,
    dz_ref: jnp.ndarray,
    cfg: BiogeoConfig,
) -> jnp.ndarray:
    """Compute PAR at each depth level with self-shading.

    Beer-Lambert law with water + chlorophyll attenuation:
    PAR(z) = PAR_surf * exp(-integral_0^z (k_w + k_chl * P) dz')

    Parameters
    ----------
    PAR_surf : array
        Surface PAR [W/m^2], shape (...).
    z_full_ref : array
        Reference depths [m], shape (nlev,), negative.
    Phyto : array
        Phytoplankton [mol N/m^3], shape (..., nlev).
    dz_ref : array
        Layer thicknesses [m], shape (nlev,).
    cfg : BiogeoConfig

    Returns
    -------
    PAR : array
        PAR at each level [W/m^2], shape (..., nlev).
    """
    # Total attenuation per layer: (k_w + k_chl * P) * dz
    atten = (cfg.k_w_atten + cfg.k_chl_atten * jnp.clip(Phyto, 0.0, None)) * dz_ref

    # Cumulative attenuation from surface (k=0 is surface)
    # For level k, PAR has been attenuated by layers 0..k-1 plus half of layer k
    cum_atten_interface = jnp.cumsum(atten, axis=-1)
    # Shift: interface attenuation at top of layer k = cumsum through k-1.
    # ``jnp.pad`` is one Pad HLO op; the previous form allocated a fresh
    # ``(..., 1)`` zero buffer + concatenate.
    _pad_axes = ((0, 0),) * (cum_atten_interface.ndim - 1)
    cum_atten_top = jnp.pad(cum_atten_interface[..., :-1], (*_pad_axes, (1, 0)))
    # Mid-level attenuation = top + half this layer
    cum_atten_mid = cum_atten_top + 0.5 * atten

    PAR = PAR_surf[..., jnp.newaxis] * jnp.exp(-cum_atten_mid)
    return PAR


def npzd_source_sink(
    NO3: jnp.ndarray,
    Phyto: jnp.ndarray,
    Zoo: jnp.ndarray,
    Det: jnp.ndarray,
    DIC: jnp.ndarray,
    ALK: jnp.ndarray,
    T_degC: jnp.ndarray,
    PAR: jnp.ndarray,
    dz_ref: jnp.ndarray,
    cfg: BiogeoConfig,
    wet_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, ...]:
    """Compute NPZD source/sink terms at every grid point and level.

    Parameters
    ----------
    NO3, Phyto, Zoo, Det, DIC, ALK : array (..., nlev)
        Biogeochemical tracer concentrations.
    T_degC : array (..., nlev)
        Temperature [degC].
    PAR : array (..., nlev)
        Photosynthetically available radiation [W/m^2].
    dz_ref : array (nlev,)
        Layer thicknesses [m].
    cfg : BiogeoConfig
    wet_mask : array (..., nlev)
        3-D sea mask, 1 = ocean cell, 0 = land / below the sea floor.
        Tendencies are zero (for finite inputs) in dry cells and in any
        cell below the first dry one; the bottom wet cell of each column
        receives what sinks onto the sea floor. ``dz_ref`` must be > 0.

    Returns
    -------
    dNO3_dt, dPhyto_dt, dZoo_dt, dDet_dt, dDIC_dt, dALK_dt : arrays
        Source/sink tendencies [units/s], same shapes as inputs.
    """
    day_to_s = 1.0 / 86400.0  # convert rates from 1/day to 1/s

    # Clip tracers to non-negative (smooth via softplus at small values)
    eps = 1.0e-12
    P = jnp.clip(Phyto, eps, None)
    Z = jnp.clip(Zoo, eps, None)
    D = jnp.clip(Det, eps, None)
    N = jnp.clip(NO3, eps, None)

    # ---- 1. Phytoplankton growth ----
    # Temperature dependence: Eppley (1972) Q10 = 1.066^T
    T_factor = 1.066 ** jnp.clip(T_degC, -2.0, 40.0)

    # Nutrient limitation (Michaelis-Menten)
    f_N = N / (N + cfg.k_N)

    # Light limitation (Webb et al. 1974 exponential form)
    # alpha_P [1/(W/m^2)/day] and mu_max [1/day] are both per-day,
    # so their ratio alpha_P/mu_max [1/(W/m^2)] is already consistent.
    f_L = 1.0 - jnp.exp(-cfg.alpha_P * PAR / jnp.clip(cfg.mu_max, eps, None))

    # Growth rate
    mu = cfg.mu_max * day_to_s * T_factor * jnp.minimum(f_N, f_L)
    growth = mu * P  # mol N/m^3/s

    # ---- 2. Zooplankton grazing ----
    grazing = cfg.g_max * day_to_s * P ** 2 / (P ** 2 + cfg.k_P ** 2) * Z

    # ---- 3. Mortality ----
    phyto_mort = cfg.m_P * day_to_s * P
    zoo_mort = cfg.m_Z * day_to_s * Z ** 2  # quadratic

    # ---- 4. Detritus remineralization ----
    remin = cfg.remin_rate * day_to_s * D

    # ---- 5. Detritus sinking (conservative interface-flux formulation) ----
    # w_sink in m/day -> m/s. Interface flux (upwind), positive DOWN:
    # flux_out[k] = w_sink * D[k] leaves layer k through its bottom. It
    # enters layer k+1 only if that cell is wet (contiguous from the
    # surface); otherwise it reaches the sea floor and is remineralised in
    # layer k (oracle phos_flux.F: what reaches the floor is returned to the
    # bottom wet cell). Nothing leaves the column.
    w_sink_s = cfg.w_sink * day_to_s  # m/s
    wet = jnp.cumprod(wet_mask.astype(D.dtype), axis=-1)
    _pad_axes_d = ((0, 0),) * (wet.ndim - 1)
    wet_next = jnp.pad(wet[..., 1:], (*_pad_axes_d, (0, 1)))
    flux_out = w_sink_s * D * wet
    flux_down = flux_out * wet_next
    floor_remin = (flux_out - flux_down) / dz_ref  # mol N/m^3/s, bottom cell
    flux_in = jnp.pad(flux_down[..., :-1], (*_pad_axes_d, (1, 0)))
    sinking_tend = (flux_in - flux_out) / dz_ref

    # ---- Assemble tendencies ----
    # Total remineralisation returning N (and C) to solution.
    remin_all = remin + floor_remin

    # Nutrients
    dNO3_dt = -growth + remin_all + (1.0 - cfg.gamma_Z) * grazing

    # Phytoplankton
    dPhyto_dt = growth - grazing - phyto_mort

    # Zooplankton
    dZoo_dt = cfg.gamma_Z * grazing - zoo_mort

    # Detritus
    dDet_dt = phyto_mort + zoo_mort - remin + sinking_tend

    # ---- Carbon coupling (Redfield) ----
    # Organic carbon cycle: C:N = R_CN
    # DIC decreases with primary production, increases with remineralization
    net_production = growth - remin_all - (1.0 - cfg.gamma_Z) * grazing
    dDIC_bio = -cfg.R_CN * net_production

    # Implicit CaCO3 (MITgcm dic_biotic_forcing.F: CAR_S = rain_ratio *
    # particulate organic production). The NPZD's particulate channel is
    # detritus formation (phyto + zoo mortality), in carbon units via R_CN.
    # Forms only in the production levels (oracle nlev/zcrit) and is
    # dissolved lower in the same column (car_flux.F).
    prod_levels = _production_levels(dz_ref, cfg.caco3_zcrit)
    caco3_production = (cfg.R_CaP * cfg.R_CN * (phyto_mort + zoo_mort)
                        * prod_levels * wet)
    caco3_dissolution = implicit_caco3_dissolution(
        caco3_production, dz_ref, wet, cfg.caco3_remin_depth)

    dDIC_dt = dDIC_bio - caco3_production + caco3_dissolution
    # Alkalinity (Dickson total alkalinity, which carries the
    # ``-[NO3-]`` term):
    #   * NO3 uptake by phytoplankton REMOVES nitrate from solution, so
    #     -d[NO3-]/dt is positive ⇒ TA increases ⇒ +growth.
    #   * Remineralization adds NO3- back ⇒ TA decreases ⇒ -remin.
    #   * Zooplankton excretion (``(1-gamma_Z)*grazing``) returns N as
    #     NO3- to solution ⇒ -(1-gamma_Z)*grazing.
    #   * CaCO3 precipitation removes 2 mol of charge per mol CaCO3
    #     ⇒ -2*caco3_production.
    #   * CaCO3 dissolution adds 2 mol of charge ⇒ +2*caco3_dissolution.
    dALK_dt = (growth - remin_all - (1.0 - cfg.gamma_Z) * grazing
               - 2.0 * caco3_production + 2.0 * caco3_dissolution)

    dNO3_dt, dPhyto_dt, dZoo_dt, dDet_dt, dDIC_dt, dALK_dt = (
        t * wet for t in (dNO3_dt, dPhyto_dt, dZoo_dt, dDet_dt, dDIC_dt, dALK_dt))

    return dNO3_dt, dPhyto_dt, dZoo_dt, dDet_dt, dDIC_dt, dALK_dt
