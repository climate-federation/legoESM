"""End-to-end ecRad call of OpenIFS 49r1 (8f6f722), IFS configuration, JAX port.

Port of ``radiation_interface.F90:radiation`` (compiled file lines 190-495,
``bld49/src``) for the configuration of ``config49r1_rrtmg.nam``, with RRTMGP
gas optics in place of RRTMG (:mod:`.gas_optics`, user decision 2026-11-06):
``use_general_cloud_optics = false``, aerosols off,
McICA LW and SW solvers, ``do_lw_derivatives``, ``do_surface_sw_spectral_flux``
and ``do_canopy_fluxes_sw``.  The call order is the oracle's:

1. ``single_level%get_albedos``: SW albedos by the weighted interval->band
   mapping (``do_nearest_spectral_sw_albedo = false``), LW albedo = 1 -
   emissivity by the nearest-interval mapping (``do_nearest_spectral_lw_emiss``);
2. :func:`.gas_optics.gas_optics`: full levels, gas units, LW and SW gas
   optics, Planck terms, incoming SW;
3. ``crop_cloud_fraction`` + ``cloud_optics`` (per band);
4. aerosols off: ``g_sw = 0`` (line 381; LW aerosol scattering is off);
5. ``solver_mcica_lw`` / ``solver_mcica_sw``, each running the McICA
   ``cloud_generator`` first (LW seed ``iseed + 997``, SW seed ``iseed``);
   the band->g-point mapping of the cloud optics is inside the solvers;
6. ``calc_surface_spectral`` (SW band and canopy surface fluxes).

Every numerical step is a section module (no code duplicated here).  Not
ported (outside this configuration): ``radiation_reverse`` (inputs must be
top-down, index 0 = TOA, as the IFS passes them), other gas/solver/overlap
options, aerosol optics, ``calc_toa_spectral`` (McICA saves no spectral flux).

Random numbers (user decision A16, vote G1): the McICA uniforms are explicit
inputs in the generator layout (:mod:`.mcica_generator`), one 4-tuple per
spectrum.  :func:`draw_uniforms` fills them for production from
``jax.random.PRNGKey(oracle seed)`` (vote G2).

Smoothing: ONE static ``sharpness`` passed to every section (default 100;
``math.inf`` = the oracle's hard form).  No switch of its own here: the
smoothed transitions are those of the section modules.
"""

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics.radiation.ecrad_ifs import cloud_optics as co
from legoesm.atmosphere.physics.radiation.ecrad_ifs import gas_optics as go
from legoesm.atmosphere.physics.radiation.ecrad_ifs import mcica_generator as mg
from legoesm.atmosphere.physics.radiation.ecrad_ifs import mcica_solvers as ms

__physics_contract__ = {
    "summary": (
        "ecRad 49r1 (IFS, RRTMGP gas optics, McICA) end-to-end call: surface "
        "albedo/emissivity mapping, LW/SW gas optics, cloud crop and optics, "
        "McICA generator and solvers, broadband, surface-band and canopy fluxes. "
        "Port of radiation_interface.F90 radiation (OpenIFS 49r1)."
    ),
    "inputs": {
        "pressure_hl": "Pa (ncol, nlev+1), top-down", "temperature_hl": "K",
        "gas_mmr": "kg/kg (ncol, nlev, 12), ecRad gas order",
        "skin_temperature": "K", "cos_sza": "1", "solar_irradiance": "W/m^2",
        "sw_albedo, sw_albedo_direct": "1 (ncol, 6 intervals)",
        "lw_emissivity": "1 (ncol, 2 intervals)", "cloud_fraction": "1",
        "q_liquid, q_ice": "kg/kg", "re_liquid, re_ice": "m",
        "overlap_param": "1 (ncol, nlev-1)", "fractional_std": "1",
        "uniforms_lw, uniforms_sw": "1 (McICA uniform deviates, generator layout)",
    },
    "outputs": {
        "flux_up/dn_lw(_clear)": "W/m^2 (ncol, nlev+1)",
        "lw_derivative": "1",
        "flux_up/dn/dn_direct_sw(_clear)": "W/m^2 (ncol, nlev+1)",
        "spectral_flux_dn_(direct_)sw_surf(_clear)": "W/m^2 (ncol, 14 bands)",
        "canopy_flux_dn_diffuse/direct_sw_surf": "W/m^2 (ncol, 6 intervals)",
        "cloud_cover_lw/sw": "1 (-1 for SW night columns, oracle fill)",
    },
    "sign_convention": (
        "Fluxes positive in their named direction on half levels, index 0 = "
        "TOA. Net flux = dn - up; layer heating = g/cp * (net(k) - net(k+1))/dp."
    ),
    "conserves": ["energy"],
    "differentiable": True,
    "reference": (
        "OpenIFS 49r1 (8f6f722) radiation_interface.F90, radiation_single_level.F90 "
        "get_albedos; rte-rrtmgp gas optics; Hogan & Bozzo (2018) JAMES 10."
    ),
    "idealized_test": (
        "tests/atmosphere/test_ecrad_ifs_rrtmgp_gas.py: clear-sky fluxes against "
        "the RTE Fortran fixture, heating/flux and TOA/surface consistency; "
        "solvers and cloud optics against 49r1 stage dumps in their own tests."
    ),
}

# --- config49r1_rrtmg.nam and the radiation_config.F90 defaults it leaves ---
_CLOUD_FRACTION_THRESHOLD = 1.0e-6         # namelist cloud_fraction_threshold
_CLOUD_INHOM_DECORR_SCALING = 0.5          # radiation_config.F90:214 default
_MIN_GAS_OD_LW = 1.0e-15                   # radiation_config.F90:235 default
SW_ALBEDO_WAVELENGTH_BOUND = (0.25e-6, 0.44e-6, 0.69e-6, 1.19e-6, 2.38e-6)  # m, namelist
I_SW_ALBEDO_INDEX = (1, 2, 3, 4, 5, 6)     # namelist
# i_emiss_from_band_lw = maxloc(lw_emiss_weights, dim=1) (radiation_config.F90:1977-1978)
# for the namelist lw_emiss_wavelength_bound = 8, 13 um and i_lw_emiss_index = 1,2,1;
# as printed by the oracle's setup ("Mapping from 16 longwave intervals to
# emissivity intervals: 1 1 1 1 1 2 2 2 1 1 1 1 1 1 1 1", stage_meridian/out.log:65).
I_EMISS_FROM_BAND_LW = (1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1)
# The 49r1 gravity (yomcst_ecrad.F90:33), equal to ecRad's AccelDueToGravity.
_GRAVITY = constants.g_ecrad

OUTPUT_KEYS = (
    "flux_up_lw", "flux_dn_lw", "flux_up_lw_clear", "flux_dn_lw_clear", "lw_derivative",
    "flux_up_sw", "flux_dn_sw", "flux_dn_direct_sw", "flux_up_sw_clear", "flux_dn_sw_clear",
    "flux_dn_direct_sw_clear", "spectral_flux_dn_sw_surf", "spectral_flux_dn_direct_sw_surf",
    "spectral_flux_dn_sw_surf_clear", "spectral_flux_dn_direct_sw_surf_clear",
    "canopy_flux_dn_diffuse_sw_surf", "canopy_flux_dn_direct_sw_surf",
    "cloud_cover_lw", "cloud_cover_sw",
)


def lw_albedo_from_emissivity(lw_emissivity):
    """get_albedos, nearest-emissivity branch (radiation_single_level.F90:364-365):
    lw_albedo(g) = 1 - lw_emissivity(i_emiss_from_band_lw(band(g))); (ncol, 140)."""
    idx = np.asarray(I_EMISS_FROM_BAND_LW, dtype=np.int64)[go.band_from_g_lw()] - 1
    return 1.0 - lw_emissivity[:, idx]


def draw_uniforms(iseed, nlev):
    """Production McICA uniforms (vote G2): PRNGKey(iseed + 997) for LW and
    PRNGKey(iseed) for SW, per column (the oracle's seeds,
    radiation_mcica_lw.F90:189, radiation_mcica_sw.F90:194).  iseed (ncol,)
    integer.  Returns (uniforms_lw, uniforms_sw) in the generator layout."""
    iseed = jnp.asarray(iseed)
    n_lw, n_sw = go.band_from_g_lw().size, go.band_from_g_sw().size
    lw = jax.vmap(lambda s: mg.draw_uniforms(s, n_lw, nlev))(iseed + mg.LW_SEED_OFFSET)
    sw = jax.vmap(lambda s: mg.draw_uniforms(s, n_sw, nlev))(iseed)
    return lw, sw


def radiation(pressure_hl, temperature_hl, gas_mmr, skin_temperature, cos_sza,
              solar_irradiance, sw_albedo, sw_albedo_direct, lw_emissivity,
              cloud_fraction, q_liquid, q_ice, re_liquid, re_ice, overlap_param,
              fractional_std, uniforms_lw, uniforms_sw, sharpness=100.0):
    """radiation_interface.F90:radiation for config49r1_rrtmg.nam.

    Shapes (column first, levels top-down): pressure_hl, temperature_hl
    (ncol, nlev+1); gas_mmr (ncol, nlev, 12); cloud fields (ncol, nlev);
    overlap_param (ncol, nlev-1); skin_temperature, cos_sza (ncol,);
    solar_irradiance scalar; sw_albedo(_direct) (ncol, 6); lw_emissivity
    (ncol, 2); uniforms_lw/sw = (rand_top, rand_cloud, rand_inhom1,
    rand_inhom2) as :func:`.mcica_generator.cloud_generator` takes them.
    Returns a dict keyed by :data:`OUTPUT_KEYS` (the oracle's output names).
    """
    # 1. get_albedos (radiation_interface.F90:314)
    band_lw, band_sw = go.band_from_g_lw(), go.band_from_g_sw()
    sw_alb_diffuse = ms.sw_albedo_to_gpoints(sw_albedo, SW_ALBEDO_WAVELENGTH_BOUND,
                                             I_SW_ALBEDO_INDEX, band_sw)
    sw_alb_direct = ms.sw_albedo_to_gpoints(sw_albedo_direct, SW_ALBEDO_WAVELENGTH_BOUND,
                                            I_SW_ALBEDO_INDEX, band_sw)
    lw_albedo = lw_albedo_from_emissivity(lw_emissivity)

    # 2. gas optics
    gas = go.gas_optics(pressure_hl, temperature_hl, gas_mmr, skin_temperature, lw_albedo,
                        solar_irradiance, _GRAVITY, _MIN_GAS_OD_LW, sharpness)

    # 3. crop_cloud_fraction + cloud_optics (lines 346, 363)
    cloud = co.cloud_optics(pressure_hl, cloud_fraction, q_liquid, q_ice, re_liquid, re_ice,
                            sharpness)
    fraction = cloud.cloud_fraction

    # 4. use_aerosols = false: g_sw = 0 (line 381)
    g_sw = jnp.zeros_like(gas.od_sw)

    # 5. McICA: generator then solver, LW (line 414) and SW (line 451)
    def generate(uniforms):
        od_scaling, tcc = mg.cloud_generator(
            fraction, overlap_param, fractional_std, *uniforms,
            decorrelation_scaling=_CLOUD_INHOM_DECORR_SCALING,
            frac_threshold=_CLOUD_FRACTION_THRESHOLD, sharpness=sharpness)
        return jnp.swapaxes(od_scaling, 1, 2), tcc        # (ncol, nlev, ng)

    scaling_lw, tcc_lw = generate(uniforms_lw)
    flux_lw = ms.solver_mcica_lw(
        gas.od_lw, cloud.od_lw, cloud.ssa_lw, cloud.g_lw, gas.planck_hl, gas.lw_emission,
        lw_albedo, fraction, scaling_lw, tcc_lw, band=band_lw,
        cloud_fraction_threshold=_CLOUD_FRACTION_THRESHOLD, sharpness=sharpness)
    scaling_sw, tcc_sw = generate(uniforms_sw)
    flux_sw = ms.solver_mcica_sw(
        gas.od_sw, gas.ssa_sw, g_sw, cloud.od_sw, cloud.ssa_sw, cloud.g_sw, sw_alb_direct,
        sw_alb_diffuse, gas.incoming_sw, cos_sza, fraction, scaling_sw, tcc_sw, band=band_sw,
        cloud_fraction_threshold=_CLOUD_FRACTION_THRESHOLD, sharpness=sharpness)

    # 6. calc_surface_spectral (line 488)
    spectral = ms.surface_spectral_sw(flux_sw, SW_ALBEDO_WAVELENGTH_BOUND, I_SW_ALBEDO_INDEX, band_sw)
    out = {**flux_lw, **flux_sw, **spectral}
    return {k: out[k] for k in OUTPUT_KEYS}

