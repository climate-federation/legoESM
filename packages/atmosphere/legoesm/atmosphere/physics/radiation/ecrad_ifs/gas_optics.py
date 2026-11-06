"""RRTMGP gas optics for the ecRad lane (user decision 2026-11-06, replacing RRTMG).

ecRad's solvers, cloud optics and McICA are unchanged; this adapter feeds them
from the same RRTMGP optics path as ``rrtmgp.rte.two_stream``.  Cloud optics stay
on RRTMG's bands: the 16 LW bands map one to one (edges differ at 2380/2390 and
2600/2680 cm-1), and RRTMGP SW band b is RRTMG band (b - 1) mod 14, whose
820-2600 cm-1 band RRTMG orders last.  Band outputs keep RRTMG's order and edges.

Gases as RRTM_PREPARE_GASES fed them: mass mixing ratios to volume by the
molecular weights below, O2 fixed.  RRTMGP gases the lane does not carry take the
RRTMGP library's value.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
from legoesm.atmosphere.physics.radiation.rrtmgp import constants as rc
from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
from legoesm.atmosphere.physics.radiation.rrtmgp.rte import monochromatic_two_stream as mts
from legoesm.core.smooth import smooth_max

__physics_contract__ = {
    "summary": (
        "RRTMGP gas optics on ecRad's column-first, top-down layout: optical "
        "depth, Rayleigh albedo, Planck level and surface terms, solar source."
    ),
    "inputs": {
        "pressure_hl": "Pa (ncol, nlev+1), top-down", "temperature_hl": "K",
        "gas_mmr": "kg/kg (ncol, nlev, 12), ecRad gas order",
        "skin_temperature": "K", "lw_albedo": "1 (ncol, ng_lw)",
        "solar_irradiance": "W/m^2", "gravity": "m/s^2",
    },
    "outputs": {
        "od_lw": "1 (ncol, nlev, ng_lw)", "planck_hl": "W/m^2 (ncol, nlev+1, ng_lw)",
        "lw_emission": "W/m^2 (ncol, ng_lw)", "od_sw, ssa_sw": "1 (ncol, nlev, ng_sw)",
        "incoming_sw": "W/m^2 (ncol, ng_sw), normal incidence",
    },
    "sign_convention": "Levels index 0 = TOA; sources and optical depths non-negative.",
    "conserves": ["none"],
    "differentiable": True,
    "reference": "rte-rrtmgp d4b3c51 gas optics; OpenIFS 49r1 rrtm_prepare_gases.F90 gas units",
    "idealized_test": (
        "tests/atmosphere/test_ecrad_ifs_rrtmgp_gas.py: clear-sky fluxes against "
        "the RTE Fortran fixture, band maps, blackbody limit."
    ),
}

# --- ecRad gas indices (radiation_gas_constants.F90:27-38), zero-based ---
IH2O, ICO2, IO3, IN2O, ICO, ICH4, IO2 = 0, 1, 2, 3, 4, 5, 6
ICFC11, ICFC12, IHCFC22, ICCL4, INO2 = 7, 8, 9, 10, 11

# --- molecular weights g/mol (rrtm_prepare_gases.F90:94-104) ---
_AIR_MOLAR_MASS = 28.970
_MOLAR_MASS = {
    "h2o": (IH2O, 18.0154), "co2": (ICO2, 44.011), "o3": (IO3, 47.9982),
    "n2o": (IN2O, 44.013), "ch4": (ICH4, 16.043), "cfc11": (ICFC11, 137.3686),
    "cfc12": (ICFC12, 120.9140), "cfc22": (IHCFC22, 86.4690), "ccl4": (ICCL4, 153.8230),
}
_O2_VMR = 0.209488  # rrtm_prepare_gases.F90:144
_ABSENT_FROM_RRTM = ("co", "no2")  # WKL left zero by RRTM_PREPARE_GASES


class GasOptics(NamedTuple):
    od_lw: jnp.ndarray
    planck_hl: jnp.ndarray
    lw_emission: jnp.ndarray
    od_sw: jnp.ndarray
    ssa_sw: jnp.ndarray
    incoming_sw: jnp.ndarray


@functools.lru_cache(maxsize=1)
def _optics():
    # The first call can land inside a jit trace; tables must stay concrete.
    with jax.ensure_compile_time_eval():
        return RRTMGP.from_legoesm_config(RRTMGPConfig(include_clouds=False)).optics_lib


def band_from_g_lw():
    """RRTMG band of each LW g-point, zero-based."""
    return np.asarray(_optics().gas_optics_lw.g_point_to_bnd, dtype=np.int64)


def band_from_g_sw():
    """RRTMG band of each SW g-point, zero-based."""
    bands = np.asarray(_optics().gas_optics_sw.g_point_to_bnd, dtype=np.int64)
    return (bands - 1) % (bands.max() + 1)


def _cells(a):
    """(ncol, nlev) top-down -> (ncol, 1, nlev+2) surface-first with halo cells."""
    a = a[:, None, ::-1]
    return jnp.concatenate([a[..., :1], a, a[..., -1:]], axis=-1)


def _top_down(a):
    """(ng, ncol, n) surface-first -> (ncol, n, ng) top-down."""
    return jnp.moveaxis(a[..., ::-1], 0, -1)


def gas_optics(pressure_hl, temperature_hl, gas_mmr, skin_temperature, lw_albedo,
               solar_irradiance, gravity, min_gas_od_lw, sharpness=100.0):
    """LW and SW gas optics; shapes in ``__physics_contract__``.

    min_gas_od_lw must be a Python float (the smooth width is relative to it).
    """
    lib = _optics()
    p = _cells(0.5 * (pressure_hl[:, :-1] + pressure_hl[:, 1:]))
    t = _cells(0.5 * (temperature_hl[:, :-1] + temperature_hl[:, 1:]))
    vmr = {name: _cells(gas_mmr[..., i] * (_AIR_MOLAR_MASS / mass))
           for name, (i, mass) in _MOLAR_MASS.items()}
    vmr["o2"] = jnp.full_like(p, _O2_VMR)
    vmr.update({name: jnp.zeros_like(p) for name in _ABSENT_FROM_RRTM})
    dp = _cells(pressure_hl[:, 1:] - pressure_hl[:, :-1])
    molecules = (dp / gravity) * rc.AVOGADRO / (rc.DRY_AIR_MOL_MASS + rc.WATER_MOL_MASS * vmr["h2o"])
    level_t = temperature_hl[:, None, ::-1]

    lookup = lib.gas_optics_lw
    bound = lib.with_all_gpts(True, p, t, molecules, {lookup.idx_gases[k]: v for k, v in vmr.items()})

    def lw_gpoint(igpt):
        tau = bound.compute_lw_optical_properties(p, t, molecules, igpt)["optical_depth"]
        src = bound.compute_planck_sources(p, t, igpt, sfc_temperature=skin_temperature[:, None],
                                           level_temperature=level_t)
        level = mts.lw_combine_sources(src)["planck_src_top"]
        return tau[:, 0, 1:-1], level[:, 0, :-1], src["planck_src_sfc"][:, 0]

    tau_lw, level, sfc = jax.vmap(lw_gpoint)(jnp.arange(lib.n_gpt_lw))

    lookup = lib.gas_optics_sw
    bound = lib.with_all_gpts(False, p, t, molecules, {lookup.idx_gases[k]: v for k, v in vmr.items()})

    def sw_gpoint(igpt):
        props = bound.compute_sw_optical_properties(p, t, molecules, igpt)
        return props["optical_depth"][:, 0, 1:-1], props["ssa"][:, 0, 1:-1]

    tau_sw, ssa_sw = jax.vmap(sw_gpoint)(jnp.arange(lib.n_gpt_sw))
    incoming = solar_irradiance * jnp.asarray(lib.solar_fraction_by_gpt, tau_sw.dtype)

    return GasOptics(
        od_lw=smooth_max(_top_down(tau_lw), min_gas_od_lw, sharpness / min_gas_od_lw),
        planck_hl=jnp.pi * _top_down(level),
        lw_emission=jnp.pi * jnp.moveaxis(sfc, 0, -1) * (1.0 - lw_albedo),
        od_sw=_top_down(tau_sw),
        ssa_sw=_top_down(ssa_sw),
        incoming_sw=jnp.broadcast_to(incoming, (pressure_hl.shape[0], incoming.size)),
    )
