"""RRTMGP radiation -- compatibility shim.

All solver logic now lives in ``RRTMGP.solve_columns()`` inside the
bundled ``legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp`` module.
This file provides backward-compatible function entry points.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig  # noqa: F401  (re-export)
from legoesm.atmosphere.physics.radiation.output import RadiationOutput
from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

# Machine-checked scheme contract (see tests/test_physics_contracts.py).
__physics_contract__ = {
    "summary": (
        "RRTMGP radiation entry point (compatibility shim): forwards column "
        "arrays to RRTMGP.solve_columns and returns broadband LW+SW fluxes and "
        "radiative heating rates."
    ),
    "inputs": {
        "T": "K", "p_full": "Pa", "p_half": "Pa", "sfc_temperature": "K",
        "q_v": "kg/kg", "cos_zenith": "1 (cos solar zenith angle)",
        "sfc_albedo_override": "1 (surface shortwave albedo)",
        "sfc_emissivity_override": "1 (surface longwave emissivity)",
        "o3_vmr": "mol/mol", "cloud_path_liq": "kg/m^2",
        "cloud_path_ice": "kg/m^2",
        "cloud_path_liq_lw": "kg/m^2 (separate LW-stream path; None == SW)",
        "cloud_path_ice_lw": "kg/m^2 (separate LW-stream path; None == SW)",
        "cloud_r_eff_liq": "m",
        "cloud_r_eff_ice": "m", "cloud_fraction": "1",
        "aerosol_optical_depth": "1",
    },
    "outputs": {
        "lw_flux_up": "W/m^2", "lw_flux_down": "W/m^2",
        "sw_flux_up": "W/m^2", "sw_flux_down": "W/m^2",
        "heating_rate": "K/s", "lw_heating_rate": "K/s",
        "sw_heating_rate": "K/s", "toa_insolation": "W/m^2",
    },
    "sign_convention": (
        "heating_rate dT/dt>0 warms the layer; fluxes positive in their named "
        "direction (up/down); net radiative flux OUT of a layer cools it "
        "(heating_rate = -g/c_p * dF_net/dp); optical depth tau>=0; cos_zenith>=0 "
        "for illuminated columns (nighttime SW is zeroed). Photons enter/leave "
        "at TOA and the surface, so the column energy budget is OPEN (accounted, "
        "not conserved)."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Pincus, Mlawer & Delamere (2019), JAMES, doi:10.1029/2019MS001621 "
        "(RRTMGP); swirl_jatmos two-stream port."
    ),
    "idealized_test": (
        "tests/unit/test_physics_radiation.py + "
        "tests/atmosphere/hydrostatic/unit/test_rrtmgp_stratosphere.py: "
        "clear-sky column gives realistic LW cooling / SW heating; TOA/surface "
        "flux balance tracks the prescribed insolation and albedo."
    ),
}

# Module-level RRTMGP instance cache (keyed by config tuple).
_instance_cache: dict = {}


def _get_instance(config: RRTMGPConfig) -> RRTMGP:
    """Get or create a cached RRTMGP solver for the given config.

    Codex adversarial review 019e5467 (issue #273 follow-up):
    instance cache MUST key on the full set of solver-behavior
    fields, not just the optics-table fields.  Previously the cache
    shared its key with the optics-table cache, so a first call with
    ``use_scan=False`` stamped a solver instance whose
    ``self._config.use_scan = False``; a later call with the new
    auto-pick default ``None`` (or explicit ``True``) silently reused
    that instance and kept running the for-loop path on GPU,
    defeating the scan auto-pick the module ships.
    """
    key = RRTMGP._instance_cache_key(config)
    if key not in _instance_cache:
        _instance_cache[key] = RRTMGP.from_legoesm_config(config)
    return _instance_cache[key]


def preload_rrtmgp_optics(config: RRTMGPConfig) -> None:
    """Preload optics tables/cache outside JIT for stable runtime reuse."""
    RRTMGP.preload(config)


def rrtmgp_radiation(
    T: jnp.ndarray,
    p_full: jnp.ndarray,
    p_half: jnp.ndarray,
    sfc_temperature: jnp.ndarray,
    q_v: jnp.ndarray,
    cos_zenith: jnp.ndarray,
    config: RRTMGPConfig,
    sfc_albedo_override: jnp.ndarray | float | None = None,
    sfc_emissivity_override: jnp.ndarray | float | None = None,
    o3_vmr: jnp.ndarray | None = None,
    cloud_path_liq: jnp.ndarray | None = None,
    cloud_path_ice: jnp.ndarray | None = None,
    cloud_path_liq_lw: jnp.ndarray | None = None,
    cloud_path_ice_lw: jnp.ndarray | None = None,
    cloud_r_eff_liq: jnp.ndarray | None = None,
    cloud_r_eff_ice: jnp.ndarray | None = None,
    cloud_fraction: jnp.ndarray | None = None,
    aerosol_optical_depth: jnp.ndarray | None = None,
    aerosol_absorption_optical_depth_lw: jnp.ndarray | None = None,
    solar_spectral_fraction: jnp.ndarray | None = None,
    ghg_vmr_override: dict | None = None,
) -> RadiationOutput:
    """Compute radiation using jax-rrtmgp (compatibility shim).

    All arguments are forwarded to ``RRTMGP.solve_columns()``.
    """
    solver = _get_instance(config)
    _rad_kwargs = dict(
        T=T,
        p_full=p_full,
        p_half=p_half,
        sfc_temperature=sfc_temperature,
        q_v=q_v,
        cos_zenith=cos_zenith,
        sfc_albedo=sfc_albedo_override,
        sfc_emissivity=sfc_emissivity_override,
        o3_vmr=o3_vmr,
        cloud_path_liq=cloud_path_liq,
        cloud_path_ice=cloud_path_ice,
        cloud_path_liq_lw=cloud_path_liq_lw,
        cloud_path_ice_lw=cloud_path_ice_lw,
        cloud_r_eff_liq=cloud_r_eff_liq,
        cloud_r_eff_ice=cloud_r_eff_ice,
        cloud_fraction=cloud_fraction,
        aerosol_optical_depth=aerosol_optical_depth,
        aerosol_absorption_optical_depth_lw=aerosol_absorption_optical_depth_lw,
        solar_spectral_fraction=solar_spectral_fraction,
        ghg_vmr_override=ghg_vmr_override,
    )
    # Honour RRTMGPConfig.column_chunk_size so this public entry point caps
    # the rrtmgp XLA compile time identically to the integration path (the
    # per-block body compiles ONCE at column_chunk_size).  Columns are
    # independent → numerically exact; 0 (default) = plain single-shot solve.
    if getattr(config, "column_chunk_size", 0) and config.column_chunk_size > 0:
        return solver.solve_columns_chunked(
            column_chunk_size=config.column_chunk_size, **_rad_kwargs,
        )
    return solver.solve_columns(**_rad_kwargs)
