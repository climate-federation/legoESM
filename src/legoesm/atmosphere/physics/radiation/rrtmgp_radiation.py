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


def preload_rrtmgp_optics_mpi(config: RRTMGPConfig) -> None:
    """MPI-aware preload: rank 0 reads NetCDF files, broadcasts to others."""
    RRTMGP.preload_mpi(config)


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
    cloud_r_eff_liq: jnp.ndarray | None = None,
    cloud_r_eff_ice: jnp.ndarray | None = None,
    aerosol_optical_depth: jnp.ndarray | None = None,
    solar_spectral_fraction: jnp.ndarray | None = None,
    ghg_vmr_override: dict | None = None,
) -> RadiationOutput:
    """Compute radiation using jax-rrtmgp (compatibility shim).

    All arguments are forwarded to ``RRTMGP.solve_columns()``.
    """
    solver = _get_instance(config)
    return solver.solve_columns(
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
        cloud_r_eff_liq=cloud_r_eff_liq,
        cloud_r_eff_ice=cloud_r_eff_ice,
        aerosol_optical_depth=aerosol_optical_depth,
        solar_spectral_fraction=solar_spectral_fraction,
        ghg_vmr_override=ghg_vmr_override,
    )
