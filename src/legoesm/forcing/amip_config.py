"""AMIP experiment configuration and restart I/O.

Provides:
- AMIPExperimentConfig: serializable experiment configuration
- save_checkpoint / load_checkpoint: restart I/O for long integrations
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class AMIPExperimentConfig(NamedTuple):
    """Complete AMIP experiment configuration.

    Serializable to JSON for reproducibility. All fields have defaults
    so a minimal config only needs forcing_path.
    """
    # Grid
    resolution: int = 16
    nlev: int = 40
    dt: float = 600.0

    # Vertical coordinate
    vertical_coord: str = "hybrid"  # "sigma" or "hybrid"
    p_top_Pa: float = 200.0  # model top pressure [Pa] (hybrid only)
    stretching: float = 2.0  # sinh stretching for BL resolution (hybrid only)

    # Integration
    start_day: float = 0.0
    days: int = 200
    diag_days: int = 5
    checkpoint_days: int = 0  # 0 = no checkpointing

    # Forcing
    dataset: str = "cobe"
    forcing_path: str = ""
    sic_path: str = ""
    sst_var: str = ""     # empty = use preset
    sic_var: str = ""
    time_var: str = ""
    lat_var: str = ""
    lon_var: str = ""
    sst_offset: float = 0.0  # only used if sst_var is set (custom)
    sic_scale: float = 1.0

    # Atmosphere
    A_h_scale: float = 1.0
    hyperdiff_scale: float = 5e16
    T_init: float = 280.0
    RH_init: float = 0.6

    # Radiation
    radiation: str = "gray"  # "gray" or "rrtmg"
    rad_update_steps: int = 1  # recompute radiation every N steps (1 = every step)
    diurnal_cycle: bool = False  # use instantaneous solar zenith angle
    solar_source: str = "constant"  # "constant", "file", or "spectral_file"
    solar_file: str = ""
    solar_spectral_var: str = "solar_fraction_by_gpt"

    # Physics — gray radiation
    tau_equator: float = 7.2
    tau_pole: float = 1.8
    S_0: float = 1360.0
    sbm_tau_c: float = 7200.0
    sbm_RH_ref: float = 0.7
    sbm_cape_threshold: float = 70.0
    C_H: float = 1.5e-3
    C_E: float = 1.5e-3
    k_free_per_day: float = 0.1
    k_BL_max_per_day: float = 1.0
    sigma_b: float = 0.7

    # Physics — RRTMG gas concentrations
    co2_ppmv: float = 415.0
    ch4_ppbv: float = 1900.0
    n2o_ppbv: float = 332.0
    ozone_source: str = "standard"  # "standard", "analytical", or "none"
    ozone_forcing: str = "inline"  # "inline", "external", or "off"
    ozone_file: str = ""
    ghg_forcing: str = "constant"  # "constant" or "external"
    ghg_file: str = ""
    aerosol_forcing: str = "off"  # "off" or "external"
    aerosol_file: str = ""
    aerosol_reference_aod: float = 0.03
    volcanic_aerosol_file: str = ""
    volcanic_aerosol_scale: float = 1.0
    cloud_scheme: str = "none"  # "none", "sundqvist", or "xu_randall"
    sfc_emissivity: float = 0.98
    emissivity_ice: float = 0.99

    # Microphysics
    microphysics: str = "none"  # "none", "kessler", "sundqvist"

    # Subgrid physics selection
    convection: str = "sbm"
    turbulence: str = "none"
    gravity_wave_drag: str = "none"
    fix_moisture: bool = False
    sat_adjust_without_microphysics: bool = True

    # Topography
    topography: str = "flat"  # "flat", "gaussian", or path to NetCDF file
    topo_smoothing: int = 4  # Laplacian smoothing passes
    topo_edge_blend: float = 0.3  # edge blending strength for cubed-sphere

    # Sea ice
    T_ice: float = 271.35
    albedo_ice: float = 0.65
    albedo_ocean: float = 0.06

    # Surface albedo improvements (Task 10)
    dynamic_albedo: bool = False  # Enable temperature/zenith-dependent albedo

    # Monthly-mean diagnostics (Task 12)
    monthly_means: bool = False  # Accumulate zonal/global monthly means

    # Carbon cycle
    carbon_cycle: str = "none"  # "none", "differland", or "seasonal"

    # CMIP experiment
    experiment: str = ""  # "piControl", "historical", "ssp245", "ssp585", "amip", "1pctCO2"
    start_year: int = 1979  # calendar start year (for GHG trajectory lookup)
    cmip_output: bool = False  # write CF/CMOR NetCDF files during the run
    clear_sky_diag: bool = False  # run clear-sky radiation pass for rsutcs/rlutcs

    # Distributed execution
    distributed: bool = False
    ensemble_size: int = 1

    # Output
    output_dir: str = ""

    # Held-Suarez forcing
    held_suarez_forcing: bool = False

    # Joint ML physics parameterization
    physics_parameterization: str = "none"
    physics_parameterization_checkpoint: str = ""
    physics_parameterization_stats: str = ""
    physics_parameterization_hidden_dim: int = 128
    physics_parameterization_layers: int = 3
    physics_parameterization_seed: int = 0


def config_to_dict(config) -> dict:
    """Convert config to a plain dict for JSON serialization.

    Accepts ``AMIPExperimentConfig`` or ``ExperimentConfig``.
    Sub-config NamedTuples (GridConfig, DycoreConfig, OutputConfig) are
    recursively converted to dicts.
    """
    d = config._asdict()
    for key, val in d.items():
        if hasattr(val, '_asdict'):
            d[key] = val._asdict()
    return d


def config_from_dict(d: dict) -> AMIPExperimentConfig:
    """Reconstruct config from a dict (e.g., loaded from JSON)."""
    # Filter to only known fields
    known = set(AMIPExperimentConfig._fields)
    filtered = {k: v for k, v in d.items() if k in known}
    return AMIPExperimentConfig(**filtered)


def save_config(config: AMIPExperimentConfig, path: Path) -> None:
    """Save experiment config to JSON."""
    with open(path, "w") as f:
        json.dump(config_to_dict(config), f, indent=2)


def load_config(path: Path) -> AMIPExperimentConfig:
    """Load experiment config from JSON."""
    with open(path) as f:
        return config_from_dict(json.load(f))


# ==============================================================================
# Checkpoint / Restart
# ==============================================================================

def save_checkpoint(
    path: Path,
    state,
    q_v: jnp.ndarray,
    step: int,
    day: float,
    config,
    diag_accumulators: dict | None = None,
    q_c: jnp.ndarray | None = None,
    q_r: jnp.ndarray | None = None,
    carry_aux: dict | None = None,
) -> None:
    """Save a checkpoint for restart.

    Parameters
    ----------
    path : Path
        Output file path (.npz).
    state : HydrostaticState
        Atmospheric state (NamedTuple with Field members).
    q_v : jax.Array
        Moisture field, shape (6, n, n, nlev).
    step : int
        Current time step number.
    day : float
        Current simulation day.
    config : ExperimentConfig or AMIPExperimentConfig
        Experiment configuration (saved alongside for reproducibility).
    diag_accumulators : dict, optional
        Any accumulated diagnostic arrays to preserve across restart.
    q_c : jax.Array, optional
        Cloud water field, shape (6, n, n, nlev).
    q_r : jax.Array, optional
        Rain water field, shape (6, n, n, nlev).
    """
    # Extract state arrays
    arrays = {
        "T": np.asarray(state.T.data),
        "u": np.asarray(state.u.data),
        "v": np.asarray(state.v.data),
        "p_s": np.asarray(state.p_s.data),
        "phis": np.asarray(state.phis.data),
        "q_v": np.asarray(q_v),
        "step": np.array(step),
        "day": np.array(day),
    }

    # Save hydrometeor fields if provided
    if q_c is not None:
        arrays["q_c"] = np.asarray(q_c)
    if q_r is not None:
        arrays["q_r"] = np.asarray(q_r)

    # Save config as JSON string inside npz
    arrays["config_json"] = np.array(json.dumps(config_to_dict(config)))

    # Save diagnostic accumulators if provided
    if diag_accumulators:
        for k, v in diag_accumulators.items():
            arrays[f"diag_{k}"] = np.asarray(v)

    # Save carry auxiliary fields (held radiation, target_moisture, etc.)
    if carry_aux:
        for k, v in carry_aux.items():
            arrays[f"carry_{k}"] = np.asarray(v)

    np.savez(str(path), **arrays)


def load_checkpoint(
    path: Path,
    grid,
    sigma,
) -> tuple:
    """Load a checkpoint and reconstruct state.

    Parameters
    ----------
    path : Path
        Checkpoint file path (.npz).
    grid : CubedSphereGrid
        Grid (must match checkpoint resolution).
    sigma : SigmaCoordinate
        Vertical coordinate (must match checkpoint nlev).

    Returns
    -------
    state : HydrostaticState
    q_v : jax.Array
    step : int
    day : float
    config : AMIPExperimentConfig
    diag_accumulators : dict
    q_c : jax.Array or None
    q_r : jax.Array or None
    """
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState

    from legoesm.core.precision import get_policy
    _storage_dtype = get_policy().storage

    data = np.load(str(path), allow_pickle=True)

    T = jnp.array(data["T"], dtype=_storage_dtype)
    u = jnp.array(data["u"], dtype=_storage_dtype)
    v = jnp.array(data["v"], dtype=_storage_dtype)
    p_s = jnp.array(data["p_s"], dtype=_storage_dtype)
    phis = jnp.array(data["phis"], dtype=_storage_dtype)
    q_v = jnp.array(data["q_v"], dtype=_storage_dtype)

    step = int(data["step"])
    day = float(data["day"])

    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")

    state = HydrostaticState(
        u=Field(data=u, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=p_s, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=phis, name="phis", dims=dims_2d, units="m^2/s^2"),
    )

    # Restore config
    config_str = str(data["config_json"])
    config = config_from_dict(json.loads(config_str))

    # Restore diagnostic accumulators
    diag_accumulators = {}
    for key in data.files:
        if key.startswith("diag_"):
            diag_accumulators[key[5:]] = data[key]

    # Restore hydrometeor fields if present
    q_c = jnp.array(data["q_c"], dtype=_storage_dtype) if "q_c" in data.files else None
    q_r = jnp.array(data["q_r"], dtype=_storage_dtype) if "q_r" in data.files else None
    tracer_fields = {
        "q_v": Field(data=q_v, name="q_v", dims=dims_3d, units="kg/kg"),
    }
    if q_c is not None:
        tracer_fields["q_c"] = Field(data=q_c, name="q_c", dims=dims_3d, units="kg/kg")
    if q_r is not None:
        tracer_fields["q_r"] = Field(data=q_r, name="q_r", dims=dims_3d, units="kg/kg")
    state = state._replace(tracers=tracer_fields)

    # Restore carry auxiliary fields (held radiation, etc.)
    carry_aux = {}
    for key in data.files:
        if key.startswith("carry_"):
            carry_aux[key[6:]] = jnp.array(data[key], dtype=_storage_dtype)

    return state, q_v, step, day, config, diag_accumulators, q_c, q_r, carry_aux
