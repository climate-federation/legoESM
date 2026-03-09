"""AMIP experiment configuration and restart I/O.

Provides:
- AMIPExperimentConfig: serializable experiment configuration
- save_checkpoint / load_checkpoint: restart I/O for long integrations
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.forcing.amip import AMIPForcingConfig
from legoesm.forcing.external import ExternalForcingConfig


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
    sst_var: str = ""     # empty = use preset
    sic_var: str = ""
    sst_offset: float = 0.0  # only used if sst_var is set (custom)
    sic_scale: float = 1.0

    # Atmosphere
    hyperdiff_scale: float = 5e16
    T_init: float = 280.0
    RH_init: float = 0.6

    # Radiation
    radiation: str = "gray"  # "gray" or "rrtmg"
    rad_update_steps: int = 1  # recompute radiation every N steps (1 = every step)
    diurnal_cycle: bool = False  # use instantaneous solar zenith angle

    # Physics — gray radiation
    tau_equator: float = 7.2
    tau_pole: float = 1.8
    S_0: float = 1360.0
    sbm_tau_c: float = 7200.0
    sbm_RH_ref: float = 0.7
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
    cloud_scheme: str = "none"  # "none", "sundqvist", or "xu_randall"
    sfc_emissivity: float = 0.98
    emissivity_ice: float = 0.99

    # Sea ice
    T_ice: float = 271.35
    albedo_ice: float = 0.65
    albedo_ocean: float = 0.06

    # Output
    output_dir: str = ""


def config_to_dict(config: AMIPExperimentConfig) -> dict:
    """Convert config to a plain dict for JSON serialization."""
    return config._asdict()


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
    config: AMIPExperimentConfig,
    diag_accumulators: dict | None = None,
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
    config : AMIPExperimentConfig
        Experiment configuration (saved alongside for reproducibility).
    diag_accumulators : dict, optional
        Any accumulated diagnostic arrays to preserve across restart.
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

    # Save config as JSON string inside npz
    arrays["config_json"] = np.array(json.dumps(config_to_dict(config)))

    # Save diagnostic accumulators if provided
    if diag_accumulators:
        for k, v in diag_accumulators.items():
            arrays[f"diag_{k}"] = np.asarray(v)

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
    """
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState

    data = np.load(str(path), allow_pickle=True)

    T = jnp.array(data["T"])
    u = jnp.array(data["u"])
    v = jnp.array(data["v"])
    p_s = jnp.array(data["p_s"])
    phis = jnp.array(data["phis"])
    q_v = jnp.array(data["q_v"])

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

    return state, q_v, step, day, config, diag_accumulators
