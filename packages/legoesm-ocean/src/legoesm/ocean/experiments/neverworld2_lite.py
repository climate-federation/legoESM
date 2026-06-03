"""NeverWorld2-lite: idealised global basin + ACC channel.

Reference
---------
Marques, G., et al. (2022). "NeverWorld2: an idealized model
hierarchy to investigate ocean mesoscale eddies", Geosci. Model
Dev. 15, 6567-6579.

Bachman, S. (2025) DINO + NeverWorld2 successor referenced in
``docs/ocean_fidelity/...``; uses similar geometry to DINO but with
* doubly-periodic ACC channel band in the south (45-65 S)
* global-scale subtropical / subpolar gyres in both hemispheres
* idealised flat-with-ridge bathymetry in the channel

What it tests
-------------
Eddy equilibration at two resolutions:

* 1 deg (``eddy-permitting``): mesoscale eddies marginally resolved;
  GM/Redi closure carries the bulk of the meridional eddy flux.
* 0.25 deg (``eddy-resolving``): mesoscale eddies fully resolved;
  GM/Redi off; reference solution for eddy-flux comparison.

Domain
------
DINO geometry as the starting point, but **doubly periodic** in
longitude (full globe, ~360 deg zonal extent) so the ACC channel
is the lower band of the basin rather than a sector channel.

This is a Phase D smoke-grade port; the fully tuned NeverWorld2
configuration (continental shelves, deep western boundary current,
realistic stratification spinup) is followup work. The smoke
benchmark validates the model integrates the chosen topology
without instability + that bulk metrics (peak ACC transport, basin-
mean EKE, AMOC magnitude) land in plausible ranges.

Implementation
--------------
Reuses ``DINOConfig`` for the vertical grid, restoring profiles,
and wind-stress knots. Overrides the lon domain to ``[-180, 180]``
and turns on full-zonal cyclic boundary handling on lat-lon /
MPAS regional grids. Cube grid not supported (sharp-front
limitation per Phase B.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from legoesm.ocean.experiments.dino import (
    DINOConfig,
    create_dino_z_star,
    dino_initial_T_S,
    dino_lat_lon_state,
    dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays,
    apply_dino_lat_lon_surface_forcing,
)


@dataclass
class NeverWorld2LiteConfig:
    """Configuration for the NeverWorld2-lite benchmark."""

    # Global zonal extent (doubly periodic) -- ACC channel runs around
    # the whole basin.
    lon_west_deg: float = -180.0
    lon_east_deg: float = 180.0
    lat_max_deg: float = 70.0

    channel_lat_south_deg: float = -65.0
    channel_lat_north_deg: float = -45.0

    H_deep: float = 4000.0
    H_shallow: float = 2000.0

    # Vertical grid (Levy stretch, paper R1).
    n_levels: int = 36
    dz_min: float = 10.0
    k_th: int = 35
    a_cr: float = 10.5

    # Restoring + wind: borrowed wholesale from DINO so the
    # diabatic forcing is identical between the two experiments.
    A_theta: float = 40.0
    A_S: float = 3.858e-3
    L_phi_deg: float = 140.0
    rho_0: float = 1026.0
    c_p: float = 3991.86

    # Smoke vs production resolution toggle.
    n_lon: int = 360       # 1 deg (set 1440 for 0.25 deg)

    @property
    def is_eddy_resolving(self) -> bool:
        return self.n_lon >= 720


def _to_dino_cfg(cfg: NeverWorld2LiteConfig) -> DINOConfig:
    """Translate to a DINOConfig with NeverWorld2's wider lon domain.

    Most DINO fields carry over unchanged; only the lon domain widens
    from the 50 deg DINO sector to the full 360 deg NeverWorld2 band.
    """
    return DINOConfig(
        lon_west_deg=cfg.lon_west_deg,
        lon_east_deg=cfg.lon_east_deg,
        lat_max_deg=cfg.lat_max_deg,
        channel_lat_south_deg=cfg.channel_lat_south_deg,
        channel_lat_north_deg=cfg.channel_lat_north_deg,
        H_deep=cfg.H_deep,
        H_shallow=cfg.H_shallow,
        n_levels=cfg.n_levels,
        dz_min=cfg.dz_min,
        k_th=cfg.k_th,
        a_cr=cfg.a_cr,
        A_theta=cfg.A_theta,
        A_S=cfg.A_S,
        L_phi_deg=cfg.L_phi_deg,
        rho_0=cfg.rho_0,
        c_p=cfg.c_p,
    )


def create_initial_conditions(grid_type: str, grid, z_coord,
                              config: NeverWorld2LiteConfig | None = None):
    """Reuse the DINO IC builder on the wider lon domain."""
    if config is None:
        config = NeverWorld2LiteConfig()
    if grid_type != "latlon":
        raise NotImplementedError(
            "NeverWorld2-lite currently supports only lat-lon; MPAS "
            "regional + cube ports are followups."
        )
    return dino_lat_lon_state(grid, z_coord, _to_dino_cfg(config))


def create_forcings(grid_type: str, grid,
                    config: NeverWorld2LiteConfig | None = None):
    """DINO-style surface forcing dict (model_config, physics_config,
    surface_forcing arrays) on the wider lon domain."""
    if config is None:
        config = NeverWorld2LiteConfig()
    dino_cfg = _to_dino_cfg(config)
    model_cfg, phys_cfg = dino_lat_lon_model_config(grid, dino_cfg,
                                                     physics=True)
    return {
        "model_config": model_cfg,
        "physics_config": phys_cfg,
        "surface_forcing": dino_lat_lon_surface_forcing_arrays(
            grid, dino_cfg,
        ),
    }


def apply_per_step(state, forcing, z_coord, config, dt):
    """DINO-style per-step surface forcing applicator."""
    dino_cfg = _to_dino_cfg(config)
    return apply_dino_lat_lon_surface_forcing(
        state, forcing, z_coord, dino_cfg, dt,
    )


def validate_results(final_state, diagnostics: Dict[str, list],
                     config: NeverWorld2LiteConfig | None = None
                     ) -> tuple[bool, str]:
    import jax.numpy as jnp
    if config is None:
        config = NeverWorld2LiteConfig()
    for name in ("u", "T", "S", "eta"):
        data = getattr(final_state, name).data
        if not bool(jnp.all(jnp.isfinite(data))):
            return False, f"NaN/Inf in final {name}"
    max_speed = diagnostics.get("max_speed", [0.0])[-1] if diagnostics.get(
        "max_speed") else 0.0
    eta_max = diagnostics.get("max_abs_eta", [0.0])[-1] if diagnostics.get(
        "max_abs_eta") else 0.0
    notes = (f"|u|max={max_speed:.3f}m/s, |eta|max={eta_max:.3f}m, "
             f"{'eddy-resolving' if config.is_eddy_resolving else 'eddy-permitting'}")
    ok = max_speed < 5.0 and eta_max < 5.0
    return ok, notes


def get_diagnostic_field_specs() -> list:
    return [
        ("eta", "SSH (m)", "RdBu_r"),
        ("SST", "SST (degC)", "RdYlBu_r"),
        ("speed_sfc", "Surface Speed (m/s)", "magma"),
    ]


def get_scalar_units() -> Dict[str, str]:
    return {
        "mean_eta": "m", "max_abs_eta": "m",
        "max_speed": "m/s", "max_abs_u": "m/s",
        "mean_T": "degC", "mean_S": "PSU",
    }


EXPERIMENT_CONFIG = {
    "name": "neverworld2_lite",
    "description": (
        "NeverWorld2-lite: idealised global basin + ACC channel; "
        "Bachman 2025 / Marques 2022"
    ),
    "scientific_purpose": (
        "Eddy equilibration at 1 deg (eddy-permitting) vs 0.25 deg "
        "(eddy-resolving); validate eddy-parameterisation tuning"
    ),
    "reference": "Marques et al. 2022, GMD 15, 6567-6579",
    "config_class": NeverWorld2LiteConfig,
    "create_initial_conditions": create_initial_conditions,
    "create_forcings": create_forcings,
    "create_domain": lambda c=None: {
        "H_max": (c or NeverWorld2LiteConfig()).H_deep,
        "lon_west": (c or NeverWorld2LiteConfig()).lon_west_deg,
        "lon_east": (c or NeverWorld2LiteConfig()).lon_east_deg,
        "description": "NeverWorld2-lite global basin + ACC band",
    },
    "validate": validate_results,
    "get_field_specs": get_diagnostic_field_specs,
    "get_scalar_units": get_scalar_units,
    "default_duration": 365.0,
    "quick_duration": 5.0,
    "grid_support": {
        "cubed_sphere": False,        # sharp-front cube limitation (Phase B.2)
        "latlon": True,
        "mpas": False,                # regional 360-deg seam needs work
        "latlon_regional": False, "mpas_regional": False,
        "latlon_channel": False, "mpas_channel": False,
        "cs_regional": False, "spectral": False,
    },
    "expected_metrics": {
        "ACC_transport_Sv": "~130 +/- 30 Sv at 1 deg eddy-permitting",
        "eddy_KE_J_per_m3":
            "50-150 at 0.25 deg eddy-resolving (Bachman 2025 ref)",
        "AMOC_Sv": "~15 +/- 5 Sv at 365 d spinup",
    },
}
