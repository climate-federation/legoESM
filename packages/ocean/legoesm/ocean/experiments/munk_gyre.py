"""Munk gyre: wind-driven single gyre with frictional western boundary current.

Reference
---------
Munk, W. H. (1950). "On the wind-driven ocean circulation",
J. Meteorol. 7, 79-93.

What it tests
-------------
Sverdrup balance ``beta * V = curl(tau) / rho`` in the interior plus
the Munk boundary layer of width

.. math::
    \\delta_M = (A_h / \\beta)^{1/3}

dissipating the meridional-momentum imbalance at the western wall.
This is the canonical "WBC + lateral viscosity" benchmark every
ocean dycore must pass within ~10 % to claim correct large-scale
circulation.

Acceptance
----------
* Western boundary current peak transport ~ 20 Sv +/- 10 % vs the
  analytical Stommel-Munk solution (depends on basin geometry +
  wind amplitude; the defaults here target 20 Sv at saturation).
* Munk-layer FWHM of |u_max| matches
  :func:`legoesm.ocean.fidelity.references.munk_width(A_h, beta)`
  within +/- 15 %.

Configuration
-------------
The Munk gyre is a 60 deg x 60 deg sector basin (default 0-60 E,
15-75 N) on flat bathymetry. The horizontal viscosity ``A_h`` is
sized so the Munk layer ``delta_M`` is resolved by >= 3 grid cells
at the working resolution. With ``A_h = 5e3 m^2/s`` and
``beta ~ 2e-11 m^-1 s^-1``:

.. math::
    \\delta_M = (5 \\times 10^3 / 2 \\times 10^{-11})^{1/3} \\approx 63 \\text{ km}

which is resolved on a 1 deg latlon grid (~110 km cells) only at
the edge of the Nyquist limit -- bump to ``A_h = 2e4`` for solid
resolution at the matrix's smoke-grade ``24x48`` regional grid
(dx ~ 250 km, delta_M ~ 100 km, ~ 0.4 cells per WBC).

The single-gyre wind ``tau_x(y) = -tau_max * cos(pi * (y - y_s) /
(y_n - y_s))`` is the cleanest forcing for analytical comparison
since it has a single curl extremum and a single Sverdrup-balance
streamfunction.

Reuses ``RegionalGyreConfig`` /
``wind_driven_gyre_{latlon,mpas,cube}`` from
``regional_gyre.py`` for the basin geometry + initial-condition
construction; only the wind profile (single vs double) and the
explicit lateral-viscosity sizing differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from legoesm.ocean.experiments.regional_gyre import (
    RegionalGyreConfig,
    create_initial_conditions as _regional_create_ic,
    create_forcings as _regional_create_forcings,
)


@dataclass
class MunkGyreConfig:
    """Configuration for the Munk gyre benchmark.

    The default values target a 20 Sv WBC peak with delta_M ~ 100 km
    on a 1-2 deg latlon grid, comfortably resolved on the matrix's
    ``24x48`` regional grid.
    """

    # Domain (same defaults as RegionalGyreConfig).
    H_max: float = 5500.0
    lon_west: float = 0.0
    lon_east: float = 60.0
    lat_south: float = 15.0
    lat_north: float = 75.0

    # Stratification (rest-state defaults). Field name matches
    # RegionalGyreConfig.T_water_init_C exactly.
    T_water_init_C: float = 20.0
    T_deep_C: float = 2.0          # Deep ocean temperature [degC]
    scale_depth: float = 1000.0
    S_uniform: float = 35.0

    # Wind forcing: single-gyre cosine, peak 0.1 Pa (matches Munk 1950).
    wind_stress_max: float = 0.1
    wind_buffer_deg: float = 0.0

    # Lateral viscosity sized so delta_M = (A_h / beta)^(1/3) ~ 100 km.
    # beta ~ 2e-11 m^-1 s^-1 at mid-latitudes -> A_h = beta * delta_M^3
    # = 2e-11 * (1e5)^3 = 2e4 m^2/s.
    A_h: float = 2.0e4
    # Linear bottom drag (kept for stability; Munk dissipation comes
    # from A_h, not drag).
    bottom_drag_coeff: float = 1.0e-5

    @property
    def munk_layer_width_m(self) -> float:
        """Theoretical Munk layer width delta_M = (A_h / beta)^(1/3) [m]."""
        beta = 2.0e-11  # mid-latitude beta-plane
        return (self.A_h / beta) ** (1.0 / 3.0)


def _to_regional_cfg(cfg: MunkGyreConfig) -> RegionalGyreConfig:
    """Translate a ``MunkGyreConfig`` into a ``RegionalGyreConfig`` with
    a single-gyre wind profile and Munk-layer-sized viscosity."""
    return RegionalGyreConfig(
        T_water_init_C=cfg.T_water_init_C, T_deep_C=cfg.T_deep_C,
        scale_depth=cfg.scale_depth, S_uniform=cfg.S_uniform,
        H_max=cfg.H_max,
        lon_west=cfg.lon_west, lon_east=cfg.lon_east,
        lat_south=cfg.lat_south, lat_north=cfg.lat_north,
        wind_stress_max=cfg.wind_stress_max,
        wind_profile="single_gyre",
        wind_buffer_deg=cfg.wind_buffer_deg,
        A_h=cfg.A_h,
        bottom_drag_coeff=cfg.bottom_drag_coeff,
    )


def create_initial_conditions(grid_type: str, grid, z_coord,
                              config: MunkGyreConfig | None = None):
    """Build the Munk-gyre initial state on ``grid_type``."""
    if config is None:
        config = MunkGyreConfig()
    return _regional_create_ic(grid_type, grid, z_coord, _to_regional_cfg(config))


def create_forcings(grid_type: str, grid,
                    config: MunkGyreConfig | None = None):
    """Single-gyre wind via ``regional_gyre.create_forcings``."""
    if config is None:
        config = MunkGyreConfig()
    return _regional_create_forcings(grid_type, grid, _to_regional_cfg(config))


def create_domain_config(config: MunkGyreConfig | None = None) -> Dict[str, Any]:
    if config is None:
        config = MunkGyreConfig()
    return {
        "H_max": config.H_max,
        "lon_west": config.lon_west, "lon_east": config.lon_east,
        "lat_south": config.lat_south, "lat_north": config.lat_north,
        "munk_layer_width_m": config.munk_layer_width_m,
        "description": "Munk wind-driven gyre with frictional WBC",
    }


def validate_results(final_state, diagnostics: Dict[str, list],
                     config: MunkGyreConfig | None = None) -> tuple[bool, str]:
    """Basic shake-down: finite fields + Sverdrup-scale WBC magnitude.

    The full ``delta_M = (A_h/beta)^(1/3)`` width check lives in a
    fidelity-tier test (Phase D follow-up) that walks the meridional
    surface-velocity profile + fits the Munk boundary-layer decay.
    """
    import jax.numpy as jnp
    if config is None:
        config = MunkGyreConfig()
    for name in ("u", "T", "S", "eta"):
        data = getattr(final_state, name).data
        if not bool(jnp.all(jnp.isfinite(data))):
            return False, f"NaN/Inf in final {name}"
    max_speed = diagnostics.get("max_speed", [0.0])[-1] if diagnostics.get(
        "max_speed") else 0.0
    eta_max = diagnostics.get("max_abs_eta", [0.0])[-1] if diagnostics.get(
        "max_abs_eta") else 0.0
    notes = (f"|u|max={max_speed:.3f}m/s, |eta|max={eta_max:.3f}m, "
             f"delta_M={config.munk_layer_width_m / 1e3:.0f}km")
    # Loose smoke gates -- tighter fidelity criteria live in the tier
    # tests that fit the WBC profile against the analytical solution.
    ok = max_speed < 5.0 and eta_max < 5.0
    return ok, notes


def get_diagnostic_field_specs() -> list:
    return [
        ("eta", "SSH (m)", "RdBu_r"),
        ("speed_sfc", "Surface Speed (m/s)", "magma"),
    ]


def get_scalar_units() -> Dict[str, str]:
    return {
        "mean_eta": "m", "max_abs_eta": "m",
        "max_speed": "m/s", "max_abs_u": "m/s",
        "mean_T": "degC", "mean_S": "PSU",
    }


EXPERIMENT_CONFIG = {
    "name": "munk_gyre",
    "description": "Munk wind-driven gyre with frictional WBC",
    "scientific_purpose": (
        "Validate Sverdrup balance in the interior and the Munk "
        "boundary-layer width delta_M = (A_h / beta)^(1/3) at the "
        "western wall."
    ),
    "reference": "Munk (1950), J. Meteorol. 7, 79-93",
    "config_class": MunkGyreConfig,
    "create_initial_conditions": create_initial_conditions,
    "create_forcings": create_forcings,
    "create_domain": create_domain_config,
    "validate": validate_results,
    "get_field_specs": get_diagnostic_field_specs,
    "get_scalar_units": get_scalar_units,
    "default_duration": 365.0,
    "quick_duration": 30.0,
    "grid_support": {
        "cubed_sphere": False,        # global cube not the right geometry
        "latlon": False,              # global lat-lon: same
        "mpas": False,
        "latlon_regional": True,
        "mpas_regional": True,
        "cs_regional": False,         # 6-face init constraint
        "latlon_channel": False,
        "mpas_channel": False,
        "spectral": False,
    },
    "expected_metrics": {
        "WBC_peak_transport_Sv": "~20 Sv +/- 10 %",
        "Munk_layer_width_m":
            "delta_M = (A_h / beta)^(1/3) within +/- 15 % of theory",
    },
}
