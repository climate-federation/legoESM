"""Ocean test matrix configuration constants and results tracking."""

from __future__ import annotations

from typing import Any

from legoesm import constants as _C

# ===========================================================================
# Configuration constants
# ===========================================================================

# ~2.5 degree resolutions per grid type
GRID_RESOLUTIONS: dict[str, str] = {
    "cubed_sphere": "C24",
    "latlon": "36x72",
    "mpas": "ico3",
    "mpas_regional": "300km",
    "latlon_regional": "24x48",
    "cs_regional": "C24",
    "latlon_channel": "24x72",
    "mpas_channel": "300km",
    "spectral": "T21",
    # FESOM is mesh-file-backed: "pi" is the only packaged mesh.
    "fesom": "pi",
}

# Standard grid types for the full test matrix.
# Regional grids are only added to specific test cases (gyre experiments).
# Channel grids are zonally periodic latitude bands (e.g. Eady instability).
GRID_TYPES = ["cubed_sphere", "latlon", "mpas"]
REGIONAL_GRID_TYPES = ["mpas_regional", "latlon_regional", "cs_regional"]
CHANNEL_GRID_TYPES = ["latlon_channel", "mpas_channel"]

DEFAULT_NLEV = 10
DEFAULT_H_MAX = 5500.0
DEFAULT_DT = 300.0  # seconds (scaled for ~2.5 deg resolution CFL)
TRACER_ADVECTION_OVERRIDE = None  # Set via --tracer-advection CLI flag
NO_SPONGE = False  # Set via --no-sponge CLI flag
B_H_OVERRIDE = None  # Set via --B-h CLI flag
C_SMAG_OVERRIDE = None  # Set via --C-smag CLI flag
K_H_OVERRIDE = None  # Set via --K-h CLI flag
U_SURFACE_OVERRIDE = None  # Set via --U-surface CLI flag (Eady)
PV_SCHEME_OVERRIDE = None  # Set via --pv-scheme CLI flag (MPAS only)
APVM_DT_OVERRIDE = None    # Set via --apvm-dt CLI flag (MPAS only; seconds; 0 disables)
PV_ALPHA_OVERRIDE = None   # Set via --pv-alpha CLI flag (MPAS only; used when pv_scheme="mixed")
K_ZETA_BIH_OVERRIDE = None # Set via --K-zeta-bih CLI flag (MPAS only; m⁴/s; biharmonic ζ damping)
C_LEITH_OVERRIDE = None    # Set via --C-leith CLI flag (MPAS only; flow-dependent biharmonic viscosity)
C_LEITH_MODIFIED_OVERRIDE = None  # Set via --C-leith-modified CLI flag (uses |∇(ζ+Q_d)| instead of |∇ζ|)
BAROTROPIC_DIV_DAMP_OVERRIDE = None  # Set via --barotropic-div-damp CLI flag
                                     # (dimensionless; targets grid-scale
                                     # compressible modes — see issue #213
                                     # hi-res Eady SOM investigation recipe)
MOMENTUM_ADVECTION_OVERRIDE = None   # Set via --momentum-advection CLI flag
                                     # (latlon C-grid only: "vector_invariant",
                                     # "weno5", or "weno7")
WENO_D_TERM_OVERRIDE = None          # Set via --no-weno-d-term CLI flag
                                     # (disables the D-term in WENO momentum)

# Output format: "netcdf" (default), "zarr", or "npz" (legacy)
OUTPUT_FORMAT = "netcdf"

# Physical constants for idealized ocean test cases — use canonical values.
_A_EARTH = _C.R_earth   # Earth radius (m)
_OMEGA_E = _C.Omega      # Earth rotation rate (rad/s)
_G_EARTH = _C.g           # gravitational acceleration (m/s^2)

# Field ranges for consistent plotting across grid types
FIELD_RANGES = {
    "rest_state_stratified_with_land": {
        "eta": (-1e-6, 1e-6),      # meters - rest state should have tiny SSH
        "SST": (1.5, 21.0),        # °C - range from deep to surface T
    },
    "rest_state_stratified_no_land": {
        "eta": (-1e-6, 1e-6),      # meters - rest state should have tiny SSH (pure ocean)
        "SST": (1.5, 21.0),        # °C - range from deep to surface T
    },
    "barotropic_wave": {
        "eta": (-1.5, 1.5),        # meters - wave amplitude ~1m
        "SST": (1.5, 21.0),        # °C - background temperature range
    },
    "barotropic_gyre": {
        "eta": (-0.02, 0.02),      # meters - gyre SSH (small after quick spin-up)
        "speed_sfc": (0, 0.15),     # m/s - surface speeds
        "SST": (9.5, 10.5),        # °C - uniform 10°C (barotropic)
    },
    "barotropic_double_gyre": {
        "eta": (-0.02, 0.02),      # meters - gyre SSH (small after quick spin-up)
        "speed_sfc": (0, 0.15),     # m/s - surface speeds
        "SST": (9.5, 10.5),        # °C - uniform 10°C (barotropic)
    },
    "barotropic_double_gyre_sin2": {
        "eta": (-0.02, 0.02),      # meters - gyre SSH (sin² wind, slight E-W tilt)
        "speed_sfc": (0, 0.15),     # m/s - surface speeds
        "SST": (9.5, 10.5),        # °C - uniform 10°C (barotropic)
    },
    "baroclinic_gyre": {
        "eta": (-0.05, 0.05),      # meters - larger SSH with baroclinic dynamics
        "speed_sfc": (0, 0.5),      # m/s - higher speeds with thermal wind
        "SST": (None, None),        # °C - adaptive range for circulation-driven T patterns
    },
    "baroclinic_gyre_cos": {
        "eta": (-0.05, 0.05),      # meters - larger SSH with baroclinic dynamics
        "speed_sfc": (0, 0.5),      # m/s - higher speeds with thermal wind
        "SST": (None, None),        # °C - adaptive range for circulation-driven T patterns
    },
    "geostrophic_adjustment": {
        "eta": (-0.1, 0.1),        # meters - adjustment process
        "SST": (1.5, 21.0),        # °C - background temperature range
    },
    "phillips_two_layer": {
        "eta": (-0.2, 0.2),        # meters - 2-layer dynamics
        "SST": (8, 16),             # °C - 2-layer temperature range
    },
    "inertia_gravity_wave": {
        "eta": (-1.2, 1.2),        # meters - IGW amplitude
        "SST": (1.5, 21.0),        # °C
    },
    "lock_exchange": {
        "eta": (-0.05, 0.05),      # meters - density current adjustment
        "SST": (-1, 21),            # °C - cold/warm water exchange
    },
    "overflow": {
        "eta": (-0.1, 0.1),        # meters - dense water overflow
        "SST": (-1, 21),            # °C - cold dense water
    },
    "stommel_gyre_tracer": {
        "eta": (-0.02, 0.02),      # meters - gyre circulation (small during spin-up)
        "SST": (1.5, 21.0),        # °C
        "SSS": (33, 37),            # PSU - tracer salinity range
    },
    "acc_channel": {
        "eta": (-0.1, 0.1),        # meters - wind-driven SSH response
        "speed_sfc": (0, 0.2),      # m/s - zonal flow over ridge
        "SST": (0, 9),              # °C - Abernathey profile range
    },
    "acc_channel_rest": {
        "eta": (-1e-4, 1e-4),      # meters - should stay near zero
        "speed_sfc": (0, 0.01),     # m/s - should stay near zero
        "SST": (0, 9),              # °C - should maintain initial profile
    },
}

# ===========================================================================
# Results tracking
# ===========================================================================

ALL_RESULTS: list[dict[str, Any]] = []


def record(tc, status: str, wall_time: float, notes: str = ""):
    """Record a test result and print a summary line.

    Parameters
    ----------
    tc : TestCase
        The test case (imported from testcase module to avoid circular import;
        typed loosely here so config has no dependency on testcase).
    status : str
        One of "PASS", "FAIL", "ERROR", "SKIP".
    wall_time : float
        Wall-clock seconds for the run.
    notes : str, optional
        Additional information about the result.
    """
    icon = {"PASS": "  ", "FAIL": "**", "ERROR": "!!", "SKIP": "--"}[status]
    ALL_RESULTS.append({
        "test": tc.case, "grid": tc.grid_type,
        "resolution": tc.resolution, "status": status,
        "wall_time": wall_time, "notes": notes,
    })
    label = f"{tc.case}/{tc.grid_type}/{tc.resolution}"
    print(f"  {icon} {status:5s} | {label:<45s} | {wall_time:7.1f}s | {notes}")
