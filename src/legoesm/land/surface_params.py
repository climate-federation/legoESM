"""Spatially-varying land surface parameter container and helpers.

Provides ``LandSurfaceParams``, a NamedTuple of per-grid-cell parameter
arrays that can replace the scalar defaults in ``LandConfig`` /
``MultiLayerLandConfig``.  Three provider modes produce these arrays:

1. **Constant / Prescribed** — broadcast config scalars or load per-pixel data.
2. **PFT-based** — CLM5-style weighted-average over plant functional types.
3. **Neural** — small FCNN mapping static features to bounded parameters.

All arrays have shape ``(ncol,)`` (column-major).  For cubed-sphere slab
land with spatial shape ``(6, n, n)``, use ``reshape_params`` before
passing to ``step_land``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


# =====================================================================
# Parameter bounds  (lo, hi) — used by sigmoid transforms in providers
# =====================================================================

PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "albedo_veg": (0.05, 0.40),
    "emissivity": (0.90, 0.99),
    "z0": (0.001, 2.0),
    "W_max": (50.0, 500.0),
    "C_soil": (1.0e6, 5.0e6),
    "d_soil": (0.3, 3.0),
    "root_depth": (0.1, 5.0),
    "theta_wp": (0.05, 0.25),
    "theta_fc": (0.15, 0.45),
    "Vc_max25": (10.0, 120.0),
    "LCMA": (20.0, 120.0),
    "g1": (1.0, 15.0),
    # CLM-ML-JAX canopy scheme parameters
    "LAI": (0.0, 10.0),        # Leaf area index [m2/m2]
    "SAI": (0.0, 3.0),         # Stem area index [m2/m2]
    "htop": (0.1, 50.0),       # Canopy top height [m]
    "hbot": (0.0, 10.0),       # Canopy bottom height [m]
}

# Ordered parameter names — columns of the (ncol, n_params) matrix
PARAM_NAMES: tuple[str, ...] = tuple(PARAM_BOUNDS.keys())
N_PARAMS: int = len(PARAM_NAMES)


# =====================================================================
# LandSurfaceParams container
# =====================================================================

class LandSurfaceParams(NamedTuple):
    """Per-grid-cell land surface parameters.

    Every field is a ``jax.Array`` with shape ``(ncol,)`` or a scalar
    that broadcasts.  Step functions read from this when provided,
    falling back to config scalars when ``land_params is None``.
    """
    # Surface properties (slab + multilayer)
    albedo_veg: jax.Array     # Vegetation base albedo [0.05, 0.40]
    emissivity: jax.Array     # Surface emissivity [0.90, 0.99]
    z0: jax.Array             # Roughness length [0.001, 2.0] m
    # Slab hydrology
    W_max: jax.Array          # Bucket capacity [50, 500] kg/m2
    C_soil: jax.Array         # Soil heat capacity [1e6, 5e6] J/m3/K
    d_soil: jax.Array         # Slab depth [0.3, 3.0] m
    # Root zone (multilayer)
    root_depth: jax.Array     # Root e-folding depth [0.1, 5.0] m
    theta_wp: jax.Array       # Wilting point [0.05, 0.25]
    theta_fc: jax.Array       # Field capacity [0.15, 0.45]
    # Plant physiology
    Vc_max25: jax.Array       # Max carboxylation at 25 C [10, 120] umol/m2/s
    LCMA: jax.Array           # Leaf carbon mass per area [20, 120] gC/m2
    g1: jax.Array             # Stomatal slope [1, 15]
    # CLM-ML-JAX canopy scheme parameters (optional; None when not used)
    LAI: jax.Array | None = None   # Leaf area index [0, 10] m2/m2
    SAI: jax.Array | None = None   # Stem area index [0, 3] m2/m2
    htop: jax.Array | None = None  # Canopy foliage top height [0.1, 50] m
    hbot: jax.Array | None = None  # Canopy foliage bottom height [0, 10] m


# =====================================================================
# Helpers
# =====================================================================

def default_land_surface_params(ncol: int, config) -> LandSurfaceParams:
    """Broadcast config scalars to ``(ncol,)`` arrays.

    Parameters
    ----------
    ncol : int
        Number of grid columns.
    config : LandConfig or MultiLayerLandConfig
        Land model configuration with scalar defaults.

    Returns
    -------
    LandSurfaceParams
    """
    def _bc(val):
        return jnp.full(ncol, val)

    # Map LandSurfaceParams field names to config attributes.
    # MultiLayerLandConfig has root_depth/theta_wp/theta_fc directly;
    # LandConfig does not, so use sensible defaults.
    root_depth = getattr(config, "root_depth", 1.0)
    theta_wp = getattr(config, "theta_wp", 0.15)
    theta_fc = getattr(config, "theta_fc", 0.30)
    d_soil = getattr(config, "d_soil", 1.0)
    W_max = getattr(config, "W_max", 150.0)
    # C_soil is top-level on LandConfig but in thermal sub-config on MultiLayer
    if hasattr(config, "C_soil"):
        C_soil = config.C_soil
    else:
        C_soil = getattr(config, "thermal", None)
        C_soil = C_soil.C_soil if C_soil is not None else 2.0e6

    return LandSurfaceParams(
        albedo_veg=_bc(config.albedo_land),
        emissivity=_bc(config.emissivity_land),
        z0=_bc(config.z0_land),
        W_max=_bc(W_max),
        C_soil=_bc(C_soil),
        d_soil=_bc(d_soil),
        root_depth=_bc(root_depth),
        theta_wp=_bc(theta_wp),
        theta_fc=_bc(theta_fc),
        Vc_max25=_bc(config.stomata.Vcmax25_C3),
        LCMA=_bc(config.carbon.LCMA),
        g1=_bc(config.stomata.g1_bb),
    )


def array_to_params(arr: jnp.ndarray, param_names: tuple[str, ...] | None = None) -> LandSurfaceParams:
    """Slice columns of ``(ncol, n_params)`` array into a LandSurfaceParams.

    Parameters
    ----------
    arr : jnp.ndarray
        Shape ``(ncol, n_params)``.
    param_names : tuple of str, optional
        Column order.  Defaults to ``PARAM_NAMES``.
    """
    if param_names is None:
        param_names = PARAM_NAMES
    if arr.shape[-1] != len(param_names):
        raise ValueError(
            f"Expected {len(param_names)} columns, got {arr.shape[-1]}")
    fields = {name: arr[..., i] for i, name in enumerate(param_names)}
    return LandSurfaceParams(**fields)


def reshape_params(params: LandSurfaceParams, shape: tuple[int, ...]) -> LandSurfaceParams:
    """Reshape every ``(ncol,)`` field to *shape* (C-order / row-major).

    Used to convert flat ``(ncol,)`` arrays to cubed-sphere ``(6, n, n)``
    before passing to slab land.
    """
    return jax.tree.map(lambda x: x.reshape(shape), params)


def bounds_arrays() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(lo, hi)`` arrays of shape ``(n_params,)`` in ``PARAM_NAMES`` order."""
    lo = jnp.array([PARAM_BOUNDS[n][0] for n in PARAM_NAMES])
    hi = jnp.array([PARAM_BOUNDS[n][1] for n in PARAM_NAMES])
    return lo, hi


# =====================================================================
# Default CLM5 PFT table
# =====================================================================

# 17 CLM5 plant functional types with representative parameter values.
# Sources: CLM5 Technical Note (Lawrence et al. 2019) Tables 2.2-2.5,
# Bonan et al. (2011), and Oleson et al. (2013).
#
# Columns follow PARAM_NAMES order:
#   albedo_veg, emissivity, z0, W_max, C_soil, d_soil,
#   root_depth, theta_wp, theta_fc, Vc_max25, LCMA, g1
#
# fmt: off
CLM5_PFT_NAMES: tuple[str, ...] = (
    "bare_soil",
    "needleleaf_evergreen_temperate",
    "needleleaf_evergreen_boreal",
    "needleleaf_deciduous_boreal",
    "broadleaf_evergreen_tropical",
    "broadleaf_evergreen_temperate",
    "broadleaf_deciduous_tropical",
    "broadleaf_deciduous_temperate",
    "broadleaf_deciduous_boreal",
    "broadleaf_evergreen_shrub",
    "broadleaf_deciduous_temperate_shrub",
    "broadleaf_deciduous_boreal_shrub",
    "c3_arctic_grass",
    "c3_grass",
    "c4_grass",
    "crop_c3",
    "crop_c4",
)

# Each row: one PFT; each column: one parameter in PARAM_NAMES order
_CLM5_PFT_TABLE_RAW: list[list[float]] = [
    # bare soil
    [0.30, 0.96, 0.001,  50.0, 2.0e6, 1.0, 0.1, 0.10, 0.20,  0.0, 50.0, 1.0],
    # needleleaf evergreen temperate
    [0.12, 0.97, 1.00, 250.0, 2.5e6, 1.5, 2.0, 0.10, 0.25, 62.0, 80.0, 6.0],
    # needleleaf evergreen boreal
    [0.13, 0.97, 0.80, 200.0, 2.5e6, 1.5, 1.5, 0.10, 0.25, 43.0, 80.0, 6.0],
    # needleleaf deciduous boreal
    [0.14, 0.97, 0.80, 200.0, 2.5e6, 1.5, 1.5, 0.10, 0.25, 43.0, 70.0, 6.0],
    # broadleaf evergreen tropical
    [0.14, 0.98, 2.00, 200.0, 2.0e6, 1.0, 1.5, 0.12, 0.30, 55.0, 40.0, 9.0],
    # broadleaf evergreen temperate
    [0.15, 0.98, 1.50, 220.0, 2.0e6, 1.0, 1.8, 0.12, 0.30, 61.0, 50.0, 9.0],
    # broadleaf deciduous tropical
    [0.16, 0.98, 2.00, 200.0, 2.0e6, 1.0, 1.5, 0.12, 0.30, 41.0, 40.0, 9.0],
    # broadleaf deciduous temperate
    [0.17, 0.97, 1.00, 200.0, 2.0e6, 1.0, 1.5, 0.12, 0.30, 58.0, 45.0, 9.0],
    # broadleaf deciduous boreal
    [0.18, 0.97, 0.80, 200.0, 2.0e6, 1.0, 1.2, 0.12, 0.30, 46.0, 45.0, 9.0],
    # broadleaf evergreen shrub
    [0.18, 0.97, 0.10, 150.0, 2.0e6, 0.8, 0.8, 0.10, 0.25, 17.0, 50.0, 6.0],
    # broadleaf deciduous temperate shrub
    [0.20, 0.97, 0.10, 150.0, 2.0e6, 0.8, 0.8, 0.10, 0.25, 33.0, 50.0, 6.0],
    # broadleaf deciduous boreal shrub
    [0.20, 0.97, 0.10, 150.0, 2.0e6, 0.8, 0.8, 0.10, 0.25, 33.0, 50.0, 6.0],
    # c3 arctic grass
    [0.20, 0.96, 0.03, 100.0, 2.0e6, 0.5, 0.5, 0.10, 0.25, 43.0, 30.0, 5.0],
    # c3 grass
    [0.20, 0.96, 0.03, 100.0, 2.0e6, 0.5, 0.5, 0.10, 0.25, 43.0, 30.0, 5.0],
    # c4 grass
    [0.20, 0.96, 0.03, 100.0, 2.0e6, 0.5, 0.5, 0.10, 0.25, 24.0, 30.0, 4.0],
    # crop c3
    [0.18, 0.96, 0.06, 150.0, 2.0e6, 0.5, 0.5, 0.10, 0.25, 50.0, 35.0, 5.0],
    # crop c4
    [0.18, 0.96, 0.06, 150.0, 2.0e6, 0.5, 0.5, 0.10, 0.25, 30.0, 35.0, 4.0],
]
# fmt: on

N_PFT_CLM5: int = len(CLM5_PFT_NAMES)

# Lazy-converted to jnp array on first use to avoid import-time JAX init
_clm5_table_cache: jnp.ndarray | None = None


def clm5_pft_table() -> jnp.ndarray:
    """Return the CLM5 PFT lookup table as ``(17, 12)`` jnp.ndarray."""
    global _clm5_table_cache
    if _clm5_table_cache is None:
        _clm5_table_cache = jnp.array(_CLM5_PFT_TABLE_RAW)
    return _clm5_table_cache
