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
}
# NOTE: the optional CLM-ML-JAX canopy inputs (``LAI``, ``SAI``, ``htop``,
# ``hbot``) are deliberately NOT in ``PARAM_BOUNDS``.  ``PARAM_BOUNDS`` /
# ``PARAM_NAMES`` index the 12-column CLM5 PFT lookup table
# (``clm5_pft_table`` -> ``(n_pft, 12)``) that every ``array_to_params`` caller
# (``gap_fill``, ``param_providers``, ``clm_surface_map``, land-param training)
# maps column-for-column; adding the 4 canopy names here desynchronises them
# from the 12-column table and raises "Expected 16 columns, got 12".  The canopy
# fields are instead prescribed through the surfdata / boundary-data path:
# ``LAI``, ``SAI`` and ``htop`` are read with a ``None`` fallback in
# ``canopy/clm_ml_interface.py``; ``hbot`` is currently DERIVED there as
# ``CLMMLCanopyConfig.hbot_frac * htop`` (the field is reserved for a future
# explicit per-column bottom height and is not yet consumed).

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
    # Prescribed leaf area index [m2/m2] — the two-leaf canopy reads this as its
    # spatial LAI climatology (PFT-weighted CLM MONTHLY_LAI); the CLM-ML-JAX canopy
    # reads it as its LAI input.  None => the canopy falls back to its scalar
    # default (only for non-CLM setups); a NEW CLM/two-leaf/CLM-ML path must
    # populate it so barren land gets LAI~0, not a spurious uniform canopy.
    LAI: jax.Array | None = None   # Leaf area index [0, 10] m2/m2
    # CLM-ML-JAX canopy scheme parameters (optional; None when not used)
    SAI: jax.Array | None = None   # Stem area index [0, 3] m2/m2
    htop: jax.Array | None = None  # Canopy foliage top height [0.1, 50] m
    hbot: jax.Array | None = None  # Canopy foliage bottom height [0, 10] m
    # Big-leaf (SimpleSEB) C4 flag — the DOMINANT PFT's C4 flag (0/1) in
    # production, matching the two-leaf canopy's fC4 origin. Selects the pure C3
    # or pure C4 canonical-FvCB pathway in the coupled A-gs solver
    # (``land/stomata.py::solve_coupled_farquhar_ci``); a single big leaf is one
    # pathway. None => pure C3 (the correct default for a column with no C4 data;
    # e.g. the prescribed-PFT / idealized paths that do not populate it).
    fC4: jax.Array | None = None   # dominant-PFT C4 flag [0, 1]


# =====================================================================
# Helpers
# =====================================================================

# Default land-surface parameter fallbacks (mirror LandConfig defaults; used
# when a passed config lacks the attribute).
_THETA_WP_DEFAULT = 0.15   # wilting-point water content [m3/m3]
_THETA_FC_DEFAULT = 0.30   # field-capacity water content [m3/m3]
_W_MAX_DEFAULT = 150.0     # bucket capacity [kg/m2]
_ROOT_DEPTH_DEFAULT = 1.0  # root-zone depth [m] (LandConfig lacks the field)
_D_SOIL_DEFAULT = 1.0      # soil thermal-column depth [m] (LandConfig fallback)
_C_SOIL_DEFAULT = 2.0e6    # soil volumetric heat capacity [J/m3/K]

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
    root_depth = getattr(config, "root_depth", _ROOT_DEPTH_DEFAULT)
    theta_wp = getattr(config, "theta_wp", _THETA_WP_DEFAULT)
    theta_fc = getattr(config, "theta_fc", _THETA_FC_DEFAULT)
    d_soil = getattr(config, "d_soil", _D_SOIL_DEFAULT)
    W_max = getattr(config, "W_max", _W_MAX_DEFAULT)
    # C_soil is top-level on LandConfig but in thermal sub-config on MultiLayer
    if hasattr(config, "C_soil"):
        C_soil = config.C_soil
    else:
        C_soil = getattr(config, "thermal", None)
        C_soil = C_soil.C_soil if C_soil is not None else _C_SOIL_DEFAULT

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
        Vc_max25=_bc(config.stomata.Vc_max25),
        LCMA=_bc(config.carbon.LCMA),
        g1=_bc(config.stomata.g1_bb),
    )


def read_spatial_param(lp, name: str, fallback):
    """Read field ``name`` from spatial :class:`LandSurfaceParams` ``lp`` when
    present, else return the config scalar ``fallback``.  Shared by the slab
    and multilayer land step modules (both wire trainable per-column surface
    params the same way)."""
    return getattr(lp, name) if lp is not None else fallback


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
# NOTE: the C4 PFT rows (c4_grass, crop_c4) are consumed by the C3-only
# Farquhar biochemistry in land/stomata.py -- their photosynthesis is
# run through C3 kinetics as a documented approximation, NOT a Collatz (1992)
# C4 scheme, so C4 CO2 sensitivity / compensation point are not represented.
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
    # c4 grass  (C4 PFT run through C3 Farquhar kinetics -- approximation)
    [0.20, 0.96, 0.03, 100.0, 2.0e6, 0.5, 0.5, 0.10, 0.25, 24.0, 30.0, 4.0],
    # crop c3
    [0.18, 0.96, 0.06, 150.0, 2.0e6, 0.5, 0.5, 0.10, 0.25, 50.0, 35.0, 5.0],
    # crop c4  (C4 PFT run through C3 Farquhar kinetics -- approximation)
    [0.18, 0.96, 0.06, 150.0, 2.0e6, 0.5, 0.5, 0.10, 0.25, 30.0, 35.0, 4.0],
]
# fmt: on

N_PFT_CLM5: int = len(CLM5_PFT_NAMES)


# ---------------------------------------------------------------------------
# PFT growth-form / leaf-habit classifiers
# ---------------------------------------------------------------------------
# The CLM5_PFT_NAMES strings encode growth form and leaf habit as substrings
# (``grass`` / ``crop`` mark herbaceous PFTs; ``*_evergreen_*`` vs ``*_deciduous_*``
# mark the leaf habit), so a substring match cleanly classifies a PFT.  SINGLE
# source of truth: shared by the archetype IC builder (``carbon.global_init``,
# which groups archetypes by ``(is_woody, is_evergreen, soil_class)``) and its
# per-pixel validator (``scripts/validate/land_carbon_equilibrium.py``) so the
# two never drift apart (a drift would let the validator check different physics
# than the IC ships).
_HERBACEOUS_PFT_TAGS: tuple[str, ...] = ("grass", "crop")


def is_woody(pft_name: str) -> bool:
    """True for woody PFTs (trees/shrubs); False for grasses/crops (herbaceous).

    Woodiness selects the ``CarbonConfig.woody`` branch (wood pool + wood
    allocation) for a PFT's carbon archetype.
    """
    return not any(tag in pft_name for tag in _HERBACEOUS_PFT_TAGS)


def is_evergreen(pft_name: str) -> bool:
    """True for evergreen PFTs (continuous leaf turnover); False otherwise
    (deciduous / grass / crop).

    The leaf habit is encoded in the CLM5 PFT name (``needleleaf_evergreen_boreal``
    / ``broadleaf_evergreen_tropical`` vs the ``*_deciduous_*`` / grass / crop
    names), so a substring match on ``evergreen`` separates the continuous-turnover
    branch of ``carbon.carbon_cycle.compute_phenology`` from the deciduous
    DALEC990 Gaussian pulse.
    """
    return "evergreen" in pft_name


_COLD_DECIDUOUS_PFTS = frozenset({
    "needleleaf_deciduous_boreal",
    "c3_arctic_grass",
    "broadleaf_deciduous_boreal_shrub",
})


def is_cold_deciduous(pft_name: str) -> bool:
    """True for cold-deciduous / winter-dormant high-latitude PFTs.

    These PFTs (larch ``needleleaf_deciduous_boreal``, arctic graminoids
    ``c3_arctic_grass``, ``broadleaf_deciduous_boreal_shrub``) shed or
    metabolically shut down their foliage over the frozen season, so both canopy
    GPP and foliar maintenance respiration should stop when frozen
    (``carbon.carbon_cycle`` cold-deciduous freeze dormancy, gated by
    ``CarbonConfig.cold_deciduous`` + ``cold_deciduous_dormancy``).  An
    EXACT-name membership set, NOT a substring match: the already-productive
    ``broadleaf_deciduous_boreal`` tree is intentionally excluded (a substring on
    ``"deciduous_boreal"`` would wrongly include it).
    """
    return pft_name in _COLD_DECIDUOUS_PFTS


def is_c4(pft_name: str) -> bool:
    """True for C4-pathway PFTs (``c4_grass`` / ``crop_c4``); False for C3 (all others).

    The photosynthetic pathway is encoded in the CLM5 PFT name (``c4_grass`` / ``crop_c4``
    carry the ``c4`` tag; every other name -- including the ``c3_*`` / ``crop_c3`` C3
    grasses/crops -- does not), so a substring match on ``c4`` cleanly separates the two.
    Agrees with the per-PFT C4 flag in the canopy biome table
    (``boundary_data._internals._CLM5_TO_BIOME``).

    Why it matters: the Farquhar biochemistry in ``carbon.stomata`` is **C3-only** -- the
    C4 rows are run through C3 kinetics as a documented approximation, so the leaf
    intercellular CO2 (``Ci``) it returns for a C4 PFT does NOT represent the true C4
    CO2-concentrating leaf state.  Any diagnostic that reads ``Ci`` as a C3 quantity (e.g.
    the leaf carbon-isotope discrimination in ``carbon.d13c_forward``, where C4 plants
    discriminate FAR less than C3) must MASK the C4 PFTs rather than apply the C3 form to
    them -- a C3 formula on a C4 leaf is a magnitude error.
    """
    return "c4" in pft_name


def is_c4_pft_id(pft_id):
    """Vectorized per-id C4 mask ``(n,)`` bool from integer CLM5 PFT ids.

    The array companion to :func:`is_c4`: maps each integer CLM5 PFT id to its
    name (:data:`CLM5_PFT_NAMES`) and returns ``True`` where that name is C4
    (``c4_grass`` / ``crop_c4``).  Out-of-range ids (``< 0`` or ``>= N_PFT_CLM5``)
    map to ``False`` (never crashes).  Pure NumPy -- a STATIC classifier on the
    static PFT id, so it can be materialised once and used as a compile-time
    selector in a JAX forward.

    SINGLE source of truth for the per-id C4 split, shared by both sides of the
    leaf carbon-isotope-discrimination calibration: the modelled forward
    (:mod:`legoesm.land.carbon.d13c_forward`, which selects the FAITHFUL C4
    Farquhar-Cerling discrimination for these ids via ``jnp.where``) and the
    observed loader (:func:`legoesm.land.carbon.d13c_observations.c4_archetype_mask`).
    """
    import numpy as np

    pid = np.asarray(pft_id, dtype=int).ravel()
    return np.array(
        [bool(0 <= int(p) < N_PFT_CLM5 and is_c4(CLM5_PFT_NAMES[int(p)])) for p in pid],
        dtype=bool,
    )


# Lazy-converted to jnp array on first use to avoid import-time JAX init
_clm5_table_cache: jnp.ndarray | None = None


def clm5_pft_table() -> jnp.ndarray:
    """Return the CLM5 PFT lookup table as ``(17, 12)`` jnp.ndarray."""
    global _clm5_table_cache
    if _clm5_table_cache is None:
        _clm5_table_cache = jnp.array(_CLM5_PFT_TABLE_RAW)
    return _clm5_table_cache
