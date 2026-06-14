"""ESA CCI **Plant Functional Type** source: fractional cover -> legoESM cover + PFT.

The ESA CCI PFT product (``climate.esa.int`` land-cover project, 300 m, annual
1992-2020) gives **14 fractional-cover layers** per pixel — unlike a discrete
land-cover map, every pixel carries the fraction of each PFT.  legoESM adopts
this **observed taxonomy as canonical** (no offline crosswalk to CLM5's
climate-zoned 17 PFTs): bioclimatic distinctions ESA does not measure
(tropical/temperate/boreal, C3/C4) are handled as *continuous functions in the
physics*, not baked into the inventory.  In particular C3/C4 grass is carried as
a separate continuous :data:`c4_fraction` field (see
:mod:`legoesm.land.surface_data.sources.c4_fraction`), so grass cover is **not**
split here.

The 14 ESA layers split into:
  - **non-vegetated (4):** bare soil, built, water, snow/ice
  - **vegetation (10):** trees and shrubs x {broadleaf, needleleaf} x
    {evergreen, deciduous}, plus natural and managed grass.

This module is format-independent: :func:`derive_cover_and_pft` operates on the
14 layers *already extracted and named*, so it is unit-tested without the (large)
ESA NetCDF.  The reader/aggregation that maps the real file's variable names onto
these canonical keys lives in a separate function added once the data is local
(variable names verified against the real file, not guessed).

Mapping to legoESM cover (:class:`legoesm.land.global_surface_data.GlobalSurfaceData`):
  - ``f_glacier`` <- snow/ice
  - ``f_lake``    <- water, but inland only (ocean water is excluded via the
    land-sea mask; see :func:`derive_cover_and_pft`)
  - ``f_land``    <- bare + built + sum(vegetation)
  - ``pft_frac``  <- within-land composition over :data:`LAND_PFT_NAMES`
    (bare + 10 veg), renormalized to sum to 1.

Host-side only; not traced.

References
----------
- Harper, K. L., Lamarche, C., Hartley, A., Peylin, P., Ottlé, C., Bastrikov, V.,
  San Martín, R., Bohnenstengel, S. I., Kirches, G., Boettcher, M., Shevchuk, R.,
  Brockmann, C., and Defourny, P. (2023): A 29-year time series of annual 300 m
  resolution plant-functional-type maps for climate models. Earth Syst. Sci.
  Data, 15, 1465-1499. https://doi.org/10.5194/essd-15-1465-2023
  Portal: https://climate.esa.int/en/odp/#/project/land-cover ;
  viewer: https://maps.elie.ucl.ac.be/CCI/viewer/
- (C3/C4, used by :mod:`~legoesm.land.surface_data.sources.c4_fraction`)
  Luo, X., Zhou, H., Satriawan, T. W., Tian, J., Zhao, R., Keenan, T. F.,
  Griffith, D. M., Sitch, S., Smith, N. G., and Still, C. J. (2024): Mapping the
  global distribution of C4 vegetation using observations and optimality theory.
  Nature Communications, 15, 1219. https://doi.org/10.1038/s41467-024-45606-3
"""

from __future__ import annotations

import numpy as np

from legoesm.land.surface_data.aggregate import (
    area_weighted_block_mean,
    coarse_grid_centers,
    row_cos_weights,
)
from legoesm.land.surface_data.bioclimate import ClimateZones

# Canonical ESA layer key -> variable name in the ESA CCI PFT NetCDF.  The file's
# `WATER_INLAND` (not the total `WATER`) is the inland-water layer, so f_lake is
# read directly with no land-sea mask needed; `WATER_OCEAN`/`WATER`/`LAND` are
# redundant aggregates and intentionally unused.
ESA_NC_VARMAP: dict[str, str] = {
    "bare_soil": "BARE",
    "built": "BUILT",
    "water": "WATER_INLAND",
    "snow_ice": "SNOWICE",
    "tree_broadleaf_evergreen": "TREES-BE",
    "tree_broadleaf_deciduous": "TREES-BD",
    "tree_needleleaf_evergreen": "TREES-NE",
    "tree_needleleaf_deciduous": "TREES-ND",
    "shrub_broadleaf_evergreen": "SHRUBS-BE",
    "shrub_broadleaf_deciduous": "SHRUBS-BD",
    "shrub_needleleaf_evergreen": "SHRUBS-NE",
    "shrub_needleleaf_deciduous": "SHRUBS-ND",
    "natural_grass": "GRASS-NAT",
    "managed_grass": "GRASS-MAN",
}

# Canonical keys for the four non-vegetated ESA classes.
ESA_BARE = "bare_soil"
ESA_BUILT = "built"
ESA_WATER = "water"
ESA_SNOW_ICE = "snow_ice"
ESA_NONVEG_NAMES: tuple[str, ...] = (ESA_BARE, ESA_BUILT, ESA_WATER, ESA_SNOW_ICE)

# Canonical keys for the ten ESA vegetation PFTs (the order is the pft axis).
ESA_VEG_PFT_NAMES: tuple[str, ...] = (
    "tree_broadleaf_evergreen",
    "tree_broadleaf_deciduous",
    "tree_needleleaf_evergreen",
    "tree_needleleaf_deciduous",
    "shrub_broadleaf_evergreen",
    "shrub_broadleaf_deciduous",
    "shrub_needleleaf_evergreen",
    "shrub_needleleaf_deciduous",
    "natural_grass",
    "managed_grass",
)

# The within-land PFT-composition axis written to ``pft_frac``: bare soil at
# index 0 (CLM-style convention the container assumes), then the 10 veg PFTs.
# ``built`` is folded into bare soil (legoESM has no urban tile yet — TODO: add a
# dedicated urban class when an urban landunit exists).
LAND_PFT_NAMES: tuple[str, ...] = (ESA_BARE,) + ESA_VEG_PFT_NAMES
N_LAND_PFT: int = len(LAND_PFT_NAMES)

# All 14 layers the reader must supply.
ESA_ALL_LAYER_NAMES: tuple[str, ...] = ESA_NONVEG_NAMES + ESA_VEG_PFT_NAMES


def derive_cover_and_pft(
    layers: dict[str, np.ndarray],
    land_mask: np.ndarray | None = None,
    *,
    pct_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Derive legoESM cover fractions + within-land PFT composition from ESA layers.

    Parameters
    ----------
    layers
        Mapping from each canonical name in :data:`ESA_ALL_LAYER_NAMES` to a
        spatial array (any shape ``S``), the fractional cover of that class.
        Values are multiplied by ``pct_scale`` (use ``0.01`` if the source is in
        percent).
    land_mask
        Optional boolean array (shape ``S``) — ``True`` where the cell is land.
        Needed because ESA's ``water`` layer includes the **ocean**: only water
        on land counts as ``f_lake``.  If ``None``, a fallback heuristic treats a
        cell as land when its non-water cover (land surface + ice) is at least its
        water cover.  Supplying a real land-sea mask (e.g. the HWSD soil-coverage
        mask) is strongly preferred.

    Returns
    -------
    dict
        ``f_land``, ``f_lake``, ``f_glacier`` (each shape ``S``, [0-1]) and
        ``pft_frac`` (shape ``S + (N_LAND_PFT,)``, summing to 1 over the last
        axis where land exists, else 0).
    """
    missing = [n for n in ESA_ALL_LAYER_NAMES if n not in layers]
    if missing:
        raise ValueError(f"derive_cover_and_pft: missing ESA layers {missing}.")

    def g(name):
        return np.asarray(layers[name], dtype=np.float64) * pct_scale

    bare, built = g(ESA_BARE), g(ESA_BUILT)
    water, ice = g(ESA_WATER), g(ESA_SNOW_ICE)
    veg = np.stack([g(n) for n in ESA_VEG_PFT_NAMES], axis=-1)   # S + (10,)
    veg_sum = veg.sum(axis=-1)

    # Solid land surface = soil/veg ground (bare+built+vegetation).  Snow/ice is
    # land too but tracked separately as glacier.
    f_land = bare + built + veg_sum
    f_glacier = ice

    if land_mask is None:
        land_mask = (f_land + ice) >= water
    land_mask = np.asarray(land_mask, dtype=bool)
    f_lake = np.where(land_mask, water, 0.0)        # ocean water dropped

    # Within-land composition: [bare+built, 10 veg], renormalized to sum 1.
    comp = np.concatenate(
        [(bare + built)[..., None], veg], axis=-1                # S + (11,)
    )
    total = comp.sum(axis=-1, keepdims=True)
    pft_frac = np.where(total > 0.0, comp / np.maximum(total, 1e-30), 0.0)

    return {
        "f_land": f_land,
        "f_lake": f_lake,
        "f_glacier": f_glacier,
        "pft_frac": pft_frac,
    }


# ===========================================================================
# Reader: stream the 300 m ESA NetCDF -> coarse cover + PFT fractions
# ===========================================================================
def aggregate_esa_pft(
    nc_path: str | None = None,
    res_deg: float = 0.25,
    *,
    dataset=None,
    coarse_rows_per_chunk: int = 2,
) -> dict:
    """Aggregate the ESA CCI PFT 300 m map to ``res_deg`` cover + PFT fractions.

    Streams the (64800 x 129600 int8 percent) raster in coarse-row chunks,
    ``cos(lat)`` area-averaging the 14 cover layers to the coarse grid, then runs
    :func:`derive_cover_and_pft` on the coarse means.  Aggregating the *raw*
    per-gridcell cover percentages (then deriving) is correct because cover
    fractions are extensive and average linearly; the renormalization in
    ``derive_cover_and_pft`` is applied once, on the coarse field.

    ``dataset`` may be injected (an xarray Dataset) for testing; otherwise
    ``nc_path`` is opened.  Returns ``lat``, ``lon``, ``f_land``/``f_lake``/
    ``f_glacier`` (ny, nx), ``pft_frac`` (ny, nx, N_LAND_PFT), and the coarse raw
    ``esa_layers`` dict.  Host-side, run-once; not traced.
    """
    import xarray as xr  # noqa: F401 (xarray is the reader backend)

    ds = dataset if dataset is not None else xr.open_dataset(nc_path, decode_times=False)
    try:
        lat = np.asarray(ds["lat"].values, dtype=np.float64)   # descending
        nrows, ncols = lat.size, ds["lon"].size
        ny = int(round(180.0 / res_deg))
        nx = int(round(360.0 / res_deg))
        if nrows % ny or ncols % nx:
            raise ValueError(
                f"ESA grid {nrows}x{ncols} not divisible into {ny}x{nx} for "
                f"{res_deg} deg."
            )
        by, bx = nrows // ny, ncols // nx
        keys = ESA_ALL_LAYER_NAMES                              # 14, fixed order

        coarse = np.full((ny, nx, len(keys)), np.nan, dtype=np.float64)
        for c0 in range(0, ny, coarse_rows_per_chunk):
            c1 = min(c0 + coarse_rows_per_chunk, ny)
            r0, r1 = c0 * by, c1 * by
            block = np.empty((r1 - r0, ncols, len(keys)), dtype=np.float32)
            for li, key in enumerate(keys):
                raw = ds[ESA_NC_VARMAP[key]].isel(time=0, lat=slice(r0, r1)).values
                # int8 percent -> fraction; clip stray fill/negatives to [0,1].
                block[..., li] = np.clip(raw.astype(np.float32), 0.0, 100.0) / 100.0
            valid = np.ones((r1 - r0, ncols), dtype=bool)      # ocean dilutes via 0s
            mean, _ = area_weighted_block_mean(
                block, valid, row_cos_weights(lat[r0:r1]), by, bx
            )
            coarse[c0:c1] = mean

        clat, clon = coarse_grid_centers(res_deg, lon0=-180.0)
        layers = {key: coarse[..., li] for li, key in enumerate(keys)}
        # WATER_INLAND already excludes ocean, so no land mask is needed.
        cover = derive_cover_and_pft(
            layers, land_mask=np.ones((ny, nx), dtype=bool), pct_scale=1.0
        )
        return {"lat": clat, "lon": clon, "esa_layers": layers, **cover}
    finally:
        if dataset is None:
            ds.close()


# ===========================================================================
# Optional crosswalk: ESA-native composition -> CLM5's 17 PFTs
# ===========================================================================
# legoESM v1 maps to CLM5's 17 PFTs to reuse the existing CLM5 parameter table
# (``surface_params.clm5_pft_table``); the ESA-native composition stays the
# faithful intermediate, so this crosswalk is a single removable stage for when
# the model later adopts ESA-native PFTs + learned parameters.
#
# Needleleaf shrubs have no CLM5 equivalent, so (decision) they are folded into
# the nearest CLM5 shrub preserving growth form: NE shrub -> broadleaf-evergreen
# shrub, ND shrub -> broadleaf-deciduous-boreal shrub.

def esa_to_clm5_pft(
    esa_land_frac: np.ndarray,        # S + (N_LAND_PFT,) over LAND_PFT_NAMES
    zones: ClimateZones,              # masks shape S (tropical/temperate/boreal)
    c4_grass_frac: np.ndarray,        # S, [0-1] fraction of natural grass that is C4
    c4_crop_frac: np.ndarray,         # S, [0-1] fraction of managed grass that is C4
) -> np.ndarray:
    """Crosswalk the ESA-native land composition to CLM5's 17-PFT axis.

    Distributes each ESA class across CLM5's climate-zoned PFTs using the
    bioclimatic ``zones`` (which partition every cell), and splits grass/crop by
    the observational C4 fractions.  Area-conserving: the CLM5 fractions sum to
    the same total as ``esa_land_frac`` per cell.  Output last axis follows
    :data:`legoesm.land.surface_params.CLM5_PFT_NAMES` (length 17).
    """
    from legoesm.land.surface_params import CLM5_PFT_NAMES, N_PFT_CLM5

    esa_land_frac = np.asarray(esa_land_frac, dtype=np.float64)
    if esa_land_frac.shape[-1] != N_LAND_PFT:
        raise ValueError(
            f"esa_land_frac last axis must be {N_LAND_PFT} (LAND_PFT_NAMES), "
            f"got {esa_land_frac.shape[-1]}."
        )

    def e(name):
        return esa_land_frac[..., LAND_PFT_NAMES.index(name)]

    trop = zones.tropical.astype(np.float64)
    bor = zones.boreal.astype(np.float64)
    nonbor = 1.0 - bor                       # temperate + tropical climate
    temp = zones.temperate.astype(np.float64)
    nontrop = 1.0 - trop
    c4g = np.asarray(c4_grass_frac, dtype=np.float64)
    c4c = np.asarray(c4_crop_frac, dtype=np.float64)

    t_be, t_bd = e("tree_broadleaf_evergreen"), e("tree_broadleaf_deciduous")
    t_ne, t_nd = e("tree_needleleaf_evergreen"), e("tree_needleleaf_deciduous")
    s_be, s_bd = e("shrub_broadleaf_evergreen"), e("shrub_broadleaf_deciduous")
    s_ne, s_nd = e("shrub_needleleaf_evergreen"), e("shrub_needleleaf_deciduous")
    ng, mg = e("natural_grass"), e("managed_grass")

    clm = {
        "bare_soil": e("bare_soil"),
        # needleleaf evergreen has no tropical class -> non-boreal == temperate.
        "needleleaf_evergreen_temperate": t_ne * nonbor,
        "needleleaf_evergreen_boreal": t_ne * bor,
        "needleleaf_deciduous_boreal": t_nd,                  # only boreal exists
        "broadleaf_evergreen_tropical": t_be * trop,
        "broadleaf_evergreen_temperate": t_be * nontrop,      # no boreal BE class
        "broadleaf_deciduous_tropical": t_bd * trop,
        "broadleaf_deciduous_temperate": t_bd * temp,
        "broadleaf_deciduous_boreal": t_bd * bor,
        # NE shrub folded into broadleaf-evergreen shrub (growth form preserved).
        "broadleaf_evergreen_shrub": s_be + s_ne,
        "broadleaf_deciduous_temperate_shrub": s_bd * nonbor,
        # ND shrub folded into broadleaf-deciduous-boreal shrub.
        "broadleaf_deciduous_boreal_shrub": s_bd * bor + s_nd,
        # grass: C4 by observation; the C3 remainder is arctic in boreal climate.
        "c3_arctic_grass": ng * (1.0 - c4g) * bor,
        "c3_grass": ng * (1.0 - c4g) * nonbor,
        "c4_grass": ng * c4g,
        "crop_c3": mg * (1.0 - c4c),
        "crop_c4": mg * c4c,
    }

    out = np.zeros(esa_land_frac.shape[:-1] + (N_PFT_CLM5,), dtype=np.float64)
    for i, name in enumerate(CLM5_PFT_NAMES):
        out[..., i] = clm[name]
    return out
