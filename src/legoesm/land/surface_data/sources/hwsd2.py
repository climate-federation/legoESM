"""FAO **HWSD v2.0** soil source: SMU raster + attribute table -> coarse soil grid.

The HWSD2 raster (``HWSD2.bil``, 30 arc-second) stores only an integer **soil
mapping-unit id** per pixel.  The soil properties live in the attribute table
``HWSD2_LAYERS`` (in ``HWSD2.mdb``), where each mapping unit is a *set of soil
components* (``SEQUENCE``) with an areal ``SHARE`` (%), each described over the 7
fixed depth layers ``D1``..``D7``.

Pipeline (host-side, run once):

  1. :func:`build_smu_lookup` — for every ``(SMU, layer)`` collapse the component
     set to a single ``SHARE``-weighted mean of sand / clay / organic-carbon /
     bulk-density, masking the ``-9`` missing-data sentinel per property.  Result
     is a dense lookup indexed by SMU id and layer.
  2. :func:`aggregate_hwsd2_soil` — stream the SMU raster, map each pixel through
     the lookup, and ``cos(lat)`` area-weight to the coarse grid (per-layer,
     per-property validity so data gaps don't bias a layer's mean).
  3. :func:`build_hwsd2_surfdata` — write the coarse soil fields into a
     :func:`legoesm.land.surface_data.schema.write_surfdata` NetCDF.

Units harmonized to the schema: sand/clay as percent (loader applies
``pct_scale``); organic carbon as mass percent; bulk density converted
g/cm^3 -> kg/m^3.

References
----------
- FAO & IIASA (2023): Harmonized World Soil Database version 2.0. Rome and
  Laxenburg. ISBN 978-92-5-137499-3. https://doi.org/10.4060/cc3823en
  Data hub: https://www.fao.org/soils-portal/data-hub/soil-maps-and-databases/harmonized-world-soil-database-v20/en/
"""

from __future__ import annotations

import io
import subprocess
from typing import NamedTuple

import numpy as np

from legoesm.land.surface_data.raster import open_bil_memmap
from legoesm.land.surface_data.aggregate import aggregate_raster_streaming
from legoesm.land.surface_data.schema import write_surfdata, HWSD2_LAYER_DZ

# HWSD2 encodes special map units as NEGATIVE property codes whose WRB2 group
# names the unit.  Several are *real soils* with no measured texture (their
# texture is implied by the soil group), so rather than blanking them to NaN we
# fill them with a representative texture — most importantly Arenosols (the great
# sand seas, e.g. the Sahara), which are sandy.  Units that genuinely have no soil
# (open water, glaciers, technosols/urban, no-data) stay missing.
#   WRB2: AR=Arenosols, LP=Leptosols, SC=Solonchaks, WR=Open Water,
#         GG=Glaciers, TC=Technosols, ND=No Data.
# Representative texture (percent sand/clay, organic-carbon %, bulk g/cm^3) for the
# soil-bearing misc units (documented category fill, exempt like a lookup table):
_WRB_TEXTURE_FILL: dict[str, dict[str, float]] = {
    "AR": {"SAND": 92.0, "CLAY": 3.0, "ORG_CARBON": 0.3, "BULK": 1.55},   # sandy / dunes
    "LP": {"SAND": 45.0, "CLAY": 20.0, "ORG_CARBON": 2.0, "BULK": 1.40},  # shallow / rocky
    "SC": {"SAND": 35.0, "CLAY": 25.0, "ORG_CARBON": 1.0, "BULK": 1.45},  # salt-affected
}
# WR / GG / TC / ND and any other negative code -> no soil -> stays NaN.
HWSD2_NONSOIL_MAX = 0.0  # after the WRB fill, valid soil property values are >= 0
# Fixed depth-layer labels D1..D7.
HWSD2_LAYER_LABELS = ("D1", "D2", "D3", "D4", "D5", "D6", "D7")
_N_LAYER = len(HWSD2_LAYER_LABELS)
# Bulk density g/cm^3 -> kg/m^3 (pure unit conversion).
_GCM3_TO_KGM3 = 1000.0

# Harmonized output field <- (HWSD column, multiplicative scale).
_PROP_SPEC = (
    ("sand_pct", "SAND", 1.0),
    ("clay_pct", "CLAY", 1.0),
    ("organic", "ORG_CARBON", 1.0),
    ("bulk_density", "BULK", _GCM3_TO_KGM3),
)
_PROP_NAMES = tuple(spec[0] for spec in _PROP_SPEC)

# Columns we actually read from the (wide) HWSD2_LAYERS table.
_USECOLS = ["HWSD2_SMU_ID", "SEQUENCE", "SHARE", "LAYER", "SAND", "CLAY",
            "ORG_CARBON", "BULK", "WRB2"]


class SmuLookup(NamedTuple):
    """Dense SMU-id -> per-layer soil-property table.

    ``table`` is ``(max_smu_id + 1, 7, n_prop)`` float32 with ``NaN`` where a
    property is undefined for that mapping unit / layer (no soil component, or
    all components missing).  ``prop_names`` orders the last axis.
    """

    table: np.ndarray            # (max_smu+1, 7, n_prop) float32, NaN = undefined
    prop_names: tuple[str, ...]
    max_smu_id: int


def read_hwsd2_layers(path: str):
    """Read the ``HWSD2_LAYERS`` table into a DataFrame from ``.mdb`` or ``.csv``.

    A ``.mdb`` is exported on the fly via ``mdb-export`` (mdbtools); a ``.csv`` is
    read directly (e.g. a previously cached export).  Only the columns needed for
    the soil join are loaded.
    """
    import pandas as pd

    if path.lower().endswith(".mdb"):
        out = subprocess.run(
            ["mdb-export", path, "HWSD2_LAYERS"],
            check=True, capture_output=True, text=True,
        )
        return pd.read_csv(io.StringIO(out.stdout), usecols=_USECOLS, low_memory=False)
    return pd.read_csv(path, usecols=_USECOLS, low_memory=False)


def build_smu_lookup(df) -> SmuLookup:
    """Collapse the HWSD2_LAYERS component sets to a ``SHARE``-weighted lookup.

    For each ``(SMU, layer, property)`` the value is the ``SHARE``-weighted mean
    over the soil components whose value is present.  Negative special-unit codes
    are first replaced by a representative texture for soil-bearing WRB groups
    (:data:`_WRB_TEXTURE_FILL`, e.g. Arenosols -> sand); remaining negatives
    (water / glaciers / technosols / no-data) are excluded.
    """
    smu = df["HWSD2_SMU_ID"].to_numpy(dtype=np.int64)
    share = df["SHARE"].to_numpy(dtype=np.float64)
    # Map layer label -> 0..6; unrecognized labels become NaN and are dropped.
    label_to_idx = {lab: i for i, lab in enumerate(HWSD2_LAYER_LABELS)}
    lay = df["LAYER"].map(label_to_idx).to_numpy(dtype=np.float64)
    keep = np.isfinite(lay)
    smu, share, lay = smu[keep], share[keep], lay[keep].astype(np.int64)
    wrb = (df["WRB2"].to_numpy()[keep] if "WRB2" in df.columns
           else np.full(smu.shape, None, dtype=object))

    max_smu = int(smu.max())
    n_cell = (max_smu + 1) * _N_LAYER
    key = smu * _N_LAYER + lay                         # unique per (smu, layer)

    table = np.full((max_smu + 1, _N_LAYER, len(_PROP_SPEC)), np.nan, dtype=np.float32)
    for p, (_field, col, scale) in enumerate(_PROP_SPEC):
        v = df[col].to_numpy(dtype=np.float64)[keep]
        # Fill negative special-unit codes for soil-bearing WRB groups with a
        # representative texture; other negatives stay missing.
        neg = v < 0.0
        if neg.any():
            fill = np.full(v.shape, np.nan)
            for grp, props in _WRB_TEXTURE_FILL.items():
                fill[wrb == grp] = props[col]
            v = np.where(neg, fill, v)
        present = np.isfinite(v) & (v >= HWSD2_NONSOIL_MAX)
        wv = np.where(present, share * v, 0.0)
        wm = np.where(present, share, 0.0)
        num = np.bincount(key, weights=wv, minlength=n_cell)
        den = np.bincount(key, weights=wm, minlength=n_cell)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(den > 0.0, num / np.maximum(den, 1e-30), np.nan)
        table[:, :, p] = (mean.reshape(max_smu + 1, _N_LAYER) * scale).astype(np.float32)

    # SMU id 0 is never a real mapping unit; force it NaN so out-of-range gathers
    # (clamped to index 0) read as undefined.
    table[0] = np.nan
    return SmuLookup(table=table, prop_names=_PROP_NAMES, max_smu_id=max_smu)


def aggregate_hwsd2_soil(
    bil_path: str,
    lookup: SmuLookup,
    res_deg: float = 0.25,
    *,
    coarse_rows_per_chunk: int = 6,
) -> dict:
    """Stream the SMU raster through ``lookup`` to coarse per-layer soil fields.

    Returns a dict with each property as ``(n_layer, nlat, nlon)`` plus
    ``soil_data_frac (nlat, nlon)`` (the area fraction of the cell with *valid
    soil data* — excludes non-soil land like dunes/ice, so it is NOT the true
    land fraction), ``lat``, ``lon``.  Memory is bounded by the chunk size: only a
    strip of fine rows (times the 7x4 property channels) is resident.
    """
    mm, hdr = open_bil_memmap(bil_path)
    nodata = hdr.nodata
    lut = lookup.table                                  # (max_smu+1, 7, n_prop)
    max_smu = lookup.max_smu_id
    n_prop = lut.shape[2]
    L = _N_LAYER * n_prop

    def transform(block, lat_block):
        smu = block.astype(np.int64)
        land = (smu != nodata) & (smu >= 0) & (smu <= max_smu)
        idx = np.where(land, smu, 0)                    # clamp; index 0 is NaN row
        vals = lut[idx]                                 # (h, w, 7, n_prop) float32
        valid = land[..., None, None] & np.isfinite(vals)
        return vals.reshape(vals.shape[0], vals.shape[1], L), valid.reshape(
            valid.shape[0], valid.shape[1], L)

    mean, soil_data_frac, clat, clon = aggregate_raster_streaming(
        mm, hdr, res_deg, transform, coarse_rows_per_chunk=coarse_rows_per_chunk
    )
    ny, nx = soil_data_frac.shape
    mean = mean.reshape(ny, nx, _N_LAYER, n_prop)       # (nlat, nlon, 7, n_prop)

    out = {"lat": clat, "lon": clon, "soil_data_frac": soil_data_frac}
    for p, name in enumerate(lookup.prop_names):
        out[name] = np.moveaxis(mean[:, :, :, p], 2, 0)  # (7, nlat, nlon)
    return out


def build_hwsd2_surfdata(
    bil_path: str,
    layers_path: str,
    out_path: str,
    res_deg: float = 0.25,
    *,
    coarse_rows_per_chunk: int = 6,
) -> dict:
    """End-to-end: HWSD2 raster + attribute table -> ``legoesm_surfdata`` NetCDF.

    Writes the soil group (sand/clay/organic-carbon/bulk-density per HWSD layer,
    plus ``soil_dz``) and returns the in-memory coarse fields for inspection.
    """
    df = read_hwsd2_layers(layers_path)
    lookup = build_smu_lookup(df)
    soil = aggregate_hwsd2_soil(
        bil_path, lookup, res_deg, coarse_rows_per_chunk=coarse_rows_per_chunk
    )
    write_surfdata(
        out_path, lat=soil["lat"], lon=soil["lon"], soil_dz=HWSD2_LAYER_DZ,
        sand_pct=soil["sand_pct"], clay_pct=soil["clay_pct"],
        organic=soil["organic"], bulk_density=soil["bulk_density"],
        source="FAO HWSD v2.0 (SHARE-weighted, area-aggregated)",
    )
    return soil
