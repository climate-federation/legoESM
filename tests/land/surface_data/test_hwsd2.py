"""Unit tests for the HWSD2 SMU->soil-property join and soil aggregation.

The SHARE-weighted component reduction and the missing-data (-9) masking are
tested on a synthetic attribute table; a small synthetic raster exercises the
full lookup->aggregate path.  No real HWSD files required.
"""

import numpy as np
import pandas as pd
import pytest

from legoesm.land.surface_data.sources.hwsd2 import (
    build_smu_lookup,
    aggregate_hwsd2_soil,
    SmuLookup,
    _PROP_NAMES,
)
from legoesm.land.surface_data.raster import EnviBilHeader, write_envi_bil, open_bil_memmap


def _toy_layers():
    # SMU 5, layer D1: two components -> SHARE-weighted; BULK of comp 2 is missing.
    # SMU 7, layer D1: single component.  SMU 5 D2 present, others absent.
    rows = [
        # smu, seq, share, layer, sand, clay, orgc, bulk
        (5, 1, 70, "D1", 80, 10, 1.0, 1.40),
        (5, 2, 30, "D1", 40, 30, 2.0, -9.0),     # BULK missing for this component
        (5, 1, 100, "D2", 50, 20, 0.5, 1.30),
        (7, 1, 100, "D1", 20, 60, 3.0, 1.10),
        (9, 1, 100, "D1", -9, -9, -9.0, -9.0),   # SMU with all-missing texture (-9)
        (11, 1, 100, "D1", -5, -5, -5.0, -5.0),  # non-soil unit (water/ice, code -5)
    ]
    return pd.DataFrame(
        rows,
        columns=["HWSD2_SMU_ID", "SEQUENCE", "SHARE", "LAYER",
                 "SAND", "CLAY", "ORG_CARBON", "BULK"],
    )


def test_share_weighted_join_and_missing():
    lut = build_smu_lookup(_toy_layers())
    assert isinstance(lut, SmuLookup)
    assert lut.prop_names == _PROP_NAMES        # (sand_pct, clay_pct, organic, bulk_density)
    s, c, o, b = range(4)

    # SMU 5, D1 sand = (70*80 + 30*40)/100 = 68 ; clay = (70*10+30*30)/100 = 16
    assert np.isclose(lut.table[5, 0, s], 68.0)
    assert np.isclose(lut.table[5, 0, c], 16.0)
    # BULK: comp 2 missing (-9) -> only comp 1 counts -> 1.40 g/cm3 -> 1400 kg/m3
    assert np.isclose(lut.table[5, 0, b], 1400.0)
    # organic = (70*1 + 30*2)/100 = 1.3 %
    assert np.isclose(lut.table[5, 0, o], 1.3, atol=1e-5)
    # SMU 7 D1 single comp
    assert np.isclose(lut.table[7, 0, s], 20.0)
    # SMU 5 D2 present, D3.. absent -> NaN
    assert np.isclose(lut.table[5, 1, s], 50.0)
    assert np.isnan(lut.table[5, 2, s])
    # SMU 9: all texture missing -> NaN (not -9 leaking through)
    assert np.isnan(lut.table[9, 0, s])
    # SMU 11: non-soil code -5 also masked -> NaN (no negative leaks into means)
    assert np.isnan(lut.table[11, 0, s])
    assert np.isnan(lut.table[11, 0, b])
    # index 0 forced NaN (out-of-range gather sentinel)
    assert np.isnan(lut.table[0, 0, s])


def test_wrb_negative_codes_filled_or_masked():
    # Arenosol (AR, code -4) -> filled to sandy; Open Water (WR, -1) -> stays NaN.
    rows = [
        (20, 1, 100, "D1", -4, -4, -4.0, -4.0, "AR"),   # sandy fill
        (21, 1, 100, "D1", -1, -1, -1.0, -1.0, "WR"),   # water -> no soil
        (22, 1, 100, "D1", -3, -3, -3.0, -3.0, "LP"),   # leptosol fill
    ]
    df = pd.DataFrame(rows, columns=[
        "HWSD2_SMU_ID", "SEQUENCE", "SHARE", "LAYER",
        "SAND", "CLAY", "ORG_CARBON", "BULK", "WRB2"])
    lut = build_smu_lookup(df)
    s, b = 0, 3
    # AR -> representative sand (92) and bulk 1.55 g/cm3 -> 1550 kg/m3
    assert np.isclose(lut.table[20, 0, s], 92.0)
    assert np.isclose(lut.table[20, 0, b], 1550.0, atol=1.0)
    # WR (open water) -> no soil -> NaN
    assert np.isnan(lut.table[21, 0, s])
    # LP (leptosol) -> filled (not NaN), sandier-loam
    assert np.isclose(lut.table[22, 0, s], 45.0)


def _toy_raster(tmp_path, res=1.0):
    n_lat, n_lon = int(180 / res), int(360 / res)
    hdr = EnviBilHeader(
        nrows=n_lat, ncols=n_lon, nbands=1, dtype=np.dtype("<u2"), nodata=65535.0,
        ulxmap=-180.0 + 0.5 * res, ulymap=90.0 - 0.5 * res, xdim=res, ydim=res,
        layout="BIL",
    )
    smu = np.full((n_lat, n_lon), 65535, dtype="<u2")
    smu[40:60, 100:140] = 5          # a block of SMU 5
    smu[40:60, 140:180] = 7          # adjacent block of SMU 7
    bil = str(tmp_path / "toy_smu.bil")
    write_envi_bil(bil, smu, hdr)
    return bil


def test_aggregate_hwsd2_soil_maps_lookup_values(tmp_path):
    lut = build_smu_lookup(_toy_layers())
    bil = _toy_raster(tmp_path, res=1.0)
    soil = aggregate_hwsd2_soil(bil, lut, res_deg=2.0, coarse_rows_per_chunk=10)

    assert soil["sand_pct"].shape == (7, 90, 180)        # (n_layer, nlat, nlon)
    mm, hdr = open_bil_memmap(bil)
    # A coarse cell fully inside the SMU-5 block must read SMU-5's D1 sand (68).
    # fine block rows 40:60, cols 100:140 -> coarse (res 2) rows 20:30, cols 50:70.
    assert np.isclose(soil["sand_pct"][0, 25, 55], 68.0)
    assert np.isclose(soil["bulk_density"][0, 25, 55], 1400.0)
    # SMU-7 region
    assert np.isclose(soil["sand_pct"][0, 25, 80], 20.0)
    # ocean stays NaN; soil-data fraction is 0 there, 1 inside the blocks
    assert np.isnan(soil["sand_pct"][0, 0, 0])
    assert np.isclose(soil["soil_data_frac"][25, 55], 1.0)
    assert np.isclose(soil["soil_data_frac"][0, 0], 0.0)
