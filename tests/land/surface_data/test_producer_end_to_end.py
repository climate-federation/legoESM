"""End-to-end producer integration test: raster + attrs + CLM -> harmonized NetCDF.

Exercises the *full* producer chain in one shot, with no real data files:

  1. write a tiny synthetic HWSD2 SMU raster (BIL + .hdr) and a synthetic
     ``HWSD2_LAYERS`` CSV with two known mapping units;
  2. write a tiny synthetic CLM5 surfdata file (cover / PFT / LAI);
  3. run :func:`build_hwsd2_surfdata` (raster + CSV -> intermediate NetCDF);
  4. run :func:`build_v1_surfdata` (intermediate + CLM file -> harmonized NetCDF);
  5. load the harmonized NetCDF through the runtime loader (the consumer side)
     and check the soil + cover fields survived round-trip.

This is the test that pins the missing piece in the producer pipeline: every
other producer test exercises one of these steps, but none ran the whole
``raster -> .nc`` chain.  Match goes through the same path
:func:`scripts/data/build_legoesm_surfdata.py` uses in production.
"""

import numpy as np
import jax.numpy as jnp
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from legoesm.land.surface_data.assemble import build_v1_surfdata
from legoesm.land.surface_data.raster import EnviBilHeader, write_envi_bil
from legoesm.land.surface_data.sources.hwsd2 import build_hwsd2_surfdata
from legoesm.land.global_surface_data import get_surfdata_preset, load_global_surface_data
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.land.surface_params import CLM5_PFT_NAMES, N_PFT_CLM5


# ---------------------------------------------------------------------------
# Synthetic input writers
# ---------------------------------------------------------------------------
def _write_hwsd_attr_csv(path, *, smu_id=5, sand=68.0, clay=16.0,
                         organic=1.3, bulk=1.40):
    """One mapping unit with a single soil component, present at all 7 layers."""
    rows = []
    for k, lab in enumerate(("D1", "D2", "D3", "D4", "D5", "D6", "D7")):
        rows.append((smu_id, 1, 100, lab, sand, clay, organic, bulk, "LP"))
    df = pd.DataFrame(rows, columns=[
        "HWSD2_SMU_ID", "SEQUENCE", "SHARE", "LAYER",
        "SAND", "CLAY", "ORG_CARBON", "BULK", "WRB2",
    ])
    df.to_csv(path, index=False)


def _write_hwsd_raster(path, *, smu_id=5, res=1.0):
    """Global 1 deg SMU raster with a small block of ``smu_id`` over Amazon."""
    n_lat, n_lon = int(180 / res), int(360 / res)
    hdr = EnviBilHeader(
        nrows=n_lat, ncols=n_lon, nbands=1, dtype=np.dtype("<u2"), nodata=65535.0,
        ulxmap=-180.0 + 0.5 * res, ulymap=90.0 - 0.5 * res, xdim=res, ydim=res,
        layout="BIL",
    )
    smu = np.full((n_lat, n_lon), 65535, dtype="<u2")     # ocean / NoData
    # Amazon-ish patch: 10S - 10N, 70W - 50W.
    lat_idx = slice(int((90 - 10) / res), int((90 + 10) / res))
    lon_idx = slice(int((180 - 70) / res), int((180 - 50) / res))
    smu[lat_idx, lon_idx] = smu_id
    write_envi_bil(path, smu, hdr)


def _write_clm(path, *, nlat=4, nlon=8):
    """Synthetic CLM5 surfdata file (broadleaf evergreen tropical everywhere)."""
    be = CLM5_PFT_NAMES.index("broadleaf_evergreen_tropical")
    pct_nat_pft = np.zeros((15, nlat, nlon))
    pct_nat_pft[be] = 100.0
    natveg = np.full((nlat, nlon), 100.0)
    lai = np.zeros((12, 17, nlat, nlon)); lai[:, be] = 4.0
    latixy = np.broadcast_to(np.linspace(-80, 80, nlat)[:, None], (nlat, nlon)).copy()
    longxy = np.broadcast_to(np.linspace(0, 315, nlon)[None, :], (nlat, nlon)).copy()
    z = lambda: np.zeros((12, 17, nlat, nlon))
    zz = lambda: np.zeros((nlat, nlon))
    xr.Dataset({
        "PCT_NAT_PFT": (("natpft", "lsmlat", "lsmlon"), pct_nat_pft),
        "PCT_CFT": (("cft", "lsmlat", "lsmlon"), np.zeros((2, nlat, nlon))),
        "PCT_NATVEG": (("lsmlat", "lsmlon"), natveg),
        "PCT_CROP": (("lsmlat", "lsmlon"), zz()),
        "PCT_LAKE": (("lsmlat", "lsmlon"), zz()),
        "PCT_GLACIER": (("lsmlat", "lsmlon"), zz()),
        "LANDFRAC_PFT": (("lsmlat", "lsmlon"), np.ones((nlat, nlon))),  # all land
        "SOIL_COLOR": (("lsmlat", "lsmlon"), np.ones((nlat, nlon))),
        "MONTHLY_LAI": (("time", "lsmpft", "lsmlat", "lsmlon"), lai),
        "MONTHLY_SAI": (("time", "lsmpft", "lsmlat", "lsmlon"), z()),
        "MONTHLY_HEIGHT_TOP": (("time", "lsmpft", "lsmlat", "lsmlon"), z()),
        "MONTHLY_HEIGHT_BOT": (("time", "lsmpft", "lsmlat", "lsmlon"), z()),
        "LATIXY": (("lsmlat", "lsmlon"), latixy),
        "LONGXY": (("lsmlat", "lsmlon"), longxy),
    }).to_netcdf(path)


class _ToyGrid:
    def __init__(self, nlat=6, nlon=12):
        latc = np.linspace(-80, 80, nlat); lonc = np.linspace(0, 330, nlon)
        self.lat2d, self.lon2d = np.meshgrid(
            np.deg2rad(latc), np.deg2rad(lonc), indexing="ij")
        self.grid_area = jnp.asarray(np.full(nlat * nlon, 1e9), jnp.float32)
        self.ncol = nlat * nlon


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------
def test_full_producer_chain(tmp_path):
    """raster + attrs + CLM -> intermediate -> harmonized NetCDF -> consumer load."""
    # --- synthetic inputs
    bil = str(tmp_path / "smu.bil")
    csv = str(tmp_path / "layers.csv")
    clm = str(tmp_path / "clm.nc")
    intermediate = str(tmp_path / "hwsd_0p25.nc")
    out = str(tmp_path / "v1.nc")

    _write_hwsd_raster(bil, smu_id=5, res=1.0)
    _write_hwsd_attr_csv(csv, smu_id=5, sand=68.0, clay=16.0, organic=1.3, bulk=1.40)
    _write_clm(clm)

    # --- step 1: raster + attrs -> 0.25 deg HWSD intermediate
    soil = build_hwsd2_surfdata(bil, csv, intermediate,
                                res_deg=2.0, coarse_rows_per_chunk=10)
    assert soil["sand_pct"].shape == (7, 90, 180)              # (nlayer, nlat, nlon)
    # the Amazon patch must read SMU-5's sand value back
    finite = np.isfinite(soil["sand_pct"][0])
    assert np.any(finite)
    assert np.allclose(soil["sand_pct"][0][finite], 68.0, atol=1e-6)
    # intermediate NetCDF exists and has the soil group
    with xr.open_dataset(intermediate) as ds:
        assert "sand_pct" in ds and "soil_dz" in ds
        assert ds["sand_pct"].shape == (7, 90, 180)

    # --- step 2: intermediate + CLM -> harmonized v1 NetCDF
    build_v1_surfdata(clm, intermediate, out)
    with xr.open_dataset(out) as ds:
        # CLM-grid output: nlat=4, nlon=8 (from _write_clm)
        assert ds["sand_pct"].shape == (7, 4, 8)
        assert ds["pft_frac"].shape == (1, 17, 4, 8)
        assert ds["monthly_lai"].shape == (12, 17, 4, 8)

    # --- step 3: round-trip through the runtime loader
    grid = _ToyGrid()
    cfg = get_surfdata_preset("legoesm_surfdata")._replace(surf_path=out)
    gsd = load_global_surface_data(cfg, grid,
                                   soil_grid=make_soil_grid(SoilGridConfig()))
    assert gsd.pft_frac.shape == (1, grid.ncol, N_PFT_CLM5)
    assert gsd.lai_monthly.shape == (12, grid.ncol, N_PFT_CLM5)
    # BE-tropical PFT dominates everywhere with cover
    psum = jnp.sum(gsd.pft_frac[0], axis=-1)
    land = psum > 0.5
    be = CLM5_PFT_NAMES.index("broadleaf_evergreen_tropical")
    assert jnp.all(jnp.argmax(gsd.pft_frac[0][land], axis=-1) == be)
