"""Window liquid and ice water paths vs ERA5 on the SAME days, by region.

  cloud_water_path_window.py ERA5_TCLW.nc ERA5_TCIW.nc RUN_WINDOW_DIR [...]

ERA5 files: one-time-step window means of total column cloud liquid (code 78)
and ice (code 79) water [kg/m2], 0.25-degree grid (cdo -timmean -seldate).
RUN_WINDOW_DIR: cmor tree from cmor_window.py with Amon/clwvi (liquid + ice,
CMIP definition), Amon/clivi and fx/sftlf.  Model liquid = clwvi - clivi.
ERA5 is box-averaged into model cells as in t2m_window_metrics.  Prints
cos-lat weighted means [g/m2] per region (land = sftlf >= 50 %).
"""
import glob
import sys

import numpy as np
import xarray as xr

from t2m_window_metrics import _one, era5_on_model

REGIONS = {
    "45-70N land": lambda la, lf: (la >= 45) & (la <= 70) & (lf >= 50),
    "45-70N ocean": lambda la, lf: (la >= 45) & (la <= 70) & (lf < 50),
    "30-45N land": lambda la, lf: (la >= 30) & (la < 45) & (lf >= 50),
    "tropics 30S-30N": lambda la, lf: np.abs(la) <= 30,
}


def main(e5_lw, e5_iw, runs):
    for d in runs:
        cw = xr.open_dataset(_one(f"{d}/Amon/clwvi_*.nc"))["clwvi"].squeeze(drop=True)
        ci = xr.open_dataset(_one(f"{d}/Amon/clivi_*.nc"))["clivi"].squeeze(drop=True)
        lf = xr.open_dataset(_one(f"{d}/fx/sftlf_*.nc"))["sftlf"].squeeze(drop=True).values
        lat, lon = cw["lat"].values, cw["lon"].values
        model = {"LWP": cw.values - ci.values, "IWP": ci.values}
        era5 = {"LWP": era5_on_model(e5_lw, lat, lon, "var78"),
                "IWP": era5_on_model(e5_iw, lat, lon, "var79")}
        for k in model:
            if not (np.isfinite(model[k]).all() and np.isfinite(era5[k]).all()):
                raise SystemExit(f"non-finite {k}")
        la = lat[:, None] * np.ones((1, lon.size))
        w0 = np.cos(np.deg2rad(la))
        for name, f in REGIONS.items():
            w = w0 * f(la, lf)
            mean = lambda x: 1e3 * float((x * w).sum() / w.sum())
            print(f"{d} {name}: LWP model {mean(model['LWP']):.1f} ERA5 {mean(era5['LWP']):.1f} | "
                  f"IWP model {mean(model['IWP']):.1f} ERA5 {mean(era5['IWP']):.1f} g/m2")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
