import sys
import numpy as np
import jax.numpy as jnp
import xarray as xr
from legoesm import constants
from legoesm.coupler.config import CouplerConfig
from legoesm.coupler.coupler import ocean_tile_response
from legoesm.forcing.jra55_do import jra55_to_atm_surface, jra55_to_freshwater, load_jra55_slice

CACHE = "/work/bb1596/b384036/diffESM/data/jra55_ryf9091_1deg_noleap.zarr"
RESTARTS = sys.argv[1:]
ds = xr.open_zarr(CACHE, decode_times=False)
ref_year = int(ds.attrs.get("ref_year", 1958))
ny, nx = int(ds.sizes["lat"]), int(ds.sizes["lon"])
for v in ["prra", "prsn", "friver", "rsds", "rlds"]:
    if v in ds: print(f"  {v}: units={ds[v].attrs.get('units','?')}")
lat_deg = np.linspace(-90 + 90/ny, 90 - 90/ny, ny)
lon_deg = np.linspace(180/nx, 360 - 180/nx, nx)
lat_2d = jnp.asarray(np.deg2rad(lat_deg)[:, None]); lon_2d = jnp.asarray(np.deg2rad(lon_deg)[None, :])
w = np.cos(np.deg2rad(lat_deg))[:, None] * np.ones((1, nx))
cfg = CouplerConfig(bulk_scheme="large_yeager", z_ref=10.0, z_t_atm=2.0, z_q_atm=2.0)
absl = np.abs(lat_deg)[:, None] * np.ones((1, nx))
for rpath in RESTARTS:
    z = np.load(rpath)
    T = np.asarray(z["T"], float); sst = T[..., 0]
    ocean = np.asarray(z["land_mask"]) > 0.5
    gm = lambda a, m=ocean: float(np.sum(np.asarray(a, float)[m] * w[m]) / np.sum(w[m]))
    sst_K = jnp.asarray(sst + constants.T_freeze); zero = jnp.zeros_like(sst_K)
    acc = {}
    for d in np.arange(0.5, 365, 5.0):
        slc = load_jra55_slice(CACHE, float(d), ref_year=ref_year, cycle=True)
        atm = jra55_to_atm_surface(slc, lat_2d, lon_2d, float(d), ref_year=ref_year, co2_ppmv=400.0)
        tr = ocean_tile_response(atm, sst_K, zero, zero, cfg)
        swn = atm.sw_down * (1.0 - tr.albedo)
        qns = atm.lw_down - tr.lw_up - tr.shflx - tr.lhflx
        fw = jra55_to_freshwater(slc, tr.lhflx)
        for k, v in [("SW_down", atm.sw_down), ("albedo", tr.albedo), ("SW_net", swn), ("LW_down", atm.lw_down),
                     ("LW_up", tr.lw_up), ("SH_up", tr.shflx), ("LH_up", tr.lhflx), ("Q_ns", qns),
                     ("Q_net", swn + qns), ("P", fw.precip), ("E", fw.evap), ("R", slc.friver)]:
            acc.setdefault(k, []).append(gm(v))
        for k, m in [("Q_net |lat|<30", absl < 30), ("Q_net 30-60", (absl >= 30) & (absl < 60)), ("Q_net >60", absl >= 60)]:
            acc.setdefault(k, []).append(gm(swn + qns, ocean & m))
    print(f"\n=== {rpath}\n  global-mean SST {gm(sst):.2f} C (area-weighted)")
    for k, v in acc.items():
        val = np.mean(v)
        if k in ("P", "E", "R"): print(f"  {k:16s} {val*86400:8.3f} mm/day")
        elif k == "albedo": print(f"  {k:16s} {val:8.3f}")
        else: print(f"  {k:16s} {val:8.1f} W/m2")
print("\nExpected roughly: SW_net ~170, LW_net ~-55, SH ~10, LH ~95, Q_net ~0;  P ~3.0, E ~3.3, R ~0.3 mm/day")
