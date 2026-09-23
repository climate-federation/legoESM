"""How far apart are the cell and edge surface pressures the ERA5 initializer builds?

The cell surface pressure gets a barometric correction after the ERA5 terrain is
smoothed onto the mesh; the edge surface pressure, which places the wind levels,
does not.  Any difference means winds and mass are interpolated onto different
pressure columns, an imbalance the model must shed in its first days - which is
when the Arctic warm bias is built (7-8 hPa/day of descent on day 1 against a
physical 0.5-1).
"""
import os, sys, glob
sys.path[:0] = [os.getcwd() + "/src"] + glob.glob(os.getcwd() + "/packages/*") + [os.getcwd()]
import numpy as np, xarray as xr, jax.numpy as jnp
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.grids.regridding import compute_latlon_to_voronoi_weights, regrid_scalar
from legoesm.grids.topography import smooth_phis_voronoi
from legoesm import constants

IC = os.environ.get("ERA5_IC", "/scratch/b/b309178/era5_ic_1979-01-01.zarr")
SUB = int(os.environ.get("SUB", "6"))
PASSES = int(os.environ.get("SMOOTH_PASSES", "4"))

ds = xr.open_zarr(IC)
print("variables:", list(ds.data_vars)[:20], flush=True)
name = {v.lower(): v for v in ds.data_vars}
get = lambda *c: next(name[k] for k in c if k in name)
lat = np.asarray(ds["latitude" if "latitude" in ds else "lat"].values, dtype=np.float64)
lon = np.asarray(ds["longitude" if "longitude" in ds else "lon"].values, dtype=np.float64)
sq = lambda a: np.squeeze(np.asarray(a, dtype=np.float64))
p_s_ll = sq(ds[get("surface_pressure", "sp", "ps")].values)
phis_ll = sq(ds[get("geopotential_at_surface", "z", "phis")].values)
if phis_ll.ndim == 3:
    phis_ll = phis_ll[0]
T_ll = sq(ds[get("temperature", "t", "ta")].values)
T_sfc_ll = T_ll[-1] if T_ll.ndim == 3 else T_ll

mesh = create_voronoi_mesh(SUB)
cw = compute_latlon_to_voronoi_weights(lat, lon, np.asarray(mesh.latCell), np.asarray(mesh.lonCell))
ew = compute_latlon_to_voronoi_weights(lat, lon, np.asarray(mesh.latEdge), np.asarray(mesh.lonEdge))
R = lambda f, w: np.asarray(regrid_scalar(jnp.asarray(f), w), dtype=np.float64)
p_s_cell, p_s_edge = R(p_s_ll, cw), R(p_s_ll, ew)
phis, T_sfc = R(phis_ll, cw), R(T_sfc_ll, cw)

phis_s = np.asarray(smooth_phis_voronoi(jnp.asarray(phis), mesh.cellsOnCell,
                                        mesh.nEdgesOnCell, smoothing_passes=PASSES),
                    dtype=np.float64)
p_s_cell_corr = p_s_cell * np.exp((phis - phis_s) / (constants.R_d * T_sfc))
coe = np.asarray(mesh.cellsOnEdge)
d = (p_s_edge - 0.5 * (p_s_cell_corr[coe[0]] + p_s_cell_corr[coe[1]])) / 100.0
d_nosmooth = (p_s_edge - 0.5 * (p_s_cell[coe[0]] + p_s_cell[coe[1]])) / 100.0
latE = np.rad2deg(np.asarray(mesh.latEdge, dtype=np.float64))

# The edge-minus-cell difference below is NOT a clean measure of the missing
# correction: regridding to edges and averaging cells differ over rough terrain
# even with no smoothing at all.  The missing term itself is the barometric
# factor the edges never get, so measure THAT directly, and by region - if it
# vanishes over the Arctic Ocean it cannot drive a cap-wide descent.
corr = p_s_cell_corr - p_s_cell                              # Pa, per cell
dphis = phis - phis_s
print(f"CONTROL: phis regridded  min {phis.min():.1f} max {phis.max():.1f} m2/s2")
print(f"CONTROL: phis smoothed   min {phis_s.min():.1f} max {phis_s.max():.1f}")
print(f"CONTROL: delta phis      mean|d| {np.abs(dphis).mean():.1f} max {np.abs(dphis).max():.1f}")
print(f"CONTROL: T_sfc           min {T_sfc.min():.1f} max {T_sfc.max():.1f} K")
_fl = np.abs(phis) < 500.0
print(f"CONTROL: on |phis|<500 cells, mean|delta phis| {np.abs(dphis[_fl]).mean():.2f}, "
      f"max {np.abs(dphis[_fl]).max():.1f}")
latC = np.rad2deg(np.asarray(mesh.latCell, dtype=np.float64))
print("\nthe missing term itself (the barometric correction the EDGES never get),")
print("as it would appear at cells [hPa]")
print(f"{'band':>10}{'mean|c|':>10}{'max|c|':>9}{'frac>1hPa':>11}{'frac>10hPa':>12}")
for lo, hi, nm in ((60, 90, "60-90N"), (30, 60, "30-60N"), (-20, 20, "20S-20N"),
                   (-90, -60, "60-90S"), (-90, 90, "global")):
    m = (latC >= lo) & (latC < hi)
    c = np.abs(corr[m]) / 100.0
    print(f"{nm:>10}{c.mean():10.3f}{c.max():9.2f}{(c > 1).mean():11.3f}{(c > 10).mean():12.3f}")
# and over the Arctic specifically, split by whether there is terrain at all
m = (latC >= 60) & (latC < 90)
flat = m & (np.abs(phis) < 100.0 * constants.g / 9.81 * 0.0 + 500.0)   # |z| < ~50 m
print(f"\n60-90N cells with essentially no terrain (|phis| < 500 m2/s2): "
      f"{int(flat.sum())} of {int(m.sum())}, "
      f"mean correction {np.abs(corr[flat]).mean()/100:.4f} hPa, "
      f"max {np.abs(corr[flat]).max()/100:.3f} hPa")

print(f"\nmesh subdivision {SUB}, terrain smoothing passes {PASSES}")
print(f"the cell correction itself: mean {np.abs(p_s_cell_corr - p_s_cell).mean()/100:.3f} hPa, "
      f"max {np.abs(p_s_cell_corr - p_s_cell).max()/100:.1f} hPa")
print("\nedge surface pressure minus the cells it sits between [hPa]")
print(f"{'band':>10}{'mean|d|':>10}{'max|d|':>9}{'mean d':>9}   (without the correction: mean|d|)")
for lo, hi, nm in ((60, 90, "60-90N"), (30, 60, "30-60N"), (-20, 20, "20S-20N"),
                   (-90, -60, "60-90S"), (-90, 90, "global")):
    m = (latE >= lo) & (latE < hi)
    print(f"{nm:>10}{np.abs(d[m]).mean():10.3f}{np.abs(d[m]).max():9.2f}"
          f"{d[m].mean():+9.3f}   {np.abs(d_nosmooth[m]).mean():10.3f}")
