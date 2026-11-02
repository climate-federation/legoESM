"""Check marbl_step_3d against the column MARBL calls.

Builds a (2, 2, nlev) grid: three identical BATS columns and one land column.
The 3-D update of each wet column must equal the column-model MARBL update
(interior tendency + surface flux into the top layer, forward Euler), the land
column must be unchanged, and NPP must equal the column integral of photoC.

Usage (repo root, JAX_ENABLE_X64=1):
    python scripts/test_marbl_driver3d.py
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import importlib.util
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from legoesm.ocean.column.model import make_column_grid
from legoesm.ocean.biogeochemistry.marbl import (
    TRACER_NAMES, MarblConfig, MarblInteriorForcing, MarblSurfaceForcing, MarblTracers,
    marbl_interior_tendencies, marbl_surface_fluxes,
)
from legoesm.ocean.biogeochemistry.marbl.driver3d import Marbl2DForcing, marbl_step_3d

spec = importlib.util.spec_from_file_location(
    "rmc", Path(__file__).with_name("run_marbl_column.py"))
rmc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rmc)

NLEV, DT = 60, 3600.0
grid = make_column_grid(rmc.LAT, rmc.LON, rmc.DEPTH, n_levels=NLEV)
f64 = lambda a: jnp.asarray(a, jnp.float64)  # noqa: E731
dz, zt, zw = f64(grid.dz), f64(grid.zt), f64(grid.zw)
zt_np = np.asarray(zt)
bgc = rmc.initial_bgc(zt_np)
T, S = f64(rmc.temp_bg(zt_np) + 3.0), f64(rmc.salt_bg(zt_np))
cfg = MarblConfig()
F = dict(sw_net=220.0, u10_sqr=50.0, atm_press=1.0, ifrac=0.0, xco2=410.0,
         dust_flux=3e-11, iron_flux=4e-10, nox_flux=1.6e-6, nhy_flux=6e-7)

# ---- reference: the column-model MARBL calls -------------------------------------
interior = MarblInteriorForcing(
    potemp=T, salinity=S, pressure_bar=1026.0 * 9.80616 * zt / 1e5,
    surf_shortwave=jnp.asarray([F["sw_net"]]), PAR_col_frac=jnp.ones((1,)),
    dust_flux=f64(F["dust_flux"]), fesedflux=jnp.zeros(NLEV), feventflux=jnp.zeros(NLEV))
tend, idiag = marbl_interior_tendencies(bgc, interior, dz, zt, zw, jnp.asarray(NLEV), cfg, n_iter=50)
sforc = MarblSurfaceForcing(sst=T[0], sss=S[0], u10_sqr=f64(F["u10_sqr"]), atm_press=f64(1.0),
                            ifrac=f64(0.0), xco2=f64(410.0), xco2_alt_co2=f64(410.0),
                            dust_flux=f64(F["dust_flux"]), iron_flux=f64(F["iron_flux"]),
                            nox_flux=f64(F["nox_flux"]), nhy_flux=f64(F["nhy_flux"]))
top = MarblTracers(**{k: getattr(bgc, k)[0] for k in TRACER_NAMES})
sflux, sdiag = marbl_surface_fluxes(top, sforc, cfg, n_iter=50)
ref = {k: np.asarray((getattr(bgc, k) + DT * getattr(tend, k)).at[0].add(DT * getattr(sflux, k) / dz[0]))
       for k in TRACER_NAMES}

# ---- 3-D: three copies of the column + one land column ---------------------------
tile = lambda a: jnp.broadcast_to(a, (2, 2, NLEV))  # noqa: E731
bgc3 = MarblTracers(**{k: tile(getattr(bgc, k)) for k in TRACER_NAMES})
h_k = tile(dz).at[1, 1].set(0.0)                     # (1, 1) is land
forcing = Marbl2DForcing(**{k: jnp.full((2, 2), v) for k, v in F.items()})
new, diag = marbl_step_3d(bgc3, tile(T), tile(S), h_k, forcing, zt, zw, dz, cfg, DT, batch_size=2)

worst, ok = 0.0, True
for k in TRACER_NAMES:
    got = np.asarray(getattr(new, k))
    for (i, j) in [(0, 0), (0, 1), (1, 0)]:
        err = np.max(np.abs(got[i, j] - ref[k]) / (np.abs(ref[k]) + 1e-12))
        worst = max(worst, err)
    if not np.array_equal(got[1, 1], np.asarray(getattr(bgc3, k))[1, 1]):
        print(f"FAIL: land column changed for {k}")
        ok = False
npp_ref = np.asarray(jnp.sum(idiag.photoC * dz, axis=-1))
npp_err = np.max(np.abs(np.asarray(diag.NPP)[:, 0, 0] - npp_ref) / (np.abs(npp_ref) + 1e-30))
nan_tracers = [k for k in TRACER_NAMES if not np.all(np.isfinite(np.asarray(getattr(new, k))))]

print(f"max relative difference 3-D vs column update: {worst:.2e}")
print(f"max relative difference NPP: {npp_err:.2e}")
print(f"flux_co2 3-D {float(diag.flux_co2[0, 0]):.4e}  column {float(sdiag.flux_co2):.4e}")
print(f"non-finite tracers: {nan_tracers if nan_tracers else 'none'}")
ok = ok and worst < 1e-10 and npp_err < 1e-10 and not nan_tracers
print("PASS" if ok else "FAIL")
