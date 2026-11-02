"""First 1-D MARBL column run at BATS with idealised cyclic forcing.

Smoke test for the 32-tracer MARBL port (legoesm.ocean.biogeochemistry.marbl)
driven through legoesm.ocean.column.model (KPP + MARBL).  Forcing and initial
profiles are analytic (typical BATS seasonal cycle / subtropical gyre
profiles), so this checks stability and gross behaviour, not skill.

Runs one model year per jitted call (compiled once), stops on NaN, prints
bulk numbers per year and writes daily diagnostics + end-of-day profiles.

Usage (a login node is fine for a few years on CPU):
    python scripts/run_marbl_column.py --years 3
    JAX_PLATFORMS=cuda python scripts/run_marbl_column.py ...   # on a GPU node
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from legoesm.ocean.column.model import (
    ColumnConfig, ColumnForcing, initial_state, make_column_grid, run_column_raw,
)
from legoesm.ocean.biogeochemistry.marbl import MarblConfig, MarblTracers, TRACER_NAMES

LAT, LON, DEPTH = 31.67, -64.17, 4500.0   # BATS
DAY = 86400.0


# ---------------------------------------------------------------- forcing
def cyc(n, phase, mean, amp):
    """Cyclic interval-mean samples: mean + amp*cos(2pi(t - phase)), t in [0,1)."""
    t = (np.arange(n) + 0.5) / n
    return mean + amp * np.cos(2 * np.pi * (t - phase))


def temp_bg(z):   # ~18 C mode water at top, ~2.5 C abyss
    return 2.4 + 8.0 * (1 - np.tanh((z - 700.0) / 350.0))


def salt_bg(z):   # ~36.6 surface, ~34.95 deep
    return 34.95 + 0.825 * (1 - np.tanh((z - 800.0) / 400.0))


# ------------------------------------------------------------------- WOA
WOA_DIR = "/work/bd0854/DATA/ESMValTool2/OBS/Tier2/WOA"
WOA_FILES = {
    "thetao": "OBS_WOA_clim_2013v2_Omon_thetao_200001-200012.nc",
    "so":     "OBS_WOA_clim_2013v2_Omon_so_200001-200012.nc",
    "no3":    "OBS6_WOA_clim_2018_Oyr_no3_200001-200012.nc",
    "po4":    "OBS6_WOA_clim_2018_Oyr_po4_200001-200012.nc",
    "si":     "OBS6_WOA_clim_2018_Oyr_si_200001-200012.nc",
    "o2":     "OBS6_WOA_clim_2018_Oyr_o2_200001-200012.nc",
}


def _to_model_units(var, v, units):
    """WOA/CMOR units -> degC, PSU, mmol/m3."""
    u = units.replace(" ", "").lower()
    if var == "thetao":
        return v - 273.15 if (u in ("k", "kelvin") or np.nanmean(v) > 100) else v
    if var == "so":
        return v * 1000.0 if np.nanmax(v) < 1 else v
    if "mmol" in u:
        return v
    if "umol/kg" in u or "µmol/kg" in u or "umolkg-1" in u or "µmolkg-1" in u:
        return v * 1.025
    if "umol" in u or "µmol" in u:
        return v * 1e-3 if "m-3" in u or "/m3" in u else v
    if "ml/l" in u or "mll-1" in u:
        return v * 44.66
    if "mol" in u:
        return v * 1e3
    return v * 1e3 if np.nanmax(v) < 1 else v       # unknown: assume mol/m3 if tiny


def _woa_column(path, var, zt):
    """WOA profile(s) at BATS interpolated to zt -> (ntime, nlev), plus units string."""
    import xarray as xr
    da = xr.open_dataset(path, decode_times=False)[var]
    latn = next(d for d in da.dims if d.lower().startswith("lat"))
    lonn = next(d for d in da.dims if d.lower().startswith("lon"))
    zn = next(d for d in da.dims if d not in (latn, lonn) and not d.lower().startswith("time"))
    lon = LON % 360 if float(da[lonn].max()) > 180 else LON
    col = None
    for dlat, dlon in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
        c = da.sel({latn: LAT + dlat, lonn: lon + dlon}, method="nearest")
        if np.all(np.isfinite(c.isel({zn: 0}).values)):
            col = c
            break
    if col is None:
        raise ValueError(f"no valid WOA column near BATS in {path}")
    other = [d for d in col.dims if d != zn]
    col = col.transpose(*other, zn)
    z = np.abs(np.asarray(col[zn].values, float))
    if col[zn].attrs.get("units", "").lower() in ("cm", "centimeters"):
        z = z / 100.0
    units = da.attrs.get("units", "")
    vals = _to_model_units(var, np.asarray(col.values, float).reshape(-1, z.size), units)
    out = np.empty((vals.shape[0], zt.size))
    for i, v in enumerate(vals):
        ok = np.isfinite(v)
        out[i] = np.interp(zt, z[ok], v[ok])       # constant extrapolation below last level
    return out, units


def load_woa(zt, woa_dir):
    woa = {}
    print("WOA at BATS (model units: degC, PSU, mmol/m3):")
    print(f"  {'var':6s} {'units in file':16s} {'nt':>3s}  " + "  ".join(f"{d:>6.0f}m" for d in (0, 100, 200, 500, 1000, 2000, 4000)))
    for var, fname in WOA_FILES.items():
        prof, units = _woa_column(str(Path(woa_dir) / fname), var, zt)
        woa[var] = prof
        m = prof.mean(0)
        vals = [np.interp(d, zt, m) for d in (0, 100, 200, 500, 1000, 2000, 4000)]
        print(f"  {var:6s} {units[:16]:16s} {prof.shape[0]:3d}  " + "  ".join(f"{v:7.2f}" for v in vals))
    return woa


def make_forcing(zt, woa=None, n_sfc=365, n_clim=12):
    tau_x = cyc(n_sfc, 0.0, 0.065, 0.035)          # westerlies, ~0.10 winter, ~0.03 summer [Pa]
    tau_y = np.zeros(n_sfc)
    u10_sqr = np.abs(tau_x) / (1.22 * 1.3e-3)      # from bulk formula, ~6-8 m/s
    sw_net = cyc(n_sfc, 0.47, 200.0, 80.0)         # daily mean, peak late June [W/m2]
    q_ns = cyc(n_sfc, 0.0, -200.0, -75.0)          # -275 Jan, -125 Jul; annual net heat ~0

    sst = cyc(n_clim, 0.635, 23.0, 4.5)            # ~18.5 Mar, ~27.5 Aug
    mld = cyc(n_clim, 0.15, 90.0, 70.0)            # ~160 m late Feb, ~20 m summer
    if woa is not None and woa["thetao"].shape[0] == 12 and np.std(woa["thetao"][:, 0]) > 0.3:
        T_clim, S_clim = woa["thetao"], woa["so"]
        print("T/S relaxation: WOA monthly climatology")
    else:
        if woa is not None:
            Tb, Sb = woa["thetao"].mean(0), woa["so"].mean(0)
            print("T/S relaxation: WOA annual mean + analytic surface seasonal cycle "
                  "(WOA file has no real monthly variation)")
        else:
            Tb, Sb = temp_bg(zt), salt_bg(zt)
            print("T/S relaxation: analytic profiles")
        T_clim = Tb[None, :] + (sst[:, None] - Tb[0]) * np.exp(-zt[None, :] / mld[:, None])
        S_clim = np.repeat(Sb[None, :], n_clim, axis=0)

    const = lambda v: np.full(n_sfc, v)
    f = ColumnForcing(
        tau_x=tau_x, tau_y=tau_y, u10_sqr=u10_sqr, sw_net=sw_net, q_nonsolar=q_ns,
        freshwater=const(0.0), atm_press=const(1.0), ifrac=const(0.0),
        xco2=const(410.0),
        dust_flux=const(3.0e-11),   # ~1 g/m2/yr [kg/m2/s]
        iron_flux=const(4.0e-10),   # ~0.0125 mmol/m2/yr soluble Fe [mmol/m2/s]
        nox_flux=const(1.6e-6),     # ~50 mmol N/m2/yr
        nhy_flux=const(6.0e-7),     # ~20 mmol N/m2/yr
        T_clim=T_clim, S_clim=S_clim,
    )
    return jax.tree_util.tree_map(jnp.asarray, f), T_clim, S_clim


# ---------------------------------------------------------- initial BGC
def initial_bgc(z, woa=None):
    step = lambda zc, w: 0.5 * (1 + np.tanh((z - zc) / w))
    ez = lambda s: np.exp(-z / s)

    NO3 = 21.0 * step(500.0, 250.0)
    spC, diatC, diazC = 0.5 * ez(100.0), 0.1 * ez(100.0), 0.05 * ez(80.0)
    DOCr = np.full_like(z, 42.0)
    DOC = 20.0 * ez(150.0) + 1.0
    DIC = 2050.0 + 140.0 * step(700.0, 400.0)
    ALK = 2330.0 + 60.0 * ez(600.0)

    p = dict(
        PO4=1.35 * step(500.0, 250.0) + 0.01,
        NO3=NO3,
        SiO3=1.0 + 29.0 * step(1500.0, 700.0),
        NH4=np.full_like(z, 0.01),
        Fe=2.0e-4 + 4.0e-4 * step(500.0, 300.0),
        Lig=np.full_like(z, 1.5e-3),
        O2=250.0 - 35.0 * ez(200.0) - 80.0 * np.exp(-((z - 800.0) / 300.0) ** 2),
        DIC=DIC, DIC_ALT_CO2=DIC.copy(),
        ALK=ALK, ALK_ALT_CO2=ALK.copy(),
        DOC=DOC, DON=DOC / 12.0, DOP=DOC / 200.0,
        DOCr=DOCr, DONr=DOCr / 16.0, DOPr=DOCr / 1000.0,
        zooC=0.3 * ez(100.0),
        spC=spC, spChl=spC * 12.0 / 60.0, spP=spC / 117.0, spFe=spC * 2.0e-5,
        spCaCO3=spC * 0.07,
        diatC=diatC, diatChl=diatC * 12.0 / 50.0, diatP=diatC / 117.0,
        diatFe=diatC * 2.0e-5, diatSi=diatC * 0.13,
        diazC=diazC, diazChl=diazC * 12.0 / 60.0, diazP=diazC / 300.0,
        diazFe=diazC * 4.0e-5,
    )
    if woa is not None:
        for wk, tk in (("no3", "NO3"), ("po4", "PO4"), ("si", "SiO3"), ("o2", "O2")):
            p[tk] = np.maximum(woa[wk].mean(0), 1e-3)
        print("initial NO3, PO4, SiO3, O2: WOA18 annual at BATS; other tracers analytic")
    missing = set(TRACER_NAMES) - set(p)
    extra = set(p) - set(TRACER_NAMES)
    if missing or extra:
        raise KeyError(f"tracer mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    return MarblTracers(**{k: jnp.asarray(p[k]) for k in TRACER_NAMES})


# ------------------------------------------------------------- summary
def summarise(year, states, diags, wall):
    to_np = lambda x: np.asarray(x)
    bgc = {k: to_np(getattr(states.bgc, k)) for k in TRACER_NAMES}
    bad = [k for k, a in bgc.items() if not np.all(np.isfinite(a))]
    if not np.all(np.isfinite(to_np(states.T))):
        bad.append("T")
    neg = {k: a.min() for k, a in bgc.items() if a.min() < 0}

    npp = to_np(diags.NPP)                                    # (365, 3) mmol C/m2/s
    s2y = 86400 * 365 * 12e-3                                 # -> g C/m2/yr
    chl_sfc = bgc["spChl"][:, 0] + bgc["diatChl"][:, 0] + bgc["diazChl"][:, 0]
    print(f"\n--- year {year}  ({wall:.0f} s wall)")
    print(f"  non-finite: {bad if bad else 'none'}")
    print(f"  negative minima: " + (", ".join(f"{k}={v:.2e}" for k, v in neg.items()) if neg else "none"))
    print(f"  NPP [gC/m2/yr] sp/diat/diaz/total: "
          + " / ".join(f"{v:.0f}" for v in npp.mean(0) * s2y) + f" / {npp.sum(1).mean() * s2y:.0f}")
    print(f"  POC flux 100m [mmol C/m2/d]: {to_np(diags.POC_flux_100m).mean() * 86400:.2f}")
    print(f"  CO2 flux [mol/m2/yr, +in]: {to_np(diags.flux_co2).mean() * 86400 * 365e-3:.2f}"
          f"   pCO2 surf range: {to_np(diags.pCO2surf).min():.0f}-{to_np(diags.pCO2surf).max():.0f} ppmv")
    print(f"  surface Chl [mg/m3]: {chl_sfc.min():.3f}-{chl_sfc.max():.3f}"
          f"   surface NO3: {bgc['NO3'][:, 0].min():.3f}-{bgc['NO3'][:, 0].max():.3f}"
          f"   surface O2: {bgc['O2'][:, 0].mean():.0f}")
    print(f"  h_bl [m]: {to_np(diags.h_bl).min():.0f}-{to_np(diags.h_bl).max():.0f}"
          f"   SST: {to_np(diags.sst).min():.1f}-{to_np(diags.sst).max():.1f}")
    return not bad


# ---------------------------------------------------------------- output
def save(out, zt, states_list, diags_list):
    cat = lambda xs: np.concatenate([np.asarray(x) for x in xs], axis=0)
    days = np.arange(sum(np.asarray(s.T).shape[0] for s in states_list)) + 1
    prof = {"T": cat([s.T for s in states_list]), "S": cat([s.S for s in states_list])}
    for k in TRACER_NAMES:
        prof[k] = cat([getattr(s.bgc, k) for s in states_list])
    diag = {k: cat([getattr(d, k) for d in diags_list]) for k in diags_list[0]._fields}

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import xarray as xr
        ds = xr.Dataset(coords={"day": days, "z": zt, "pft": ["sp", "diat", "diaz"]})
        for k, a in prof.items():
            ds[k] = (("day", "z"), a)
        for k, a in diag.items():
            ds[k] = (("day", "pft"), a) if a.ndim == 2 else (("day",), a)
        ds.attrs.update(description="legoESM 1-D MARBL column, BATS, idealised cyclic forcing",
                        lat=LAT, lon=LON)
        ds.to_netcdf(out)
    except ImportError:
        out = out.with_suffix(".npz")
        np.savez(out, day=days, z=zt, **prof, **{f"diag_{k}": v for k, v in diag.items()})
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--dt", type=float, default=3600.0)
    ap.add_argument("--nlev", type=int, default=60)
    ap.add_argument("--out", type=str, default="results/marbl_column_bats/marbl_column_bats.nc")
    ap.add_argument("--woa-dir", type=str, default=WOA_DIR)
    ap.add_argument("--analytic", action="store_true",
                    help="analytic T/S and nutrient profiles instead of WOA (old behaviour)")
    ap.add_argument("--check-woa", action="store_true",
                    help="only load and print the WOA profiles, then exit")
    a = ap.parse_args()

    print(f"jax {jax.__version__} on {jax.devices()}  x64={jax.config.jax_enable_x64}")
    grid = make_column_grid(LAT, LON, DEPTH, n_levels=a.nlev)
    grid = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float64) if hasattr(a, "dtype") and jnp.issubdtype(a.dtype, jnp.floating) else a, grid)
    zt = np.asarray(grid.zt, dtype=np.float64)
    print(f"grid: {zt.size} levels, zt {zt[0]:.1f} .. {zt[-1]:.0f} m")
    if zt[-1] <= 0:
        raise ValueError("expected zt positive downward")

    woa = None if a.analytic else load_woa(zt, a.woa_dir)
    if a.check_woa:
        return
    forcing, T_clim, S_clim = make_forcing(zt, woa)
    cfg = ColumnConfig(dt=a.dt)
    mcfg = MarblConfig()
    steps = int(round(DAY / a.dt))
    st = initial_state(jnp.asarray(T_clim[0]), jnp.asarray(S_clim[0]), initial_bgc(zt, woa))

    @jax.jit
    def run_year(st, forcing):
        return run_column_raw(st, forcing, grid, cfg, mcfg, 365, steps)

    states_list, diags_list = [], []
    for y in range(1, a.years + 1):
        t0 = time.time()
        st, states, diags = run_year(st, forcing)
        jax.block_until_ready(st)
        ok = summarise(y, states, diags, time.time() - t0)
        states_list.append(states)
        diags_list.append(diags)
        if not ok:
            print("\nNaN/Inf found, stopping.")
            break
    save(Path(a.out), zt, states_list, diags_list)


if __name__ == "__main__":
    main()
