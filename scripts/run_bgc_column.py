"""1D column BGC experiment using legoESM biogeochemistry.

Replicates the OceanBioME.jl column model example.

Usage
-----
cd /work/bd1083/b309178/diffESM/legoesm_ggn
JAX_ENABLE_X64=1 .venv/bin/python scripts/run_column_bgc.py --years 2 --out column_bgc.nc
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_banded

from legoesm.ocean.biogeochemistry import (
    BiogeoConfig,
    OceanBiogeoState,
    init_biogeo_state,
    step_ocean_biogeochemistry,
)


# =============================================================================
# Grid  — shallow column (200 m) to avoid bottom accumulation
# =============================================================================

def make_column_grid(H=200.0, nlev=40, stretching=2.0):
    """Stretched grid, finer near surface."""
    sigma = np.linspace(0, 1, nlev + 1)
    z_interfaces = -H * (np.exp(stretching * sigma) - 1) / (np.exp(stretching) - 1)
    dz = np.diff(z_interfaces)
    z_full = z_interfaces[:-1] - dz / 2
    return z_full.astype(np.float64), dz.astype(np.float64)


# =============================================================================
# Forcing  (day 0 = Jan 1 = mid-winter)
# =============================================================================

def par_surface(t_day, latitude_deg=47.0):
    """Surface PAR [W/m²]. ~15 W/m² Jan, ~320 W/m² Jul."""
    doy = t_day % 365.25
    angle = 2 * np.pi * (doy - 355) / 365.25   # 0 at winter solstice (day 355)
    PAR_max = 350.0 * np.cos(np.radians(latitude_deg)) + 50.0
    PAR_min = PAR_max * 0.04
    return float(PAR_min + (PAR_max - PAR_min) * 0.5 * (1.0 - np.cos(angle)))


def mixed_layer_depth(t_day):
    """MLD [m]. Deep (~150m) Jan-Feb, shallow (~15m) Jul-Aug."""
    doy = t_day % 365.25
    angle = 2 * np.pi * (doy - 50) / 365.25    # 0 at winter max (day 50)
    return float(15.0 + 135.0 * 0.5 * (1.0 + np.cos(angle)))


def sea_surface_temperature(t_day):
    """SST [°C]. ~7°C Feb, ~19°C Aug."""
    doy = t_day % 365.25
    angle = 2 * np.pi * (doy - 50) / 365.25
    return float(7.0 + 12.0 * 0.5 * (1.0 - np.cos(angle)))


def make_T_S_column(z_full, sst, mld):
    """Temperature and salinity profiles with mixed layer and thermocline."""
    nlev = len(z_full)
    T = np.full(nlev, sst)
    S = np.full(nlev, 35.0)
    for k in range(nlev):
        z = -z_full[k]
        if z > mld:
            frac = min((z - mld) / 100.0, 1.0)
            T[k] = sst - (sst - 4.0) * frac
            S[k] = 35.0 + 0.3 * frac
    return T, S


# =============================================================================
# Vertical diffusion — scipy banded solver (unconditionally stable)
# =============================================================================

def build_diffusion_matrix(dz: np.ndarray, Kv: np.ndarray, dt: float):
    """Implicit diffusion: banded matrix for scipy.linalg.solve_banded."""
    nlev = len(dz)
    dz_half = np.zeros(nlev + 1)
    dz_half[1:-1] = 0.5 * (dz[:-1] + dz[1:])
    dz_half[0]    = dz[0]
    dz_half[-1]   = dz[-1]

    a = np.zeros(nlev)
    b = np.ones(nlev)
    c = np.zeros(nlev)

    for k in range(nlev):
        if k > 0:
            r = dt * Kv[k] / (dz[k] * dz_half[k])
            a[k] = -r
            b[k] += r
        if k < nlev - 1:
            r = dt * Kv[k + 1] / (dz[k] * dz_half[k + 1])
            c[k] = -r
            b[k] += r

    ab = np.zeros((3, nlev))
    ab[0, 1:]  = c[:-1]
    ab[1, :]   = b
    ab[2, :-1] = a[1:]
    return ab


def diffuse_tracer(tracer: np.ndarray, ab: np.ndarray) -> np.ndarray:
    return solve_banded((1, 1), ab, tracer)


def mixed_layer_diffusivity(z_full: np.ndarray, mld: float,
                             Kv_mld=2e-2, Kv_deep=1e-5) -> np.ndarray:
    """Strong mixing in MLD, background below."""
    nlev = len(z_full)
    Kv_mid = np.where(-z_full < mld, Kv_mld, Kv_deep)
    Kv_int = np.zeros(nlev + 1)
    Kv_int[0]    = Kv_mid[0]
    Kv_int[-1]   = Kv_deep          # no flux through bottom
    Kv_int[1:-1] = 0.5 * (Kv_mid[:-1] + Kv_mid[1:])
    return Kv_int


# =============================================================================
# NO3 deep restoring — prevents bottom runaway accumulation
# =============================================================================

def no3_deep_restoring(NO3: np.ndarray, z_full: np.ndarray,
                        NO3_clim_deep: float = 15e-3,
                        z_restore: float = -120.0,
                        tau_days: float = 30.0,
                        dt: float = 1200.0) -> np.ndarray:
    """Relax NO3 toward climatology below z_restore [m].

    This replaces the instantaneous bottom remineralisation closure
    with a slow restoring that prevents deep NO3 from accumulating.
    """
    tau_s = tau_days * 86400.0
    mask_deep = z_full < z_restore   # True below restoring depth
    restore_tend = np.where(mask_deep,
                            (NO3_clim_deep - NO3) / tau_s,
                            0.0)
    return NO3 + dt * restore_tend


# =============================================================================
# Main loop
# =============================================================================

def run_column(n_years=2, dt_minutes=20.0, output_file="column_bgc.nc",
               save_every_hours=24.0):

    print("=" * 60)
    print("legoESM 1D column BGC experiment")
    print("=" * 60)

    # Grid
    nlev = 40
    z_full, dz = make_column_grid(H=200.0, nlev=nlev)
    z_jax  = jnp.array(z_full)
    dz_jax = jnp.array(dz)

    print(f"Grid: {nlev} levels, H=200m, "
          f"dz_surf={dz[0]:.1f}m, dz_bot={dz[-1]:.1f}m")

    # Forcing sanity check
    print(f"\nForcing at key dates:")
    for d, label in [(0,"Jan 1"),(91,"Apr 1"),(172,"Jul 1"),(264,"Sep 21")]:
        print(f"  {label:6s}: PAR={par_surface(d):5.1f} W/m²  "
              f"MLD={mixed_layer_depth(d):5.1f}m  "
              f"SST={sea_surface_temperature(d):.1f}°C")

    # BGC config
    cfg = BiogeoConfig(
        scheme="npzd",
        pCO2_atm=400.0,
        wind_speed=7.0,
        mu_max=1.5,           # back to default — now diffusion is stable
        k_N=0.5e-3,
        alpha_P=0.025,
        g_max=0.6,
        k_P=0.1e-3,
        gamma_Z=0.7,
        m_P=0.05,
        m_Pq=0.1,
        m_Zl=0.02,
        m_Z=0.1,
        remin_rate=0.05,
        w_sink=5.0,
        NO3_init_surf=5e-3,
        NO3_init_deep=15e-3,  # matches restoring target
        Phyto_init=0.05e-3,
        Zoo_init=0.02e-3,
        Det_init=0.005e-3,
    )

    state = init_biogeo_state((nlev,), z_jax, cfg)
    ocean_mask = jnp.ones((), dtype=jnp.float64)

    print(f"\nInitial:  NO3_surf={float(state.NO3[0])*1e3:.2f} mmol/m³  "
          f"Phyto={float(state.Phyto[0])*1e6:.1f} µmol/m³")

    # Time
    dt = dt_minutes * 60.0
    t_end = n_years * 365.25 * 86400.0
    n_steps = int(t_end / dt)
    save_every = max(1, int(save_every_hours * 3600.0 / dt))
    print(f"Time: {n_years} yr, dt={dt_minutes} min, "
          f"{n_steps} steps, saving every {save_every_hours}h")

    # Output storage
    times_days   = []
    NO3_out, Phyto_out, Zoo_out, Det_out = [], [], [], []
    DIC_out, ALK_out = [], []
    pCO2_out, flux_co2_out, PAR_surf_out, MLD_out, SST_out = [], [], [], [], []

    # JIT compile
    print("\nCompiling BGC step...")
    t0 = time.time()
    step_bgc_jit = jax.jit(step_ocean_biogeochemistry, static_argnames=("cfg",))
    T0, S0 = make_T_S_column(z_full, sea_surface_temperature(0), mixed_layer_depth(0))
    _ = step_bgc_jit(
        state, jnp.array(T0), jnp.array(S0), dz_jax, z_jax,
        ocean_mask, dt, cfg, PAR_surf=jnp.array(par_surface(0.0)),
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # Main loop
    print("\nRunning...")
    t0_run = time.time()
    progress_every = max(1, n_steps // 20)

    for step in range(n_steps):
        t_day = step * dt / 86400.0

        sst = sea_surface_temperature(t_day)
        mld = mixed_layer_depth(t_day)
        par = par_surface(t_day)
        T_col, S_col = make_T_S_column(z_full, sst, mld)

        # BGC source/sink
        state, co2_diag = step_bgc_jit(
            state,
            jnp.array(T_col), jnp.array(S_col),
            dz_jax, z_jax,
            ocean_mask, dt, cfg,
            PAR_surf=jnp.array(par),
        )

        # Vertical diffusion
        Kv = mixed_layer_diffusivity(z_full, mld)
        ab = build_diffusion_matrix(dz, Kv, dt)

        NO3_np   = diffuse_tracer(np.array(state.NO3),   ab)
        Phyto_np = diffuse_tracer(np.array(state.Phyto), ab)
        Zoo_np   = diffuse_tracer(np.array(state.Zoo),   ab)
        Det_np   = diffuse_tracer(np.array(state.Det),   ab)
        DIC_np   = diffuse_tracer(np.array(state.DIC),   ab)
        ALK_np   = diffuse_tracer(np.array(state.ALK),   ab)

        # Deep NO3 restoring — prevents bottom accumulation
        NO3_np = no3_deep_restoring(NO3_np, z_full, dt=dt)

        # Clip negatives (diffusion can introduce tiny negatives)
        NO3_np   = np.maximum(NO3_np,   0.0)
        Phyto_np = np.maximum(Phyto_np, 0.0)
        Zoo_np   = np.maximum(Zoo_np,   0.0)
        Det_np   = np.maximum(Det_np,   0.0)

        state = OceanBiogeoState(
            DIC   = jnp.array(DIC_np),
            ALK   = jnp.array(ALK_np),
            NO3   = jnp.array(NO3_np),
            Phyto = jnp.array(Phyto_np),
            Zoo   = jnp.array(Zoo_np),
            Det   = jnp.array(Det_np),
        )

        # Save
        if step % save_every == 0:
            times_days.append(t_day)
            NO3_out.append(NO3_np.copy())
            Phyto_out.append(Phyto_np.copy())
            Zoo_out.append(Zoo_np.copy())
            Det_out.append(Det_np.copy())
            DIC_out.append(DIC_np.copy())
            ALK_out.append(ALK_np.copy())
            pCO2_out.append(float(co2_diag.pCO2_ocean))
            flux_co2_out.append(float(co2_diag.flux_co2))
            PAR_surf_out.append(par)
            MLD_out.append(mld)
            SST_out.append(sst)

        if step % progress_every == 0 and step > 0:
            elapsed = time.time() - t0_run
            eta = elapsed / (step / n_steps) * (1 - step / n_steps)
            print(f"  Yr {t_day/365.25:.2f}  "
                  f"P={float(state.Phyto[0])*1e3:.3f} mmol/m³  "
                  f"NO3={float(state.NO3[0])*1e3:.2f} mmol/m³  "
                  f"pCO2={float(co2_diag.pCO2_ocean):.1f} uatm  "
                  f"ETA {eta:.0f}s")

    print(f"\nDone: {n_steps} steps in {time.time()-t0_run:.1f}s")

    _write_netcdf(
        output_file,
        times_days=np.array(times_days),
        z_full=z_full, dz=dz,
        NO3=np.array(NO3_out),   Phyto=np.array(Phyto_out),
        Zoo=np.array(Zoo_out),   Det=np.array(Det_out),
        DIC=np.array(DIC_out),   ALK=np.array(ALK_out),
        pCO2=np.array(pCO2_out), flux_co2=np.array(flux_co2_out),
        PAR_surf=np.array(PAR_surf_out), MLD=np.array(MLD_out),
        SST=np.array(SST_out), cfg=cfg,
    )
    print(f"Output: {Path(output_file).resolve()}")


# =============================================================================
# NetCDF
# =============================================================================

def _write_netcdf(path, **kw):
    try:
        import netCDF4 as nc
    except ImportError:
        np.savez(path.replace(".nc", ".npz"),
                 **{k: v for k, v in kw.items() if isinstance(v, np.ndarray)})
        return

    nt, nlev = kw["NO3"].shape
    ds = nc.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("time", nt)
    ds.createDimension("depth", nlev)

    tv = ds.createVariable("time",  "f8", ("time",));  tv[:] = kw["times_days"]
    tv.units = "days since start"
    zv = ds.createVariable("depth", "f8", ("depth",)); zv[:] = kw["z_full"]
    zv.units = "m (negative downward)"
    dv = ds.createVariable("dz",    "f8", ("depth",)); dv[:] = kw["dz"]
    dv.units = "m"

    for name, long, units in [
        ("NO3",   "Nitrate",            "mol N m-3"),
        ("Phyto", "Phytoplankton",      "mol N m-3"),
        ("Zoo",   "Zooplankton",        "mol N m-3"),
        ("Det",   "Detritus",           "mol N m-3"),
        ("DIC",   "Dissolved inorg. C", "mol C m-3"),
        ("ALK",   "Total alkalinity",   "mol eq m-3"),
    ]:
        v = ds.createVariable(name, "f4", ("time", "depth"), zlib=True)
        v[:] = kw[name]; v.long_name = long; v.units = units

    for name, long, units in [
        ("pCO2",     "Surface pCO2",     "uatm"),
        ("flux_co2", "Air-sea CO2 flux", "mol C m-2 s-1"),
        ("PAR_surf", "Surface PAR",      "W m-2"),
        ("MLD",      "Mixed-layer depth","m"),
        ("SST",      "Sea surface temp", "degC"),
    ]:
        v = ds.createVariable(name, "f4", ("time",))
        v[:] = kw[name]; v.long_name = long; v.units = units

    cfg = kw["cfg"]
    ds.description = "legoESM 1D column NPZD BGC"
    ds.mu_max = cfg.mu_max; ds.pCO2_atm = cfg.pCO2_atm
    ds.close()
    print(f"Written: {path}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--years",      type=int,   default=2)
    p.add_argument("--dt",         type=float, default=20.0,
                   help="Timestep in minutes (default: 20)")
    p.add_argument("--out",        type=str,   default="column_bgc.nc")
    p.add_argument("--save-every", type=float, default=24.0)
    args = p.parse_args()

    run_column(
        n_years=args.years,
        dt_minutes=args.dt,
        output_file=args.out,
        save_every_hours=args.save_every,
    )
