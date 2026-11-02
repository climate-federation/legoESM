"""
Offline NPZDv2 BGC run forced by ORAS5 reanalysis (OPeNDAP).

T/S from ORAS5 streamed monthly. BGC stepped at 6h intervals.
No ocean dynamics — BGC source/sink only (no advection).

Usage:
    sbatch scripts/submit_bgc_offline.sh
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as _pd
import xarray as xr

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_woa import init_bgc_from_woa
from legoesm.ocean.biogeochemistry.config import BiogeoConfig
from legoesm.ocean.biogeochemistry import init_biogeo_state
from legoesm.ocean.biogeochemistry.carbon_cycle import step_ocean_biogeochemistry

# ── Config ────────────────────────────────────────────────────────────────────
N_LAT, N_LON = 180, 360
N_LEV        = 35
H_MAX        = 5500.0
DT           = 3600.0 * 6          # 6-hour BGC timestep [s]
YEARS        = 10
START_YEAR   = 1990
WOA_DIR      = Path("/work/bd0854/DATA/ESMValTool2/OBS/Tier2/WOA")
OUT_DIR      = Path("results/bgc_offline_oras5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORAS5_BASE = "http://apdrc.soest.hawaii.edu:80/dods/public_data/Reanalysis_Data/ORAS5/1x1_grid"
PAR_SURF_MEAN = 170.0 * 0.43       # mean surface PAR [W/m²]
K_W           = 0.04               # water attenuation [1/m]

# ── Grid ──────────────────────────────────────────────────────────────────────
print("Setting up grid...")
grid     = create_latlon_grid(N_LAT, N_LON)
z_coord  = create_ocean_z_star(n_levels=N_LEV, H_max=H_MAX)
z_centers = np.abs(np.array(z_coord.z_full_ref))
dz_jnp   = jnp.array(np.diff(np.concatenate([[0], z_centers])))
z_ref_jnp = jnp.array(z_coord.z_full_ref)

# Precompute PAR profile (static — no seasonal cycle for simplicity)
par_prof = PAR_SURF_MEAN * np.exp(-K_W * z_centers)  # (N_LEV,)
PAR_surf_scalar = float(par_prof[0])

# ── BGC config ────────────────────────────────────────────────────────────────
bgc_cfg = BiogeoConfig(scheme="npzd_v2")

# ── WOA18 initialization ──────────────────────────────────────────────────────
print("Loading WOA18 nutrients...")
woa_bgc = init_bgc_from_woa(
    grid, z_coord,
    no3_path=WOA_DIR / "OBS6_WOA_clim_2018_Oyr_no3_200001-200012.nc",
    po4_path=WOA_DIR / "OBS6_WOA_clim_2018_Oyr_po4_200001-200012.nc",
    si_path =WOA_DIR / "OBS6_WOA_clim_2018_Oyr_si_200001-200012.nc",
)

bgc_state = init_biogeo_state((N_LAT, N_LON, N_LEV), z_coord.z_full_ref, bgc_cfg)
bgc_state = bgc_state._replace(
    NO3=woa_bgc["NO3"].astype(jnp.float64),
    PO4=woa_bgc["PO4"].astype(jnp.float64),
    Si =woa_bgc["Si" ].astype(jnp.float64),
    Fe =woa_bgc["Fe" ].astype(jnp.float64),
)
print(f"  NO3 surface: {float(bgc_state.NO3[...,0].mean())*1e3:.2f} mmol/m3")
print(f"  Fe  surface: {float(bgc_state.Fe[...,0].mean())*1e9:.2f} nmol/m3")

# ── ORAS5 OPeNDAP ─────────────────────────────────────────────────────────────
print("Opening ORAS5 OPeNDAP streams...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ds_T = xr.open_dataset(f"{ORAS5_BASE}/votemper/opa0", engine="pydap")
    ds_S = xr.open_dataset(f"{ORAS5_BASE}/vosaline/opa0", engine="pydap")

oras5_levs = ds_T["lev"].values   # (75,) positive down


def load_oras5_month(year: int, month: int) -> tuple[np.ndarray, np.ndarray]:
    """Load ORAS5 T,S for one month and interpolate to model z-star levels."""
    t_idx = (year - 1979) * 12 + (month - 1)
    t_idx = min(t_idx, 479)   # cap at last available month (Dec 2018)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T_raw = np.array(ds_T["votemper"].isel(time=t_idx).values)  # (75, 180, 360)
        S_raw = np.array(ds_S["vosaline"].isel(time=t_idx).values)

    # Vectorised vertical interpolation: (75, 180*360) → (N_LEV, 180*360)
    ncols = N_LAT * N_LON
    T_flat = T_raw.reshape(75, ncols)          # (75, ncols)
    S_flat = S_raw.reshape(75, ncols)

    T_out = np.zeros((N_LEV, ncols))
    S_out = np.zeros((N_LEV, ncols))

    for k, z_k in enumerate(z_centers):
        # Linear interpolation across ORAS5 levels at each column
        T_out[k] = np.array([
            np.interp(z_k, oras5_levs, T_flat[:, j],
                      left=T_flat[0, j], right=T_flat[-1, j])
            for j in range(ncols)
        ])
        S_out[k] = np.array([
            np.interp(z_k, oras5_levs, S_flat[:, j],
                      left=S_flat[0, j], right=S_flat[-1, j])
            for j in range(ncols)
        ])

    T_3d = T_out.reshape(N_LEV, N_LAT, N_LON).transpose(1, 2, 0)   # (lat,lon,lev)
    S_3d = S_out.reshape(N_LEV, N_LAT, N_LON).transpose(1, 2, 0)

    # Fill invalid
    T_3d = np.where(np.isfinite(T_3d), T_3d, 2.0)
    S_3d = np.where(np.isfinite(S_3d) & (S_3d > 0.5), S_3d, 34.7)
    return T_3d, S_3d


def land_mask_from_T(T_3d: np.ndarray) -> np.ndarray:
    return (T_3d[:, :, 0] > -5.0).astype(np.float64)


# ── Main loop ─────────────────────────────────────────────────────────────────
steps_per_month = int(30.5 * 86400 / DT)
total_months    = YEARS * 12
ts_records      = []

print(f"\nStarting {YEARS}-year offline BGC ({total_months} months, "
      f"{steps_per_month} steps/month at DT={DT/3600:.0f}h)...")
t0 = time.time()

for month_idx in range(total_months):
    year   = START_YEAR + month_idx // 12
    month  = month_idx % 12 + 1

    # Cycle ORAS5 over 1990-2009 (available)
    oras5_year = 1990 + (month_idx % (20 * 12)) // 12

    t1 = time.time()
    T_now, S_now = load_oras5_month(oras5_year, month)
    mask_2d      = jnp.array(land_mask_from_T(T_now))
    T_jnp        = jnp.array(T_now)
    S_jnp        = jnp.array(S_now)

    # Surface PAR (2D, constant for now — seasonal variation via MLD future work)
    PAR_surf_2d = jnp.full((N_LAT, N_LON), PAR_SURF_MEAN)

    # ── BGC steps ─────────────────────────────────────────────────────────
    for _ in range(steps_per_month):
        bgc_state, _ = step_ocean_biogeochemistry(
            bgc_state,
            T_jnp, S_jnp,
            dz_jnp, z_ref_jnp,
            mask_2d, DT, bgc_cfg,
            PAR_surf=PAR_surf_2d,
        )

    # ── Diagnostics ───────────────────────────────────────────────────────
    no3  = float(jnp.nanmean(bgc_state.NO3[:,:,0])) * 1e3
    chl  = float(jnp.nanmean(
        bgc_state.Chl_d[:,:,0] + bgc_state.Chl_n[:,:,0])) * 1e6
    pd   = float(jnp.nanmean(bgc_state.Pd[:,:,0])) * 1e3
    fe   = float(jnp.nanmean(bgc_state.Fe[:,:,0])) * 1e9
    dic  = float(jnp.nanmean(bgc_state.DIC[:,:,0])) * 1e3
    load_t = time.time() - t1

    print(f"  {year}/{month:02d} | NO3={no3:.2f} | Chl={chl:.4f} mg/m3 | "
          f"Pd={pd:.4f} | Fe={fe:.2f} nM | DIC={dic:.1f} | {load_t:.0f}s/mo")

    ts_records.append(dict(year=year, month=month,
                           NO3=no3, Chl=chl, Pd=pd, Fe=fe, DIC=dic))

    # ── Monthly checkpoint ─────────────────────────────────────────────────
    ckpt_path = OUT_DIR / f"restart_{year}{month:02d}.npz"
    save_dict = {f"biogeo_{f}": np.array(getattr(bgc_state, f))
                 for f in bgc_state._fields if getattr(bgc_state, f) is not None}
    save_dict["land_mask"] = np.array(mask_2d)
    save_dict["T"] = T_now
    np.savez(str(ckpt_path), **save_dict)

# ── Timeseries ────────────────────────────────────────────────────────────────
pd_df = _pd.DataFrame(ts_records)
pd_df.to_csv(OUT_DIR / "timeseries.csv", index=False)
print(f"\nDone in {(time.time()-t0)/3600:.1f}h. Results in {OUT_DIR}/")
