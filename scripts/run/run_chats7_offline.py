"""Drive legoESM's CLM-ML-JAX adapter offline at the CHATS7 walnut orchard,
matching the Fortran CLM-ML v2 CHATS reference (Bonan et al. 2026, AFM).

Purpose (E1 of docs/MLC_experiment_plan/experiment_plan.md).
-----------------------------------------------------------

Reads the in-tree tower forcing NetCDFs at
``clm-ml-jax/src/input_files/tower-forcing/CHATS7/{2007-04,2007-05}.nc``,
loops over 30-min timesteps calling
:func:`legoesm.land.canopy.clm_ml_interface.compute_clm_ml_canopy_fluxes`
with adapter overrides that match the Fortran namelist
(``met_type=3``, ``runge_kutta_type=41``, ``num_ml_steps=30`` → dtime_ml=60 s),
and writes ``flux.out``, ``fsun.out``, ``aux.out`` files that match the
Fortran + JAX-standalone schema so
``scripts/validate/validate_clm_ml_canopy.py`` can diff them.

Site parameters (CHATS7, walnut orchard, CA; Patton et al. 2011):
  * lat 38.49°N, lon -121.84°E
  * PFT 7 (broadleaf deciduous temperate tree, BDT)
  * canopy top 10 m, LAI 2.6 peak (May-2007)
  * reference height z_ref = 23 m (from ``ZBOT`` in the forcing NetCDF)

Usage
-----
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run/run_chats7_offline.py \
        --output-dir results/chats7_adapter_2007-05 \
        [--days 31] [--smoke]

``--smoke`` runs only the first day (48 timesteps) for quick sanity checks.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time as time_mod
from types import SimpleNamespace
from typing import IO

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import xarray as xr


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_pkg_root = REPO_ROOT / "packages"
_paths = [REPO_ROOT / "src"]
_paths += [p for p in sorted(_pkg_root.iterdir()) if p.is_dir() and (p / "legoesm").exists()]
_paths.append(REPO_ROOT / "clm-ml-jax" / "src")
for _p in _paths:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# --------------------------------------------------------------------------
# netCDF4 shim: on Apple Silicon + Python 3.14 the pre-built netCDF4 wheel is
# x86_64 and refuses to load. clm-ml-jax's ``LookupPsihatINI`` uses only
# ``nc.Dataset(path, "r").dimensions[...]`` and ``.variables[...][:]``,
# which scipy.io.netcdf covers exactly.  Install the shim once, at import
# time, BEFORE any ``from clm_ml_jax…`` import triggers netCDF4 loading.
try:
    import netCDF4 as _nc  # noqa: F401
except ImportError:
    import scipy.io as _sio
    import types as _types

    class _NcDim:
        def __init__(self, length: int):
            self._length = int(length)

        def __len__(self):
            return self._length

    class _NcDataset:
        def __init__(self, path: str, mode: str = "r"):
            self._f = _sio.netcdf_file(str(path), mode)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._f.close()

        @property
        def dimensions(self):
            return {k: _NcDim(v) for k, v in self._f.dimensions.items()}

        @property
        def variables(self):
            return {k: v for k, v in self._f.variables.items()}

    _shim = _types.ModuleType("netCDF4")
    _shim.Dataset = _NcDataset
    sys.modules["netCDF4"] = _shim

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio

# ---------------------------------------------------------------------------
# CHATS7 site constants
# ---------------------------------------------------------------------------
CHATS7_FORCING_DIR = (
    REPO_ROOT / "clm-ml-jax" / "src" / "input_files" / "tower-forcing" / "CHATS7"
)
CHATS7_LAT_DEG = 38.49
CHATS7_LON_DEG = -121.84
CHATS7_HTOP_M = 10.0     # Canopy top height (Bonan et al. 2026)
# Fortran/JAX-standalone uses ELAI + ESAI from the CLM h1 initial-state file
# (``CHATS_A15.clm2.h1.2007-04-01-00000_22Jul25.nc``): ELAI grows 0 → ~2.0 across
# April-May 2007; ESAI = 0.7 constant. The default here matches PAI=2.7 at the
# start of May 2007, giving parity with Fortran fsun.out pai; use
# ``--lai-time-series`` to read the actual time-varying ELAI/ESAI schedule.
CHATS7_LAI_MAY = 2.0     # LAI at start of May 2007 (matches CHATS_A15 ELAI)
CHATS7_SAI = 0.7         # Stem area index (matches CHATS_A15 ESAI, constant)
CHATS7_PFT_CLM = 7       # broadleaf_deciduous_temperate_tree
CHATS7_CO2_PPMV = 383.0  # matches Fortran ``TowerMetCO2()`` constant

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _spencer_cos_zen(lat_deg: float, lon_deg: float, doy_frac: float) -> float:
    """Cosine of solar zenith angle (Spencer 1971 declination + hour angle).

    ``doy_frac`` is the 0-based fractional day of year in UTC
    (0.0 = Jan 1 00:00 UTC). The adapter itself inverts CLM's Kepler
    ``shr_orb_cosz`` on the returned ``cos_zen``, so any smooth
    approximation would work; Spencer is the same one the adapter
    uses in its legacy test helper.
    """
    lat_r = math.radians(lat_deg)
    B = 2.0 * math.pi * doy_frac / 365.0  # coeff-ok: Julian year length (Spencer 1971)
    decl = (
        0.006918                           # coeff-ok: Spencer (1971) declination polynomial
        - 0.399912 * math.cos(B)           # coeff-ok: Spencer (1971) declination polynomial
        + 0.070257 * math.sin(B)           # coeff-ok: Spencer (1971) declination polynomial
        - 0.006758 * math.cos(2 * B)       # coeff-ok: Spencer (1971) declination polynomial
        + 0.000907 * math.sin(2 * B)       # coeff-ok: Spencer (1971) declination polynomial
        - 0.002697 * math.cos(3 * B)       # coeff-ok: Spencer (1971) declination polynomial
        + 0.00148 * math.sin(3 * B)        # coeff-ok: Spencer (1971) declination polynomial
    )
    frac = doy_frac % 1.0
    utc_hour = frac * 24.0                 # coeff-ok: exact hours-per-day (24 h/day)
    lon_norm = lon_deg - 360.0 if lon_deg > 180.0 else lon_deg
    ha_deg = (utc_hour - 12.0) * 15.0 + lon_norm  # coeff-ok: exact 360°/24h = 15°/h
    ha_r = math.radians(ha_deg)
    return max(
        math.sin(lat_r) * math.sin(decl)
        + math.cos(lat_r) * math.cos(decl) * math.cos(ha_r),
        0.0,
    )


def _forcing_to_atm2sfc(row: dict, doy_frac_utc: float):
    """Build a 1-column AtmToSurface from a CHATS7 forcing row.

    ``row`` has keys TBOT, WIND, RH, FSDS, FLDS, PSRF, PRECTmms. Missing
    q_bot is derived from RH using ``legoesm.thermo.saturation_mixing_ratio``.
    """
    from legoesm.core.coupling_fields import AtmToSurface

    T = float(row["TBOT"])
    p = float(row["PSRF"])
    # RH-based specific humidity: q = RH * q_sat(T, p)
    q_sat = float(saturation_mixing_ratio(jnp.array([T]), jnp.array([p]))[0])
    q = q_sat * float(row["RH"]) / 100.0
    # Virtual-T corrected moist-air density (matches AtmToSurface.rho_lowest)
    T_v = T * (1.0 + 0.608 * q)  # coeff-ok: virtual-T conversion factor (M_a/M_v - 1) ≈ 0.608
    rho = p / (constants.R_d * T_v)
    cos_zen = _spencer_cos_zen(CHATS7_LAT_DEG, CHATS7_LON_DEG, doy_frac_utc)
    return AtmToSurface(
        sw_down=jnp.full((1,), float(row["FSDS"])),
        lw_down=jnp.full((1,), float(row["FLDS"])),
        precip_total=jnp.full((1,), float(row["PRECTmms"])),  # kg/m²/s = mm/s
        precip_snow=jnp.zeros((1,)),
        T_lowest=jnp.full((1,), T),
        q_lowest=jnp.full((1,), q),
        u_lowest=jnp.full((1,), float(row["WIND"])),
        v_lowest=jnp.zeros((1,)),
        p_lowest=jnp.full((1,), p),
        p_surface=jnp.full((1,), p),
        rho_lowest=jnp.full((1,), rho),
        cos_zenith=jnp.full((1,), cos_zen),
        co2_ppmv=jnp.full((1,), CHATS7_CO2_PPMV),
        has_radiation=jnp.ones((1,)),
        has_precipitation=jnp.ones((1,)),
    )


def _build_land_params():
    """Minimal ``LandSurfaceParams``-like object for CHATS7 (LAI + SAI + htop)."""
    return SimpleNamespace(
        LAI=jnp.full((1,), CHATS7_LAI_MAY),
        SAI=jnp.full((1,), CHATS7_SAI),
        htop=jnp.full((1,), CHATS7_HTOP_M),
    )


def _build_land_config(fast: bool = False):
    from legoesm.land.canopy.config import CLMMLCanopyConfig
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.soil_grid import SoilGridConfig

    if fast:
        # Adapter defaults: Euler @ dt=1800s, no ML sub-cycling.  ~30x faster
        # than Fortran-matched RK4+60s, at some parity cost.
        canopy = CLMMLCanopyConfig(pft_clm=CHATS7_PFT_CLM)
    else:
        canopy = CLMMLCanopyConfig(
            pft_clm=CHATS7_PFT_CLM,
            met_type=3,           # 3-point centred, matches nl.CHATS7.1day:15
            runge_kutta_type=41,  # RK4, standalone default (MLclm_varctl.py:78)
            num_ml_steps=30,      # dtime_ml = 1800/30 = 60 s, standalone default
        )
    land = MultiLayerLandConfig(
        surface_scheme=canopy,
        soil_grid=SoilGridConfig(),
        z_ref=23.0,           # CHATS7 tower observation height
    )
    return land


# ---------------------------------------------------------------------------
# Output writer — mirrors clm-ml-jax/src/offline_driver/CLMml_driver.py::output
# ---------------------------------------------------------------------------

def _fmt10(x: float) -> str:
    return f"{float(x):10.3f}"


def _write_flux_row(nout: IO, curr_calday: float, mlcan, dt_seconds: float,
                    met_type: int) -> None:
    """flux.out row (18 columns).

    ``curr_calday`` here is the START of the CLM timestep (matches the
    NetCDF forcing time stamp). Fortran writes the CENTER of the interval
    for ``met_type == 3``, so we add ``+0.5 * dt`` (matching the Fortran
    ``time_stamp = curr_calday - 0.5 * dtstep/86400`` when ``curr_calday``
    is END-of-step; here we start from START-of-step so the sign flips).
    """
    from clm_src_main.clm_varpar import inir, ivis
    from multilayer_canopy.MLWaterVaporMod import LatVap

    if met_type == 3:
        time_stamp = curr_calday + 0.5 * dt_seconds / 86400.0
    else:
        time_stamp = curr_calday + dt_seconds / 86400.0  # END-of-step

    p = 1
    swup = float(mlcan.albcan_canopy[p, ivis]) * (
        float(mlcan.swskyb_forcing[p, ivis]) + float(mlcan.swskyd_forcing[p, ivis])
    ) + float(mlcan.albcan_canopy[p, inir]) * (
        float(mlcan.swskyb_forcing[p, inir]) + float(mlcan.swskyd_forcing[p, inir])
    )
    lat_vap = LatVap(float(mlcan.tref_forcing[p]))
    lhflx_tr = float(mlcan.trveg_canopy[p]) * lat_vap
    lhflx_ev = float(mlcan.evveg_canopy[p]) * lat_vap
    ic_top = int(mlcan.ntop_canopy[p])
    tair = float(mlcan.tair_profile[p, ic_top])
    nout.write(
        f"{time_stamp:12.7f}"
        + _fmt10(mlcan.rnet_canopy[p])
        + _fmt10(mlcan.stflx_air_canopy[p])
        + _fmt10(mlcan.shflx_canopy[p])
        + _fmt10(mlcan.lhflx_canopy[p])
        + _fmt10(mlcan.gppveg_canopy[p])
        + _fmt10(mlcan.ustar_canopy[p])
        + _fmt10(swup)
        + _fmt10(mlcan.lwup_canopy[p])
        + _fmt10(tair)
        + _fmt10(mlcan.gsoi_soil[p])
        + _fmt10(mlcan.rnsoi_soil[p])
        + _fmt10(mlcan.shsoi_soil[p])
        + _fmt10(mlcan.lhsoi_soil[p])
        + _fmt10(lhflx_tr)
        + _fmt10(lhflx_ev)
        + _fmt10(mlcan.beta_canopy[p])
        + _fmt10(mlcan.stflx_veg_canopy[p])
        + "\n"
    )


def _write_fsun_row(nout: IO, mlcan) -> None:
    """fsun.out row (32 columns)."""
    from clm_src_main.clm_varpar import ivis
    p = 1
    fields = [
        float(mlcan.solar_zen_forcing[p]) * 180.0 / math.pi,
        float(mlcan.swskyb_forcing[p, ivis]) + float(mlcan.swskyd_forcing[p, ivis]),
        float(mlcan.lai_canopy[p]) + float(mlcan.sai_canopy[p]),
        float(mlcan.laisun_canopy[p]),
        float(mlcan.laisha_canopy[p]),
        float(mlcan.swveg_canopy[p, ivis]),
        float(mlcan.swvegsun_canopy[p, ivis]),
        float(mlcan.swvegsha_canopy[p, ivis]),
        float(mlcan.gppveg_canopy[p]),
        float(mlcan.gppvegsun_canopy[p]),
        float(mlcan.gppvegsha_canopy[p]),
        float(mlcan.lhveg_canopy[p]),
        float(mlcan.lhvegsun_canopy[p]),
        float(mlcan.lhvegsha_canopy[p]),
        float(mlcan.shveg_canopy[p]),
        float(mlcan.shvegsun_canopy[p]),
        float(mlcan.shvegsha_canopy[p]),
        float(mlcan.vcmax25veg_canopy[p]),
        float(mlcan.vcmax25sun_canopy[p]),
        float(mlcan.vcmax25sha_canopy[p]),
        float(mlcan.gsveg_canopy[p]),
        float(mlcan.gsvegsun_canopy[p]),
        float(mlcan.gsvegsha_canopy[p]),
        float(mlcan.windveg_canopy[p]),
        float(mlcan.windvegsun_canopy[p]),
        float(mlcan.windvegsha_canopy[p]),
        float(mlcan.tlveg_canopy[p]),
        float(mlcan.tlvegsun_canopy[p]),
        float(mlcan.tlvegsha_canopy[p]),
        float(mlcan.taveg_canopy[p]),
        float(mlcan.tavegsun_canopy[p]),
        float(mlcan.tavegsha_canopy[p]),
    ]
    nout.write("".join(_fmt10(f) for f in fields) + "\n")


def _write_aux_row(nout: IO, mlcan) -> None:
    """aux.out row (6 columns)."""
    p = 1
    top = int(mlcan.ntop_canopy[p])
    mid_ic = max(
        1,
        int(mlcan.nbot_canopy[p])
        + (int(mlcan.ntop_canopy[p]) - int(mlcan.nbot_canopy[p]) + 1) // 2
        - 1,
    )
    nout.write(
        f"{float(mlcan.btran_soil[p]):10.4f}"
        + _fmt10(mlcan.lsc_profile[p, top])
        + _fmt10(mlcan.psis_soil[p])
        + _fmt10(mlcan.lwp_mean_profile[p, top])
        + _fmt10(mlcan.lwp_mean_profile[p, mid_ic])
        + _fmt10(mlcan.fracminlwp_canopy[p])
        + "\n"
    )


def _write_profile_rows(nout: IO, curr_calday: float, mlcan,
                        dt_seconds: float, met_type: int) -> None:
    """profile.out — vertical profiles (28 cols per canopy layer per timestep).

    Mirrors the standalone Fortran writer at
    ``clm-ml-jax/src/offline_driver/CLMml_driver.py::output`` (nout3).
    """
    from clm_src_main.clm_varpar import inir, ivis  # noqa: F401
    from multilayer_canopy.MLclm_varcon import mmh2o, mmdry
    from multilayer_canopy.MLclm_varpar import isha, isun

    if met_type == 3:
        time_stamp = curr_calday + 0.5 * dt_seconds / 86400.0
    else:
        time_stamp = curr_calday + dt_seconds / 86400.0

    p = 1
    missing_value = -999.0
    zero_value = 0.0

    def _qair(ic_: int) -> float:
        e = float(mlcan.eair_profile[p, ic_])
        pr = float(mlcan.pref_forcing[p])
        return 1000.0 * (mmh2o / mmdry) * e / (pr - (1.0 - mmh2o / mmdry) * e)

    def _ra(ic_: int) -> float:
        return float(mlcan.rhomol_forcing[p]) / float(mlcan.gac_profile[p, ic_])

    def _lad(ic_: int) -> float:
        return float(mlcan.dpai_profile[p, ic_]) / float(mlcan.dz_profile[p, ic_])

    def _line(ic_: int, is_above: bool) -> str:
        tair_ = float(mlcan.tair_profile[p, ic_])
        qair_ = _qair(ic_)
        ra_ = _ra(ic_)
        mv, zv = missing_value, zero_value
        prefix = (
            f"{time_stamp:12.7f}"
            + _fmt10(mlcan.zs_profile[p, ic_])
        )
        wind_ta_q_ra = (
            _fmt10(float(mlcan.wind_profile[p, ic_]))
            + _fmt10(tair_)
            + _fmt10(qair_)
            + _fmt10(ra_)
            + "\n"
        )
        if is_above:
            return prefix + _fmt10(zv) * 4 + _fmt10(mv) * 18 + wind_ta_q_ra
        dpai_ = float(mlcan.dpai_profile[p, ic_])
        frac_ = float(mlcan.fracsun_profile[p, ic_])
        if dpai_ <= 0.0:
            return prefix + _fmt10(frac_) + _fmt10(zv) * 3 + _fmt10(mv) * 18 + wind_ta_q_ra
        lad_ = _lad(ic_)
        return (
            prefix
            + _fmt10(frac_)
            + _fmt10(lad_)
            + _fmt10(lad_ * frac_)
            + _fmt10(lad_ * (1.0 - frac_))
            + _fmt10(mlcan.rnleaf_leaf[p, ic_, isun])
            + _fmt10(mlcan.rnleaf_leaf[p, ic_, isha])
            + _fmt10(mlcan.shleaf_leaf[p, ic_, isun])
            + _fmt10(mlcan.shleaf_leaf[p, ic_, isha])
            + _fmt10(mlcan.lhleaf_leaf[p, ic_, isun])
            + _fmt10(mlcan.lhleaf_leaf[p, ic_, isha])
            + _fmt10(mlcan.anet_leaf[p, ic_, isun])
            + _fmt10(mlcan.anet_leaf[p, ic_, isha])
            + _fmt10(mlcan.apar_leaf[p, ic_, isun])
            + _fmt10(mlcan.apar_leaf[p, ic_, isha])
            + _fmt10(mlcan.gs_leaf[p, ic_, isun])
            + _fmt10(mlcan.gs_leaf[p, ic_, isha])
            + _fmt10(mlcan.lwp_hist_leaf[p, ic_, isun])
            + _fmt10(mlcan.lwp_hist_leaf[p, ic_, isha])
            + _fmt10(mlcan.tleaf_hist_leaf[p, ic_, isun])
            + _fmt10(mlcan.tleaf_hist_leaf[p, ic_, isha])
            + _fmt10(mlcan.vcmax25_leaf[p, ic_, isun])
            + _fmt10(mlcan.vcmax25_leaf[p, ic_, isha])
            + wind_ta_q_ra
        )

    ncan = int(mlcan.ncan_canopy[p])
    ntop = int(mlcan.ntop_canopy[p])
    for ic in range(ncan, ntop, -1):
        nout.write(_line(ic, is_above=True))
    for ic in range(ntop, 0, -1):
        nout.write(_line(ic, is_above=False))


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def _load_forcing_month(nc_path: pathlib.Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ds = xr.open_dataset(nc_path, engine="h5netcdf")
    time = ds["time"].values
    fields = {
        v: ds[v].values.squeeze()
        for v in ("TBOT", "WIND", "RH", "FSDS", "FLDS", "PSRF", "PRECTmms")
    }
    ds.close()
    return time, fields


def _time_to_doy_frac(ts: np.datetime64) -> float:
    """0-based fractional day-of-year in UTC (0.0 = Jan 1 00:00)."""
    ts_dt = ts.astype("datetime64[s]").astype(object)
    year_start = np.datetime64(f"{ts_dt.year}-01-01T00:00:00", "s")
    seconds = (ts.astype("datetime64[s]") - year_start).astype("int64").item()
    return seconds / 86400.0  # coeff-ok: seconds-per-day (24*3600)


def run_chats7(
    forcing_dir: pathlib.Path,
    output_dir: pathlib.Path,
    n_analysis_steps: int,
    n_spinup_steps: int = 0,
    dt: float = 1800.0,
    month_tag: str = "CHATS7_2007-05",
    fast: bool = False,
) -> None:
    """Run the E1 CHATS7 offline driver and write .out files."""
    from legoesm.land.canopy.clm_ml_interface import compute_clm_ml_canopy_fluxes
    from legoesm.land.canopy.state import CanopyState

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[E1] output → {output_dir}", flush=True)

    apr_time, apr = (
        _load_forcing_month(forcing_dir / "2007-04.nc")
        if n_spinup_steps > 0
        else (np.array([]), {})
    )
    may_time, may = _load_forcing_month(forcing_dir / "2007-05.nc")
    print(
        f"[E1] loaded forcing: spinup={len(apr_time) if n_spinup_steps else 0} apr, "
        f"analysis={len(may_time)} may (using {n_analysis_steps})",
        flush=True,
    )

    land_config = _build_land_config(fast=fast)
    canopy_config = land_config.surface_scheme
    land_params = _build_land_params()

    grid_n_layers = 10
    T_soil0 = jnp.full((1, grid_n_layers), 290.0)
    theta_soil0 = jnp.full((1, grid_n_layers), 0.25)
    from legoesm.land.soil_hydraulics import psi_from_theta
    psi_soil0 = psi_from_theta(theta_soil0, land_config.hydraulics)

    canopy_state: CanopyState | None = None

    def _step(row: dict, doy_frac_utc: float, doy_calday: float):
        nonlocal canopy_state
        forcing = _forcing_to_atm2sfc(row, doy_frac_utc)
        surface_out, canopy_state = compute_clm_ml_canopy_fluxes(
            T_soil_top=T_soil0[:, 0],
            forcing=forcing,
            canopy_config=canopy_config,
            land_config=land_config,
            land_params=land_params,
            w_frac_rz=jnp.full((1,), 0.6),
            wind_speed=jnp.full((1,), max(float(row["WIND"]), 1.0)),
            canopy_state=canopy_state,
            dt=dt,
            T_soil=T_soil0,
            psi_soil=psi_soil0,
            theta_soil=theta_soil0,
            lat=jnp.array([CHATS7_LAT_DEG]),
            doy=doy_calday,
        )
        return surface_out

    t0 = time_mod.perf_counter()
    if n_spinup_steps > 0:
        for i in range(min(n_spinup_steps, len(apr_time))):
            ts = apr_time[i]
            doy_frac = _time_to_doy_frac(ts)
            row = {k: v[i] for k, v in apr.items()}
            _step(row, doy_frac, doy_frac)
            if (i + 1) % 48 == 0:
                elapsed = time_mod.perf_counter() - t0
                print(f"[E1] spinup step {i + 1}/{n_spinup_steps} ({elapsed:.1f}s)",
                      flush=True)

    fpath = output_dir / f"{month_tag}_flux.out"
    xpath = output_dir / f"{month_tag}_fsun.out"
    apath = output_dir / f"{month_tag}_aux.out"
    ppath = output_dir / f"{month_tag}_profile.out"
    with (
        open(fpath, "w") as nout1,
        open(xpath, "w") as nout4,
        open(apath, "w") as nout2,
        open(ppath, "w") as nout3,
    ):
        for i in range(min(n_analysis_steps, len(may_time))):
            ts = may_time[i]
            doy_frac = _time_to_doy_frac(ts)
            row = {k: v[i] for k, v in may.items()}
            _step(row, doy_frac, doy_frac)
            curr_calday = doy_frac + 1.0
            mlcan = canopy_state.mlcanopy
            _write_flux_row(nout1, curr_calday, mlcan, dt,
                            met_type=canopy_config.met_type)
            _write_fsun_row(nout4, mlcan)
            _write_aux_row(nout2, mlcan)
            _write_profile_rows(nout3, curr_calday, mlcan, dt,
                                met_type=canopy_config.met_type)
            nout1.flush(); nout4.flush(); nout2.flush(); nout3.flush()
            if (i + 1) % 24 == 0:
                elapsed = time_mod.perf_counter() - t0
                print(
                    f"[E1] analysis step {i + 1}/{n_analysis_steps} "
                    f"({elapsed:.1f}s wall; SH={float(mlcan.shflx_canopy[1]):.1f} "
                    f"LH={float(mlcan.lhflx_canopy[1]):.1f} W/m²)",
                    flush=True,
                )
    print(f"[E1] wrote {fpath.name}, {xpath.name}, {apath.name}", flush=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run CLM-ML-JAX adapter offline at CHATS7 for E1 parity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--forcing-dir", type=pathlib.Path, default=CHATS7_FORCING_DIR)
    p.add_argument("--output-dir", type=pathlib.Path,
                   default=REPO_ROOT / "results" / "chats7_adapter_2007-05")
    p.add_argument("--days", type=int, default=31)
    p.add_argument("--smoke", action="store_true",
                   help="only run first day (48 steps); no spinup")
    p.add_argument("--steps", type=int, default=None,
                   help="override analysis step count (for very short tests)")
    p.add_argument("--spinup-days", type=int, default=30)
    p.add_argument("--fast", action="store_true",
                   help="Adapter defaults (Euler @ dt, no ML sub-cycling) — ~30x faster")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.steps is not None:
        n_analysis = args.steps
        n_spinup = 0
    elif args.smoke:
        n_analysis = 48
        n_spinup = 0
    else:
        n_analysis = args.days * 48
        n_spinup = args.spinup_days * 48
    run_chats7(
        forcing_dir=args.forcing_dir,
        output_dir=args.output_dir,
        n_analysis_steps=n_analysis,
        n_spinup_steps=n_spinup,
        fast=args.fast,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
