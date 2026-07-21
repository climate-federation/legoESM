"""Standalone canopy-land diagnostic — sweep PFT × canopy height × soil dryness.

Drives ``step_canopy_land_with_diagnostics`` through a 3-day idealised diurnal
cycle for six contrasting cases and saves everything needed to reproduce or
re-analyse the run to netCDF::

    outputs/canopy_diagnostic/canopy_diagnostic.nc
    outputs/canopy_diagnostic/canopy_diagnostic.png
    outputs/canopy_diagnostic/canopy_diagnostic_night.png

The netCDF contains, per ``(time, case)``:

* atmospheric forcing (sw_down, lw_down, T_air, q_air, wind, p_surface, cos_z)
* land parameters (LAI, hc, fC4, Vcmax25, albedo, etc., replicated over time)
* top-layer state (T_soil[0], theta_soil[0], snow_depth, T_soil profile)
* coupler response (T_surface, lhflx, shflx, lw_up, co2_flux, albedo)
* full surface-budget diagnostics (Rn_int, Rn_ext, SW_net, LW_net, LE, H, G,
  residuals, component LE/H, Tf_Sun/Sh, Ts_solve, fSun, n_iters, GPP)

After the ``Ts_bc``/residual-G refactor (see canopy_land.py), this script
verifies that the surface energy balance closes and that nighttime behaviour
is now physical: diagnosed ``G`` flips sign at night, ``LE`` is allowed to
be negative (dew), and Newton convergence does not diverge when SW → 0.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.canopy import CanopyConfig, CanopyLandParams
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import (
    init_multilayer_land_state,
    step_multilayer_land_with_diagnostics,
)
from legoesm.land.soil_hydraulics import psi_from_theta
from legoesm.land.surface_scheme import TwoLeafCanopyConfig

# Phase 3 shims: ``CanopyLandConfig``, ``init_canopy_land_state``, and
# ``step_canopy_land_with_diagnostics`` were removed when the canopy
# became a surface scheme of ``MultiLayerLandConfig``.  These shims
# preserve the rest of this script unchanged.
init_canopy_land_state = init_multilayer_land_state


def CanopyLandConfig(*, multilayer=None, canopy=None) -> MultiLayerLandConfig:
    """Phase 3 script-local shim — ``CanopyLandConfig(multilayer, canopy)``
    becomes ``MultiLayerLandConfig(surface_scheme=canopy)``.
    """
    base = multilayer if multilayer is not None else MultiLayerLandConfig()
    cc = canopy if canopy is not None else CanopyConfig()
    return base._replace(surface_scheme=cc)


def step_canopy_land_with_diagnostics(state, forcing, cfg, U_min, dt, **kwargs):
    """Return ``(new_state, response, carbon_state, surface_out)`` where
    ``surface_out`` is the canopy-populated ``SurfaceFluxOutput`` whose
    fields directly cover all 22 ``CanopyDiagnostics`` keys consumed below.
    """
    return step_multilayer_land_with_diagnostics(
        state, forcing, cfg, U_min, dt, **kwargs)


# --------------------------------------------------------------------------
# Case definitions — PFT × canopy height × soil dryness
# --------------------------------------------------------------------------

CASES = [
    # name                  LAI   hc    fC4  Vc3   Vc4   TgC   theta0
    ("tropical_DBF",         5.0, 20.0, 0.0, 66.0,  0.0, 27.0, 0.38),
    ("temperate_DBF",        4.0, 15.0, 0.0, 57.0,  0.0, 20.0, 0.30),
    ("temperate_GRA_C3",     2.0,  0.5, 0.0, 78.0,  0.0, 18.0, 0.25),
    ("boreal_ENF",           3.0, 12.0, 0.0, 54.0,  0.0, 12.0, 0.35),
    ("C4_savanna",           1.5,  3.0, 1.0,  0.0, 40.0, 28.0, 0.22),
    # dry_shrub: kept just above theta_wp=0.15 so fStress_soil > 0; lower
    # values send Rsoil → ∞ and crash the soil EB.
    ("dry_shrub",            1.0,  1.5, 0.0, 62.0,  0.0, 22.0, 0.17),
]

CASE_NAMES = [c[0] for c in CASES]
NCOL = len(CASES)


def build_params(ncol: int) -> CanopyLandParams:
    arr = lambda j: jnp.asarray([c[j] for c in CASES])
    LAI = arr(1)
    hc  = arr(2)
    fC4 = arr(3)
    Vc3 = arr(4)
    Vc4 = arr(5)
    TgC = arr(6)

    return CanopyLandParams(
        LAI=LAI, hc=hc, fC4=fC4,
        FNonVeg=jnp.zeros(ncol),
        CI=jnp.array([0.70, 0.72, 0.85, 0.75, 0.80, 0.75]),
        kn=jnp.full(ncol, 0.3),
        Vcmax25_C3_leaf=Vc3,
        Vcmax25_C4_leaf=Vc4,
        m_C3=jnp.full(ncol, 9.0),
        m_C4=jnp.full(ncol, 4.0),
        b0_C3=jnp.full(ncol, 0.01),
        b0_C4=jnp.full(ncol, 0.04),
        alf=jnp.full(ncol, 0.3),
        TgC=TgC,
        ALB_VIS=jnp.array([0.05, 0.06, 0.10, 0.07, 0.15, 0.18]),
        ALB_NIR=jnp.array([0.20, 0.22, 0.25, 0.22, 0.30, 0.35]),
        emissivity=jnp.full(ncol, 0.97),
        rz0m=jnp.array([0.055, 0.055, 0.12, 0.06, 0.12, 0.12]),
        rd=jnp.array([0.67, 0.67, 0.68, 0.67, 0.68, 0.68]),
    )


# --------------------------------------------------------------------------
# Idealised forcing
# --------------------------------------------------------------------------

def diurnal_forcing(t_hr: float, ncol: int) -> tuple[AtmToSurface, dict]:
    """Idealised diurnal forcing at hour t_hr within the day.

    Returns both the coupler struct AND a dict of the forcing scalars for
    archiving.  All columns share the same forcing (same time of day) but have
    per-case q_lowest to reflect the ambient climate of each PFT.
    """
    cosz = max(0.0, float(np.sin(np.pi * (t_hr - 6.0) / 12.0)))
    sw_top = 1000.0 * cosz
    # Realistic diurnal T_air phase: minimum at sunrise (~6am), max at
    # ~3pm.  An asymmetric profile (warmer half of day longer than cooler
    # half) is approximated by a sin centred so the minimum coincides
    # with sunrise.  The earlier ``sin((t_hr-9)·π/12)`` formula put the
    # T_air minimum at 3am — physically the air does not start warming
    # until after sunrise, so prescribing it to warm 3 hours before
    # sunrise creates an artificial widening of the (T_surf − T_air)
    # gap and a spurious "kink" in nocturnal H toward dawn.
    T_air_common = 293.0 + 6.0 * float(np.sin((t_hr - 12.0) * np.pi / 12.0))

    # Per-case T_air and q_air so that each PFT sits in its own climate
    T_air = np.array([
        T_air_common + 3.0,   # tropical_DBF (warmer)
        T_air_common + 0.0,
        T_air_common - 1.0,
        T_air_common - 8.0,   # boreal (cold)
        T_air_common + 4.0,   # C4 savanna
        T_air_common + 1.0,
    ], dtype=np.float64)
    q_air = np.array([
        0.018, 0.012, 0.010, 0.005, 0.012, 0.006
    ], dtype=np.float64)

    forcing = AtmToSurface(
        T_lowest   = jnp.asarray(T_air),
        q_lowest   = jnp.asarray(q_air),
        u_lowest   = jnp.full(ncol, 3.0),
        v_lowest   = jnp.full(ncol, 0.0),
        p_lowest   = jnp.full(ncol, 98000.0),
        p_surface  = jnp.full(ncol, 101325.0),
        rho_lowest = jnp.full(ncol, 1.18),
        sw_down    = jnp.full(ncol, sw_top),
        lw_down    = jnp.full(ncol, 360.0),
        cos_zenith = jnp.full(ncol, cosz),
        precip_total=jnp.zeros(ncol),
        precip_snow =jnp.zeros(ncol),
        co2_ppmv    = jnp.full(ncol, 420.0),
        has_radiation=True,
        has_precipitation=False,
    )

    record = dict(
        sw_down=np.full(ncol, sw_top),
        lw_down=np.full(ncol, 360.0),
        cos_zenith=np.full(ncol, cosz),
        T_air=T_air,
        q_air=q_air,
        wind_speed=np.full(ncol, 3.0),
        p_surface=np.full(ncol, 101325.0),
    )
    return forcing, record


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

def run(n_days: int = 3, dt_s: float = 1800.0):
    ncol = NCOL
    cfg = CanopyLandConfig(
        multilayer=MultiLayerLandConfig(),
        canopy=CanopyConfig(max_iters=50, tol=1e-4),
    )
    state = init_canopy_land_state(ncol, cfg, T_init=290.0)

    # Case-specific initial soil moisture.  Richards uses psi as the
    # prognostic variable, so both theta and psi must be set consistently —
    # overriding only theta silently de-syncs them and the first Richards
    # step snaps theta back to whatever psi represents.
    theta_init_arr = jnp.asarray([c[7] for c in CASES])
    theta_init_profile = jnp.broadcast_to(
        theta_init_arr[:, None], state.theta_soil.shape)
    psi_init_profile = psi_from_theta(theta_init_profile, cfg.hydraulics)
    state = state._replace(
        theta_soil=theta_init_profile,
        psi_soil=psi_init_profile,
    )

    params = build_params(ncol)

    step_fn = jax.jit(
        lambda s, f: step_canopy_land_with_diagnostics(
            s, f, cfg, 1.0, dt_s,
            lat=jnp.zeros(ncol), doy=180.0, land_params=params,
        )
    )

    # Start at local noon to give Newton a well-conditioned first step.
    # Starting at midnight leaves the Newton closure with no SW forcing to
    # anchor the sunlit-leaf state variable — the solver silently diverges
    # to unphysical values (Tf_Sun ~ 80-200 K) and the nighttime state is
    # seeded from garbage.  Starting at noon then marching forward lets the
    # solver take an easy first step and remain in the basin of attraction.
    start_hour = 12.0

    n_steps = int(n_days * 24 * 3600 / dt_s)
    n_layers = state.T_soil.shape[1]

    # Storage (numpy, accumulated host-side)
    def zcol():  return np.zeros((n_steps, ncol))
    def zprof(): return np.zeros((n_steps, ncol, n_layers))

    forcing_keys = ["sw_down", "lw_down", "cos_zenith", "T_air", "q_air",
                    "wind_speed", "p_surface"]
    forcing_arr  = {k: zcol() for k in forcing_keys}

    response_keys = ["T_surface", "albedo", "lhflx", "shflx", "lw_up",
                     "co2_flux", "q_surface", "z0"]
    response_arr = {k: zcol() for k in response_keys}

    # Phase 3: diag now reads from a SurfaceFluxOutput.  Map output → field names.
    # Old name (key in diag_arr / netCDF)  ->  attribute name on SurfaceFluxOutput
    diag_keymap = {
        "Rn_ext":       "Rn_ext",
        "Rn_int":       "Rn_int",
        "SW_net":       "sw_net",
        "LW_net":       "lw_net",
        "LE_tot":       "lhflx",
        "H_tot":        "shflx",
        "G":            "G_soil",
        "LE_canopy":    "LE_canopy",
        "LE_soil":      "LE_soil",
        "H_canopy":     "H_canopy",
        "H_soil":       "H_soil",
        "Rn_canopy":    "Rn_canopy",
        "Rn_soil":      "Rn_soil",
        "residual_int": "residual_int",
        "residual_ext": "residual_ext",
        "GPP":          "gpp",
        "fSun":         "fSun",
        "n_iters":      "n_iters",
        "Tf_Sun":       "Tf_Sun",
        "Tf_Sh":        "Tf_Sh",
        "Ts_solve":     "Ts_solve",
        "T_surface":    "T_surface",
    }
    diag_keys = list(diag_keymap.keys())
    diag_arr = {k: zcol() for k in diag_keys}

    state_keys  = ["snow_depth", "snow_age",
                   "runoff_surface", "runoff_subsurface"]
    state_arr   = {k: zcol() for k in state_keys}
    T_soil_arr  = zprof()
    theta_arr   = zprof()
    psi_arr     = zprof()

    time_s = np.arange(n_steps, dtype=np.float64) * dt_s

    for k in range(n_steps):
        t_hr = (start_hour + k * dt_s / 3600.0) % 24.0
        forcing, rec = diurnal_forcing(t_hr, ncol)
        state, response, _, diag = step_fn(state, forcing)

        for key in forcing_keys:
            forcing_arr[key][k] = rec[key]
        response_arr["T_surface"][k] = np.asarray(response.T_surface)
        response_arr["albedo"   ][k] = np.asarray(response.albedo)
        response_arr["lhflx"    ][k] = np.asarray(response.lhflx)
        response_arr["shflx"    ][k] = np.asarray(response.shflx)
        response_arr["lw_up"    ][k] = np.asarray(response.lw_up)
        response_arr["co2_flux" ][k] = np.asarray(response.co2_flux)
        response_arr["q_surface"][k] = np.asarray(response.q_surface)
        response_arr["z0"       ][k] = np.asarray(response.z0)

        for key in diag_keys:
            attr_name = diag_keymap[key]
            diag_arr[key][k] = np.asarray(getattr(diag, attr_name))

        state_arr["snow_depth"       ][k] = np.asarray(state.snow_depth)
        state_arr["snow_age"         ][k] = np.asarray(state.snow_age)
        state_arr["runoff_surface"   ][k] = np.asarray(state.runoff_surface)
        state_arr["runoff_subsurface"][k] = np.asarray(state.runoff_subsurface)
        T_soil_arr[k] = np.asarray(state.T_soil)
        theta_arr [k] = np.asarray(state.theta_soil)
        psi_arr   [k] = np.asarray(state.psi_soil)

    return dict(
        time_s=time_s, n_layers=n_layers,
        forcing=forcing_arr, response=response_arr, diag=diag_arr,
        state_scalar=state_arr,
        T_soil=T_soil_arr, theta_soil=theta_arr, psi_soil=psi_arr,
        params=params, cfg=cfg,
    )


# --------------------------------------------------------------------------
# netCDF writer
# --------------------------------------------------------------------------

def to_dataset(result: dict) -> xr.Dataset:
    time_hr = result["time_s"] / 3600.0
    case    = np.asarray(CASE_NAMES, dtype="U32")
    layer   = np.arange(result["n_layers"], dtype=np.int32)

    dims_tc = ("time", "case")
    dims_tcl= ("time", "case", "layer")

    data_vars: dict = {}

    # Forcing
    for k, arr in result["forcing"].items():
        data_vars[f"forcing_{k}"] = (dims_tc, arr)

    # Coupler response
    for k, arr in result["response"].items():
        data_vars[f"response_{k}"] = (dims_tc, arr)

    # Diagnostics (CanopyDiagnostics fields) — prefix: diag_
    for k, arr in result["diag"].items():
        data_vars[f"diag_{k}"] = (dims_tc, arr)

    # Scalar land state
    for k, arr in result["state_scalar"].items():
        data_vars[f"state_{k}"] = (dims_tc, arr)

    data_vars["state_T_soil"]     = (dims_tcl, result["T_soil"])
    data_vars["state_theta_soil"] = (dims_tcl, result["theta_soil"])
    data_vars["state_psi_soil"]   = (dims_tcl, result["psi_soil"])

    # Per-case static parameters (broadcast over time=0 for quick access)
    params = result["params"]
    static_fields = [
        "LAI", "hc", "fC4", "CI", "kn",
        "Vcmax25_C3_leaf", "Vcmax25_C4_leaf",
        "m_C3", "m_C4", "b0_C3", "b0_C4",
        "alf", "TgC", "ALB_VIS", "ALB_NIR", "emissivity",
        "rz0m", "rd", "FNonVeg",
    ]
    for name in static_fields:
        val = getattr(params, name, None)
        if val is None:
            continue
        arr = np.asarray(val)
        if arr.shape == (NCOL,):
            data_vars[f"param_{name}"] = (("case",), arr)

    theta_init = np.array([c[7] for c in CASES], dtype=np.float64)
    data_vars["param_theta_init"] = (("case",), theta_init)

    coords = dict(
        time=("time", time_hr, {"long_name": "elapsed time", "units": "hours"}),
        case=("case", case),
        layer=("layer", layer),
    )
    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    # Variable attributes — units & short descriptions
    units = {
        "sw_down": "W m-2", "lw_down": "W m-2", "T_air": "K", "q_air": "kg kg-1",
        "wind_speed": "m s-1", "p_surface": "Pa", "cos_zenith": "-",
        "T_surface": "K", "albedo": "-", "lhflx": "W m-2", "shflx": "W m-2",
        "lw_up": "W m-2", "co2_flux": "gC m-2 s-1", "q_surface": "kg kg-1",
        "z0": "m",
        "Rn_ext": "W m-2", "Rn_int": "W m-2", "SW_net": "W m-2", "LW_net": "W m-2",
        "LE_tot": "W m-2", "H_tot": "W m-2", "G": "W m-2",
        "LE_canopy": "W m-2", "LE_soil": "W m-2",
        "H_canopy": "W m-2", "H_soil": "W m-2",
        "Rn_canopy": "W m-2", "Rn_soil": "W m-2",
        "residual_int": "W m-2", "residual_ext": "W m-2",
        "GPP": "gC m-2 s-1", "fSun": "-", "n_iters": "count",
        "Tf_Sun": "K", "Tf_Sh": "K", "Ts_solve": "K",
        "snow_depth": "m", "snow_age": "s",
        "runoff_surface": "m s-1", "runoff_subsurface": "m s-1",
        "T_soil": "K", "theta_soil": "m3 m-3", "psi_soil": "m",
    }
    for var in list(ds.data_vars):
        for tag in ("forcing_", "response_", "diag_", "state_", "param_"):
            if var.startswith(tag):
                base = var[len(tag):]
                if base in units:
                    ds[var].attrs["units"] = units[base]
                break

    ds.attrs.update({
        "title": "legoESM canopy land diagnostic — 6-case diurnal sweep",
        "description": (
            "3-day idealised diurnal cycle for six contrasting PFT × canopy "
            "height × soil dryness cases.  Full surface energy budget "
            "diagnostics included for nighttime-closure analysis.  G is "
            "diagnosed as the SEB residual (not G_alpha · Rn)."
        ),
        "source": "scripts/run_canopy_diagnostic.py",
        "n_days": 3,
        "dt_s": 1800.0,
        "LE_module": result["cfg"].canopy.LE_module,
        "coupling_scheme": "FULLY_COUPLED",
    })
    return ds


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

COLORS = ["tab:green", "tab:olive", "gold", "tab:cyan", "tab:orange", "tab:brown"]


def plot_main(ds: xr.Dataset, out_path: Path) -> None:
    t = ds["time"].values
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharex=True)

    # Row 0: SW forcing + T_surface
    ax = axes[0, 0]
    ax.plot(t, ds["forcing_sw_down"].isel(case=0).values, color="gold", lw=2)
    ax.set_ylabel("SW_down  [W m$^{-2}$]")
    ax.set_title("Atmospheric SW forcing (shared diurnal cycle)")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for j, name in enumerate(CASE_NAMES):
        ax.plot(t, ds["response_T_surface"].isel(case=j).values - constants.T_freeze,
                color=COLORS[j], label=name)
    ax.set_ylabel(r"T$_{surface}$  [$\degree$C]")
    ax.set_title("Surface temperature")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)

    # Row 1: LE and H
    ax = axes[1, 0]
    for j, name in enumerate(CASE_NAMES):
        ax.plot(t, ds["diag_LE_tot"].isel(case=j).values, color=COLORS[j])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("LE_tot  [W m$^{-2}$]")
    ax.set_title("Total latent heat flux")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for j in range(NCOL):
        ax.plot(t, ds["diag_H_tot"].isel(case=j).values, color=COLORS[j])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("H_tot  [W m$^{-2}$]")
    ax.set_title("Total sensible heat flux")
    ax.grid(alpha=0.3)

    # Row 2: G and Rn
    ax = axes[2, 0]
    for j in range(NCOL):
        ax.plot(t, ds["diag_G"].isel(case=j).values, color=COLORS[j])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("G  [W m$^{-2}$]")
    ax.set_title(r"Ground heat flux  (diagnosed SEB residual)")
    ax.grid(alpha=0.3)

    ax = axes[2, 1]
    for j in range(NCOL):
        ax.plot(t, ds["diag_Rn_int"].isel(case=j).values, color=COLORS[j])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Rn (internal)  [W m$^{-2}$]")
    ax.set_title("Net radiation (solver-internal total)")
    ax.grid(alpha=0.3)

    # Row 3: residuals + GPP
    ax = axes[3, 0]
    for j in range(NCOL):
        ax.plot(t, ds["diag_residual_int"].isel(case=j).values, color=COLORS[j])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Rn - LE - H - G  [W m$^{-2}$]")
    ax.set_title("Internal closure residual (should be ~0)")
    ax.set_xlabel("Hour of simulation")
    ax.grid(alpha=0.3)

    ax = axes[3, 1]
    for j in range(NCOL):
        ax.plot(t, ds["diag_GPP"].isel(case=j).values * 1e3, color=COLORS[j])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("GPP  [mgC m$^{-2}$ s$^{-1}$]")
    ax.set_title("Gross primary productivity")
    ax.set_xlabel("Hour of simulation")
    ax.grid(alpha=0.3)

    fig.suptitle("Canopy land diagnostic — 3-day diurnal cycle, 6 cases",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"saved: {out_path}")
    plt.close(fig)


def plot_night_budget(ds: xr.Dataset, out_path: Path) -> None:
    """Focus on day-2 night (hours 24-36 selected to the SW=0 subset)."""
    t   = ds["time"].values
    sw  = ds["forcing_sw_down"].isel(case=0).values
    day2 = (t >= 24) & (t < 48)
    night = (sw < 1.0) & day2

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)

    ax = axes[0, 0]
    for j in range(NCOL):
        ax.plot(t[day2], ds["diag_LE_tot"].isel(case=j).values[day2],
                color=COLORS[j], label=CASE_NAMES[j])
    ax.axvspan(t[night].min() if night.any() else 0,
               t[night].max() if night.any() else 0,
               color="k", alpha=0.08, label="night")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("LE_tot  [W m$^{-2}$]")
    ax.set_title("Day-2 LE — sign now free (dew allowed at night)")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for j in range(NCOL):
        ax.plot(t[day2], ds["diag_H_tot"].isel(case=j).values[day2],
                color=COLORS[j])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("H_tot  [W m$^{-2}$]")
    ax.set_title("Day-2 H (bulk transfer from Ts_bc and Tc)")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for j in range(NCOL):
        ax.plot(t[day2], ds["diag_G"].isel(case=j).values[day2],
                color=COLORS[j])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("G  [W m$^{-2}$]")
    ax.set_title(r"Day-2 G — diagnosed SEB residual (positive into soil)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for j in range(NCOL):
        ax.plot(t[day2], ds["diag_Rn_int"].isel(case=j).values[day2],
                color=COLORS[j])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Rn (internal)  [W m$^{-2}$]")
    ax.set_title("Day-2 Rn_int")
    ax.grid(alpha=0.3)

    # G/Rn ratio (data-driven now — no longer a construction constant)
    ax = axes[2, 0]
    Rn  = ds["diag_Rn_soil"].values
    G   = ds["diag_G"].values
    ratio = np.where(np.abs(Rn) > 5.0, G / Rn, np.nan)
    for j in range(NCOL):
        ax.plot(t[day2], ratio[day2, j], color=COLORS[j])
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_ylabel(r"G / $R_{n,soil}$  [-]")
    ax.set_title("Day-2 G / Rn_soil — data, not parameter")
    ax.set_ylim(-2, 2)
    ax.set_xlabel("Hour of simulation")
    ax.grid(alpha=0.3)

    # Night residual distribution
    ax = axes[2, 1]
    for j in range(NCOL):
        r = ds["diag_residual_int"].isel(case=j).values[day2]
        ax.plot(t[day2], r, color=COLORS[j])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Rn - LE - H - G  [W m$^{-2}$]")
    ax.set_title("Day-2 internal closure residual")
    ax.set_xlabel("Hour of simulation")
    ax.grid(alpha=0.3)

    fig.suptitle("Nighttime energy budget — day 2", fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"saved: {out_path}")
    plt.close(fig)


# --------------------------------------------------------------------------
# Summary printout
# --------------------------------------------------------------------------

def print_summary(ds: xr.Dataset) -> None:
    t = ds["time"].values
    sw = ds["forcing_sw_down"].isel(case=0).values
    day2 = (t >= 24) & (t < 48)
    night_mask = day2 & (sw < 1.0)
    day_mask   = day2 & (sw > 50.0)

    def stats(var, mask):
        arr = ds[var].values[mask]
        return arr.mean(axis=0)

    print("\n=== Day-2 means ===")
    print(f"{'case':20s}  {'period':6s}  {'Rn_int':>8s}  {'LE':>8s}  {'H':>8s}  {'G':>8s}  {'resid':>9s}  {'n_it':>5s}")
    for j, name in enumerate(CASE_NAMES):
        for label, mask in (("day", day_mask), ("night", night_mask)):
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue
            Rn = ds["diag_Rn_int"].values[mask, j].mean()
            LE = ds["diag_LE_tot"].values[mask, j].mean()
            H  = ds["diag_H_tot" ].values[mask, j].mean()
            G  = ds["diag_G"     ].values[mask, j].mean()
            res= ds["diag_residual_int"].values[mask, j].mean()
            ni = ds["diag_n_iters"].values[mask, j].mean()
            print(f"{name:20s}  {label:6s}  {Rn:8.1f}  {LE:8.1f}  {H:8.1f}  {G:8.1f}  {res:9.2e}  {ni:5.1f}")


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    jax.config.update("jax_enable_x64", True)

    result = run(n_days=3, dt_s=1800.0)
    ds = to_dataset(result)

    out_dir = Path(__file__).resolve().parents[1] / "outputs" / "canopy_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    nc_path = out_dir / "canopy_diagnostic.nc"
    ds.to_netcdf(nc_path)
    print(f"saved: {nc_path}")

    plot_main(ds, out_dir / "canopy_diagnostic.png")
    plot_night_budget(ds, out_dir / "canopy_diagnostic_night.png")

    print_summary(ds)
