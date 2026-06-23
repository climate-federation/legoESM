#!/usr/bin/env python
"""Standalone offline two-leaf-canopy run at eddy-covariance flux-tower sites.

Drives legoESM's two-leaf canopy from a DifferBESS v2 per-site driver NetCDF
(read via ``legoesm.land.boundary_data.ec_site``) and compares the modelled
GPP / LE / H against the tower observations bundled in the driver.

Modes
-----
``diagnostic`` (this stage): prescribe the soil skin temperature + root-zone
moisture stress from the driver and evaluate the canopy energy/carbon/water
closure independently at each timestep (``jax.vmap`` over time, chunked to bound
memory) — the apples-to-apples analogue of DifferBESS's vmapped forward pass.
(``prognostic`` — running the full multilayer soil — is added in a later stage.)

Validation is against the observed fluxes only, restricted to the adapter's
``valid`` timesteps (every driving input genuinely observed) AND finite obs.

Usage
-----
    JAX_ENABLE_X64=1 python scripts/run/run_ec_site.py \
        --driver-nc <DifferBESS>/data/sitelevel/nc/US-Ton_driver_v2.nc \
                    <DifferBESS>/data/sitelevel/nc/US-MMS_driver_v2.nc \
        --mode diagnostic --out diagnostics/ec_site
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.boundary_data.ec_site import read_ec_site_driver, ECSiteDriver
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import (
    init_multilayer_land_state,
    step_multilayer_land_with_diagnostics,
)
from legoesm.land.richards import RichardsConfig
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig, psi_from_theta
from legoesm.land.surface_scheme import TwoLeafCanopyConfig
from legoesm.land.surface_scheme.two_leaf_canopy import compute_two_leaf_canopy_fluxes

# Soil presets for the offline land config.  ``default`` = legoESM loam defaults
# (6.4 m free-draining column).  The site presets set texture-appropriate
# hydraulics + a 2 m column; ``siltloam``/``sandyloam`` match US-MMS soil
# (Clapp & Hornberger / Cosby texture classes).  A zero-flux bottom (no deep
# drainage) keeps the column from bleeding to wilting at a site with a shallow
# restrictive layer / water table.
_SOIL_HYDRAULICS = {
    "siltloam": dict(theta_sat=0.485, theta_r=0.05, K_sat=7.2e-7, b_ch=5.30,
                     psi_sat=-0.786),
    "sandyloam": dict(theta_sat=0.41, theta_r=0.041, K_sat=7.2e-6, b_ch=4.74,
                      psi_sat=-0.218),
}
_SOIL_FC_WP = {"siltloam": (0.33, 0.13), "sandyloam": (0.21, 0.10)}


def _build_land_config(canopy_config: TwoLeafCanopyConfig, soil: str,
                       bottom_bc: str, depth_m: float) -> MultiLayerLandConfig:
    """Assemble the multilayer land config for the offline EC-site run."""
    kw = dict(surface_scheme=canopy_config)
    if depth_m > 0:
        kw["soil_grid"] = SoilGridConfig(total_depth=depth_m)
    if bottom_bc != "free_drainage":
        kw["richards"] = RichardsConfig(bottom_bc=bottom_bc)
    if soil != "default":
        kw["hydraulics"] = SoilHydraulicsConfig(**_SOIL_HYDRAULICS[soil])
        fc, wp = _SOIL_FC_WP[soil]
        kw["theta_fc"], kw["theta_wp"] = fc, wp
    return MultiLayerLandConfig(**kw)


# gC per umol CO2 (carbon molar mass conversion); matches the canopy's own
# ``GPP = (An_Sun + An_Sh) * 12.0e-6`` so model<->obs GPP units round-trip.
_GC_PER_UMOL_CO2 = 12.0e-6
_DEFAULT_CHUNK = 8760            # timesteps per vmap chunk (bounds memory)

# gC per umol CO2 (carbon molar mass conversion); matches the canopy's own
# ``GPP = (An_Sun + An_Sh) * 12.0e-6`` so model<->obs GPP units round-trip.
_GC_PER_UMOL_CO2 = 12.0e-6
_DEFAULT_CHUNK = 8760            # timesteps per vmap chunk (bounds memory)
# Production wind-speed floor [m/s].  The coupler / land step ALWAYS drive the
# canopy with sqrt(u^2 + v^2 + U_min^2) (multilayer_land.py; coupler U_min default
# = 1.0), so the canopy never sees raw calm wind.  We mirror that here for a
# faithful offline run rather than passing the raw tower wind.
_U_MIN = 1.0


def _diagnostic_fluxes(d: ECSiteDriver, canopy_config: TwoLeafCanopyConfig,
                       land_config: MultiLayerLandConfig, chunk: int):
    """vmap the canopy over time with the soil state PRESCRIBED from the driver.

    Returns (gpp_gC, le_wm2, h_wm2, t_surface) each shape (n_time,).
    """
    def _step(T_soil_t, forcing_t, params_t, w_frac_t):
        # Mirror production: floor wind as sqrt(u^2 + v^2 + U_min^2) (the canopy
        # never sees raw calm wind in the coupled model).
        wind_t = jnp.sqrt(forcing_t.u_lowest ** 2 + forcing_t.v_lowest ** 2
                          + _U_MIN ** 2)
        out = compute_two_leaf_canopy_fluxes(
            T_soil_top=T_soil_t, forcing=forcing_t,
            canopy_config=canopy_config, land_config=land_config,
            canopy_params=params_t, w_frac_rz=w_frac_t, wind_speed=wind_t,
            wind_dir_x=jnp.ones_like(wind_t), wind_dir_y=jnp.zeros_like(wind_t),
            # Diagnostic: hold the soil skin temperature fixed at the prescribed
            # value (no soil-thermal evolution) so this isolates the canopy.
            soil_thermal_fn=lambda G, dt_: T_soil_t,
            dt=d.dt_s,
            LAI_override=params_t.LAI, TgC_override=params_t.TgC,
        )
        return out.gpp, out.lhflx, out.shflx, out.T_surface

    vstep = jax.jit(jax.vmap(_step))
    n = int(d.forcing.T_lowest.shape[0])
    gpp, le, h, ts = [], [], [], []
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        f = jax.tree_util.tree_map(lambda a: a[sl], d.forcing)
        p = jax.tree_util.tree_map(lambda a: a[sl], d.canopy_params)
        g, l, hh, t = vstep(d.T_soil_top[sl], f, p, d.w_frac_rz[sl])
        gpp.append(np.asarray(g).ravel()); le.append(np.asarray(l).ravel())
        h.append(np.asarray(hh).ravel()); ts.append(np.asarray(t).ravel())
    return (np.concatenate(gpp), np.concatenate(le),
            np.concatenate(h), np.concatenate(ts))


def _any_nonfinite(tree) -> jnp.ndarray:
    """Scalar bool: True if any leaf of ``tree`` holds a non-finite value."""
    flags = [jnp.any(~jnp.isfinite(leaf))
             for leaf in jax.tree_util.tree_leaves(tree)]
    return jnp.any(jnp.stack(flags)) if flags else jnp.asarray(False)


def _prognostic_fluxes(d: ECSiteDriver, canopy_config: TwoLeafCanopyConfig,
                       land_config: MultiLayerLandConfig, U_min: float,
                       nudge_tau_days: float = 0.0):
    """Integrate the FULL multilayer land forward in time (``lax.scan``).

    Unlike diagnostic mode (per-step ``vmap`` with the soil PRESCRIBED), the soil
    moisture (Richards) and soil temperature evolve prognostically and the
    moisture stress ``w_frac_rz`` is computed from the carried ``theta`` — the
    driver's ``T_soil_top`` / ``w_frac_rz`` are used only for the initial state.
    The canopy is invoked inside the land step, which also applies the ``U_min``
    wind floor (so the raw forcing wind is passed through unchanged).

    Soil-moisture nudging (``nudge_tau_days > 0``): after each step the prognostic
    ``theta`` is relaxed toward the OBSERVED volumetric soil moisture with an
    e-folding timescale ``tau`` (``theta += dt/tau * (theta_obs - theta)``).  The
    energy balance, canopy and carbon stay fully prognostic (so H is still solved,
    unlike diagnostic mode), but the water stress tracks observed soil moisture
    rather than the free-running soil hydrology — a standard land-flux evaluation
    technique that isolates the surface flux physics from soil-hydrology drift.

    NaN handling — because the soil state is CARRIED, a single non-finite update
    would otherwise poison every subsequent step.  The carry is therefore made
    atomic: if any state leaf goes non-finite on a step, the *entire* state is
    reverted to the previous step's state (``jnp.where(reverted, old, new)``), so
    a bad step cannot propagate.  That step's emitted fluxes are left non-finite
    and surface via ``model_nan`` in the skill report; ``reverted`` counts how
    many steps were rolled back.

    Returns (gpp_gC, le_wm2, h_wm2, t_surface, reverted) each shape (n_time,).
    """
    # --- initial soil state from the driver's first finite soil obs ---
    # Initialise from the OBSERVED volumetric soil moisture directly (theta_soil
    # = SWC/100), NOT by inverting w_frac_rz: the reader derives w_frac_rz with
    # its own theta_wp/theta_fc, which differ from this config's, so a round-trip
    # would bias the IC.  theta is clipped to this config's hydraulic range.
    Ts = np.asarray(d.T_soil_top).ravel()
    T_init = float(Ts[np.isfinite(Ts)][0])                       # [K]
    th = np.asarray(d.theta_soil).ravel()
    th0 = th[np.isfinite(th)]
    theta_r = float(land_config.hydraulics.theta_r)
    theta_sat = float(land_config.hydraulics.theta_sat)
    theta_init = float(np.clip(th0[0], theta_r + 1e-3, theta_sat - 1e-3))
    state0 = init_multilayer_land_state(
        1, land_config, T_init=T_init, theta_init=theta_init,
        TgC_init=T_init - constants.T_freeze)

    # Static nudging weight (dt and tau are compile-time constants -> no retrace).
    nudge_alpha = (min(d.dt_s / (nudge_tau_days * 86400.0), 1.0)
                   if nudge_tau_days > 0 else 0.0)
    # Nudge only SUB-SURFACE layers (z > 5 cm): the root zone tracks observed
    # moisture (transpiration stress) while the thin surface layer dries freely
    # between rains, so the (top-layer-based) bare-soil evaporation is realistic.
    _zc = np.cumsum(np.asarray(make_soil_grid(land_config.soil_grid).dz))
    nudge_layer_mask = jnp.asarray(_zc >= 0.05)

    def _scan_step(state, xs):
        forcing_t, params_t, doy_t, theta_obs_t = xs
        new_state, _resp, _carbon, out = step_multilayer_land_with_diagnostics(
            state, forcing_t, land_config, U_min, d.dt_s,
            lat=None, carbon_state=None, doy=doy_t, land_params=params_t)
        reverted = _any_nonfinite(new_state)
        safe_state = jax.tree_util.tree_map(
            lambda n, o: jnp.where(reverted, o, n), new_state, state)
        if nudge_alpha > 0.0:
            # Relax theta toward observed SWC (all layers; only where obs finite),
            # then make psi consistent.  Keeps water stress realistic while the
            # energy/canopy/carbon stay prognostic.
            th = safe_state.theta_soil
            obs = theta_obs_t[:, None]
            th_nudged = jnp.where(
                jnp.isfinite(obs) & nudge_layer_mask[None, :],
                jnp.clip(th + nudge_alpha * (obs - th), theta_r, theta_sat),
                th)
            safe_state = safe_state._replace(
                theta_soil=th_nudged,
                psi_soil=psi_from_theta(th_nudged, land_config.hydraulics))
        # On revert the post-flux soil update was rejected, so the step's
        # diagnostics are NOT trustworthy even when `out` itself is finite
        # (the failure is downstream of the surface flux calc).  Mask the
        # emitted fluxes to NaN so they count as model_nan and are excluded
        # from skill metrics rather than silently scored.
        nan = jnp.array(jnp.nan, dtype=jnp.float64)
        masked = lambda v: jnp.where(reverted, nan, v)
        emit = (masked(out.gpp), masked(out.lhflx), masked(out.shflx),
                masked(out.T_surface), reverted.astype(jnp.float64))
        return safe_state, emit

    xs = (d.forcing, d.canopy_params, d.doy, jnp.asarray(d.theta_soil))
    run = jax.jit(lambda s0, x: jax.lax.scan(_scan_step, s0, x))
    _final, (gpp, le, h, ts, reverted) = run(state0, xs)
    rav = lambda a: np.asarray(a).ravel()
    return rav(gpp), rav(le), rav(h), rav(ts), rav(reverted)


def _skill(model: np.ndarray, obs: np.ndarray, valid: np.ndarray) -> dict:
    """RMSE / bias / Pearson r / count over valid + jointly-finite samples."""
    m, o = np.asarray(model).ravel(), np.asarray(obs).ravel()
    vmask = np.asarray(valid).ravel()
    # Steps the model itself failed (non-finite) on a valid forcing step — reported
    # separately so the masked-out failures are not hidden in an optimistic skill.
    model_nan = int((vmask & ~np.isfinite(m)).sum())
    mask = vmask & np.isfinite(m) & np.isfinite(o)
    k = int(mask.sum())
    if k < 2:
        return dict(n=k, model_nan=model_nan, rmse=float("nan"),
                    bias=float("nan"), r=float("nan"))
    mm, oo = m[mask], o[mask]
    rmse = float(np.sqrt(np.mean((mm - oo) ** 2)))
    bias = float(np.mean(mm - oo))
    r = float(np.corrcoef(mm, oo)[0, 1]) if np.std(mm) > 0 and np.std(oo) > 0 else float("nan")
    return dict(n=k, model_nan=model_nan, rmse=rmse, bias=bias, r=r)


def _write_output(out_dir: str, d: ECSiteDriver, model: dict,
                  mode: str = "diagnostic",
                  reverted: np.ndarray | None = None,
                  score_valid: np.ndarray | None = None) -> str:
    import xarray as xr
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{d.site_id}_ec_{mode}.nc")
    t = np.arange(model["gpp_umol"].shape[0])
    data = {
        "gpp_mod": ("time", model["gpp_umol"]), "gpp_obs": ("time", d.obs["gpp_umol"]),
        "le_mod": ("time", model["le_wm2"]), "le_obs": ("time", d.obs["le_wm2"]),
        "h_mod": ("time", model["h_wm2"]), "h_obs": ("time", d.obs["h_wm2"]),
        "valid": ("time", d.valid.astype("i1")),
    }
    if reverted is not None:
        # prognostic: per-step flag, 1 where the soil update was NaN-reverted.
        data["reverted"] = ("time", np.asarray(reverted).astype("i1"))
    if d.met_filled is not None:
        # forcing provenance: 1 where an atmospheric forcing field was gap-filled.
        data["met_filled"] = ("time", np.asarray(d.met_filled).astype("i1"))
    if d.soil_filled is not None:
        data["soil_filled"] = ("time", np.asarray(d.soil_filled).astype("i1"))
    if score_valid is not None:
        # the EXACT mask used for the reported skill metrics, so downstream users
        # can reproduce the metric sample set from the file alone.
        data["score_valid"] = ("time", np.asarray(score_valid).astype("i1"))
    ds = xr.Dataset(
        data,
        coords={"time": t},
        attrs=dict(site=d.site_id, pft=d.pft, climate=d.climate, igbp=int(d.igbp),
                   lat_deg=float(np.rad2deg(d.lat_rad)), lon_deg=float(d.lon_deg),
                   dt_s=float(d.dt_s), mode=mode,
                   gpp_units="umolCO2/m2/s", le_units="W/m2", h_units="W/m2"),
    )
    ds.to_netcdf(path)
    return path


def _slice_driver(d: ECSiteDriver, start: int, k: int) -> ECSiteDriver:
    """``[start : start+k]`` timestep view of a driver (quick runs / smoke test)."""
    sl = slice(start, start + k)
    return d._replace(
        forcing=jax.tree_util.tree_map(lambda a: a[sl], d.forcing),
        canopy_params=jax.tree_util.tree_map(lambda a: a[sl], d.canopy_params),
        T_soil_top=d.T_soil_top[sl], w_frac_rz=d.w_frac_rz[sl],
        theta_soil=d.theta_soil[sl],
        wind_speed=d.wind_speed[sl], valid=d.valid[sl], doy=d.doy[sl],
        obs={kk: vv[sl] for kk, vv in d.obs.items()},
        met_filled=(None if d.met_filled is None else d.met_filled[sl]),
        soil_filled=(None if d.soil_filled is None else d.soil_filled[sl]),
    )


def run_site(driver_nc: str, mode: str, out_dir: str, chunk: int,
             max_steps: int | None = None, start_step: int = 0,
             soil: str = "default", bottom_bc: str = "free_drainage",
             soil_depth_m: float = 0.0, nudge_tau_days: float = 0.0) -> dict:
    if mode not in ("diagnostic", "prognostic"):
        raise ValueError(f"mode {mode!r} not supported (diagnostic|prognostic)")
    d = read_ec_site_driver(driver_nc)
    if max_steps is not None or start_step:
        n = int(d.forcing.T_lowest.shape[0])
        k = (n - start_step) if max_steps is None else max_steps
        d = _slice_driver(d, start_step, k)
    canopy_config = TwoLeafCanopyConfig(max_iters=30)
    land_config = _build_land_config(canopy_config, soil, bottom_bc, soil_depth_m)

    reverted = None
    if mode == "diagnostic":
        gpp_gC, le, h, _ = _diagnostic_fluxes(d, canopy_config, land_config, chunk)
    else:
        gpp_gC, le, h, _ts, reverted = _prognostic_fluxes(
            d, canopy_config, land_config, _U_MIN, nudge_tau_days=nudge_tau_days)
    model = {
        "gpp_umol": gpp_gC / _GC_PER_UMOL_CO2,   # gC/m2/s -> umolCO2/m2/s (obs units)
        "le_wm2": le, "h_wm2": h,
    }
    # Score on genuinely-OBSERVED forcing: when running a gap-free driver, exclude
    # steps whose forcing was gap-filled so skill is not credited to synthesised
    # forcing.  Atmospheric fill (met_filled) is per-step forcing in BOTH modes.
    # Soil fill (soil_filled) is per-step forcing ONLY in diagnostic mode (TS/SWC
    # prescribed); in prognostic mode the soil is integrated, so soil fill matters
    # only for the initial state (reported separately below).  For a standard
    # driver_v2 (flags are None) this is a no-op.  Provenance is persisted to out.
    score_valid = d.valid
    if d.met_filled is not None:
        score_valid = score_valid & (np.asarray(d.met_filled).ravel() == 0)
    if mode == "diagnostic" and d.soil_filled is not None:
        score_valid = score_valid & (np.asarray(d.soil_filled).ravel() == 0)
    metrics = {
        "GPP": _skill(model["gpp_umol"], d.obs["gpp_umol"], score_valid),
        "LE": _skill(model["le_wm2"], d.obs["le_wm2"], score_valid),
        "H": _skill(model["h_wm2"], d.obs["h_wm2"], score_valid),
    }
    path = _write_output(out_dir, d, model, mode=mode, reverted=reverted,
                         score_valid=score_valid)
    forcing_note = ("" if d.met_filled is None else
                    f"; scored on observed forcing only "
                    f"({100 * (d.met_filled == 0).mean():.0f}% of steps)")
    print(f"\n=== {d.site_id}  (PFT={d.pft}, {d.climate}; mode={mode}; "
          f"valid {100 * d.valid.mean():.0f}% of {d.valid.size} steps{forcing_note}) ===")
    if reverted is not None:
        nrev = int(np.nansum(reverted))
        print(f"  prognostic soil: {nrev} step(s) NaN-reverted "
              f"({100 * nrev / reverted.size:.3f}% of run)")
        # The soil IC is taken from the first finite soil obs (step 0 on a
        # gap-free driver); report whether that value was itself gap-filled.
        if d.soil_filled is not None:
            ic_src = "FILLED" if int(np.asarray(d.soil_filled).ravel()[0]) else "observed"
            print(f"  prognostic soil: initial state from {ic_src} TS/SWC at step 0")
    for flux, mtr in metrics.items():
        print(f"  {flux:3s}  n={mtr['n']:>7d}  model_nan={mtr['model_nan']:>5d}  "
              f"RMSE={mtr['rmse']:8.3f}  bias={mtr['bias']:+8.3f}  r={mtr['r']:.3f}")
    print(f"  -> {path}")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--driver-nc", nargs="+", required=True,
                    help="one or more DifferBESS <SITE>_driver_v2.nc paths")
    ap.add_argument("--mode", default="diagnostic",
                    choices=["diagnostic", "prognostic"])
    ap.add_argument("--out", default="diagnostics/ec_site")
    ap.add_argument("--chunk", type=int, default=_DEFAULT_CHUNK,
                    help="timesteps per vmap chunk (memory control)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="limit to N timesteps from --start-step (quick runs)")
    ap.add_argument("--start-step", type=int, default=0,
                    help="first timestep index to run (skip early gappy record)")
    ap.add_argument("--soil", default="default",
                    choices=["default", "siltloam", "sandyloam"],
                    help="soil texture preset for the prognostic hydraulics")
    ap.add_argument("--bottom-bc", default="free_drainage",
                    choices=["free_drainage", "zero_flux"],
                    help="Richards bottom boundary condition (prognostic)")
    ap.add_argument("--soil-depth-m", type=float, default=0.0,
                    help="total soil column depth [m] (0 = legoESM default ~6.4 m)")
    ap.add_argument("--nudge-tau-days", type=float, default=0.0,
                    help="prognostic mode: relax soil moisture toward observed SWC "
                         "with this e-folding timescale [days] (0 = free-running)")
    args = ap.parse_args()
    for nc in args.driver_nc:
        run_site(nc, args.mode, args.out, args.chunk,
                 max_steps=args.max_steps, start_step=args.start_step,
                 soil=args.soil, bottom_bc=args.bottom_bc,
                 soil_depth_m=args.soil_depth_m, nudge_tau_days=args.nudge_tau_days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
