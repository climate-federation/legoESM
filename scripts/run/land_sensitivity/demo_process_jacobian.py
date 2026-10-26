#!/usr/bin/env python
"""Demo: instantaneous process-to-process sensitivity at one land column.

Spins the multilayer-land + two-leaf-canopy column up under the synthetic LMIP
forcing, then runs a rain-free dry-down.  At every local noon of the dry-down
ONE land step is differentiated with reverse-mode AD:

* the Jacobian of the surface fluxes (canopy latent heat = transpiration demand,
  soil evaporation, GPP, sensible heat, actual total latent heat) with respect
  to the soil water of EVERY layer, reported per kg/m2 of water in that layer
  ("which layer the plant is drinking from", and how it moves as the soil dries);
* the elasticity of the actual latent heat flux to the column's soil water (all
  layers scaled together, ``d ln LE / d ln W``) and to air humidity
  (``d ln LE / d ln q``), i.e. the water-limited vs energy/demand-limited
  regime, plus ``dLE/dT_air``.

Every AD number is a single reverse pass per output.  At two noons (``--fd-days``)
the AD values are checked against central finite differences; a failed check
aborts the run (the instrument decides what we believe).

Usage
-----
    JAX_ENABLE_X64=1 python scripts/run/land_sensitivity/demo_process_jacobian.py \
        --texture loam --out-dir diagnostics/land_sensitivity/process_jacobian
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import NamedTuple

_PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.land.multilayer_land import step_multilayer_land_with_diagnostics
from legoesm.thermo import saturation_vapor_pressure_aerk, vapor_pressure_from_specific_humidity

from legoesm import constants
from scripts.run.land_sensitivity import column_setup as cs

FLUX_NAMES = ("LE_canopy", "LE_soil", "GPP", "H", "LE_total")
FLUX_UNITS = ("W/m2", "W/m2", "gC/m2/s", "W/m2", "W/m2")
NOON_HOUR_UTC = 12.0        # local noon at lon = 0
# Finite-difference steps for the checks (x64 throughout).
FD_EPS_THETA = 1e-5         # [m3/m3]
FD_EPS_SCALE = 1e-4         # [-] on the water / humidity scale factors
FD_EPS_T_AIR = 1e-3         # [K]
FD_ATOL_FLUX = 1e-6         # [W/m2 per unit] absolute floor for exactly-zero gradients
FD_ATOL_GPP = 1e-12         # [gC/m2/s per unit]
FD_ATOL_ELASTICITY = 1e-8   # [-]
PA_PER_KPA = 1.0e3          # exact unit conversion
# Tolerance for re-deriving the saved all-layer water elasticity from the saved
# per-layer Jacobian (the same derivatives, combined two ways): round-off only.
CHAIN_RULE_RTOL = 1e-8


class NoonSensitivity(NamedTuple):
    day: np.ndarray                 # (m,) day of the dry-down, 1-based
    theta: np.ndarray               # (m, n_layers) soil water at noon (before the step)
    fluxes: np.ndarray              # (m, 5) FLUX_NAMES at noon
    jac_per_theta: np.ndarray       # (m, 5, n_layers) d flux / d theta_k
    jac_per_water: np.ndarray       # (m, 5, n_layers) d flux / d W_k [per kg/m2]
    jac_column_water: np.ndarray    # (m, 5) d flux per kg/m2 of water added to the WHOLE
                                    #   column in proportion to its present profile
    jac_humidity_scale: np.ndarray  # (m, 5) d flux / d s_q, air specific humidity scaled by
                                    #   s_q at fixed air temperature (every flux, not only LE)
    elasticity_water: np.ndarray    # (m,) d ln LE_total / d ln W (all layers scaled together)
    elasticity_humidity: np.ndarray  # (m,) d ln LE_total / d ln q_air
    dle_dt_air: np.ndarray          # (m,) d LE_total / d T_air [W/m2/K]
    column_water: np.ndarray        # (m,) soil water of the whole column [kg/m2]


def make_noon_sensitivity_fn(config, params, dt_s: float, dz):
    """Build the jitted noon sensitivity function.

    ``sensitivity(state, forcing)`` returns ``(fluxes, J_theta, dF/ds_w, dF/ds_q,
    dLE/ds_w, dLE/ds_q, dLE/dT)``: the five fluxes, their Jacobian with respect to
    every layer's water content, their derivatives with respect to a uniform
    scaling of the column water and of the air humidity, and the total-LE
    derivatives used for the elasticities.

    ``state`` is the column state at noon (before the step) and ``forcing`` the
    noon forcing.  Soil water is perturbed with its matric potential kept
    consistent (``psi_from_theta``), so the derivative is with respect to the
    layer's water content as a physical quantity.
    """
    dz = jnp.asarray(dz)

    def fluxes_of(state, forcing, theta, s_water, s_q, t_off):
        th = theta[None, :] * s_water
        st = state._replace(theta_soil=th, psi_soil=cs.psi_from_theta(th, config.hydraulics))
        f = forcing._replace(T_lowest=forcing.T_lowest + t_off,
                             q_lowest=forcing.q_lowest * s_q)
        _, resp, _, out = step_multilayer_land_with_diagnostics(
            st, f, config, cs.DEFAULT_U_MIN_M_S, dt_s, land_params=params)
        return jnp.stack([out.LE_canopy[0], out.LE_soil[0], out.gpp[0],
                          resp.shflx[0], resp.lhflx[0]])

    def le_total(state, forcing, theta, s_water, s_q, t_off):
        return fluxes_of(state, forcing, theta, s_water, s_q, t_off)[4]

    @jax.jit
    def sensitivity(state, forcing):
        theta = state.theta_soil[0]
        one, zero = jnp.asarray(1.0), jnp.asarray(0.0)
        fl = fluxes_of(state, forcing, theta, one, one, zero)
        j_theta = jax.jacrev(fluxes_of, argnums=2)(state, forcing, theta, one, one, zero)
        j_sw = jax.jacrev(fluxes_of, argnums=3)(state, forcing, theta, one, one, zero)
        j_sq = jax.jacrev(fluxes_of, argnums=4)(state, forcing, theta, one, one, zero)
        d_sw, d_sq, d_t = jax.grad(le_total, argnums=(3, 4, 5))(
            state, forcing, theta, one, one, zero)
        return fl, j_theta, j_sw, j_sq, d_sw, d_sq, d_t

    return sensitivity, fluxes_of, le_total


def noon_indices(time_axis: cs.TimeAxis) -> np.ndarray:
    idx = np.where(np.isclose(time_axis.hour, NOON_HOUR_UTC))[0]
    if idx.size == 0:
        raise ValueError("no step falls on local noon; choose a dt_s that divides 12 h")
    return idx


def compute_noon_sensitivities(config, params, *, lat_rad: float, dt_s: float,
                               state_drydown0, time_axis: cs.TimeAxis, states,
                               dz, climate: cs.ColumnClimate = cs.DEFAULT_CLIMATE,
                               ) -> tuple[NoonSensitivity, dict]:
    """Evaluate the noon Jacobians along a dry-down whose post-step states are
    ``states`` (stacked over ``time_axis``); ``state_drydown0`` is the state
    before the first dry-down step.  Returns the results and the raw callables
    (``sensitivity``, ``fluxes_of``, ``le_total``) for the FD checks."""
    sensitivity, fluxes_of, le_total = make_noon_sensitivity_fn(config, params, dt_s, dz)
    idx = noon_indices(time_axis)
    rows = []
    for i in idx:
        st = state_drydown0 if i == 0 else cs.state_at(states, int(i) - 1)
        f = cs.column_forcing(lat_rad, 0.0, float(time_axis.doy[i]), NOON_HOUR_UTC, 0.0,
                              climate=climate)
        fl, j_theta, j_sw, j_sq, d_sw, d_sq, d_t = sensitivity(st, f)
        rows.append((int(i), st, f, np.asarray(fl), np.asarray(j_theta),
                     float(d_sw), float(d_sq), float(d_t), np.asarray(j_sw), np.asarray(j_sq)))
    theta = np.stack([np.asarray(r[1].theta_soil[0]) for r in rows])
    fluxes = np.stack([r[3] for r in rows])
    j_theta = np.stack([r[4] for r in rows])
    j_sw = np.stack([r[8] for r in rows])                       # (m, 5) d flux / d s_water
    j_sq = np.stack([r[9] for r in rows])                       # (m, 5) d flux / d s_q
    le = fluxes[:, 4]
    column_water = np.asarray(cs.column_water_kg_m2(jnp.asarray(theta), dz))
    res = NoonSensitivity(
        day=np.asarray([time_axis.t_s[r[0]] / cs.SECONDS_PER_DAY for r in rows]) + 0.5,
        theta=theta, fluxes=fluxes, jac_per_theta=j_theta,
        jac_per_water=j_theta / (constants.rho_water * np.asarray(dz))[None, None, :],
        # scaling every layer by s adds s*W_col kg/m2 spread like the profile,
        # so d flux / d s divided by W_col is the response per kg/m2 of that water
        jac_column_water=j_sw / column_water[:, None],
        jac_humidity_scale=j_sq,
        elasticity_water=np.asarray([r[5] for r in rows]) / le,
        elasticity_humidity=np.asarray([r[6] for r in rows]) / le,
        dle_dt_air=np.asarray([r[7] for r in rows]),
        column_water=column_water,
    )
    if not np.all(np.isfinite(res.jac_per_theta)) or not np.all(np.isfinite(res.fluxes)):
        raise FloatingPointError("non-finite noon sensitivity — refusing to report")
    # The total-LE row of the all-flux Jacobian is the same derivative as the
    # elasticity's numerator (two AD calls on the same function must agree).
    np.testing.assert_allclose(j_sw[:, 4] / le, res.elasticity_water, rtol=1e-10, atol=1e-12,
                               err_msg="jacrev(fluxes)[LE_total] != grad(le_total)")
    np.testing.assert_allclose(j_sq[:, 4] / le, res.elasticity_humidity, rtol=1e-10,
                               atol=1e-12, err_msg="jacrev(fluxes, s_q)[LE_total] != grad")
    raw = {"rows": rows, "sensitivity": sensitivity, "fluxes_of": fluxes_of,
           "le_total": le_total, "noon_idx": idx}
    return res, raw


def finite_difference_checks(res: NoonSensitivity, raw: dict, *, check_rows: list[int],
                             layers=(1, 3), rtol: float) -> list[cs.GradientCheck]:
    """Central-difference checks of the AD Jacobian at the selected noon rows.

    The step is jitted once here; the eager canopy Newton loop is far too slow
    for the ~30 evaluations the checks need.
    """
    fluxes_of, le_total = jax.jit(raw["fluxes_of"]), jax.jit(raw["le_total"])
    one, zero = jnp.asarray(1.0), jnp.asarray(0.0)
    checks: list[cs.GradientCheck] = []
    for r in check_rows:
        _, st, f, *_ = raw["rows"][r]
        theta = st.theta_soil[0]
        day = res.day[r]
        for k in layers:
            for fi, atol in ((0, FD_ATOL_FLUX), (1, FD_ATOL_FLUX), (2, FD_ATOL_GPP)):
                fd, smooth = cs.finite_difference(
                    lambda th, fi=fi: fluxes_of(st, f, th, one, one, zero)[fi],
                    theta, k, FD_EPS_THETA, atol=atol)
                checks.append(cs.gradient_check(
                    f"day {day:.1f} d{FLUX_NAMES[fi]}/dtheta[{k}]",
                    res.jac_per_theta[r, fi, k], fd, rtol=rtol, atol=atol, fd_consistent=smooth))
        le = res.fluxes[r, 4]
        fd_w, smooth = cs.finite_difference(
            lambda s: le_total(st, f, theta, s[0], one, zero), jnp.asarray([1.0]), 0,
            FD_EPS_SCALE, atol=FD_ATOL_ELASTICITY * le)
        checks.append(cs.gradient_check(
            f"day {day:.1f} elasticity water", res.elasticity_water[r], fd_w / le,
            rtol=rtol, atol=FD_ATOL_ELASTICITY, fd_consistent=smooth))
        fd_q, smooth = cs.finite_difference(
            lambda s: le_total(st, f, theta, one, s[0], zero), jnp.asarray([1.0]), 0,
            FD_EPS_SCALE, atol=FD_ATOL_ELASTICITY * le)
        checks.append(cs.gradient_check(
            f"day {day:.1f} elasticity humidity", res.elasticity_humidity[r], fd_q / le,
            rtol=rtol, atol=FD_ATOL_ELASTICITY, fd_consistent=smooth))
        for fi in (0, 1):                       # transpiration and soil evaporation separately
            fd_qf, smooth = cs.finite_difference(
                lambda s, fi=fi: fluxes_of(st, f, theta, one, s[0], zero)[fi],
                jnp.asarray([1.0]), 0, FD_EPS_SCALE, atol=FD_ATOL_FLUX)
            checks.append(cs.gradient_check(
                f"day {day:.1f} d{FLUX_NAMES[fi]}/d s_q", res.jac_humidity_scale[r, fi], fd_qf,
                rtol=rtol, atol=FD_ATOL_FLUX, fd_consistent=smooth))
        fd_t, smooth = cs.finite_difference(
            lambda t: le_total(st, f, theta, one, one, t[0]), jnp.asarray([0.0]), 0,
            FD_EPS_T_AIR, atol=FD_ATOL_FLUX)
        checks.append(cs.gradient_check(
            f"day {day:.1f} dLE/dT_air", res.dle_dt_air[r], fd_t, rtol=rtol, atol=FD_ATOL_FLUX,
            fd_consistent=smooth))
    return checks


def run_experiment(args) -> tuple[NoonSensitivity, dict, list[cs.GradientCheck], dict]:
    lat_rad = float(np.deg2rad(args.lat_deg))
    config = cs.make_column_config(args.texture)
    params = cs.make_canopy_params(1, biome=args.biome, lai=args.lai)
    dz, z_node, z_iface = cs.soil_layers(config)
    climate = cs.ColumnClimate(args.mean_annual_t_k, args.seasonal_amp_k)
    t_init = float(climate.mean_annual_t_k)
    state0 = cs.initial_state(config, t_init_k=t_init)
    steps_per_day = int(round(cs.SECONDS_PER_DAY / args.dt_s))
    timings = {}

    # --- spin-up under steady rain (forward only) ---
    t0 = time.perf_counter()
    spin_axis = cs.make_time_axis(args.spinup_days * steps_per_day, args.dt_s,
                                  start_doy=args.start_doy - args.spinup_days)
    run_spin = jax.jit(lambda s: cs.rollout(
        s, config, params, lat_rad=lat_rad, lon_rad=0.0, time_axis=spin_axis,
        dt_s=args.dt_s, precip=args.precip_rate, climate=climate)[0])
    state_dry0 = run_spin(state0)
    jax.block_until_ready(state_dry0)
    timings["spinup_s"] = time.perf_counter() - t0

    # --- rain-free dry-down, keeping the state after every step ---
    t0 = time.perf_counter()
    dry_axis = cs.make_time_axis(args.drydown_days * steps_per_day, args.dt_s,
                                 start_doy=args.start_doy)
    run_dry = jax.jit(lambda s: cs.rollout(
        s, config, params, lat_rad=lat_rad, lon_rad=0.0, time_axis=dry_axis,
        dt_s=args.dt_s, precip=0.0, climate=climate, return_states=True))
    _final, hist, states = run_dry(state_dry0)
    jax.block_until_ready(states)
    timings["drydown_s"] = time.perf_counter() - t0
    if int(jnp.max(hist.n_iters)) >= config.surface_scheme.max_iters:
        raise RuntimeError("canopy Newton solve hit max_iters during the dry-down; "
                           "the IFT gradient is zeroed on unconverged steps")

    # --- noon sensitivities (one reverse pass per output per noon) ---
    t0 = time.perf_counter()
    res, raw = compute_noon_sensitivities(
        config, params, lat_rad=lat_rad, dt_s=args.dt_s, state_drydown0=state_dry0,
        time_axis=dry_axis, states=states, dz=dz, climate=climate)
    timings["sensitivity_s"] = time.perf_counter() - t0

    # --- finite-difference checks ---
    t0 = time.perf_counter()
    check_rows = [int(np.argmin(np.abs(res.day - d))) for d in args.fd_days]
    checks = finite_difference_checks(res, raw, check_rows=check_rows, rtol=args.fd_rtol)
    timings["fd_checks_s"] = time.perf_counter() - t0

    meta = {"z_node_m": z_node, "z_interface_m": z_iface, "dz_m": dz,
            "theta_wp": config.theta_wp, "theta_fc": config.theta_fc,
            "theta_sat": float(config.hydraulics.theta_sat),
            "t_init_k": t_init, "timings": timings,
            "drydown_history": hist}
    return res, meta, checks, timings


def plot(res: NoonSensitivity, meta: dict, path: pathlib.Path, *, texture: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z_iface = meta["z_interface_m"]
    z_node = np.asarray(meta["z_node_m"])
    day_edges = np.concatenate([res.day - 0.5, [res.day[-1] + 0.5]])
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    blue, orange, aqua = cs.SERIES_COLORS
    n_layers = z_node.shape[0]
    # one sequential hue for depth (light = surface, dark = deep), same as the heatmaps
    depth_colors = matplotlib.colormaps[cs.CMAP_SEQUENTIAL](np.linspace(0.35, 1.0, n_layers))

    def heat(ax, fi, title):
        z = res.jac_per_water[:, fi, :].T  # (layers, days)
        vmax = float(np.nanmax(np.abs(z)))
        if np.all(z >= -1e-12):
            m = ax.pcolormesh(day_edges, z_iface, z, cmap=cs.CMAP_SEQUENTIAL, vmin=0.0, vmax=vmax)
        else:
            m = ax.pcolormesh(day_edges, z_iface, z, cmap=cs.CMAP_DIVERGING, vmin=-vmax, vmax=vmax)
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.set_ylim(z_iface[-1], max(z_iface[1], 1e-2))
        ax.set_xlabel("day of dry-down")
        ax.set_ylabel("depth [m]")
        ax.set_title(title, loc="left")
        fig.colorbar(m, ax=ax, label=f"{FLUX_UNITS[fi]} per kg/m2 of water")

    heat(axes[0, 0], 0, "transpiration demand vs soil water, by layer")
    heat(axes[0, 1], 1, "soil evaporation vs soil water, by layer")
    heat(axes[0, 2], 2, "GPP vs soil water, by layer")

    def series(ax, fi, title):
        """The heatmap's columns as time series: one line per layer plus the
        whole-column response (water added in proportion to the profile)."""
        for k in range(n_layers):
            ax.plot(res.day, res.jac_per_water[:, fi, k], color=depth_colors[k], lw=1.5)
        for k in (0, n_layers - 1):
            ax.annotate(f"{z_node[k] * 100:.0f} cm", (res.day[-1], res.jac_per_water[-1, fi, k]),
                        xytext=(4, 0), textcoords="offset points", color=depth_colors[k],
                        va="center", fontsize=8)
        ax.plot(res.day, res.jac_column_water[:, fi], color="0.15", lw=2.5,
                label="whole column (profile-shaped)")
        ax.axhline(0.0, color="0.7", lw=1)
        ax.set_xlabel("day of dry-down")
        ax.set_ylabel(f"{FLUX_UNITS[fi]} per kg/m2 of water")
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, loc="best", fontsize=8)

    series(axes[1, 0], 0, "transpiration demand: same, as time series")
    series(axes[1, 1], 1, "soil evaporation: same, as time series")
    series(axes[1, 2], 2, "GPP: same, as time series")

    ax = axes[2, 0]
    ax.plot(res.day, res.elasticity_water, color=blue, lw=2, label="soil water (all layers)")
    ax.plot(res.day, res.elasticity_humidity, color=orange, lw=2, label="air humidity")
    ax.axhline(0.0, color="0.7", lw=1)
    ax.set_xlabel("day of dry-down")
    ax.set_ylabel("elasticity of LE  [d ln LE / d ln x]")
    ax.set_title("what limits evaporation: water vs demand", loc="left")
    ax.legend(frameon=False)

    ax = axes[2, 1]
    ax.plot(res.day, res.column_water, color=blue, lw=2)
    ax.set_xlabel("day of dry-down")
    ax.set_ylabel("column soil water [kg/m2]")
    ax.set_title("the dry-down", loc="left")

    ax = axes[2, 2]
    ax.plot(res.day, res.fluxes[:, 4], color=blue, lw=2, label="LE total (actual)")
    ax.plot(res.day, res.fluxes[:, 0], color=orange, lw=2, label="transpiration demand")
    ax.plot(res.day, res.fluxes[:, 1], color=aqua, lw=2, label="soil evaporation")
    ax.set_xlabel("day of dry-down")
    ax.set_ylabel("noon flux [W/m2]")
    ax.set_title("noon fluxes", loc="left")
    ax.legend(frameon=False)

    for ax in axes.ravel():
        ax.grid(True, color="0.9", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"One land step differentiated at each noon — {texture} soil, "
                 f"deciduous forest, synthetic forcing", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ===========================================================================
# ET sensitivity along the dry-down, split by process, from the SAVED Jacobians
# ===========================================================================
# (label, index into FLUX_NAMES); transpiration + soil evaporation = ET
ET_PARTS = (("ET", 4), ("transpiration", 0), ("soil evaporation", 1))
DEFAULT_WATER_DEPTHS_CM = (30.0, 50.0)
M_PER_CM = 1.0e-2           # exact unit conversion
# The split must add up: the model's actual latent heat is transpiration plus
# soil evaporation unless a soil-water reservoir cap binds, which would make a
# per-process split of the total misleading.
PART_SUM_RTOL = 1e-8


class DrydownETSensitivity(NamedTuple):
    day: np.ndarray               # (m,) day of the dry-down
    depth_m: float                # soil depth the extra water is spread over [m]
    flux_w_m2: np.ndarray         # (m, 3) noon ET, transpiration, soil evaporation [W/m2]
    d_flux_d_water: np.ndarray    # (m, 3) [W/m2 per mm] water added evenly over depth_m
    d_flux_d_vpd: np.ndarray      # (m, 3) [W/m2 per kPa] VPD raised by drying the air, fixed T
    theta_mean: np.ndarray        # (m,) mean volumetric water content over depth_m [m3/m3]
    theta_wp: float               # wilting point [m3/m3]
    theta_fc: float               # field capacity [m3/m3]
    vpd_kpa: np.ndarray           # (m,) noon vapour-pressure deficit at the forcing height


def layer_fraction_within_depth(z_interface_m, depth_m: float) -> np.ndarray:
    """Fraction of each soil layer lying above ``depth_m`` [-] (1 above, 0 below)."""
    zi = np.asarray(z_interface_m, dtype=np.float64)
    top, bottom = zi[:-1], zi[1:]
    if np.any(bottom <= top):
        raise ValueError("soil layer interfaces must increase with depth")
    return np.clip((np.minimum(bottom, depth_m) - top) / (bottom - top), 0.0, 1.0)


def noon_forcing_along_drydown(saved_args: dict):
    """``(day, T_air [K], q_air [kg/kg], p_surface [Pa])`` at each dry-down noon of a
    saved run, rebuilt on the time axis and forcing path :func:`run_experiment` used."""
    lat_rad = float(np.deg2rad(saved_args["lat_deg"]))
    climate = cs.ColumnClimate(saved_args["mean_annual_t_k"], saved_args["seasonal_amp_k"])
    dt_s = float(saved_args["dt_s"])
    steps_per_day = int(round(cs.SECONDS_PER_DAY / dt_s))
    axis = cs.make_time_axis(int(saved_args["drydown_days"]) * steps_per_day, dt_s,
                             start_doy=float(saved_args["start_doy"]))
    idx = noon_indices(axis)
    t_air, q_air, p = [], [], []
    for i in idx:
        f = cs.column_forcing(lat_rad, 0.0, float(axis.doy[i]), NOON_HOUR_UTC, 0.0,
                              climate=climate)
        t_air.append(float(np.ravel(f.T_lowest)[0]))
        q_air.append(float(np.ravel(f.q_lowest)[0]))
        # The two-leaf canopy converts humidity with the SURFACE pressure
        # (``Ps = forcing.p_surface``), which is also the pressure the forcing
        # generator used to set q; ``p_lowest`` (95 kPa) is not used for it.
        p.append(float(np.ravel(f.p_surface)[0]))
    day = axis.t_s[idx] / cs.SECONDS_PER_DAY + 0.5
    return day, np.asarray(t_air), np.asarray(q_air), np.asarray(p)


def _require_close(name: str, a, b, rtol: float) -> None:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    scale = np.maximum(np.abs(b), 1e-12)
    worst = float(np.max(np.abs(a - b) / scale))
    if not np.isfinite(worst) or worst > rtol:
        raise RuntimeError(f"{name}: max relative difference {worst:.2e} > {rtol:g}; "
                           "saved arrays are inconsistent")


def et_drydown_sensitivity(saved: dict, saved_json: dict, *,
                           depth_m: float) -> DrydownETSensitivity:
    """Noon ET, transpiration and soil-evaporation sensitivity to soil water and to VPD.

    Only exact recombinations of the saved reverse-mode derivatives:

    * **soil water**: each flux's per-layer Jacobian with respect to water
      content, taken along "the same volumetric increment in every layer above
      ``depth_m``" (a layer the depth cuts is weighted by the fraction of it
      above), divided by the water that adds, ``rho_w * depth_m`` [W/m2 per mm].
    * **VPD**: each flux's saved derivative with respect to a scaling of the
      air's specific humidity at fixed air temperature, divided by the VPD change
      that scaling causes, ``-de/ds``, differentiated through
      ``thermo.vapor_pressure_from_specific_humidity`` at the surface pressure the
      canopy uses [W/m2 per kPa].

    Raises if the saved run predates the per-process humidity derivatives, if any
    saved finite-difference check failed, if the per-layer or per-flux arrays do
    not reproduce the separately differentiated whole-column / total-ET
    derivatives, if transpiration + soil evaporation does not equal ET, or if the
    rebuilt noon time axis does not match the saved one.
    """
    if "jac_humidity_scale" not in saved:
        raise ValueError("saved run has no per-process humidity derivatives "
                         "(jac_humidity_scale); rerun demo_process_jacobian.py")
    zi = np.asarray(saved["z_interface_m"])
    if not 0.0 < depth_m <= float(zi[-1]):
        raise ValueError(f"depth_m must be in (0, {float(zi[-1])}] m, got {depth_m}")
    args = saved_json["args"]
    cs.require_all_pass([cs.GradientCheck(**c) for c in saved_json["gradient_checks"]])
    config = cs.make_column_config(args["texture"])
    theta_wp, theta_fc = float(config.theta_wp), float(config.theta_fc)
    if abs(theta_wp - float(saved_json["theta_wp"])) > 1e-12:
        raise ValueError(f"wilting point {theta_wp} != saved {saved_json['theta_wp']}; the "
                         "saved run used a different column configuration")
    day, t_air, q_air, p = noon_forcing_along_drydown(args)
    if day.shape != saved["day"].shape or not np.allclose(day, saved["day"], rtol=0, atol=1e-9):
        raise ValueError("rebuilt dry-down noons do not match the saved day axis")

    idx = [k for _, k in ET_PARTS]
    fluxes = np.asarray(saved["fluxes"])[:, idx]                   # (m, 3)
    jac = np.asarray(saved["jac_per_theta"])[:, idx, :]            # (m, 3, layers)
    jsq = np.asarray(saved["jac_humidity_scale"])[:, idx]          # (m, 3)
    theta = np.asarray(saved["theta"])
    le = fluxes[:, 0]
    # the same derivatives computed a second way by the run (guards against a
    # mis-indexed or stale array)
    _require_close("per-layer Jacobian x profile vs saved water elasticity",
                   np.einsum("mk,mk->m", jac[:, 0], theta) / le, saved["elasticity_water"],
                   CHAIN_RULE_RTOL)
    _require_close("per-flux humidity derivative vs saved humidity elasticity",
                   jsq[:, 0] / le, saved["elasticity_humidity"], CHAIN_RULE_RTOL)
    _require_close("transpiration + soil evaporation vs ET", fluxes[:, 1] + fluxes[:, 2],
                   fluxes[:, 0], PART_SUM_RTOL)

    dz = np.asarray(saved["dz_m"])
    frac = layer_fraction_within_depth(zi, depth_m)
    covered = float(np.sum(dz * frac))
    d_water = jac @ frac / (constants.rho_water * covered)         # (m, 3) W/m2 per mm

    de_ds = np.asarray(jax.vmap(
        jax.grad(lambda s, q, pp: vapor_pressure_from_specific_humidity(q * s, pp)),
        in_axes=(None, 0, 0))(jnp.asarray(1.0), jnp.asarray(q_air), jnp.asarray(p)))
    # VPD = e_s(T) - e with T fixed, so dVPD/ds = -de/ds
    d_vpd = jsq / (-de_ds)[:, None] * PA_PER_KPA                   # (m, 3) W/m2 per kPa
    for name, d in (("soil-water", d_water), ("VPD", d_vpd)):
        _require_close(f"{name} sensitivity: transpiration + soil evaporation vs ET",
                       d[:, 1] + d[:, 2], d[:, 0], PART_SUM_RTOL)

    e_air = np.asarray(vapor_pressure_from_specific_humidity(jnp.asarray(q_air), jnp.asarray(p)))
    vpd_kpa = (np.asarray(saturation_vapor_pressure_aerk(jnp.asarray(t_air))) - e_air) / PA_PER_KPA
    return DrydownETSensitivity(
        day=day, depth_m=float(depth_m), flux_w_m2=fluxes, d_flux_d_water=d_water,
        d_flux_d_vpd=d_vpd, theta_mean=np.sum(theta * dz * frac, axis=1) / covered,
        theta_wp=theta_wp, theta_fc=theta_fc, vpd_kpa=vpd_kpa)


def plot_et_drydown_sensitivity(sens: DrydownETSensitivity, path: pathlib.Path) -> None:
    """Top: sensitivity to soil water and to VPD, ET split into transpiration and
    soil evaporation; middle: the two drivers; bottom: the noon fluxes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # entity colours match the noon-flux panel of the main process_jacobian figure
    blue, orange, aqua = cs.SERIES_COLORS
    part_colors = {"ET": blue, "transpiration": orange, "soil evaporation": aqua}
    part_widths = {"ET": 2.5, "transpiration": 1.8, "soil evaporation": 1.8}
    depth_cm = sens.depth_m / M_PER_CM
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1.0, 1.0])
    ax_w = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[0, 1], sharex=ax_w)
    ax_th = fig.add_subplot(gs[1, 0], sharex=ax_w)
    ax_vpd = fig.add_subplot(gs[1, 1], sharex=ax_w)
    ax_f = fig.add_subplot(gs[2, :], sharex=ax_w)

    def parts(ax, values):
        for j, (name, _) in enumerate(ET_PARTS):
            ax.plot(sens.day, values[:, j], color=part_colors[name], lw=part_widths[name],
                    label=name)
        ax.axhline(0.0, color="0.7", lw=1)

    parts(ax_w, sens.d_flux_d_water)
    ax_w.set_ylabel("W/m2 per mm of soil water")
    ax_w.set_title(f"sensitivity to soil water in the top {depth_cm:.0f} cm", loc="left")
    ax_w.legend(frameon=False, fontsize=9)
    parts(ax_v, sens.d_flux_d_vpd)
    ax_v.set_ylabel("W/m2 per kPa of VPD")
    ax_v.set_title("sensitivity to VPD (air temperature fixed)", loc="left")

    ax_th.plot(sens.day, sens.theta_mean, color="0.2", lw=2)
    for value, label in ((sens.theta_fc, "field capacity"), (sens.theta_wp, "wilting point")):
        ax_th.axhline(value, color="0.55", lw=1, ls="--")
        ax_th.annotate(label, (sens.day[-1], value), xytext=(0, 3), textcoords="offset points",
                       ha="right", va="bottom", fontsize=8, color="0.4")
    ax_th.set_ylabel("m3/m3")
    ax_th.set_title(f"mean soil water content, top {depth_cm:.0f} cm", loc="left")
    ax_vpd.plot(sens.day, sens.vpd_kpa, color="0.2", lw=2)
    ax_vpd.set_ylabel("kPa")
    ax_vpd.set_title("noon vapour-pressure deficit", loc="left")

    parts(ax_f, sens.flux_w_m2)
    ax_f.set_ylabel("W/m2")
    ax_f.set_title("noon latent heat flux", loc="left")
    ax_f.set_xlabel("day of dry-down")
    for ax in (ax_w, ax_v, ax_th, ax_vpd, ax_f):
        ax.grid(True, color="0.9", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def et_sensitivity_filename(depth_m: float) -> str:
    return f"et_drydown_sensitivity_{depth_m / M_PER_CM:.0f}cm.png"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--texture", default="loam", choices=sorted(cs.TEXTURES))
    ap.add_argument("--biome", default=cs.DEFAULT_BIOME)
    ap.add_argument("--lai", type=float, default=cs.DEFAULT_LAI)
    ap.add_argument("--lat-deg", type=float, default=40.0, help="latitude (solar geometry)")
    ap.add_argument("--mean-annual-t-k", type=float, default=cs.DEFAULT_MEAN_ANNUAL_T_K,
                    help="mean annual air temperature of the column climate [K]")
    ap.add_argument("--seasonal-amp-k", type=float, default=cs.DEFAULT_T_SEASONAL_AMP_K,
                    help="seasonal half-amplitude of air temperature [K] (NH-phased)")
    ap.add_argument("--start-doy", type=float, default=152.0, help="first dry-down day (June 1)")
    ap.add_argument("--spinup-days", type=int, default=60)
    ap.add_argument("--drydown-days", type=int, default=45)
    ap.add_argument("--dt-s", type=float, default=cs.DEFAULT_DT_S)
    ap.add_argument("--precip-rate", type=float, default=cs.DEFAULT_PRECIP_KG_M2_S,
                    help="steady spin-up rain [kg/m2/s]")
    ap.add_argument("--fd-days", type=float, nargs="+", default=[5.0, 30.0],
                    help="dry-down days at which AD is checked against finite differences")
    ap.add_argument("--fd-rtol", type=float, default=1e-3)
    ap.add_argument("--out-dir", default="diagnostics/land_sensitivity/process_jacobian")
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    ap.add_argument("--et-sensitivity-from-saved", action="store_true",
                    help="no model run: read process_jacobian.npz/.json from --out-dir and "
                         "plot noon ET, transpiration and soil-evaporation sensitivity to "
                         "soil water and VPD along that dry-down")
    ap.add_argument("--water-depths-cm", type=float, nargs="+",
                    default=list(DEFAULT_WATER_DEPTHS_CM),
                    help="soil depths the extra water is spread over, one figure each [cm]")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.et_sensitivity_from_saved:
        saved = dict(np.load(out / "process_jacobian.npz"))
        saved_json = json.loads((out / "process_jacobian.json").read_text())
        for depth_cm in args.water_depths_cm:
            sens = et_drydown_sensitivity(saved, saved_json, depth_m=depth_cm * M_PER_CM)
            path = out / et_sensitivity_filename(sens.depth_m)
            plot_et_drydown_sensitivity(sens, path)
            print(f"--- top {depth_cm:.0f} cm: W/m2 per mm soil water | W/m2 per kPa VPD "
                  "(ET, transpiration, soil evaporation)")
            for i in range(0, len(sens.day), max(1, len(sens.day) // 10)):
                w, v = sens.d_flux_d_water[i], sens.d_flux_d_vpd[i]
                print(f"day {sens.day[i]:4.1f}  water {w[0]:6.3f} {w[1]:6.3f} {w[2]:6.3f}  "
                      f"vpd {v[0]:6.2f} {v[1]:6.2f} {v[2]:6.2f}  "
                      f"theta {sens.theta_mean[i]:.3f}")
            print(f"wrote {path}")
        return 0
    res, meta, checks, timings = run_experiment(args)
    print(cs.format_checks(checks))
    cs.require_all_pass(checks)
    np.savez(out / "process_jacobian.npz", **res._asdict(),
             z_node_m=meta["z_node_m"], z_interface_m=meta["z_interface_m"], dz_m=meta["dz_m"],
             flux_names=np.asarray(FLUX_NAMES))
    cs.write_json(out / "process_jacobian.json", cs.provenance(
        vars(args), timings=timings, theta_wp=meta["theta_wp"], theta_fc=meta["theta_fc"],
        theta_sat=meta["theta_sat"], t_init_k=meta["t_init_k"],
        gradient_checks=[c._asdict() for c in checks]))
    (out / "gradient_checks.txt").write_text(cs.format_checks(checks) + "\n")
    if args.plot:
        plot(res, meta, out / "process_jacobian.png", texture=args.texture)
    print("day  W_col[kg/m2]  LE[W/m2]  E_water  E_humidity  dLE/dT[W/m2/K]")
    for i in range(0, len(res.day), max(1, len(res.day) // 10)):
        print(f"{res.day[i]:4.1f} {res.column_water[i]:11.1f} {res.fluxes[i, 4]:9.1f} "
              f"{res.elasticity_water[i]:8.3f} {res.elasticity_humidity[i]:10.3f} "
              f"{res.dle_dt_air[i]:12.3f}")
    print(f"timings: {', '.join(f'{k}={v:.1f}s' for k, v in timings.items())}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
