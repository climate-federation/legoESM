#!/usr/bin/env python
"""Demo: adjoint memory kernels — when does rain (or warmth) still matter?

Runs one land column through a season and differentiates an END-OF-SEASON
quantity with respect to the WHOLE per-step forcing series in ONE reverse pass.
The result is a memory kernel over the season: how much an extra unit of rain
(or a one-step warming) at step ``t`` changes the target at the end.

Targets (one reverse pass each, both kernels come out of the same pass):

* ``water`` — column soil water at the end of the season [kg/m2].  The rain
  kernel is then the dimensionless fraction of an extra kg/m2 of rain at step
  ``t`` that is still stored at season end.
* ``gpp`` — cumulative gross primary production over the season [gC/m2]; the
  rain kernel is gC/m2 per extra kg/m2 of rain at step ``t``.

The season is steady rain with a dry spell (``--dry-spell``), run for two soil
textures so the memory length of sand and clay can be compared.  A perturbation
experiment would need one forward run PER STEP to build the same kernel.  The AD
kernels are checked against central finite differences at ``--fd-days``; a
failed check aborts the run.

Usage
-----
    JAX_ENABLE_X64=1 python scripts/run/land_sensitivity/demo_adjoint_memory.py \
        --textures sand clay --out-dir diagnostics/land_sensitivity/adjoint_memory
"""

from __future__ import annotations

import argparse
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

from legoesm import constants
from scripts.run.land_sensitivity import column_setup as cs

# water / gpp: the original season targets; the four water fates (window totals of
# transpiration, soil evaporation, runoff + drainage, and the water still held at the
# end) add up to every kg/m2 of rain, which demo_pulse_memory.py checks.
WATER_FATES = ("transpiration", "soil_evaporation", "runoff", "storage")
TARGETS = ("water", "gpp", *WATER_FATES)
DEFAULT_TARGETS = ("water", "gpp")
TARGET_UNITS = {"water": "kg/m2", "gpp": "gC/m2", "transpiration": "kg/m2",
                "soil_evaporation": "kg/m2", "runoff": "kg/m2", "storage": "kg/m2"}
FD_EPS_PRECIP = 1e-7      # [kg/m2/s] (~0.5 % of the default rate)
FD_EPS_T_AIR = 1e-3       # [K]
FD_ATOL_WATER = 1e-6      # [kg/m2 per unit input]
FD_ATOL_GPP = 1e-12       # [gC/m2 per unit input]


class MemoryKernels(NamedTuple):
    texture: str
    target: str
    value: float                  # the target's value at the baseline forcing
    rain_kernel: np.ndarray       # (n,) d target / d (rain mass in step t)  [target per kg/m2]
    t_air_kernel: np.ndarray      # (n,) d target / d (T_air offset in step t) [target per K]
    rain_kernel_daily: np.ndarray  # (days,) daily MEAN of rain_kernel
    t_air_kernel_daily: np.ndarray  # (days,) daily SUM of t_air_kernel (per K applied all day)


def season_precip(time_axis: cs.TimeAxis, base_rate: float,
                  dry_spell_days: tuple[float, float] | None) -> jnp.ndarray:
    """Steady rain at ``base_rate`` with zero rain inside ``[start, end)`` days."""
    day = time_axis.t_s / cs.SECONDS_PER_DAY
    p = np.full(day.shape, base_rate, dtype=np.float64)
    if dry_spell_days is not None:
        start, end = dry_spell_days
        if not 0.0 <= start < end:
            raise ValueError(f"dry spell must satisfy 0 <= start < end, got {dry_spell_days}")
        p[(day >= start) & (day < end)] = 0.0
    return jnp.asarray(p)


def water_fate_totals(final, hist, dz, dt_s: float) -> jnp.ndarray:
    """Window totals of the four fates of the water [kg/m2], in ``WATER_FATES`` order:
    transpiration and soil evaporation (their latent heat / L_v), surface runoff plus
    bottom drainage, and every water store at the end of the window."""
    return jnp.stack([
        jnp.sum(hist.le_canopy) * dt_s / constants.L_v,
        jnp.sum(hist.le_soil) * dt_s / constants.L_v,
        jnp.sum(hist.runoff) * dt_s,
        cs.column_storage_kg_m2(final, dz),
    ])


def make_target_fns(config, params, *, lat_rad: float, time_axis: cs.TimeAxis,
                    dt_s: float, dz, state0, remat: bool,
                    climate: cs.ColumnClimate = cs.DEFAULT_CLIMATE):
    """``{target: value_fn(precip, t_air_offset)}`` over the season from ``state0``."""

    def _run(precip, t_off):
        return cs.rollout(state0, config, params, lat_rad=lat_rad, lon_rad=0.0,
                          time_axis=time_axis, dt_s=dt_s, precip=precip, climate=climate,
                          t_air_offset=t_off, remat=remat)

    def water(precip, t_off):
        final, _ = _run(precip, t_off)
        return cs.column_water_kg_m2(final.theta_soil[0], dz)

    def gpp(precip, t_off):
        _, hist = _run(precip, t_off)
        return jnp.sum(hist.gpp) * dt_s

    def fate(i):
        def total(precip, t_off):
            final, hist = _run(precip, t_off)
            return water_fate_totals(final, hist, dz, dt_s)[i]
        return total

    return {"water": water, "gpp": gpp,
            **{name: fate(i) for i, name in enumerate(WATER_FATES)}}


def kernels_from_gradients(texture: str, target: str, value, g_precip, g_t_air,
                           dt_s: float, steps_per_day: int) -> MemoryKernels:
    """Memory kernels from the gradient of a target with respect to the per-step rain
    RATE and air-temperature offset; raises on a non-finite value or gradient."""
    g_p, g_t = np.asarray(g_precip), np.asarray(g_t_air)
    if not (np.isfinite(value) and np.all(np.isfinite(g_p)) and np.all(np.isfinite(g_t))):
        raise FloatingPointError(f"non-finite kernel for {texture}/{target}")
    rain_kernel = g_p / dt_s          # per kg/m2 of rain in that step
    days = g_p.shape[0] // steps_per_day
    return MemoryKernels(
        texture=texture, target=target, value=float(value),
        rain_kernel=rain_kernel, t_air_kernel=g_t,
        rain_kernel_daily=rain_kernel[:days * steps_per_day].reshape(days, steps_per_day).mean(1),
        t_air_kernel_daily=g_t[:days * steps_per_day].reshape(days, steps_per_day).sum(1),
    )


def compute_kernels(texture: str, target: str, fn, precip, dt_s: float,
                    steps_per_day: int) -> MemoryKernels:
    n = int(precip.shape[0])
    t_off = jnp.zeros(n)
    value, (g_p, g_t) = jax.jit(jax.value_and_grad(fn, argnums=(0, 1)))(precip, t_off)
    return kernels_from_gradients(texture, target, value, g_p, g_t, dt_s, steps_per_day)


def finite_difference_checks(kern: MemoryKernels, fn, precip, *, dt_s: float,
                             steps: list[int], rtol: float) -> list[cs.GradientCheck]:
    """Central-difference checks of the rain and air-temperature kernels at ``steps``.

    ``fn(precip, t_off)`` is the (jitted) season value function; the rain kernel
    is stored per kg/m2 of rain, so it is multiplied back by ``dt_s`` to compare
    with the derivative with respect to the rate.
    """
    n = int(precip.shape[0])
    t_off = jnp.zeros(n)
    atol = FD_ATOL_GPP if kern.target == "gpp" else FD_ATOL_WATER   # kg/m2 for the rest
    checks = []
    for k in steps:
        # Rain cannot go negative: inside a dry spell use a one-sided difference.
        one_sided = bool(float(precip[k]) < FD_EPS_PRECIP)
        fd_p, smooth = cs.finite_difference(lambda p: fn(p, t_off), precip, k, FD_EPS_PRECIP,
                                            atol=atol, one_sided=one_sided)
        checks.append(cs.gradient_check(
            f"{kern.texture}/{kern.target} d/d rain[step {k}]",
            kern.rain_kernel[k] * dt_s, fd_p, rtol=rtol, atol=atol, fd_consistent=smooth))
        fd_t, smooth = cs.finite_difference(lambda t: fn(precip, t), t_off, k, FD_EPS_T_AIR,
                                            atol=atol)
        checks.append(cs.gradient_check(
            f"{kern.texture}/{kern.target} d/d T_air[step {k}]",
            kern.t_air_kernel[k], fd_t, rtol=rtol, atol=atol, fd_consistent=smooth))
    return checks


def run_experiment(args):
    lat_rad = float(np.deg2rad(args.lat_deg))
    steps_per_day = int(round(cs.SECONDS_PER_DAY / args.dt_s))
    season_axis = cs.make_time_axis(args.season_days * steps_per_day, args.dt_s,
                                    start_doy=args.start_doy)
    precip = season_precip(season_axis, args.precip_rate, tuple(args.dry_spell))
    fd_steps = [int(np.clip(round(d * steps_per_day), 0, precip.shape[0] - 1))
                for d in args.fd_days]
    kernels: list[MemoryKernels] = []
    checks: list[cs.GradientCheck] = []
    timings: dict[str, float] = {}
    params = cs.make_canopy_params(1, biome=args.biome, lai=args.lai)
    climate = cs.ColumnClimate(args.mean_annual_t_k, args.seasonal_amp_k)
    t_init = float(climate.mean_annual_t_k)
    for texture in args.textures:
        config = cs.make_column_config(texture)
        dz, _, _ = cs.soil_layers(config)
        state0 = cs.initial_state(config, t_init_k=t_init)
        t0 = time.perf_counter()
        spin_axis = cs.make_time_axis(args.spinup_days * steps_per_day, args.dt_s,
                                      start_doy=args.start_doy - args.spinup_days)
        state_season0 = jax.jit(lambda s: cs.rollout(
            s, config, params, lat_rad=lat_rad, lon_rad=0.0, time_axis=spin_axis,
            dt_s=args.dt_s, precip=args.precip_rate, climate=climate)[0])(state0)
        jax.block_until_ready(state_season0)
        timings[f"{texture}/spinup_s"] = time.perf_counter() - t0
        fns = make_target_fns(config, params, lat_rad=lat_rad, time_axis=season_axis,
                              dt_s=args.dt_s, dz=dz, state0=state_season0, remat=args.remat,
                              climate=climate)
        for target in args.targets:
            fn = fns[target]
            t0 = time.perf_counter()
            kern = compute_kernels(texture, target, fn, precip, args.dt_s, steps_per_day)
            timings[f"{texture}/{target}/adjoint_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            checks += finite_difference_checks(kern, jax.jit(fn), precip, dt_s=args.dt_s,
                                               steps=fd_steps, rtol=args.fd_rtol)
            timings[f"{texture}/{target}/fd_s"] = time.perf_counter() - t0
            kernels.append(kern)
    meta = {"season_axis": season_axis, "precip": np.asarray(precip),
            "steps_per_day": steps_per_day, "t_init_k": t_init, "timings": timings}
    return kernels, checks, meta


def plot(kernels: list[MemoryKernels], meta: dict, path: pathlib.Path, *, dry_spell) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = list(dict.fromkeys(k.target for k in kernels))
    textures = list(dict.fromkeys(k.texture for k in kernels))
    colors = dict(zip(textures, cs.SERIES_COLORS))
    n_rows = len(targets) + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 3.2 * n_rows), sharex=True)
    axes = np.atleast_1d(axes)
    ylabels = {"water": "kg/m2 stored at season end\nper kg/m2 of extra rain",
               "gpp": "gC/m2 of season GPP\nper kg/m2 of extra rain",
               "transpiration": "kg/m2 transpired by season end\nper kg/m2 of extra rain",
               "soil_evaporation": "kg/m2 evaporated by season end\nper kg/m2 of extra rain",
               "runoff": "kg/m2 run off by season end\nper kg/m2 of extra rain",
               "storage": "kg/m2 held at season end\nper kg/m2 of extra rain"}
    titles = {"water": "rain memory of the soil water store",
              "gpp": "rain memory of productivity",
              "transpiration": "rain memory of transpiration",
              "soil_evaporation": "rain memory of soil evaporation",
              "runoff": "rain memory of runoff",
              "storage": "rain memory of total water storage"}
    for ax, target in zip(axes[:-1], targets):
        for k in kernels:
            if k.target != target:
                continue
            days = np.arange(k.rain_kernel_daily.shape[0]) + 0.5
            ax.plot(days, k.rain_kernel_daily, color=colors[k.texture], lw=2, label=k.texture)
            ax.annotate(k.texture, (days[-1], k.rain_kernel_daily[-1]), xytext=(4, 0),
                        textcoords="offset points", color=colors[k.texture], va="center")
        ax.set_ylabel(ylabels[target])
        ax.set_title(titles[target], loc="left")
    ax = axes[-1]
    for k in kernels:
        if k.target != targets[0]:
            continue
        days = np.arange(k.t_air_kernel_daily.shape[0]) + 0.5
        ax.plot(days, k.t_air_kernel_daily, color=colors[k.texture], lw=2, label=k.texture)
    ax.set_ylabel(f"{TARGET_UNITS[targets[0]]} per K of\n"
                  "one-day air warming")
    ax.set_title(f"warmth memory ({titles[targets[0]].split(' of ')[1]})", loc="left")
    ax.set_xlabel("day of season")
    for ax in axes:
        if dry_spell is not None:
            ax.axvspan(dry_spell[0], dry_spell[1], color="0.92", lw=0, label="dry spell")
        ax.axhline(0.0, color="0.7", lw=1)
        ax.grid(True, color="0.9", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("One backward pass per curve: sensitivity of a season-end quantity "
                 "to every hour of forcing", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--textures", nargs="+", default=["sand", "clay"], choices=sorted(cs.TEXTURES))
    ap.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS), choices=TARGETS)
    ap.add_argument("--biome", default=cs.DEFAULT_BIOME)
    ap.add_argument("--lai", type=float, default=cs.DEFAULT_LAI)
    ap.add_argument("--lat-deg", type=float, default=40.0, help="latitude (solar geometry)")
    ap.add_argument("--mean-annual-t-k", type=float, default=cs.DEFAULT_MEAN_ANNUAL_T_K,
                    help="mean annual air temperature of the column climate [K]")
    ap.add_argument("--seasonal-amp-k", type=float, default=cs.DEFAULT_T_SEASONAL_AMP_K,
                    help="seasonal half-amplitude of air temperature [K] (NH-phased)")
    ap.add_argument("--start-doy", type=float, default=121.0, help="first season day (May 1)")
    ap.add_argument("--spinup-days", type=int, default=60)
    ap.add_argument("--season-days", type=int, default=120)
    ap.add_argument("--dry-spell", type=float, nargs=2, default=[40.0, 80.0],
                    metavar=("START_DAY", "END_DAY"))
    ap.add_argument("--dt-s", type=float, default=cs.DEFAULT_DT_S)
    ap.add_argument("--precip-rate", type=float, default=cs.DEFAULT_PRECIP_KG_M2_S)
    ap.add_argument("--fd-days", type=float, nargs="+", default=[60.0, 115.0],
                    help="season days at which the kernels are checked against finite differences")
    ap.add_argument("--fd-rtol", type=float, default=1e-3)
    ap.add_argument("--no-remat", dest="remat", action="store_false",
                    help="do not checkpoint each step of the season in the reverse pass "
                         "(default: checkpoint; the checkpointed reverse pass is the one "
                         "the finite differences arbitrate for, see demo_global_param_map)")
    ap.add_argument("--out-dir", default="diagnostics/land_sensitivity/adjoint_memory")
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kernels, checks, meta = run_experiment(args)
    print(cs.format_checks(checks))
    cs.require_all_pass(checks)
    arrays = {"day_of_season": meta["season_axis"].t_s / cs.SECONDS_PER_DAY,
              "precip_kg_m2_s": meta["precip"]}
    summary = {}
    for k in kernels:
        key = f"{k.texture}_{k.target}"
        arrays[f"{key}_rain_kernel"] = k.rain_kernel
        arrays[f"{key}_t_air_kernel"] = k.t_air_kernel
        arrays[f"{key}_rain_kernel_daily"] = k.rain_kernel_daily
        arrays[f"{key}_t_air_kernel_daily"] = k.t_air_kernel_daily
        summary[key] = {"value": k.value,
                        "rain_kernel_max": float(k.rain_kernel.max()),
                        "rain_kernel_last_step": float(k.rain_kernel[-1])}
        print(f"{key:12s} value={k.value:.4g}  rain kernel: last step {k.rain_kernel[-1]:.4g}, "
              f"max {k.rain_kernel.max():.4g}, day-1 mean {k.rain_kernel_daily[0]:.4g}")
    np.savez(out / "adjoint_memory.npz", **arrays)
    cs.write_json(out / "adjoint_memory.json", cs.provenance(
        vars(args), timings=meta["timings"], t_init_k=meta["t_init_k"], summary=summary,
        gradient_checks=[c._asdict() for c in checks]))
    (out / "gradient_checks.txt").write_text(cs.format_checks(checks) + "\n")
    if args.plot:
        plot(kernels, meta, out / "adjoint_memory.png", dry_spell=tuple(args.dry_spell))
    print(f"timings: {', '.join(f'{k}={v:.1f}s' for k, v in meta['timings'].items())}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
