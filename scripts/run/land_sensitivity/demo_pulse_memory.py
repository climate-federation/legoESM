#!/usr/bin/env python
"""Demo: memory of a rain pulse in transpiration and soil evaporation (adjoint kernels).

One land column (multilayer land + two-leaf canopy) is spun up under steady rain,
then receives a single rain pulse at the start of a rain-free dry-down.  Over that
window the water has four fates: transpiration, soil evaporation, runoff plus
bottom drainage, and water still stored at the end.  One reverse-mode Jacobian of
the four window totals (a single compile; four backward sweeps run together)
differentiates each of them with respect to the rain rate of EVERY hour of the
window.  The result is a memory kernel: of one extra mm of rain falling in
hour t, how many mm leave as transpiration (or soil evaporation, ...) by the end
of the window.  At the pulse hours the kernel is the marginal fate of the pulse
water; later in the dry-down it is the fate of a small shower that did not fall.

Every extra mm has to go somewhere, so the four kernels must add to one in every
hour.  That mass-conservation identity and the column's own water budget are
always checked.  Comparing the transpiration and soil-evaporation kernels with
central finite differences is optional (``--fd-days``; off by default because each
check costs extra forward runs of the whole window).  A failed check aborts the run
after the arrays are written.

Reverse mode only: the canopy energy-balance Newton solve has an
implicit-function-theorem ``custom_vjp``, and JAX refuses forward-mode (JVP)
differentiation of a ``custom_vjp`` function, so the response of every later day
to the pulse is not available from one forward-mode pass.

Usage
-----
    JAX_ENABLE_X64=1 python scripts/run/land_sensitivity/demo_pulse_memory.py \
        --textures loam --out-dir diagnostics/land_sensitivity/pulse_memory
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
from scripts.run.land_sensitivity import demo_adjoint_memory as adj

FATES = adj.WATER_FATES      # the order of adj.water_fate_totals
FATE_LABELS = {"transpiration": "transpiration", "soil_evaporation": "soil evaporation",
               "runoff": "runoff + drainage", "storage": "still stored at the end"}
CHECKED_FATES = ("transpiration", "soil_evaporation")   # arbitrated by finite differences
PROCESSES = ("transpiration", "soil_evaporation")
# Every extra kg/m2 of rain ends in one of the four fates (mass conservation of the
# derivative); the column's own budget P - ET - R - dS closes to ~1e-4 kg/m2 over 45 days.
# Measured 2026-09-15 (loam, 10 mm / 3 h pulse): the four fate kernels add to exactly 1
# in every rain-free hour but miss by up to 2e-4 in the hours when rain is falling
# (0.99980, 1.00005, 1.00001).  Central finite differences of the forward model at the
# smaller step reproduce those sums to 1e-7 for every fate, so the gap is in the forward
# soil-water solve during rain (fixed-iteration Picard + pond clamp), not a missing
# derivative path.  A missing or double-counted fate misses by >= 1e-2, so 1e-3 still
# catches bookkeeping errors in these kernels.
PARTITION_ATOL = 1e-3        # [kg/m2 per kg/m2]
BUDGET_ATOL_KG_M2 = 1e-3     # [kg/m2] over the window
LATENT_SPLIT_ATOL_W_M2 = 1e-6   # [W/m2] latent heat must equal transpiration + soil evaporation
REPORT_DAYS = (0, 1, 3, 5, 10, 20, 30, 40)


def pulse_precip(time_axis: cs.TimeAxis, *, pulse_mm: float, pulse_hours: float,
                 dt_s: float) -> tuple[jnp.ndarray, int]:
    """Rain rate [kg/m2/s] per step: ``pulse_mm`` spread evenly over the first
    ``pulse_hours`` of the window and none afterwards; also returns the number of
    pulse steps."""
    n = int(time_axis.t_s.shape[0])
    steps = pulse_hours * cs.SECONDS_PER_HOUR / dt_s
    n_pulse = int(round(steps))
    if pulse_mm < 0.0:
        raise ValueError(f"pulse_mm must be >= 0, got {pulse_mm}")
    if n_pulse < 1 or abs(steps - n_pulse) > 1e-9 or n_pulse > n:
        raise ValueError(f"pulse_hours {pulse_hours} must be a whole number (>= 1) of "
                         f"{dt_s:g} s steps within the {n}-step window")
    rate = np.zeros(n, dtype=np.float64)
    rate[:n_pulse] = pulse_mm / (n_pulse * dt_s)
    return jnp.asarray(rate), n_pulse


class PulseMemory(NamedTuple):
    texture: str
    day: np.ndarray            # (days,) centre of each day of the window, day 0 = pulse day
    kernel_daily: dict         # fate -> (days,) mm of that fate by window end per extra mm
                               #   of rain spread over that day
    pulse_fate: dict           # fate -> mm per extra mm of the pulse
    window_value: dict         # fate -> baseline window total (storage: end-of-window) [mm]
    base_daily: dict           # process -> (days,) baseline mm/day
    partition_error: float     # max over hours of |sum of fate kernels - 1|
    budget_residual: float     # P - ET - runoff - dStorage over the window [kg/m2]
    kernels: dict              # fate -> adj.MemoryKernels (hourly rain and T_air kernels)


def run_experiment(args):
    lat_rad = float(np.deg2rad(args.lat_deg))
    steps_per_day = int(round(cs.SECONDS_PER_DAY / args.dt_s))
    if abs(steps_per_day * args.dt_s - cs.SECONDS_PER_DAY) > 1e-9:
        raise ValueError(f"dt_s {args.dt_s} must divide a day")
    n_days = int(args.window_days)
    window_axis = cs.make_time_axis(n_days * steps_per_day, args.dt_s, start_doy=args.start_doy)
    precip, n_pulse = pulse_precip(window_axis, pulse_mm=args.pulse_mm,
                                   pulse_hours=args.pulse_hours, dt_s=args.dt_s)
    n_steps = int(precip.shape[0])
    fd_steps = [int(np.clip(round(d * steps_per_day), 0, n_steps - 1)) for d in args.fd_days]
    params = cs.make_canopy_params(1, biome=args.biome, lai=args.lai)
    climate = cs.ColumnClimate(args.mean_annual_t_k, args.seasonal_amp_k)
    t_init = float(climate.mean_annual_t_k)
    results: list[PulseMemory] = []
    checks: list[cs.GradientCheck] = []
    timings: dict[str, float] = {}
    for texture in args.textures:
        config = cs.make_column_config(texture)
        if config.interception is not None:
            raise ValueError("interception is on: wet-canopy evaporation would be a fifth water "
                             "fate that these kernels do not account for")
        dz, _, _ = cs.soil_layers(config)
        t0 = time.perf_counter()
        spin_axis = cs.make_time_axis(args.spinup_days * steps_per_day, args.dt_s,
                                      start_doy=args.start_doy - args.spinup_days)
        state_w0 = jax.jit(lambda s, cfg=config: cs.rollout(
            s, cfg, params, lat_rad=lat_rad, lon_rad=0.0, time_axis=spin_axis,
            dt_s=args.dt_s, precip=args.precip_rate, climate=climate)[0])(
                cs.initial_state(config, t_init_k=t_init))
        jax.block_until_ready(state_w0)
        timings[f"{texture}/spinup_s"] = time.perf_counter() - t0

        # --- baseline window: water budget, latent split, solver convergence ---
        final, hist = jax.jit(lambda s, cfg=config: cs.rollout(
            s, cfg, params, lat_rad=lat_rad, lon_rad=0.0, time_axis=window_axis,
            dt_s=args.dt_s, precip=precip, climate=climate))(state_w0)
        if int(jnp.max(hist.n_iters)) >= config.surface_scheme.max_iters:
            raise RuntimeError(f"{texture}: canopy Newton solve hit max_iters in the window; "
                               "its implicit gradient is zeroed on unconverged steps")
        split = float(jnp.max(jnp.abs(hist.lhflx - (hist.le_canopy + hist.le_soil))))
        if split > LATENT_SPLIT_ATOL_W_M2:
            raise RuntimeError(f"{texture}: latent heat differs from transpiration + soil "
                               f"evaporation by up to {split:.3g} W/m2 (a reservoir cap binds)")
        budget = float(
            jnp.sum(precip) * args.dt_s - jnp.sum(hist.lhflx) * args.dt_s / constants.L_v
            - jnp.sum(hist.runoff) * args.dt_s
            - (cs.column_storage_kg_m2(final, dz) - cs.column_storage_kg_m2(state_w0, dz)))
        if abs(budget) > BUDGET_ATOL_KG_M2:
            raise RuntimeError(f"{texture}: window water budget does not close "
                               f"(residual {budget:.3g} kg/m2)")

        # --- the four fates in ONE reverse-mode Jacobian (one compile per soil) ---
        def fate_totals(p, t_off, cfg=config, s0=state_w0, dz_=dz):
            final_, hist_ = cs.rollout(s0, cfg, params, lat_rad=lat_rad, lon_rad=0.0,
                                       time_axis=window_axis, dt_s=args.dt_s, precip=p,
                                       climate=climate, t_air_offset=t_off, remat=args.remat)
            return adj.water_fate_totals(final_, hist_, dz_, args.dt_s)

        values = np.asarray(adj.water_fate_totals(final, hist, dz, args.dt_s))
        t0 = time.perf_counter()
        jac_p, jac_t = jax.jit(jax.jacrev(fate_totals, argnums=(0, 1)))(
            precip, jnp.zeros(n_steps))
        jax.block_until_ready(jac_p)
        timings[f"{texture}/adjoint_s"] = time.perf_counter() - t0
        kern = {fate: adj.kernels_from_gradients(texture, fate, values[i], jac_p[i], jac_t[i],
                                                 args.dt_s, steps_per_day)
                for i, fate in enumerate(FATES)}
        partition = sum(kern[f].rain_kernel for f in FATES)
        partition_error = float(np.max(np.abs(partition - 1.0)))

        if fd_steps:
            t0 = time.perf_counter()
            fns = adj.make_target_fns(config, params, lat_rad=lat_rad, time_axis=window_axis,
                                      dt_s=args.dt_s, dz=dz, state0=state_w0,
                                      remat=args.remat, climate=climate)
            for fate in CHECKED_FATES:
                checks += adj.finite_difference_checks(kern[fate], jax.jit(fns[fate]), precip,
                                                       dt_s=args.dt_s, steps=fd_steps,
                                                       rtol=args.fd_rtol)
            timings[f"{texture}/fd_s"] = time.perf_counter() - t0

        def daily_mm(latent):
            return (np.asarray(latent)[:n_days * steps_per_day]
                    .reshape(n_days, steps_per_day).sum(1) * args.dt_s / constants.L_v)

        results.append(PulseMemory(
            texture=texture, day=np.arange(n_days) + 0.5,
            kernel_daily={f: kern[f].rain_kernel_daily for f in FATES},
            # the pulse's extra mm is spread evenly over its steps
            pulse_fate={f: float(np.sum(kern[f].rain_kernel[:n_pulse]) / n_pulse) for f in FATES},
            window_value={f: kern[f].value for f in FATES},
            base_daily={"transpiration": daily_mm(hist.le_canopy),
                        "soil_evaporation": daily_mm(hist.le_soil)},
            partition_error=partition_error, budget_residual=budget, kernels=kern))
    meta = {"window_axis": window_axis, "precip": np.asarray(precip), "n_pulse": n_pulse,
            "steps_per_day": steps_per_day, "t_init_k": t_init, "timings": timings}
    return results, checks, meta


def require_partition(results: list[PulseMemory], atol: float = PARTITION_ATOL) -> None:
    """Raise if the four fate kernels do not add to one in some hour for some texture."""
    bad = [f"{r.texture}: {r.partition_error:.3g}" for r in results if r.partition_error > atol]
    if bad:
        raise RuntimeError("fate kernels do not add to one (extra rain is created or lost): "
                           + ", ".join(bad))


def load_results(out_dir: pathlib.Path) -> tuple[list[PulseMemory], dict]:
    """Rebuild the per-texture results from the ``pulse_memory.npz`` / ``.json`` a run
    wrote to ``out_dir`` (no model run); returns ``(results, saved_json)``."""
    import json
    saved = json.loads((out_dir / "pulse_memory.json").read_text())
    z = np.load(out_dir / "pulse_memory.npz")
    dt_s = float(saved["args"]["dt_s"])
    steps_per_day = int(round(cs.SECONDS_PER_DAY / dt_s))
    results = []
    for texture, summ in saved["summary"].items():
        kern = {f: adj.kernels_from_gradients(
                    texture, f, summ["window_value"][f], z[f"{texture}_{f}_rain_kernel"] * dt_s,
                    z[f"{texture}_{f}_t_air_kernel"], dt_s, steps_per_day)
                for f in FATES}
        n_days = kern[FATES[0]].rain_kernel_daily.shape[0]
        results.append(PulseMemory(
            texture=texture, day=np.arange(n_days) + 0.5,
            kernel_daily={f: kern[f].rain_kernel_daily for f in FATES},
            pulse_fate=summ["pulse_fate"], window_value=summ["window_value"],
            base_daily={proc: z[f"{texture}_{proc}_base_daily_mm"] for proc in PROCESSES},
            partition_error=summ["partition_error"],
            budget_residual=summ["budget_residual_kg_m2"], kernels=kern))
    return results, saved


def plot(results: list[PulseMemory], path: pathlib.Path, *, pulse_mm: float,
         pulse_hours: float, kernel_fates=FATES, max_day: float | None = None) -> None:
    """Top: fate of an extra mm of rain on each day (``kernel_fates`` only); bottom:
    baseline daily transpiration and soil evaporation.  ``max_day`` limits both panels
    to the first days of the window; the fates are still totals by the window end."""
    bad = [f for f in kernel_fates if f not in FATES]
    if bad or not kernel_fates:
        raise ValueError(f"kernel_fates must be a non-empty subset of {FATES}, got {kernel_fates}")
    if max_day is not None and max_day <= 0:
        raise ValueError(f"max_day must be > 0, got {max_day}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, aqua = cs.SERIES_COLORS
    # process colours match the ET-split figure (transpiration orange, soil evaporation aqua)
    style = {"transpiration": dict(color=orange, lw=2.2),
             "soil_evaporation": dict(color=aqua, lw=2.2),
             "runoff": dict(color="0.45", lw=1.5, ls="--"),
             "storage": dict(color="0.15", lw=1.5)}
    n_cols = len(results)
    fig, axes = plt.subplots(2, n_cols, figsize=(6.5 * n_cols, 8.5), sharex=True, squeeze=False)
    window_days = int(results[0].day.shape[0])
    for c, r in enumerate(results):
        keep = r.day < (max_day if max_day is not None else np.inf)
        ax = axes[0, c]
        for fate in kernel_fates:
            ax.plot(r.day[keep], r.kernel_daily[fate][keep], label=FATE_LABELS[fate],
                    **style[fate])
        ax.axhline(0.0, color="0.7", lw=1)
        ax.set_ylabel("mm per extra mm of rain")
        ax.set_title(f"{r.texture}: fate by day {window_days} of an extra mm of rain on day t",
                     loc="left")
        ax = axes[1, c]
        for proc in PROCESSES:
            ax.plot(r.day[keep], r.base_daily[proc][keep], label=FATE_LABELS[proc],
                    **style[proc])
        ax.set_ylabel("mm/day")
        ax.set_xlabel("day after the rain pulse")
        ax.set_title(f"{r.texture}: daily fluxes after a {pulse_mm:g} mm pulse", loc="left")
    for ax in axes.ravel():
        if max_day is not None:
            ax.set_xlim(0.0, max_day)
        ax.axvspan(0.0, pulse_hours / cs.HOURS_PER_DAY, color="0.88", lw=0)
        ax.grid(True, color="0.9", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[1, 0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--textures", nargs="+", default=["loam"], choices=sorted(cs.TEXTURES),
                    help="sand is available but its kernels are not smooth inside rain-free "
                         "periods (finite differences cannot arbitrate there)")
    ap.add_argument("--biome", default=cs.DEFAULT_BIOME)
    ap.add_argument("--lai", type=float, default=cs.DEFAULT_LAI)
    ap.add_argument("--lat-deg", type=float, default=40.0, help="latitude (solar geometry)")
    ap.add_argument("--mean-annual-t-k", type=float, default=cs.DEFAULT_MEAN_ANNUAL_T_K,
                    help="mean annual air temperature of the column climate [K]")
    ap.add_argument("--seasonal-amp-k", type=float, default=cs.DEFAULT_T_SEASONAL_AMP_K,
                    help="seasonal half-amplitude of air temperature [K] (NH-phased)")
    ap.add_argument("--start-doy", type=float, default=152.0, help="pulse day (June 1)")
    ap.add_argument("--spinup-days", type=int, default=60)
    ap.add_argument("--window-days", type=int, default=45, help="rain-free days after the pulse")
    ap.add_argument("--dt-s", type=float, default=cs.DEFAULT_DT_S)
    ap.add_argument("--precip-rate", type=float, default=cs.DEFAULT_PRECIP_KG_M2_S,
                    help="steady spin-up rain [kg/m2/s]")
    ap.add_argument("--pulse-mm", type=float, default=20.0, help="rain in the pulse [mm]")
    ap.add_argument("--pulse-hours", type=float, default=6.0,
                    help="duration of the pulse at the start of the window [h]")
    ap.add_argument("--fd-days", type=float, nargs="*", default=[],
                    help="window days at which the transpiration and soil-evaporation kernels "
                         "are checked against finite differences, e.g. 0.1 10.5 (default: none)")
    ap.add_argument("--fd-rtol", type=float, default=1e-3)
    ap.add_argument("--no-remat", dest="remat", action="store_false",
                    help="do not checkpoint each step in the reverse pass")
    ap.add_argument("--out-dir", default="diagnostics/land_sensitivity/pulse_memory")
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    ap.add_argument("--plot-fates", nargs="+", default=list(PROCESSES), choices=FATES,
                    help="fates drawn in the top panel (default: transpiration, soil evaporation)")
    ap.add_argument("--plot-days", type=float, default=35.0,
                    help="show only the first N days of the window in both panels")
    ap.add_argument("--replot-from-saved", action="store_true",
                    help="no model run: rebuild the figure from pulse_memory.npz/.json "
                         "in --out-dir")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.replot_from_saved:
        results, saved = load_results(out)
        plot(results, out / "pulse_memory.png", pulse_mm=float(saved["args"]["pulse_mm"]),
             pulse_hours=float(saved["args"]["pulse_hours"]),
             kernel_fates=tuple(args.plot_fates), max_day=args.plot_days)
        print(f"wrote {out / 'pulse_memory.png'}")
        return 0
    results, checks, meta = run_experiment(args)
    arrays = {"day_of_window": meta["window_axis"].t_s / cs.SECONDS_PER_DAY,
              "precip_kg_m2_s": meta["precip"]}
    summary = {}
    for r in results:
        for fate in FATES:
            arrays[f"{r.texture}_{fate}_rain_kernel"] = r.kernels[fate].rain_kernel
            arrays[f"{r.texture}_{fate}_rain_kernel_daily"] = r.kernel_daily[fate]
            arrays[f"{r.texture}_{fate}_t_air_kernel"] = r.kernels[fate].t_air_kernel
        for proc in PROCESSES:
            arrays[f"{r.texture}_{proc}_base_daily_mm"] = r.base_daily[proc]
        summary[r.texture] = {"pulse_fate": r.pulse_fate, "window_value": r.window_value,
                              "partition_error": r.partition_error,
                              "budget_residual_kg_m2": r.budget_residual}
    np.savez(out / "pulse_memory.npz", **arrays)
    cs.write_json(out / "pulse_memory.json", cs.provenance(
        vars(args), timings=meta["timings"], t_init_k=meta["t_init_k"],
        n_pulse_steps=meta["n_pulse"], summary=summary,
        gradient_checks=([c._asdict() for c in checks] if checks else "skipped")))
    if checks:
        (out / "gradient_checks.txt").write_text(cs.format_checks(checks) + "\n")
        print(cs.format_checks(checks))
        cs.require_all_pass(checks)
    else:
        print("finite-difference checks skipped (no --fd-days): the kernels are checked only "
              "by the water-budget identity")
    require_partition(results)
    for r in results:
        print(f"\n[{r.texture}] budget residual {r.budget_residual:+.2e} kg/m2, "
              f"partition error {r.partition_error:.2e}")
        print("  fate of an extra mm of the pulse: " + ", ".join(
            f"{FATE_LABELS[f]} {r.pulse_fate[f]:.3f}" for f in FATES))
        idx = [d for d in REPORT_DAYS if d < r.day.shape[0]]
        print(f"  {'day of extra rain':24s}" + "".join(f"{d:>7d}" for d in idx))
        for fate in FATES:
            print(f"  {FATE_LABELS[fate]:24s}" + "".join(
                f"{r.kernel_daily[fate][d]:7.3f}" for d in idx))
    if args.plot:
        plot(results, out / "pulse_memory.png", pulse_mm=args.pulse_mm,
             pulse_hours=args.pulse_hours, kernel_fates=tuple(args.plot_fates),
             max_day=args.plot_days)
    print(f"timings: {', '.join(f'{k}={v:.1f}s' for k, v in meta['timings'].items())}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
