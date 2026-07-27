"""#1226 TWIN-MODE box heat-budget run: causally-clean version of
box_budget_run.py.

box_budget_run.py's from-rest 2-year budget cannot separate an OPERATOR
difference (lego's tendency term itself differs from NEMO's) from a STATE
difference (the two models have simply diverged onto different T/S/u/v by
year 2, so even an identical operator would give a different number). Every
per-process operator already matches NEMO to 0.99-1.0 correlation on a
MATCHED state (tracer_tendency_compare.py) -- so a from-rest-only budget is
ambiguous about causality.

This driver removes that ambiguity: it initializes legoESM directly from
NEMO's own developed day-180 restart (kamm_twin_90d.py's
``_build_twin_state``, ``--bridge-tke --bridge-before`` -- day-0 bit-identical
to NEMO on wet cells, non-rest, leap-frog-before-level populated), then runs
the box-heat-budget accumulator for 90 days. Because the two models start
IDENTICAL and diverge only gradually, a term anomaly present in the EARLY
window (days 0-10, states still nearly identical) is attributable to the
OPERATOR; an anomaly that only appears LATE (days 30-90, after the states
have separated) is state-mediated, not proof of an operator difference.

Usage
-----
    CUDA_VISIBLE_DEVICES=<gpu> JAX_ENABLE_X64=1 \\
        python box_budget_twin90.py <recipe> <out.npz> [--days 90] [--rows LO:HI]

Saves ``<out.npz>`` with the same per-band/per-term time series as
box_budget_run.py, PLUS the early(0-10d)/mid(10-30d)/late(30-90d) window
split, and ``<out>_terms.png``.
"""
import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from kamm_twin_90d import _build_twin_state, RUN_TRAJ, RUN_STEPDUMP, DT, STEPS_PER_DAY  # noqa: E402

from legoesm.ocean.experiments.dino import apply_dino_lat_lon_surface_forcing  # noqa: E402
from legoesm.ocean.fidelity.box_heat_budget import (  # noqa: E402
    TERM_NAMES,
    BoxHeatBudgetAccumulator,
)

DEPTH_BANDS_M = ((0.0, 200.0), (200.0, 1000.0), (1000.0, None))
# (label, day_lo, day_hi) -- day_hi exclusive; matches the coordinator's
# early/mid/late split (early = still-matched states -> operator signal;
# late = separated states -> state-mediated signal).
WINDOWS = (("early_0_10d", 0.0, 10.0), ("mid_10_30d", 10.0, 30.0), ("late_30_90d", 30.0, 90.0))
RHO0_NEMO, CP_NEMO = 1026.0, 3991.86795711963


def _parse_rows(spec: str) -> slice:
    lo, hi = spec.split(":")
    return slice(int(lo), int(hi))


def run(recipe: str, out_path: str, n_days: int, row_slice: slice,
        run_traj: str = RUN_TRAJ, run_stepdump: str = RUN_STEPDUMP):
    br, cfg, mc, model, forcing, sf, st = _build_twin_state(
        recipe, run_traj, run_stepdump, bridge_tke=True, bridge_before=True,
    )
    nsteps = STEPS_PER_DAY * n_days
    dyn = jax.jit(lambda st, t: model.step(st, DT, surface_forcing=sf, t_seconds=t))

    acc = BoxHeatBudgetAccumulator(
        br.geometry, br.z_coord, mc, cfg, forcing, DT, model,
        row_slice=row_slice, depth_bands_m=DEPTH_BANDS_M,
    )
    # Per-interval Joules for early/mid/late re-windowing: the accumulator
    # itself now stores these (acc.time_series_term_J / .time_series_residual_J,
    # #1226) -- no local re-derivation needed.

    t_seconds = 0.0
    acc.sample(st, dt_step=STEPS_PER_DAY * DT, t_seconds=t_seconds)
    for k in range(nsteps):
        st = apply_dino_lat_lon_surface_forcing(st, forcing, br.z_coord, cfg, DT,
                                                t_seconds=(k + 1) * DT)
        t_seconds = (k + 1) * DT
        st = dyn(st, jnp.asarray(t_seconds))
        if (k + 1) % STEPS_PER_DAY == 0:
            acc.sample(st, dt_step=STEPS_PER_DAY * DT, t_seconds=t_seconds)
            Td = np.asarray(st.T.data)
            m = np.asarray(st.land_mask.data) > 0.5
            if not np.isfinite(Td[m]).all():
                raise SystemExit("NaN/Inf encountered -- aborting")
            if (k + 1) % (STEPS_PER_DAY * 10) == 0:
                print(f"  day {(k+1)*DT/86400:5.1f}  T[{Td[m].min():.1f},{Td[m].max():.1f}]", flush=True)

    total_seconds = t_seconds
    summary = acc.summary(total_seconds)
    accum_W_series = acc.time_series_term_J
    residual_series = acc.time_series_residual_J
    conv_ratio = (RHO0_NEMO * CP_NEMO) / (acc.rho0 * acc.cp)

    # Day of each recorded interval (interval k closes at day k+1, since the
    # first .sample() at day 0 records no interval).
    interval_days = np.arange(1, len(accum_W_series["adv_h"][0]) + 1, dtype=float)

    print("\n" + "=" * 100)
    print(f"TWIN-MODE BOX HEAT BUDGET -- {recipe}, {n_days}d twin (NEMO day-180 restart, "
          f"bridge-tke+bridge-before), rows {row_slice}, box_area={summary['box_area_m2']:.4e} m^2")
    print("=" * 100)

    window_reports = {}
    for band_i, band in enumerate(DEPTH_BANDS_M):
        print(f"\nband {band} m:")
        window_reports[band] = {}
        for wname, lo, hi in WINDOWS:
            sel = (interval_days > lo) & (interval_days <= hi)
            n_sel = int(sel.sum())
            window_seconds = n_sel * STEPS_PER_DAY * DT
            print(f"  [{wname}] n_days={n_sel}:")
            row = {}
            for term in TERM_NAMES:
                J = float(np.sum(np.array(accum_W_series[term][band_i])[sel])) if n_sel else float("nan")
                w_m2 = J / window_seconds / summary["box_area_m2"] if n_sel else float("nan")
                row[term] = w_m2
                print(f"    {term:10s}  {w_m2:+9.3f} W/m^2  (NEMO-conv {w_m2 * conv_ratio:+9.3f})")
            resid_J = float(np.sum(np.array(residual_series[band_i])[sel])) if n_sel else float("nan")
            resid_w_m2 = resid_J / window_seconds / summary["box_area_m2"] if n_sel else float("nan")
            row["residual"] = resid_w_m2
            print(f"    {'residual':10s}  {resid_w_m2:+9.3f} W/m^2  (NEMO-conv {resid_w_m2 * conv_ratio:+9.3f})"
                  "  <- closure diagnostic, not a physical term")
            window_reports[band][wname] = row
        b = summary["terms"][band]
        print(f"  [full 90d] residual = {b['residual']['W_per_m2']:+.3e} W/m^2")

    print("\nCAVEAT: early(0-10d) is the causally-clean window (states still "
          "nearly identical to NEMO's day-180 restart) -- an anomaly present "
          "there is an OPERATOR signal. late(30-90d) reflects both the "
          "operator AND accumulated state divergence -- not operator-only.")
    print("VERTMIX is now DIRECTLY MEASURED (#1226 instrument fix: the "
          "realized implicit-mixing increment, not a residual). RESIDUAL "
          "(reported separately below, per band) is the closure diagnostic "
          "-- should be small relative to the other terms; large values "
          "indicate instrument error for that band/window, not physics.")

    np.savez(
        out_path,
        box_area_m2=summary["box_area_m2"],
        bands=np.array(DEPTH_BANDS_M, dtype=object),
        time_series_t=np.array(acc.time_series_t),
        interval_days=interval_days,
        **{f"H_band{i}": np.array(acc.time_series_H[i]) for i in range(len(DEPTH_BANDS_M))},
        **{
            f"{term}_band{i}_interval_J": np.array(accum_W_series[term][i])
            for term in TERM_NAMES for i in range(len(DEPTH_BANDS_M))
        },
        **{
            f"residual_band{i}_interval_J": np.array(residual_series[i])
            for i in range(len(DEPTH_BANDS_M))
        },
        **{
            f"{term}_band{i}_full90d_J": summary["terms"][band][term]["J"]
            for i, band in enumerate(DEPTH_BANDS_M) for term in TERM_NAMES
        },
        **{
            f"residual_band{i}_full90d_J": summary["terms"][band]["residual"]["J"]
            for i, band in enumerate(DEPTH_BANDS_M)
        },
        **{
            f"window_{wname}_band{i}_{term}_W_per_m2": window_reports[band][wname][term]
            for i, band in enumerate(DEPTH_BANDS_M)
            for wname, _, _ in WINDOWS for term in TERM_NAMES + ("residual",)
        },
    )
    print(f"\nSAVED {out_path}")

    plot_path = out_path.rsplit(".", 1)[0] + "_terms.png"
    _plot(window_reports, acc, recipe, n_days, row_slice, plot_path)
    return summary, window_reports


def _plot(window_reports, acc, recipe, n_days, row_slice, out_png):
    fig, axes = plt.subplots(1, len(DEPTH_BANDS_M), figsize=(5 * len(DEPTH_BANDS_M), 4.5), sharey=True)
    wnames = [w[0] for w in WINDOWS]
    x = np.arange(len(wnames))
    width = 0.8 / len(TERM_NAMES)
    for bi, band in enumerate(DEPTH_BANDS_M):
        ax = axes[bi] if len(DEPTH_BANDS_M) > 1 else axes
        for j, term in enumerate(TERM_NAMES):
            vals = [window_reports[band][w][term] for w in wnames]
            ax.bar(x + (j - len(TERM_NAMES) / 2 + 0.5) * width, vals, width=width,
                   label=term if bi == 0 else None)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(wnames, rotation=20, fontsize=8)
        ax.set_title(f"{band[0]}-{band[1] if band[1] is not None else 'bottom'} m")
        if bi == 0:
            ax.set_ylabel("W/m^2 (legoESM native)")
            ax.legend(fontsize=7)
    fig.suptitle(f"Twin-mode box budget by window\n{recipe}, {n_days}d, rows {row_slice}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"SAVED {out_png}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("recipe")
    p.add_argument("out", help="output .npz path")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--rows", default="12:47")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run(args.recipe, args.out, args.days, _parse_rows(args.rows))


if __name__ == "__main__":
    main()
