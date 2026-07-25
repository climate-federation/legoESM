"""#1226 online box heat-budget run: WHICH term carries the +38.5 W/m^2
excess heat convergence into legoESM's DINO southern channel box?

A single winter twin snapshot (tracer_tendency_compare.py) cannot resolve
this -- the instantaneous term diffs are the wrong sign for the multi-year
residual. This driver runs the nemo_dino_kamm_mlf recipe from rest for a
multi-year window with legoesm.ocean.fidelity.box_heat_budget.
BoxHeatBudgetAccumulator sampling DAILY (32 steps at DT=2700s), then reports
the closure-validated per-term box-integrated heat budget (W/m^2, by depth
band) over the channel rows CH=slice(12,47) -- the SAME box convention
momentum_budget_diff.py / budget_{pointwise,fullframe}.py /
tracer_tendency_compare.py already use.

Diagnostics-only: the accumulator reads state, never mutates it -- the
model trajectory is IDENTICAL to kamm_run5y_v3.py's plain run (see
tests/ocean/unit/test_box_heat_budget.py::test_accumulator_does_not_alter_trajectory
for the mechanical proof on a small grid; this driver does not re-prove it
on the full DINO grid -- too expensive -- but the code path is identical).

Usage
-----
    CUDA_VISIBLE_DEVICES=<gpu> JAX_ENABLE_X64=1 \\
        python box_budget_run.py <n_years> <out.npz> [--rows LO:HI]

``n_years`` is an integer number of 360-day years (11520 steps/year at
DT=2700s). Saves ``<out.npz>`` with the per-band/per-term accumulated
Joules + W/m^2 + heat-content time series, and ``<out>_terms.png`` with the
per-term bar chart + accumulation time series.
"""
import argparse
import dataclasses
import time

import jax
import jax.numpy as jnp
import numpy as np
import netCDF4 as _nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import (
    apply_dino_lat_lon_surface_forcing,
    dino_config_for_recipe,
    dino_lat_lon_model_config,
    dino_lat_lon_state,
    dino_lat_lon_surface_forcing_arrays,
    dino_step_surface_forcing,
)
from legoesm.ocean.fidelity.box_heat_budget import (
    TERM_NAMES,
    BoxHeatBudgetAccumulator,
)
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo

RUN_TRAJ = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ"
DT = 2700.0
STEPS_PER_YEAR = 11520
STEPS_PER_DAY = 32  # 86400 / 2700
ROW_SLICE_DEFAULT = slice(12, 47)  # "channel rows" -- #1226/#1317 convention
DEPTH_BANDS_M = ((0.0, 200.0), (200.0, 1000.0), (1000.0, None))
# NEMO native convention (see heat_discriminator.py) for a NEMO-comparable W/m^2.
RHO0_NEMO, CP_NEMO = 1026.0, 3991.86795711963


def _parse_rows(spec: str) -> slice:
    lo, hi = spec.split(":")
    return slice(int(lo), int(hi))


def run(recipe: str, n_years: int, out_path: str, row_slice: slice):
    g = read_nemo_mesh_mask(f"{RUN_TRAJ}/mesh_mask.nc", nn_hls=0)
    s = read_nemo_restart(f"{RUN_TRAJ}/DINO_00000320_restart.nc", nn_hls=0)  # geometry donor
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
    cfg = dataclasses.replace(dino_config_for_recipe(recipe),
                              lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    st = dino_lat_lon_state(br.geometry, br.z_coord, cfg, land_mask_override=br.land_mask)

    # --- hard topo census gate (kamm_run5y_v3.py) ---
    mm = _nc.Dataset(f"{RUN_TRAJ}/mesh_mask.nc")
    tm = np.moveaxis(np.asarray(mm["tmask"][0]).squeeze(), 0, -1) > 0.5
    wet = np.asarray(br.z_coord.is_active) & (np.asarray(st.land_mask.data) > 0.5)[:, :, None]
    if not np.array_equal(wet, tm):
        raise SystemExit(f"TOPO CENSUS FAIL: {int(np.sum(wet != tm))} cells differ")
    print(f"census OK: {int(tm.sum())} wet cells")

    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    sf = dino_step_surface_forcing(forcing)
    print(f"slope_scheme={mc.gm_redi.slope_scheme if mc.gm_redi else None} "
          f"gm_bolus_advection={getattr(mc.gm_redi, 'gm_bolus_advection', None)}")

    n_steps = n_years * STEPS_PER_YEAR
    dyn = jax.jit(lambda st, t: model.step(st, DT, surface_forcing=sf, t_seconds=t))

    acc = BoxHeatBudgetAccumulator(
        br.geometry, br.z_coord, mc, cfg, forcing, DT, model,
        row_slice=row_slice, depth_bands_m=DEPTH_BANDS_M,
    )
    sample_dt = STEPS_PER_DAY * DT
    t0 = time.time()
    t_seconds = 0.0
    acc.sample(st, dt_step=sample_dt, t_seconds=t_seconds)
    for k in range(n_steps):
        st = apply_dino_lat_lon_surface_forcing(st, forcing, br.z_coord, cfg, DT,
                                                t_seconds=(k + 1) * DT)
        t_seconds = (k + 1) * DT
        st = dyn(st, jnp.asarray(t_seconds))
        if (k + 1) % STEPS_PER_DAY == 0:
            acc.sample(st, dt_step=sample_dt, t_seconds=t_seconds)
        if (k + 1) % (STEPS_PER_DAY * 30) == 0:
            Td = np.asarray(st.T.data)
            m = np.asarray(st.land_mask.data) > 0.5
            elapsed = time.time() - t0
            print(f"  day {(k+1)*DT/86400:6.1f}  T[{Td[m].min():.1f},{Td[m].max():.1f}] "
                  f"finite={np.isfinite(Td[m]).all()}  wall={elapsed:6.1f}s", flush=True)
            if not np.isfinite(Td[m]).all():
                raise SystemExit("NaN/Inf encountered -- aborting")

    total_seconds = t_seconds
    summary = acc.summary(total_seconds)
    # NEMO-comparable W/m^2: rho0*cp is a pure multiplicative convention on
    # every accumulated Joule -- rescale rather than re-run the accumulation.
    conv_ratio = (RHO0_NEMO * CP_NEMO) / (acc.rho0 * acc.cp)

    print("\n" + "=" * 90)
    print(f"BOX HEAT BUDGET -- {recipe}, {n_years}yr, rows {row_slice}, "
          f"box_area={summary['box_area_m2']:.4e} m^2, n_samples={summary['n_samples']}")
    print("CAVEAT: all 5 terms (incl. VERTMIX, #1226 direct-measurement fix) "
          "are endpoint-rate daily samples (box_heat_budget.py docstring); "
          "RESIDUAL is now the closure diagnostic (should be ~0; a nonzero "
          "value is instrument error, not an unattributed physical term).")
    print("=" * 90)
    for band in DEPTH_BANDS_M:
        b = summary["terms"][band]
        print(f"\nband {band} m:")
        for term in TERM_NAMES:
            w_m2 = b[term]["W_per_m2"]
            print(f"  {term:10s}  {w_m2:+9.3f} W/m^2  (NEMO-conv {w_m2 * conv_ratio:+9.3f})")
        print(f"  {'dH/dt':10s}  {b['dH_J']/total_seconds/summary['box_area_m2']:+9.3f} W/m^2")
        print(f"  closure_residual = {b['closure_residual_W_per_m2']:+.3e} W/m^2")

    np.savez(
        out_path,
        box_area_m2=summary["box_area_m2"],
        n_samples=summary["n_samples"],
        total_seconds=total_seconds,
        bands=np.array(DEPTH_BANDS_M, dtype=object),
        time_series_t=np.array(acc.time_series_t),
        **{f"H_band{i}": np.array(acc.time_series_H[i]) for i in range(len(DEPTH_BANDS_M))},
        **{
            f"{term}_band{i}_J": summary["terms"][band][term]["J"]
            for i, band in enumerate(DEPTH_BANDS_M) for term in TERM_NAMES
        },
        **{
            f"closure_residual_band{i}_J": summary["terms"][band]["closure_residual_J"]
            for i, band in enumerate(DEPTH_BANDS_M)
        },
    )
    print(f"\nSAVED {out_path}")

    plot_path = out_path.rsplit(".", 1)[0] + "_terms.png"
    _plot_terms(summary, acc, conv_ratio, recipe, n_years, row_slice, plot_path)
    return summary


def _plot_terms(summary, acc, conv_ratio, recipe, n_years, row_slice, out_png):
    """Per-term bar chart by depth band + heat-content accumulation time
    series (one panel per band). Diagnostics-only plot -- no numerics."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    x = np.arange(len(DEPTH_BANDS_M))
    width = 0.8 / len(TERM_NAMES)
    for j, term in enumerate(TERM_NAMES):
        vals = [summary["terms"][band][term]["W_per_m2"] for band in DEPTH_BANDS_M]
        ax.bar(x + (j - len(TERM_NAMES) / 2 + 0.5) * width, vals, width=width, label=term)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lo}-{hi if hi is not None else 'bottom'} m" for lo, hi in DEPTH_BANDS_M])
    ax.set_ylabel("W/m^2 (legoESM native rho0*cp)")
    ax.set_title(f"Per-term box heat budget\n{recipe}, {n_years}yr, rows {row_slice}")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    t_days = np.array(acc.time_series_t) / 86400.0
    for i, band in enumerate(DEPTH_BANDS_M):
        H = np.array(acc.time_series_H[i])
        ax2.plot(t_days, (H - H[0]) / 1e18, label=f"{band[0]}-{band[1] if band[1] is not None else 'bottom'} m")
    ax2.set_xlabel("day")
    ax2.set_ylabel("Delta heat content [EJ]")
    ax2.set_title("Box heat-content accumulation")
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"SAVED {out_png}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("n_years", type=int)
    p.add_argument("out", help="output .npz path")
    p.add_argument("--recipe", default="nemo_dino_kamm_mlf")
    p.add_argument("--rows", default="12:47", help="row_slice LO:HI (channel-rows convention)")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run(args.recipe, args.n_years, args.out, _parse_rows(args.rows))


if __name__ == "__main__":
    main()
