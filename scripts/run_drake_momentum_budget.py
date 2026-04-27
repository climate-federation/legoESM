#!/usr/bin/env python
"""Online momentum-tendency diagnostics for the Drake band.

Restarts from the 50yr GM/Redi endpoint and runs 1 sim-year (default),
calling ``model.tendencies_with_diagnostics(state)`` at each step to
capture per-term momentum tendencies.  Accumulates time means and
writes per-term 3D fields plus a band-summary npz.

Closure is guaranteed at the *per-step* level by
``test_momentum_diagnostics_closure.py``.  At the time-mean level the
RK-stage averaging introduces an O(dt) error which is negligible at
dt=600 s over a 1-yr run.

The path-3 barotropic-substep drag (`(1 − dt·r/H)` factor on U_bar in
``barotropic_latlon_cgrid.py:340–346``) is reconstructed post-hoc from
the time-mean U_baro: ``F_path3 = -ρ·r·U_baro``.

Outputs (results/ocean/momentum_budget_online/):
  - tendency_3d_means.npz: per-term time-mean tendencies, full 3D
  - tendency_band_summary.npz: per-term Drake-band-integrated stress (Pa)
  - momentum_budget_closure.png: bar chart of band-mean Pa per term

Usage:
    JAX_ENABLE_X64=1 python scripts/run_drake_momentum_budget.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.state import (
    LatLonCGridOceanConfig, MomentumTendencyDiagnostics,
)
from legoesm.ocean.experiments.global_overturning import (
    GlobalOverturningConfig, create_initial_conditions, create_forcings,
    create_eos_config, create_gm_redi_config,
)


RESTART_PATH = Path(
    "results/ocean/global_overturning_50yr_gmredi/restart_day018250.npz"
)
OUTPUT_DIR = Path("results/ocean/momentum_budget_online")
J_DRAKE = np.arange(2, 7)
RHO_0 = 1027.0


# --- Term names: u-side and v-side, excluding totals ---
DIAG_U_NAMES = [f for f in MomentumTendencyDiagnostics._fields
                if f.endswith("_u") and not f.startswith("total")]
DIAG_V_NAMES = [f for f in MomentumTendencyDiagnostics._fields
                if f.endswith("_v") and not f.startswith("total")]


def _restore_state(template, restart_path):
    npz = np.load(restart_path, allow_pickle=False)
    new = {}
    for f in template._fields:
        obj = getattr(template, f)
        if obj is None or not hasattr(obj, "data"):
            continue
        if f not in npz.files:
            raise KeyError(f"Restart missing field {f}")
        new[f] = obj.replace(data=jnp.asarray(npz[f], dtype=obj.data.dtype))
    return template._replace(**new), float(npz["time_days"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_years = 1.0
    dt = 600.0
    n_steps = int(total_years * 365.0 * 86400 / dt)
    diag_print_every = max(1, n_steps // 50)

    config = GlobalOverturningConfig(use_gm_redi=True)
    z_coord = create_ocean_z_star(
        n_levels=config.n_levels, H_max=config.H_max,
        dz_surface=config.dz_surface, dz_deep=config.dz_deep,
    )
    grid = create_latlon_grid(36, 72)
    physics = create_forcings("latlon", grid, config)
    eos_config = create_eos_config(config)
    gm_redi_cfg = create_gm_redi_config(config)
    ocean_config = LatLonCGridOceanConfig(
        n_barotropic_substeps=30,
        physics=physics,
        A_h=config.A_h, A_v=config.A_v, K_v=config.K_v,
        bottom_drag_r=config.bottom_drag_coeff,
        eos="linear", eos_linear=eos_config,
        gm_redi=gm_redi_cfg,
    )
    model = LatLonCGridOceanModel(grid, z_coord, ocean_config)

    template = create_initial_conditions("latlon", grid, z_coord, config)
    state, day_offset = _restore_state(template, RESTART_PATH)
    print(f"Restarted at sim day {day_offset:.0f} (year {day_offset/365:.2f})")
    print(f"Run length: {total_years} yr ({n_steps:,} steps at dt={dt}s)")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Initialize accumulators (one np.float64 array per term, full 3D).
    diag_sum = {name: None for name in DIAG_U_NAMES + DIAG_V_NAMES}
    sum_total_u = None
    sum_total_v = None
    state_sum = {
        "T": np.zeros_like(np.asarray(state.T.data), dtype=np.float64),
        "S": np.zeros_like(np.asarray(state.S.data), dtype=np.float64),
        "u": np.zeros_like(np.asarray(state.u.data), dtype=np.float64),
        "v": np.zeros_like(np.asarray(state.v.data), dtype=np.float64),
        "eta": np.zeros_like(np.asarray(state.eta.data), dtype=np.float64),
    }
    n_samples = 0

    print("Starting integration...")
    t0 = time.time()
    last_print = t0
    for i in range(n_steps):
        # 1. Diagnose tendencies at start of step (same state model uses).
        _, diag = model.tendencies_with_diagnostics(state, dt=dt)
        # 2. Accumulate each component (3D float64 sum).
        for name in DIAG_U_NAMES + DIAG_V_NAMES:
            arr = np.asarray(getattr(diag, name).data, dtype=np.float64)
            if diag_sum[name] is None:
                diag_sum[name] = arr.copy()
            else:
                diag_sum[name] += arr
        tu = np.asarray(diag.total_u.data, dtype=np.float64)
        tv = np.asarray(diag.total_v.data, dtype=np.float64)
        if sum_total_u is None:
            sum_total_u = tu.copy(); sum_total_v = tv.copy()
        else:
            sum_total_u += tu; sum_total_v += tv

        # 3. Accumulate state.
        state_sum["T"] += np.asarray(state.T.data)
        state_sum["S"] += np.asarray(state.S.data)
        state_sum["u"] += np.asarray(state.u.data)
        state_sum["v"] += np.asarray(state.v.data)
        state_sum["eta"] += np.asarray(state.eta.data)
        n_samples += 1

        # 4. Advance the actual model (uses fresh tendency internally).
        state = model.step(state, dt)

        if (i + 1) % diag_print_every == 0:
            now = time.time()
            if now - last_print > 30:
                yr = (i + 1) * dt / 86400.0 / 365.0
                elapsed = now - t0
                eta_s = elapsed / yr * total_years - elapsed if yr > 0 else 0
                print(f"  Year {yr:.3f}/{total_years} | "
                      f"step {i+1:,}/{n_steps:,} | "
                      f"ETA {eta_s/60:.1f} min", flush=True)
                last_print = now

    jax.block_until_ready(state.eta.data)
    wall = time.time() - t0
    print(f"\nIntegration done in {wall:.0f}s ({wall/60:.1f} min)")

    # Time means.
    tm = {name: diag_sum[name] / n_samples for name in diag_sum}
    tm_total_u = sum_total_u / n_samples
    tm_total_v = sum_total_v / n_samples
    tm_state = {k: state_sum[k] / n_samples for k in state_sum}

    # Closure verification
    sum_terms_u = sum(tm[n] for n in DIAG_U_NAMES)
    sum_terms_v = sum(tm[n] for n in DIAG_V_NAMES)
    err_u = float(np.max(np.abs(sum_terms_u - tm_total_u)))
    err_v = float(np.max(np.abs(sum_terms_v - tm_total_v)))
    norm_u = float(np.max(np.abs(tm_total_u)) + 1e-30)
    norm_v = float(np.max(np.abs(tm_total_v)) + 1e-30)
    print(f"\nClosure check (time-mean):")
    print(f"  ||Σ_u terms - total_u||_inf = {err_u:.3e}  "
          f"(rel {err_u/norm_u:.3e})")
    print(f"  ||Σ_v terms - total_v||_inf = {err_v:.3e}  "
          f"(rel {err_v/norm_v:.3e})")

    # Save 3D time means
    save_3d = {f"tend_{n}": tm[n] for n in tm}
    save_3d["tend_total_u"] = tm_total_u
    save_3d["tend_total_v"] = tm_total_v
    save_3d["state_T_mean"] = tm_state["T"]
    save_3d["state_S_mean"] = tm_state["S"]
    save_3d["state_u_mean"] = tm_state["u"]
    save_3d["state_v_mean"] = tm_state["v"]
    save_3d["state_eta_mean"] = tm_state["eta"]
    np.savez_compressed(OUTPUT_DIR / "tendency_3d_means.npz", **save_3d)
    print(f"  Saved {OUTPUT_DIR / 'tendency_3d_means.npz'}")

    # Compute Drake-band budget summary (band-mean equivalent stress, Pa).
    lat = np.asarray(grid.lat) * 180/np.pi
    dlon = float(grid.dlon)
    R = float(grid.radius)
    cos_lat = np.cos(np.clip(np.asarray(grid.lat), -np.pi/2 + 1e-9, np.pi/2 - 1e-9))
    dx_u = cos_lat * R * dlon
    dy = R * float(grid.dlat)
    dz = np.asarray(z_coord.dz_ref, dtype=np.float64)
    H_total = float(dz.sum())
    u_mask = np.asarray(state.u_mask.data, dtype=np.float64)

    # For each u-side term, integrate over band wet u-faces, full depth.
    # Force = ρ·tendency·dx·dy·dz; equivalent stress = force / band area.
    n_wet_u = np.sum(u_mask, axis=1)
    area_band = float(np.sum((dx_u * n_wet_u * dy)[J_DRAKE]))

    band_summary_Pa = {}
    for name in DIAG_U_NAMES + ["total_u"]:
        arr3d = save_3d[f"tend_{name}"] if name in DIAG_U_NAMES \
                else save_3d["tend_total_u"]
        # Force per row = ρ · Σ_i Σ_k arr3d[j,i,k] · dz[k] · u_mask[j,i] · dx_u[j] · dy
        force_per_row = (RHO_0 * np.sum(arr3d * u_mask[:, :, None] * dz[None, None, :],
                                         axis=(1, 2))
                         * dx_u * dy)
        band_summary_Pa[name] = float(np.sum(force_per_row[J_DRAKE])) / area_band

    # Path-3 contribution reconstructed from time-mean U_baro.
    u_mean = tm_state["u"]                                       # (n_lat, n_lon+1, nlev)
    U_baro_face = np.sum(u_mean * dz[None, None, :], axis=-1) / H_total  # (n_lat, n_lon+1)
    path3_per_face = -RHO_0 * config.bottom_drag_coeff * U_baro_face * u_mask
    force_path3 = np.sum(path3_per_face, axis=1) * dx_u * dy
    band_summary_Pa["bt_path3_drag_u"] = float(
        np.sum(force_path3[J_DRAKE])) / area_band

    # Write band summary
    np.savez_compressed(
        OUTPUT_DIR / "tendency_band_summary.npz",
        **{f"band_{k}_Pa": np.array(v) for k, v in band_summary_Pa.items()},
        area_band=np.array(area_band),
    )
    print(f"  Saved {OUTPUT_DIR / 'tendency_band_summary.npz'}")

    # Plot
    _plot_band_summary(band_summary_Pa)

    # Print summary
    print("\n=== Drake-band depth-and-zonally-integrated zonal-momentum budget ===")
    print(f"  band area: {area_band:.3e} m²")
    sum_baro = sum(v for k, v in band_summary_Pa.items() if k != "total_u")
    closure_resid = sum_baro - band_summary_Pa["total_u"] \
        - band_summary_Pa["bt_path3_drag_u"] + band_summary_Pa["bt_path3_drag_u"]
    # Σ baroclinic terms = total_u (by construction).  Adding path-3 gives the FULL.
    full = band_summary_Pa["total_u"] + band_summary_Pa["bt_path3_drag_u"]
    print(f"  {'term':<24} {'Pa':>10}")
    for name in DIAG_U_NAMES:
        print(f"  {name:<24} {band_summary_Pa[name]:+10.4f}")
    print(f"  {'-'*36}")
    print(f"  {'total_u (baroclinic)':<24} {band_summary_Pa['total_u']:+10.4f}"
          f"   (Σ baroclinic terms; closes to ~1e-12 by construction)")
    print(f"  {'bt_path3_drag_u (post-hoc)':<24} {band_summary_Pa['bt_path3_drag_u']:+10.4f}")
    print(f"  {'-'*36}")
    print(f"  {'FULL momentum source':<24} {full:+10.4f}"
          f"   (must be ~0 in steady state)")


def _plot_band_summary(band_summary_Pa):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = [(k, v) for k, v in band_summary_Pa.items() if k != "total_u"]
    items.sort(key=lambda kv: kv[1])

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = ["C3" if v < 0 else "C0" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Equivalent stress (Pa)")
    ax.set_title(
        "Drake-band depth-integrated zonal-momentum budget — online diagnostics\n"
        "(red = westward sink, blue = eastward source; sum should ≈ 0 in steady state)")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "momentum_budget_closure.png", dpi=150)
    plt.close()
    print(f"  Saved {OUTPUT_DIR / 'momentum_budget_closure.png'}")


if __name__ == "__main__":
    main()
