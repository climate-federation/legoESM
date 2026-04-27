#!/usr/bin/env python
"""Test (A): re-run online momentum budget with stronger gravity-wave damping.

Identical to ``run_drake_momentum_budget.py`` but turns on
``barotropic_div_damp = 0.1`` (was 0).  Tests the hypothesis that the
+0.10 Pa Coriolis × V_baro sink in our Drake-band budget is powered by
gravity-wave-induced Reynolds correlations, not a real mean meridional
circulation.

Output: ``results/ocean/momentum_budget_online_divdamp/``
"""

from __future__ import annotations

import os, sys, time
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
OUTPUT_DIR = Path("results/ocean/momentum_budget_online_divdamp")
J_DRAKE = np.arange(2, 7)

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
        if f not in npz.files: raise KeyError(f"Restart missing {f}")
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
        # ---- THE TEST KNOB ----
        barotropic_div_damp=0.1,        # was 0 in baseline
    )
    print(f"barotropic_div_damp = {ocean_config.barotropic_div_damp}")
    model = LatLonCGridOceanModel(grid, z_coord, ocean_config)

    template = create_initial_conditions("latlon", grid, z_coord, config)
    state, day_offset = _restore_state(template, RESTART_PATH)
    print(f"Restarted at sim day {day_offset:.0f} (year {day_offset/365:.2f})")
    print(f"Run length: {total_years} yr ({n_steps:,} steps at dt={dt}s)")
    print()

    diag_sum = {name: None for name in DIAG_U_NAMES + DIAG_V_NAMES}
    sum_total_u = None; sum_total_v = None
    state_sum = {k: np.zeros_like(np.asarray(getattr(state, k).data),
                                   dtype=np.float64)
                 for k in ["T", "S", "u", "v", "eta"]}
    n_samples = 0

    print("Starting integration...")
    t0 = time.time(); last_print = t0
    for i in range(n_steps):
        _, diag = model.tendencies_with_diagnostics(state, dt=dt)
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

        for k in state_sum:
            state_sum[k] += np.asarray(getattr(state, k).data)
        n_samples += 1

        state = model.step(state, dt)

        if (i+1) % diag_print_every == 0:
            now = time.time()
            if now - last_print > 30:
                yr = (i+1) * dt / 86400 / 365
                eta_s = (now - t0)/yr*total_years - (now - t0) if yr > 0 else 0
                print(f"  Year {yr:.3f}/{total_years} | "
                      f"step {i+1:,}/{n_steps:,} | "
                      f"ETA {eta_s/60:.1f} min", flush=True)
                last_print = now

    jax.block_until_ready(state.eta.data)
    wall = time.time() - t0
    print(f"\nDone in {wall:.0f}s ({wall/60:.1f} min)")

    tm = {name: diag_sum[name]/n_samples for name in diag_sum}
    tm_total_u = sum_total_u/n_samples
    tm_total_v = sum_total_v/n_samples
    tm_state = {k: state_sum[k]/n_samples for k in state_sum}

    save = {f"tend_{n}": tm[n] for n in tm}
    save["tend_total_u"] = tm_total_u
    save["tend_total_v"] = tm_total_v
    for k in tm_state:
        save[f"state_{k}_mean"] = tm_state[k]
    np.savez_compressed(OUTPUT_DIR / "tendency_3d_means.npz", **save)
    print(f"Saved {OUTPUT_DIR / 'tendency_3d_means.npz'}")


if __name__ == "__main__":
    main()
