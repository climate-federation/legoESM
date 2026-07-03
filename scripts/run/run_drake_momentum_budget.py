#!/usr/bin/env python
"""Online momentum-tendency diagnostics for the Drake band — baseline run.

Restarts from the 50yr GM/Redi endpoint and runs 1 sim-year, calling
``model.tendencies_with_diagnostics(state)`` at each step to capture
per-term momentum tendencies.  Accumulates time means and writes
per-term 3D fields plus a Drake-band summary npz and a bar-chart plot.

The inner stepping loop is JIT-compiled in blocks via
``scripts._drake_momentum_budget_runner.run_diagnostic_loop`` (see that
module for design notes).  Closure is guaranteed at the per-step level
by ``test_momentum_diagnostics_closure.py``; the time-mean closure is
verified at the end of the run.

Outputs (results/ocean/momentum_budget_online/):
  - tendency_3d_means.npz: per-term time-mean tendencies (full 3D)
  - tendency_band_summary.npz: per-term Drake-band stress (Pa)
  - momentum_budget_closure.png: bar chart of band-mean Pa per term

Usage:
    JAX_ENABLE_X64=1 python scripts/run_drake_momentum_budget.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("JAX_ENABLE_X64", "1")

from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.global_overturning import (
    GlobalOverturningConfig, create_initial_conditions, create_forcings,
    create_eos_config, create_gm_redi_config, global_overturning_model_config,
)

from _drake_momentum_budget_runner import (
    restore_state_from_npz,
    run_diagnostic_loop,
    compute_and_save_band_summary,
)


RESTART_PATH = Path(
    "results/ocean/global_overturning_50yr_gmredi/restart_day018250.npz"
)
OUTPUT_DIR = Path("results/ocean/momentum_budget_online")


def main():
    total_years = 1.0
    dt = 600.0
    n_steps = int(total_years * 365.0 * 86400 / dt)

    config = GlobalOverturningConfig(use_gm_redi=True)
    z_coord = create_ocean_z_star(
        n_levels=config.n_levels, H_max=config.H_max,
        dz_surface=config.dz_surface, dz_deep=config.dz_deep,
    )
    grid = create_latlon_grid(36, 72)
    physics = create_forcings("latlon", grid, config)
    eos_config = create_eos_config(config)
    gm_redi_cfg = create_gm_redi_config(config)
    ocean_config = global_overturning_model_config(
        config, physics=physics, eos_config=eos_config, gm_redi_cfg=gm_redi_cfg,
    )
    model = LatLonCGridOceanModel(grid, z_coord, ocean_config)

    template = create_initial_conditions("latlon", grid, z_coord, config)
    state, day_offset = restore_state_from_npz(template, RESTART_PATH)
    print(f"Restarted at sim day {day_offset:.0f} (year {day_offset / 365:.2f})")
    print(f"Run length: {total_years} yr ({n_steps:,} steps at dt={dt}s)")
    print(f"Output: {OUTPUT_DIR}")

    result = run_diagnostic_loop(
        model, state, dt, n_steps, OUTPUT_DIR, label="baseline",
    )

    # Drake-band summary + plot (baseline only).
    compute_and_save_band_summary(
        OUTPUT_DIR,
        save_3d=result["save_3d"],
        state_means=result["state_means"],
        bottom_drag_coeff=config.bottom_drag_coeff,
        grid=grid,
        z_coord=z_coord,
        final_state=result["final_state"],
    )


if __name__ == "__main__":
    main()
