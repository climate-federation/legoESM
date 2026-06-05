#!/usr/bin/env python
"""Stage 3 verification run — implicit Crank–Nicolson barotropic solver.

Identical to ``run_drake_momentum_budget.py`` but with
``barotropic_solver = 'implicit_cn'`` instead of the explicit substep
loop.  Crit 1 acceptance for issue ``barotropic_mode_noise``.

Output: ``results/ocean/momentum_budget_online_implicit/``
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
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.experiments.global_overturning import (
    GlobalOverturningConfig, create_initial_conditions, create_forcings,
    create_eos_config, create_gm_redi_config,
)

from _drake_momentum_budget_runner import (
    restore_state_from_npz,
    run_diagnostic_loop,
)


RESTART_PATH = Path(
    "results/ocean/global_overturning_50yr_gmredi/restart_day018250.npz"
)
OUTPUT_DIR = Path("results/ocean/momentum_budget_online_implicit")


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
    ocean_config = LatLonCGridOceanConfig(
        n_barotropic_substeps=30,                 # unused by implicit solver
        physics=physics,
        A_h=config.A_h, A_v=config.A_v, K_v=config.K_v,
        bottom_drag_r=config.bottom_drag_coeff,
        eos="linear", eos_linear=eos_config,
        gm_redi=gm_redi_cfg,
        # ---- THE CHANGE ----
        barotropic_solver="implicit_cn",
    )
    print(f"barotropic_solver = {ocean_config.barotropic_solver}")
    print(f"theta_eta = {ocean_config.barotropic_implicit_theta_eta}")
    print(f"theta_pgf = {ocean_config.barotropic_implicit_theta_pgf}")
    model = LatLonCGridOceanModel(grid, z_coord, ocean_config)

    template = create_initial_conditions("latlon", grid, z_coord, config)
    state, day_offset = restore_state_from_npz(template, RESTART_PATH)
    print(f"Restarted at sim day {day_offset:.0f} (year {day_offset / 365:.2f})")
    print(f"Run length: {total_years} yr ({n_steps:,} steps at dt={dt}s)")
    print(f"Output: {OUTPUT_DIR}")

    run_diagnostic_loop(
        model, state, dt, n_steps, OUTPUT_DIR, label="implicit_cn",
    )


if __name__ == "__main__":
    main()
