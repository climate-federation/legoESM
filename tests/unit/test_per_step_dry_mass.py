"""Per-step reference lane: DRY-mass convention (review 2026-10-10 F13
remainder).

Before: ``ModelDriver._run_per_step`` applied the physics water and the
saturation-adjustment rain-out at fixed ``p_s`` and the dycore's own end-step
fixer held the TOTAL ``p_s`` integral, so precipitated water stayed behind as
dry air.  Now the moisture update carries its mass (the compiled lane's
shared helpers) and a target-anchored DRY-mass fixer replaces the dycore's.

Run through the REAL per-step lane (cube C4 L5, x64), supersaturated start
(``rh_init`` 1.5) so the saturation adjustment rains on the first step:

* the global dry mass is unchanged (old code: grows by the rain);
* the global ``∫ p_s dA`` falls by exactly the water that left;
* the anchor persists as ``target_dry_mass``.

The default vertical coordinate is HYBRID, so the dry mass is measured on the
coordinate's true layer masses (codex r1: ``p_s * dsigma`` is not the layer
mass away from the reference pressure); a pure-sigma run checks the same.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from legoesm.core.conservation import global_area_sum, global_dry_mass
from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver

jax.config.update("jax_enable_x64", True)


def _masses(d):
    tr = {k: d.tracers[k] for k in ("q_v", "q_c", "q_r")}
    lay = d.sigma          # the run's TRUE (default hybrid) layer masses
    dry = float(global_dry_mass(d.state.p_s.data, tr, lay, d.grid))
    tot = float(global_area_sum(d.state.p_s.data, d.grid))
    return dry, tot


@pytest.mark.parametrize("grid_type,vcoord", [
    ("cubed_sphere", "hybrid"), ("latlon", "hybrid"), ("cubed_sphere", "sigma")])
def test_per_step_lane_keeps_dry_mass_and_rain_leaves_p_s(tmp_path, grid_type,
                                                          vcoord):
    if grid_type == "cubed_sphere":
        res, dyc = 4, DycoreConfig(energy_fixer=False, dt=1800.0, fix_mass=True)
    else:
        res, dyc = 8, DycoreConfig(energy_fixer=False, dt=600.0, fix_mass=True,
                                   discretization="latlon_cgrid")
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type=grid_type, resolution=res, nlev=5,
                        vertical_coord=vcoord),
        dycore=dyc,
        output=OutputConfig(output_dir="", diag_days=1, checkpoint_days=0),
        days=1, dataset="analytical", radiation="gray", rh_init=1.5,
        microphysics="none", convection="none", turbulence="none",
        precision="fp64",
    )
    d = ModelDriver(cfg, output_dir=str(tmp_path))
    d.setup()
    dry0, tot0 = _masses(d)
    assert d.run(compiled=False) == "COMPLETED"
    dry1, tot1 = _masses(d)

    # it rained (non-vacuity of the next two checks): the column mass fell
    assert (tot0 - tot1) / tot0 > 1e-6, (tot0 - tot1) / tot0
    # the dry mass is the anchor (old: kept total, dry grew by the rain)
    assert abs(dry1 - dry0) / (tot0 - tot1) < 1e-8, (dry1 - dry0)
    np.testing.assert_allclose(
        float(d._carry_aux["target_dry_mass"]), dry0, rtol=1e-14)
