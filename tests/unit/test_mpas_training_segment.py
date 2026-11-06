"""The WB MPAS training segment (build_mpas_training_segment).

Pinned: step k's physics sees the sample's own clock,
day_to_calendar(day_of_year + k*dt/86400); condensate starts at zero for an
ERA5 sample and is carried out; a physics cadence is refused.
Needs JAX_ENABLE_X64=1.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.driver.config import (  # noqa: E402
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
from legoesm.driver.model_driver import ModelDriver  # noqa: E402
from legoesm.forcing.time_utils import day_to_calendar  # noqa: E402
from legoesm.training.era5_to_state import MPASCarry  # noqa: E402
from legoesm.training.training_driver import (  # noqa: E402
    build_mpas_training_segment)

DT = 600.0


@pytest.fixture(scope="module")
def exp(tmp_path_factory):
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=2, nlev=4,
                        vertical_coord="sigma"),
        dycore=DycoreConfig(dt=DT, discretization="mpas"),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=0),
        days=1.0, start_day=0.0, dataset="analytical", radiation="gray",
        convection="none", turbulence="none", microphysics="kessler",
        precision="fp64", distributed=False)
    d = ModelDriver(cfg, output_dir=str(tmp_path_factory.mktemp("mpas_seg")))
    d.setup()
    return d.export_mpas_step()


def _ic(exp):
    s = exp["state"]
    return MPASCarry(u=s.u.data, T=s.T.data, p_s=s.p_s.data, phis=s.phis.data,
                     q_v=s.tracers["q_v"].data)


def test_clock_and_condensate(exp):
    from legoesm.core.state import HydrostaticTendencies
    seen = []

    def physics_fn(state, mesh, sigma, phys_state=None, forcing=None):
        jax.debug.callback(lambda d, s: seen.append((float(d), float(s))),
                           forcing["day_of_year"], forcing["seconds_of_day"])
        z = lambda f: f.replace(data=jnp.zeros_like(f.data))  # noqa: E731
        return HydrostaticTendencies(du_dt=z(state.u), dT_dt=z(state.T),
                                     dp_s_dt=z(state.p_s), dphis_dt=z(state.phis))

    seg = build_mpas_training_segment(exp, physics_fn)
    day0 = 183.5 + 300.0 / 86400.0             # Jul 2 12:05 UTC, 0-based
    out = seg.raw(_ic(exp), 3, {"day_of_year": jnp.asarray(day0)})
    jax.block_until_ready(out.T)
    want = [tuple(float(x) for x in day_to_calendar(day0 + k * DT / 86400.0))
            for k in range(3)]
    np.testing.assert_allclose(seen, want, rtol=0, atol=1e-6)
    assert seen[0][1] == pytest.approx(43500.0, abs=1e-6)   # its own hour
    # ERA5 sample: no condensate in, zero condensate carried out
    assert out.q_c is not None and out.q_r is not None
    assert float(jnp.abs(out.q_c).max()) == 0.0


def test_physics_cadence_is_refused(exp):
    with pytest.raises(ValueError, match="cadence"):
        build_mpas_training_segment(dict(exp, hold_phys=True), None)
