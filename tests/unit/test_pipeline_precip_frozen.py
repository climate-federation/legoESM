"""The PhysicsPipeline publishes the FROZEN part of its surface precipitation
and the segment lanes carry it to the land / coupled surface (review
2026-10-10 F29 remainder; the MPAS lane's ``precip_frozen`` rule).

* ``PhysicsOutput.precip_frozen`` = microphysics frozen flux + convective
  frozen flux, each bounded by its own precipitation;
* the per-step and compiled lanes publish it as ``seg_precip_frozen`` for a
  coupled surface (the compiled lane time-integrates it like ``precip_accum``);
* the segment carry's lag / accumulator refuse physics that does not publish it.

The microphysics here is Kessler (liquid only, so its own frozen flux is 0)
wrapped to report a fixed frozen fraction of its precipitation: the published
flux must follow that fraction exactly, which no temperature re-split can.
"""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver

jax.config.update("jax_enable_x64", True)

FRAC = 0.25


def _driver(tmp_path):
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=5),
        dycore=DycoreConfig(energy_fixer=False, dt=1800.0),
        output=OutputConfig(output_dir="", diag_days=1, checkpoint_days=0),
        days=1, dataset="analytical", radiation="gray", rh_init=1.3,
        microphysics="kessler", convection="none", turbulence="none",
        precision="fp64",
    )
    d = ModelDriver(cfg, output_dir=str(tmp_path))
    d.setup()
    real = d.physics.micro_fn

    def _frozen_fraction(*a, **k):
        out = real(*a, **k)
        return out._replace(precip_frozen=FRAC * out.precipitation)

    d.physics.micro_fn = _frozen_fraction
    d._requires_surface_flux_export = True     # a coupled consumer
    return d


@pytest.mark.parametrize("compiled", [False, True])
def test_lanes_publish_the_physics_frozen_precip(tmp_path, compiled):
    d = _driver(tmp_path)
    assert d.run(compiled=compiled) == "COMPLETED"
    P = np.asarray(d._carry_aux["seg_precip"])
    F = np.asarray(d._carry_aux["seg_precip_frozen"])
    assert float(P.max()) > 0.0          # it rained (non-vacuity)
    np.testing.assert_allclose(F, FRAC * P, rtol=1e-10, atol=1e-20)


def test_frozen_precip_update_lag_accum_and_refusal():
    from legoesm.driver.compiled_segments import frozen_precip_update
    f = jnp.asarray([1.0, 2.0, 3.0])
    none = SimpleNamespace(precip_frozen_prev=None, precip_frozen_accum=None)
    assert frozen_precip_update(none, SimpleNamespace(), 10.0) == (None, None)
    carry = SimpleNamespace(precip_frozen_prev=jnp.zeros(3),
                            precip_frozen_accum=jnp.ones(3))
    prev, acc = frozen_precip_update(
        carry, SimpleNamespace(precip_frozen=f), 10.0)
    np.testing.assert_allclose(np.asarray(prev), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.asarray(acc), [11.0, 21.0, 31.0])
    prev, acc = frozen_precip_update(
        carry, SimpleNamespace(precip_frozen=f[:1]), 10.0, idx=jnp.asarray([2]))
    np.testing.assert_allclose(np.asarray(prev), [0.0, 0.0, 1.0])
    np.testing.assert_allclose(np.asarray(acc), [1.0, 1.0, 11.0])
    with pytest.raises(ValueError, match="precip_frozen"):
        frozen_precip_update(carry, SimpleNamespace(precip_frozen=None), 10.0)
