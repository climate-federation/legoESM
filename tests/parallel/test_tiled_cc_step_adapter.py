"""P4 increment 1: the cell-centre tiled-step adapter twins the serial step.

``make_tiled_cc_step`` wraps the tiled D-grid SSP-RK3 core with the SERIAL
step's own cc<->corner conversions.  With the driver-default config (all
post-step damps zero, fixers off — the compiled-segment contract), the
serial ``model.step`` degenerates to exactly the tiled core, so the adapter
must match it to RK3-reorder fp tolerance on 24 virtual devices.

Runs on host CPU devices (env set before jax import).
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=24")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationModel, CDGridPrimitiveEquationConfig,
)
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.atmosphere.dynamics.tiled_step_adapter import make_tiled_cc_step

N, NLEV, KT = 8, 4, 2
DT = 60.0


def _mesh():
    from legoesm.parallel.mesh import create_device_mesh
    if len(jax.devices()) < 6 * KT * KT:
        pytest.skip(f"needs {6*KT*KT} devices (XLA_FLAGS host device count)")
    return create_device_mesh(n_devices=6 * KT * KT).mesh


def _model_and_state():
    grid = create_cubed_sphere(N)
    coord = create_sigma_coordinate(NLEV)
    cfg = CDGridPrimitiveEquationConfig(
        use_conservation_fixer=False, fix_mass=False,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)
    state = held_suarez_init(grid, coord)
    return model, state


def test_adapter_matches_serial_default_config():
    mesh = _mesh()
    model, state = _model_and_state()

    ref = state
    for _ in range(3):
        ref = model.step(ref, DT)

    tiled_step = make_tiled_cc_step(model, mesh, kt=KT, dt=DT)
    out = state
    for _ in range(3):
        out = tiled_step(out)

    for name in ("u", "v", "T", "p_s"):
        a = np.asarray(getattr(out, name).data)
        b = np.asarray(getattr(ref, name).data)
        scale = max(1.0, float(np.max(np.abs(b))))
        assert np.max(np.abs(a - b)) / scale < 1e-9, (
            f"{name}: tiled cc adapter diverged from serial "
            f"(max rel {np.max(np.abs(a - b)) / scale:.2e})")

    # Non-vacuity: the step actually advanced the state.
    assert float(np.max(np.abs(
        np.asarray(out.u.data) - np.asarray(state.u.data)))) > 1e-8


@pytest.mark.parametrize("field,value,match", [
    ("damp_v", 0.02, "damp_v"),
    ("sponge_implicit", True, "implicit sponge"),
    ("implicit_grav_wave_damping", 0.5, "gravity-wave damping"),
])
def test_adapter_refuses_out_of_envelope(field, value, match):
    mesh = _mesh()
    model, _ = _model_and_state()
    model.config = model.config._replace(**{field: value})
    with pytest.raises(NotImplementedError, match=match):
        make_tiled_cc_step(model, mesh, kt=KT, dt=DT)
