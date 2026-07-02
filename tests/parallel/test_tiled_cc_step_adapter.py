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
    # RAW (6, kt, kt) mesh with the factory's expected axis names, and the
    # LOCAL halo backend pinned — the same setup the stage gate
    # (test_tiled_fv3_hydrostatic_step) uses.  create_device_mesh(24) is
    # NOT equivalent: it arms global device-config state that reroutes the
    # serial reference's halo pads (measured: cc-angle x corner-wind shape
    # clash in pad_halo_vector_4d, job 8688245).
    import numpy as _np
    from jax.sharding import Mesh
    from legoesm.grids.halo import set_halo_backend
    if len(jax.devices()) < 6 * KT * KT:
        pytest.skip(f"needs {6*KT*KT} devices (XLA_FLAGS host device count)")
    set_halo_backend("local")
    dev = _np.array(jax.devices()[: 6 * KT * KT]).reshape(6, KT, KT)
    return Mesh(dev, axis_names=("face", "tile_i", "tile_j"))


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

    # f32-honest parity bounds.  Forensics (jobs 8748753/8748974/8749684/
    # 8758884): the tendency cores are BIT-IDENTICAL serial-vs-tiled (on
    # both the initial and an evolved state), the SSP-RK3 formulas match,
    # and the remaining per-step differences are EXACT float32 ulps of the
    # field scales (T: 1 ulp = 2^-15 at ~250 K; p_s: 2 ulps at ~1e5 Pa)
    # arising from accumulation-order rounding under the f32 storage
    # policy, with the wind fields responding through the PGF (~1e-5 abs).
    # These are rounding-point equivalences, not physics.  Bounds = ~10x
    # the measured 3-step drift; a REAL term regression (e.g. the sponge,
    # measured 5.6e-6 in u before it was added) exceeds them.
    _atol = {"u": 5e-5, "v": 5e-5, "T": 3e-4, "p_s": 0.2}
    for name in ("u", "v", "T", "p_s"):
        a = np.asarray(getattr(out, name).data)
        b = np.asarray(getattr(ref, name).data)
        assert np.max(np.abs(a - b)) < _atol[name], (
            f"{name}: tiled cc adapter diverged from serial "
            f"(max abs {np.max(np.abs(a - b)):.2e} > {_atol[name]:.0e})")

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


def test_dedup_tiled_corners_synthetic():
    """Block-concatenated tiled corners (with the duplicated shared face)
    reassemble to the exact global corner array."""
    from legoesm.atmosphere.dynamics.tiled_step_adapter import (
        dedup_tiled_corners,
    )
    kt, nl = 2, 4
    n = kt * nl
    g = jnp.arange(6 * (n + 1) * (n + 1) * 2, dtype=jnp.float64).reshape(
        6, n + 1, n + 1, 2)
    blk = nl + 1
    # Build the tiled layout: tile (ti,tj) = global [ti*nl:ti*nl+blk] block.
    rows = jnp.concatenate(
        [g[:, ti * nl: ti * nl + blk] for ti in range(kt)], axis=1)
    tiled = jnp.concatenate(
        [rows[:, :, tj * nl: tj * nl + blk] for tj in range(kt)], axis=2)
    assert tiled.shape == (6, kt * blk, kt * blk, 2)
    out = dedup_tiled_corners(tiled, kt, nl)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(g))
