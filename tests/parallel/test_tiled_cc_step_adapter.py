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
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationModel, CDGridPrimitiveEquationConfig,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.atmosphere.dynamics.gcm.tiled_step_adapter import make_tiled_cc_step

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
    # These are rounding-point equivalences, not physics.  Bounds = ~2-3x
    # the measured 3-step rounding drift.  HONEST catch envelope: a term
    # regression at the 2e-5-abs class (3 steps) fails here; FINER term
    # regressions (the sponge class measured 5.6e-6 @3 steps) are below
    # this gate's floor and are covered instead by the envelope refusals
    # in make_tiled_cc_step + the bit-identity stage gates
    # (test_tiled_fv3_hydrostatic_step) which run above f32 rounding.
    _atol = {"u": 2e-5, "v": 2e-5, "T": 1e-4, "p_s": 0.06}
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
    ("time_integrator", "ssp_rk54", "time_integrator"),
    ("div_damp_coeff", 1e6, "divergence damping"),
    ("hyperdiff_coeff", 1e15, "hyperdiffusion"),
])
def test_adapter_refuses_out_of_envelope(field, value, match):
    mesh = _mesh()
    model, _ = _model_and_state()
    model.config = model.config._replace(**{field: value})
    with pytest.raises(NotImplementedError, match=match):
        make_tiled_cc_step(model, mesh, kt=KT, dt=DT)


def test_adapter_refuses_inner_mass_fixer():
    """The DEFAULT config (use_conservation_fixer=True, fix_mass=True)
    must refuse loudly: the segment driver externalizes the fixer; a
    direct default-config caller would silently lose the per-step
    fix_ps_mass (codex round-13 HIGH #1)."""
    mesh = _mesh()
    model, _ = _model_and_state()
    model.config = model.config._replace(
        use_conservation_fixer=True, fix_mass=True)
    with pytest.raises(NotImplementedError, match="mass fixer"):
        make_tiled_cc_step(model, mesh, kt=KT, dt=DT)


def test_single_shot_step_refuses_tracers():
    """The single-shot cc step is dynamics-only: a tracer-carrying state
    must refuse LOUDLY — silently re-attaching the tracers unchanged would
    freeze them while serial advances/floors them (divergence-by-omission,
    mirroring make_tiled_cc_loop's dry refusal).  An EMPTY tracer dict is
    equivalent to no tracers and must still step."""
    mesh = _mesh()
    model, state = _model_and_state()
    step = make_tiled_cc_step(model, mesh, kt=KT, dt=DT)

    moist = state._replace(tracers={
        "q_v": state.T.replace(data=jnp.zeros_like(state.T.data),
                               name="q_v"),
    })
    with pytest.raises(ValueError, match="dynamics-only"):
        step(moist)

    # No-tracer states step fine (the full-parity gate above exercises
    # tracers=None end-to-end); pin the empty-dict case explicitly.
    out = step(state._replace(tracers={}))
    assert np.all(np.isfinite(np.asarray(out.T.data)))


def test_dedup_tiled_corners_synthetic():
    """Block-concatenated tiled corners (with the duplicated shared face)
    reassemble to the exact global corner array."""
    from legoesm.atmosphere.dynamics.gcm.tiled_step_adapter import (
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
