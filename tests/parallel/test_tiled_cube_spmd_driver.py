"""ModelDriver hookup of the BLOCKED tiled cube SPMD lane
(``_run_tiled_cube_spmd`` + ``_tiled_cube_column_physics_fn``).

Glue-level end-to-end via the lightweight driver-stub pattern
(test_atm_latlon_spmd_driver precedent): the run method reads only
config / model / sigma / state / _device_config / _segment_callback, so no
data-loading ModelDriver construction is needed.  The lane's NUMERICS are
covered by the gated blocked-loop tests (test_tiled_blocked_loop) — here we
validate the production glue: mesh/kt validation, physics selection,
segment cadence + callback, self.state gather, status, refusals.

Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=24``.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=24")

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from legoesm.driver.config import ExperimentConfig
from legoesm.driver.model_driver import ModelDriver
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationModel, CDGridPrimitiveEquationConfig,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import set_halo_backend
from legoesm.grids.vertical import create_sigma_coordinate

N, NLEV, KT = 8, 4, 2
N_DEV = 6 * KT * KT
DT = 60.0


class _DriverStub:
    """Minimal stand-in exposing exactly what the tiled cube lane reads,
    with the real ModelDriver methods bound (test_atm_latlon_spmd_driver
    pattern)."""
    _tiled_cube_column_physics_fn = ModelDriver._tiled_cube_column_physics_fn
    _tiled_cube_unified_active = ModelDriver._tiled_cube_unified_active
    _run_tiled_cube_spmd = ModelDriver._run_tiled_cube_spmd
    _save_lightweight_timeseries = ModelDriver._save_lightweight_timeseries

    def __init__(self, model, state, cfg, device_config, tracers=None):
        self.config = cfg
        self.model = model
        self.grid = model.grid
        self.sigma = model.sigma_coord
        self.state = state
        self.tracers = dict(tracers or {})   # raw-array property store
        self._device_config = device_config
        self._segment_callback = None
        self._current_day = 0.0


def _exp_cfg(n_steps, **over):
    cfg = ExperimentConfig()
    fields = dict(
        days=n_steps * DT / 86400.0,
        radiation="none", convection="none", turbulence="none",
        microphysics="none", gravity_wave_drag="none", cloud_scheme="none",
        grid=cfg.grid._replace(grid_type="cubed_sphere"),
        dycore=cfg.dycore._replace(dt=DT),
        output=cfg.output._replace(diag_days=0, checkpoint_days=0),
    )
    fields.update(over)
    return cfg._replace(**fields)


def _model_and_state():
    grid = create_cubed_sphere(N)
    coord = create_sigma_coordinate(NLEV)
    mcfg = CDGridPrimitiveEquationConfig(
        use_conservation_fixer=True, fix_mass=True,
        anchor_mass_to_initial=False, zero_mean_ps_tendency=True)
    model = CDGridPrimitiveEquationModel(grid, coord, mcfg)
    return model, held_suarez_init(grid, coord)


def _device_config():
    from legoesm.parallel.mesh import create_device_mesh
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs {N_DEV} devices "
                    f"(XLA_FLAGS=--xla_force_host_platform_device_count=24)")
    set_halo_backend("local")
    return create_device_mesh(n_devices=N_DEV,
                              devices=jax.devices()[:N_DEV])


# ---------------------------------------------------------------------------
# Physics selector (pure config logic — no devices)
# ---------------------------------------------------------------------------

class _CfgStub:
    def __init__(self, cfg, sigma=None):
        self.config = cfg
        self.sigma = sigma


def test_physics_fn_dynamics_only():
    fn = ModelDriver._tiled_cube_column_physics_fn(_CfgStub(_exp_cfg(3)))
    assert fn is None


def test_physics_fn_kessler_alone_builds_bridge():
    coord = create_sigma_coordinate(NLEV)
    fn = ModelDriver._tiled_cube_column_physics_fn(
        _CfgStub(_exp_cfg(3, microphysics="kessler"), sigma=coord))
    # The bridge honours the column contract on a tiny tile.
    import jax.numpy as jnp
    T = jnp.full((1, 2, 2, NLEV), 280.0)
    p_s = jnp.full((1, 2, 2), 1.0e5)
    q = jnp.full((1, 2, 2, NLEV), 5.0e-3)
    out = fn(T, p_s, q, 0.1 * q, 0.01 * q)
    assert len(out) == 4 and all(o.shape == T.shape for o in out)


def test_physics_fn_rejects_unified_physics_and_hs():
    # Unified physics now DISPATCHES to the operator-split lane before the
    # selector runs; calling the selector directly hits its dispatch-
    # disagreement bug guard (unified configs must never reach it).
    with pytest.raises(NotImplementedError, match="dispatch"):
        ModelDriver._tiled_cube_column_physics_fn(
            _CfgStub(_exp_cfg(3, radiation="gray")))
    with pytest.raises(NotImplementedError, match="held_suarez"):
        ModelDriver._tiled_cube_column_physics_fn(
            _CfgStub(_exp_cfg(3, held_suarez_forcing=True)))


# ---------------------------------------------------------------------------
# np24 end-to-end glue
# ---------------------------------------------------------------------------

def test_run_tiled_cube_spmd_completes_and_matches_direct_loop():
    dc = _device_config()
    model, hs = _model_and_state()
    n_steps = 3
    stub = _DriverStub(model, hs, _exp_cfg(n_steps), dc)
    seen = []
    stub._segment_callback = lambda drv, day, dt_seg: seen.append(
        (day, dt_seg))

    status = stub._run_tiled_cube_spmd()
    assert status == "COMPLETED"
    # n_steps < one-day segment -> exactly ONE (short) segment.
    assert len(seen) == 1
    day, dt_seg = seen[0]
    assert dt_seg == pytest.approx(n_steps * DT)
    assert day == pytest.approx(stub.config.start_day
                                + n_steps * DT / 86400.0)

    # Glue parity: the lane's gathered state == the direct SCANNED segment
    # (same builders, same step count) — exact equality, same code path
    # (M3b increment 1: the lane advances each segment via ONE compiled
    # lax.scan, scan_tiled_cc_steps; scan-vs-per-step numerics parity is
    # gated in tests/parallel/test_cube_tile_native_segment.py).
    from legoesm.atmosphere.dynamics.gcm.tiled_step_adapter import (
        make_tiled_cc_loop, scan_tiled_cc_steps,
    )
    enter, step, exit_ = make_tiled_cc_loop(model, dc.mesh, kt=KT, dt=DT)
    blk = scan_tiled_cc_steps(step, n_steps)(enter(hs))
    ref = exit_(blk, hs)
    np.testing.assert_array_equal(np.asarray(stub.state.T.data),
                                  np.asarray(ref.T.data))
    np.testing.assert_array_equal(np.asarray(stub.state.u.data),
                                  np.asarray(ref.u.data))
    np.testing.assert_array_equal(np.asarray(stub.state.p_s.data),
                                  np.asarray(ref.p_s.data))


def test_run_tiled_cube_spmd_uneven_tail_segments(tmp_path):
    """5 steps at a 2-step segment cadence -> segments 2+2+1: exercises
    the per-length ``_scanned`` cache REUSE (the second 2-step segment,
    donated carry from the first), the distinct-length TAIL executable
    (``_scanned(1)``), and donation across differently-compiled segment
    functions (codex M3b review: no prior test drove the tail/cache
    path)."""
    dc = _device_config()
    model, hs = _model_and_state()
    n_steps = 5
    cfg = _exp_cfg(n_steps)
    cfg = cfg._replace(output=cfg.output._replace(
        diag_days=2 * DT / 86400.0))   # segment gcd -> 2 steps
    stub = _DriverStub(model, hs, cfg, dc)
    stub._output_dir = tmp_path
    seen = []
    stub._segment_callback = lambda drv, day, dt_seg: seen.append(dt_seg)

    status = stub._run_tiled_cube_spmd()
    assert status == "COMPLETED"
    # 2 + 2 + 1 steps -> three segment callbacks with those dt lengths.
    assert seen == [pytest.approx(2 * DT), pytest.approx(2 * DT),
                    pytest.approx(1 * DT)]
    # The gathered state is current + finite after the tail segment.
    assert np.all(np.isfinite(np.asarray(stub.state.T.data)))
    assert stub._current_day == pytest.approx(
        stub.config.start_day + n_steps * DT / 86400.0)


def test_run_tiled_cube_spmd_moist_from_driver_tracer_store():
    """The REAL driver init shape: tracers live in ``self.tracers`` (raw
    arrays), ``state.tracers is None``.  The Kessler lane must attach the
    Field tracers itself (codex BLOCKER), advance them, and keep the
    canonical ``self.tracers`` store in sync after the segment gather
    (codex MAJOR)."""
    import jax.numpy as jnp

    dc = _device_config()
    model, hs = _model_and_state()
    assert getattr(hs, "tracers", None) is None
    rng = np.random.default_rng(5)
    q0 = {nm: jnp.asarray(np.abs(
              s + 1e-4 * rng.standard_normal((6, N, N, NLEV))))
          for nm, s in zip(("q_v", "q_c", "q_r"), (5e-3, 5e-4, 5e-5))}
    stub = _DriverStub(model, hs, _exp_cfg(3, microphysics="kessler"),
                       dc, tracers=q0)

    status = stub._run_tiled_cube_spmd()
    assert status == "COMPLETED"
    # State now carries advanced tracer Fields...
    assert set(stub.state.tracers) == {"q_v", "q_c", "q_r"}
    # ...and the property store was synced to the ADVANCED fields (not the
    # initial arrays) — dynamics advection guarantees they moved.
    for nm in ("q_v", "q_c", "q_r"):
        np.testing.assert_array_equal(
            np.asarray(stub.tracers[nm]),
            np.asarray(stub.state.tracers[nm].data))
    assert float(np.max(np.abs(
        np.asarray(stub.tracers["q_v"]) - np.asarray(q0["q_v"])))) > 0.0
    assert float(np.min(np.asarray(stub.tracers["q_v"]))) >= 0.0


def test_run_tiled_cube_spmd_moist_missing_tracers_refuses():
    dc = _device_config()
    model, hs = _model_and_state()
    stub = _DriverStub(model, hs, _exp_cfg(3, microphysics="kessler"), dc)
    with pytest.raises(NotImplementedError, match="missing"):
        stub._run_tiled_cube_spmd()


def test_run_tiled_cube_spmd_dispatches_unified_to_operator_split():
    dc = _device_config()
    model, hs = _model_and_state()
    calls = []

    class _Stub(_DriverStub):
        def _run_operator_split_tiled_cube(self, start_step, start_day,
                                           mesh, kt):
            calls.append((mesh.devices.shape, kt))
            return "COMPLETED"

    stub = _Stub(model, hs, _exp_cfg(3, radiation="gray"), dc)
    assert stub._run_tiled_cube_spmd() == "COMPLETED"
    assert calls == [((6, KT, KT), KT)]


def test_run_tiled_cube_spmd_writers(tmp_path):
    """Diagnostics + checkpoint writers fire on segment boundaries.

    4 steps at DT with diag every 2 steps and checkpoint every 4: the
    segment gcd drops to 2 steps, the lightweight ``timeseries.npz``
    collects 2 samples, and the checkpoint hook (run()'s
    ``_checkpoint_callback``-or-``save_checkpoint`` contract) fires once
    at the absolute step 4 boundary."""
    dc = _device_config()
    model, hs = _model_and_state()
    n_steps = 4
    cfg = _exp_cfg(n_steps)
    cfg = cfg._replace(output=cfg.output._replace(
        diag_days=2 * DT / 86400.0, checkpoint_days=4 * DT / 86400.0))
    stub = _DriverStub(model, hs, cfg, dc)
    stub._output_dir = tmp_path
    ckpts = []
    stub._checkpoint_callback = lambda step, day: ckpts.append((step, day))

    status = stub._run_tiled_cube_spmd()
    assert status == "COMPLETED"

    ts = np.load(tmp_path / "timeseries.npz")
    np.testing.assert_allclose(
        np.asarray(ts["days"]),
        [2 * DT / 86400.0, 4 * DT / 86400.0], rtol=1e-9)
    for key in ("T_atm", "max_wind", "dry_mass_ps"):
        assert np.all(np.isfinite(np.asarray(ts[key]))), key
    assert (tmp_path / "results.txt").exists()
    assert len(ckpts) == 1
    step_at, day_at = ckpts[0]
    assert step_at == 4
    assert day_at == pytest.approx(cfg.start_day + 4 * DT / 86400.0)


def test_run_tiled_cube_spmd_checkpoint_falls_back_to_save_checkpoint():
    """No ``_checkpoint_callback`` set -> the lane invokes the driver's own
    ``save_checkpoint(step, day)`` on the gathered state (run()'s
    ``callback-or-save_checkpoint`` contract).  The writer itself is the
    SHARED driver method every run path uses — its serialization is
    covered by the driver's own restart tests; here we pin the lane's
    dispatch + cadence."""
    dc = _device_config()
    model, hs = _model_and_state()
    n_steps = 2
    cfg = _exp_cfg(n_steps)
    cfg = cfg._replace(output=cfg.output._replace(
        checkpoint_days=2 * DT / 86400.0))

    calls = []

    class _CkptStub(_DriverStub):
        def save_checkpoint(self, step, day):
            # The gathered cc state must exist and be current when the
            # writer fires (never the blocked in-loop state).
            assert self.state.T.data.shape == (6, N, N, NLEV)
            calls.append((step, day))

    stub = _CkptStub(model, hs, cfg, dc)
    status = stub._run_tiled_cube_spmd()
    assert status == "COMPLETED"
    assert calls == [(2, pytest.approx(cfg.start_day + 2 * DT / 86400.0))]
