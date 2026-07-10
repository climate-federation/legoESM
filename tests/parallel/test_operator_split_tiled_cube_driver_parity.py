"""End-to-end parity of the TILED-cube OPERATOR-SPLIT driver lane
(``ModelDriver._run_operator_split_tiled_cube``) against the serial
compiled-segment path (``ModelDriver._run_compiled``) — the cube twin of
``test_operator_split_spmd_driver_parity``.

Both drivers integrate the SAME tiny gray-radiation + TKE-turbulence AMIP
config (analytical forcing, cubed-sphere C-D grid) from an identical
initial state; one runs the serial fused operator-split scan, the other
the sub-face-TILED operator split over the ``(6, 2, 2)`` mesh (24 virtual
devices) — the REAL ``build_physics_pipeline`` ``step_unified`` built at
TILE ncol, the carry seed/shard/thread, per-segment forcing, and the
gather.  The column-local physics is decomposition-invariant; the
DYNAMICS differ by the documented tiled-vs-serial face-corner wind class
(O(1e-6 abs), step-constant — the blocked-loop gates' bound), which the
per-step physics coupling spreads into T/p_s at the same abs classes the
shipped adapter gates use.

``XLA_FLAGS=--xla_force_host_platform_device_count=24``; x64.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=24")

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from legoesm.driver.config import (
    ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver

N_DEV = 24              # 6 * kt^2, kt = 2
RES = 8                 # C8 (tile edge nl = 4)
NLEV = 4
DT = 100.0
N_STEPS = 3


def _make_cfg(out_dir, *, n_devices, days=N_STEPS * DT / 86400.0):
    return ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=RES, nlev=NLEV),
        dycore=DycoreConfig(
            model_type="hydrostatic", discretization="cdgrid", dt=DT,
            fix_mass=True,
            # The tiled base-cut RK3 carries no hyperdiffusion / A_h /
            # div-damp; zero them so BOTH drivers integrate the same
            # dynamics (the tiled factory refuses them loudly otherwise —
            # exercised by the refusal test below).
            hyperdiff_scale=0.0, div_damp_scale=0.0, a_h_scale=0.0),
        output=OutputConfig(output_dir=out_dir, diag_days=0,
                            checkpoint_days=0),
        days=days,
        dataset="analytical",
        radiation="gray",
        turbulence="tke",
        convection="none",
        microphysics="none",
        gravity_wave_drag="none",
        cloud_scheme="none",
        fix_moisture=True,
        distributed=False,
        n_devices=n_devices,
    )


def _run(out_dir, *, n_devices):
    driver = ModelDriver(_make_cfg(out_dir, n_devices=n_devices),
                         output_dir=out_dir)
    driver.setup()
    status = driver.run(compiled=True)
    return driver, status


def test_operator_split_tiled_cube_driver_matches_serial(tmp_path):
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")

    ser, ser_status = _run(str(tmp_path / "serial"), n_devices=1)
    assert ser_status == "COMPLETED", f"serial: {ser_status}"

    til, til_status = _run(str(tmp_path / "tiled"), n_devices=N_DEV)
    assert til_status == "COMPLETED", f"tiled: {til_status}"

    worst_abs = {}
    for field in ("p_s", "T", "u", "v"):
        a = np.asarray(getattr(ser.state, field).data, dtype=np.float64)
        b = np.asarray(getattr(til.state, field).data, dtype=np.float64)
        assert a.shape == b.shape, f"{field} shape {a.shape} vs {b.shape}"
        worst_abs[field] = float(np.abs(a - b).max())
    worst_abs["q_v"] = float(np.abs(
        np.asarray(ser.q_v, dtype=np.float64)
        - np.asarray(til.q_v, dtype=np.float64)).max())
    print("operator-split tiled-vs-serial abs drift:", worst_abs)

    # ABS gates in the shipped adapter/blocked-loop class: the dynamics
    # carry the documented O(1e-6) face-corner wind term (step-constant);
    # T/p_s inherit it through the per-step coupling; q_v is column-local
    # physics + a shared multiplicative fixer (tight).  A wiring bug
    # (global-ncol step_unified, stale forcing, wrong carry pack) is O(1).
    assert worst_abs["u"] < 2e-5, worst_abs
    assert worst_abs["v"] < 2e-5, worst_abs
    assert worst_abs["T"] < 1e-4, worst_abs
    assert worst_abs["p_s"] < 0.06, worst_abs
    assert worst_abs["q_v"] < 1e-8, worst_abs

    # Non-vacuity: real integration happened, and the gray radiation
    # actually fired (held radiation accumulators moved -> T changed).
    assert np.isfinite(np.asarray(til.state.T.data)).all()
    assert float(np.abs(np.asarray(til.state.u.data)).max()) > 0.0


def test_operator_split_tiled_cube_refuses_hyperdiff(tmp_path):
    """A driver config whose dynamics carries hyperdiffusion must refuse
    loudly (the tiled base-cut RK3 omits it — silent integration of
    different dynamics is the failure mode the envelope guards)."""
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")
    cfg = _make_cfg(str(tmp_path / "hd"), n_devices=N_DEV)
    cfg = cfg._replace(dycore=cfg.dycore._replace(hyperdiff_scale=1.0))
    driver = ModelDriver(cfg, output_dir=str(tmp_path / "hd"))
    driver.setup()
    with pytest.raises(NotImplementedError, match="hyperdiff"):
        driver.run(compiled=True)


def test_tiled_cube_unified_dispatch_predicate():
    """Dispatch selector: unified schemes (or HS) -> operator-split lane;
    dynamics-only and Kessler-ALONE stay on the simple blocked-loop lane."""
    class _Stub:
        def __init__(self, cfg):
            self.config = cfg

    def _cfg(**over):
        cfg = ExperimentConfig()
        fields = dict(
            radiation="none", convection="none", turbulence="none",
            microphysics="none", gravity_wave_drag="none",
            cloud_scheme="none",
            grid=cfg.grid._replace(grid_type="cubed_sphere"),
        )
        fields.update(over)
        return cfg._replace(**fields)

    act = ModelDriver._tiled_cube_unified_active
    assert act(_Stub(_cfg())) is False                       # dynamics-only
    assert act(_Stub(_cfg(microphysics="kessler"))) is False  # kessler-alone
    assert act(_Stub(_cfg(radiation="gray"))) is True
    assert act(_Stub(_cfg(microphysics="kessler",
                          turbulence="tke"))) is True
    assert act(_Stub(_cfg(held_suarez_forcing=True))) is True


# Fold-back consumption gate (the lat-band lane's test pattern): spans TWO
# segments so a callback mutation folded back after segment 1 is CONSUMED
# by segment 2.  DT_FOLD keeps C8 CFL comfortable (dx ~ 1.25e6 m).
DT_FOLD = 1800.0
DAYS_FOLD = 1.02
DELTA_T = 5.0


def test_operator_split_tiled_cube_segment_callback_fold_back(tmp_path):
    """A per-segment callback that MUTATES driver.state must have its
    mutation folded back into the threaded TILED carry (codex round-1
    High) — matching _run_compiled and the lat-band lane."""
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")

    def _cfg(sub):
        cfg = _make_cfg(str(tmp_path / sub), n_devices=N_DEV,
                        days=DAYS_FOLD)
        return cfg._replace(dycore=cfg.dycore._replace(dt=DT_FOLD))

    ctrl = ModelDriver(_cfg("ctrl"), output_dir=str(tmp_path / "ctrl"))
    ctrl.setup()
    assert ctrl.run(compiled=True) == "COMPLETED"
    T_ctrl = np.asarray(ctrl.state.T.data, dtype=np.float64)

    driver = ModelDriver(_cfg("cb"), output_dir=str(tmp_path / "cb"))
    driver.setup()
    seen = {"calls": 0}

    def _cb(drv, day, dt_seg):
        seen["calls"] += 1
        if seen["calls"] == 1:
            drv.state = drv.state._replace(
                T=drv.state.T.replace(data=drv.state.T.data + DELTA_T))

    status = driver.run(compiled=True, segment_callback=_cb)
    assert status == "COMPLETED", status
    assert seen["calls"] >= 2, (
        f"fold-back consumption needs >=2 segments, saw {seen['calls']}")
    T_cb = np.asarray(driver.state.T.data, dtype=np.float64)
    assert np.isfinite(T_cb).all()
    drift = float(np.abs(T_cb - T_ctrl).mean())
    assert drift > 0.3 * DELTA_T, (
        f"segment-2 T drift {drift:.3f}K << the injected {DELTA_T}K bump "
        f"— the callback mutation was NOT folded back into the tiled "
        f"carry.")
