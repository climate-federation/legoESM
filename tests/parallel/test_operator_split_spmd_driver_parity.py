"""End-to-end parity of the run_amip lat-band-SPMD OPERATOR-SPLIT lane
(``ModelDriver._run_operator_split_spmd``) against the serial compiled-segment
path (``ModelDriver._run_compiled``).

Both drivers integrate the SAME tiny gray-radiation + TKE-turbulence AMIP config
(analytical forcing, lat-lon C-grid) from an identical initial state; one runs
the serial fused operator-split scan, the other the lat-band-sharded operator
split over a 1-D ``"lat"`` mesh (2 devices).  Because the operator-split physics
is PURELY column-local (decomposition-invariant) and the ONLY collectives are
the mass/moisture fixers (lat-band-psum-aware) + the moisture halo, the two
trajectories agree to the limited-FV-PPM cut-boundary residual the sharded
dynamics carries at band interfaces — NOT an O(1) split-wiring bug.  This is the
driver-wiring gate: it exercises the real ``build_physics_pipeline`` step_unified
(built on a band grid), the carry seed/shard/thread, the per-segment forcing, and
the gather — the pieces the shape-agnostic ``test_atm_latlon_operator_split_spmd``
mock cannot.

4 host CPU devices via ``XLA_FLAGS=--xla_force_host_platform_device_count=4``;
``JAX_ENABLE_X64=1``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from legoesm.driver.config import (
    ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver

N_DEV = 2
DT = 100.0
N_STEPS = 3
RES = 8          # n_lat = 8 (divisible by N_DEV=2 -> 4 rows/band), n_lon = 16
NLEV = 4


def _make_cfg(out_dir, *, spmd, dt=DT, days=N_STEPS * DT / 86400.0):
    return ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=RES, nlev=NLEV),
        dycore=DycoreConfig(
            model_type="hydrostatic", discretization="latlon_cgrid", dt=dt,
            fix_mass=True),
        output=OutputConfig(output_dir=out_dir, diag_days=0, checkpoint_days=0),
        days=days,
        dataset="analytical",
        radiation="gray",
        turbulence="tke",
        convection="none",
        microphysics="none",
        gravity_wave_drag="none",
        cloud_scheme="none",
        fix_moisture=True,
        enable_latlon_spmd=spmd,
        distributed=False,
        n_devices=(N_DEV if spmd else 1),
    )


def _run(out_dir, *, spmd):
    driver = ModelDriver(_make_cfg(out_dir, spmd=spmd), output_dir=out_dir)
    driver.setup()
    status = driver.run(compiled=True)
    return driver, status


def test_operator_split_spmd_driver_matches_serial(tmp_path):
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")

    ser, ser_status = _run(str(tmp_path / "serial"), spmd=False)
    assert ser_status == "COMPLETED", f"serial: {ser_status}"

    spmd, spmd_status = _run(str(tmp_path / "spmd"), spmd=True)
    assert spmd_status == "COMPLETED", f"spmd: {spmd_status}"

    # Report per-field drift (dynamical winds carry the band-cut residual; the
    # thermodynamic column state is physics-dominated -> tighter).
    worst = {}
    for field in ("p_s", "T", "u", "v"):
        a = np.asarray(getattr(ser.state, field).data, dtype=np.float64)
        b = np.asarray(getattr(spmd.state, field).data, dtype=np.float64)
        assert a.shape == b.shape, f"{field} shape {a.shape} vs {b.shape}"
        denom = np.maximum(np.abs(a).max(), 1e-30)
        worst[field] = float(np.abs(a - b).max() / denom)
    qa = np.asarray(ser.q_v, dtype=np.float64)
    qb = np.asarray(spmd.q_v, dtype=np.float64)
    worst["q_v"] = float(np.abs(qa - qb).max() / max(np.abs(qa).max(), 1e-30))
    print("operator-split SPMD-vs-serial relative drift:", worst)

    # The COLUMN-LOCAL thermodynamic state — p_s (mass-fixer target computed on
    # the full grid; the in-step global_area_sum is lat-band-psum-aware), T and
    # q_v — is decomposition-INVARIANT, so it matches to reduction round-off
    # (bitwise 0 on this hardware).  Only the winds carry the limited-FV-PPM
    # cut residual at the two band interfaces.  Field-specific gates: a wiring
    # bug (wrong carry field / stale GHG / global-ncol step_unified) would push
    # even the thermodynamic fields O(1).
    for field in ("p_s", "T", "q_v"):
        assert worst[field] < 1e-9, (
            f"{field} relative drift {worst[field]:.3e} — column-local physics "
            f"must be decomposition-invariant; wiring regression? {worst}")
    for field in ("u", "v"):
        assert worst[field] < 1e-5, (
            f"{field} relative drift {worst[field]:.3e} exceeds the FV-PPM "
            f"band-cut bound (~1e-7); wiring regression? {worst}")

    # Non-vacuity: the run actually integrated (state != initial, finite).
    assert np.isfinite(np.asarray(spmd.state.T.data)).all()
    assert float(np.abs(np.asarray(spmd.state.u.data)).max()) > 0.0


# Fold-back test spans TWO segments so a mutation folded back after segment 1 is
# actually CONSUMED by segment 2 (the original single-segment test mutated after
# the only segment and could not catch a dropped fold-back).  DT_FOLD gives
# seg_len=30 steps/day (CFL~0.35 on this coarse grid, finite), DAYS_FOLD>1 forces
# a 2nd segment (a 1-step tail) that integrates the folded-back carry.
DT_FOLD = 2880.0
DAYS_FOLD = 1.05
DELTA_T = 5.0    # first-callback T bump [K]; large enough to survive integration


def test_operator_split_spmd_segment_callback_fold_back(tmp_path):
    """A per-segment callback that MUTATES driver.state must have its mutation
    folded back into the threaded sharded carry so the NEXT segment integrates the
    mutated state — matching _run_compiled, which re-packs from self.* each
    segment.  Non-vacuous consumption gate: bump T by DELTA_T on the FIRST
    callback only (after segment 1), run a 2nd segment, and compare to a
    no-callback control.  A WORKING fold-back feeds segment 2 the +DELTA_T carry
    -> final T differs from control by ~DELTA_T; a DROPPED fold-back leaves
    segment 2 on the un-bumped carry (and the 2nd callback does not bump) ->
    final T ~= control.  The old single-segment test could not tell these apart."""
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")

    def _cfg(sub):
        return _make_cfg(str(tmp_path / sub), spmd=True,
                         dt=DT_FOLD, days=DAYS_FOLD)

    # Control: identical run, no callback.
    ctrl = ModelDriver(_cfg("ctrl"), output_dir=str(tmp_path / "ctrl"))
    ctrl.setup()
    assert ctrl.run(compiled=True) == "COMPLETED"
    T_ctrl = np.asarray(ctrl.state.T.data, dtype=np.float64)

    # Callback run: bump T on the first callback only.
    driver = ModelDriver(_cfg("cb"), output_dir=str(tmp_path / "cb"))
    driver.setup()
    seen = {"calls": 0, "shapes_ok": True}

    def _cb(drv, day, dt_seg):
        seen["calls"] += 1
        T = np.asarray(drv.state.T.data)
        # (a) the callback sees a finite, cell-centered gathered state.
        seen["shapes_ok"] &= bool(np.isfinite(T).all())
        # (b) bump on the FIRST call only -> segment 2 must consume the fold-back.
        if seen["calls"] == 1:
            drv.state = drv.state._replace(
                T=drv.state.T.replace(data=drv.state.T.data + DELTA_T))

    status = driver.run(compiled=True, segment_callback=_cb)
    assert status == "COMPLETED", status
    assert seen["calls"] >= 2, (
        f"fold-back consumption needs >=2 segments, saw {seen['calls']} "
        f"callback(s) — DAYS_FOLD/DT_FOLD must span a 2nd segment")
    assert seen["shapes_ok"]

    T_cb = np.asarray(driver.state.T.data, dtype=np.float64)
    assert np.isfinite(T_cb).all()
    # Segment 2 integrated the +DELTA_T carry -> mean |T_cb - T_ctrl| ~ DELTA_T.
    # A dropped fold-back would leave this ~0 (segment 2 on the un-bumped carry).
    drift = float(np.abs(T_cb - T_ctrl).mean())
    assert drift > 0.3 * DELTA_T, (
        f"segment-2 T drift {drift:.3f}K vs control is far below the injected "
        f"{DELTA_T}K bump — the first-segment callback mutation was NOT folded "
        f"back into the sharded carry (dropped fold-back).")
