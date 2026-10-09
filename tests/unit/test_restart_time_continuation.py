"""Bit-exact restart continuation (FIX_RESTART_TIME, docs/dev-notes/FIX_RESTART_TIME.md).

The #413 series made the PhysicsState carry persist exactly through
checkpoints, but the restart TRAJECTORY still diverged from an
uninterrupted run because of two forcing-time bugs (found by the
fix/persist-physics validation harness and ported here):

1. The MPAS loop sampled the daily forcing at the restart link's FIRST
   STEP (mid-day) instead of the canonical day boundary, so a chained
   run saw slightly different seasonal SST than the straight run.
2. The cube/lat-lon/spectral loops index time as
   ``START_DAY + ABSOLUTE_step·DT/86400`` (epoch convention), but every
   production caller passes the CHECKPOINT day from ``load_checkpoint``
   back into ``run()`` — double-counting the pre-restart elapsed time and
   running every restarted link with forcing shifted forward by the
   checkpoint day.

These tests assert the strongest property: an interrupted+restarted run
equals the uninterrupted run BITWISE — prognostic state AND the physics
carries.  Pre-port, both tests fail on main (cube ~4e-3 K, MPAS ~1e-8).

Bitwise-equality preconditions (by design, documented):
* MPAS: none — its forcing is daily-cadence + per-step traced scalars.
* cube (compiled loop): the restart must land on a segment boundary and
  all runs must share the same checkpoint/diag cadence — the compiled
  loop samples per-SEGMENT forcing at the segment-start day, so mismatched
  segmentation samples different forcing BY CONSTRUCTION (see the
  ``compute_segment_length`` pins below).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402  (kept for parity with sibling tests)

from legoesm.atmosphere.physics.physics_state import PhysicsState  # noqa: E402
from legoesm.driver.compiled_segments import compute_segment_length  # noqa: E402
from legoesm.driver.config import (  # noqa: E402
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
    OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver  # noqa: E402


# ======================================================================
# Scheduler pins the restart tests rely on (codex iteration-1 finding:
# the cadence interaction must be asserted, not assumed)
# ======================================================================

def test_segment_length_disabled_cadences_pin():
    """diag=ckpt=0 ⇒ 1-step segments; a checkpoint cadence alone sets the
    segment length.  The continuation tests below align ALL runs on one
    cadence because per-segment forcing makes mismatched cadences sample
    different forcing days by construction."""
    assert compute_segment_length(0, 0, 3) == 1
    assert compute_segment_length(0, 144, 3) == 144
    assert compute_segment_length(288, 288, 2) == 288


# ======================================================================
# MPAS: 2+2-step chain == 4-step straight run, bitwise
# ======================================================================

MPAS_RES, MPAS_NLEV, DT = 3, 20, 300.0  # icosahedral level 3 = 642 cells
TWO_STEPS_DAYS = 601.0 / 86400.0   # int(601/300) = 2 steps, rounding-safe
FOUR_STEPS_DAYS = 1201.0 / 86400.0


def _build_mpas_driver(tmpdir: str, days: float,
                       start_day: float = 0.0) -> ModelDriver:
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=MPAS_RES,
                        nlev=MPAS_NLEV, vertical_coord="hybrid"),
        dycore=DycoreConfig(discretization="mpas", dt=DT),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=1),
        days=days, start_day=start_day,
        dataset="analytical", radiation="gray",
        # tke = stateful turbulence so the carry rides the chain; GWD
        # stays "none" (prognostic_spectral still raises
        # NotImplementedError on the MPAS bridge on main).
        convection="none", turbulence="tke", precision="fp64",
        distributed=False,
    )
    d = ModelDriver(cfg, output_dir=tmpdir)
    d.setup()
    return d


def test_mpas_bitexact_restart_continuation(tmp_path):
    """Interrupted+restarted MPAS run == uninterrupted run, BITWISE.

    Fails pre-port: the link-start daily-forcing sampling gave the
    chained run different seasonal SST (T diverged ~1e-8 over 2 steps
    while every carry slot round-tripped 0.0).
    """
    dA = _build_mpas_driver(str(tmp_path / "straight"), FOUR_STEPS_DAYS)
    assert dA.run() == "COMPLETED"

    outB = str(tmp_path / "chain1")
    dB = _build_mpas_driver(outB, TWO_STEPS_DAYS)
    assert dB.run() == "COMPLETED"
    ckpt = sorted(glob.glob(os.path.join(outB, "checkpoint_day_*.npz")))[-1]
    # MPAS restart contract: cfg.days is the days THIS link advances.
    dB2 = _build_mpas_driver(str(tmp_path / "chain2"), TWO_STEPS_DAYS)
    step, day = dB2.load_checkpoint(ckpt)
    assert dB2.run(start_step=step, start_day=day) == "COMPLETED"

    for name, a, b in (
        ("T", dA.state.T.data, dB2.state.T.data),
        ("u", dA.state.u.data, dB2.state.u.data),
        ("p_s", dA.state.p_s.data, dB2.state.p_s.data),
    ):
        np.testing.assert_array_equal(
            np.asarray(a), np.asarray(b),
            err_msg=f"{name}: chained restart != straight run (bitwise)",
        )
    psA = dA._mpas_phys_state
    psB = dB2._mpas_phys_state
    assert psA is not None and psB is not None
    for f in PhysicsState._fields:
        np.testing.assert_array_equal(
            np.asarray(getattr(psA, f)), np.asarray(getattr(psB, f)),
            err_msg=f"PhysicsState.{f}: chained restart != straight run",
        )


def test_mpas_negative_epoch_chain_crossing_day_zero(tmp_path):
    """Negative fractional epoch: the chain crossing day 0 still matches
    the straight run bitwise.

    Integration coverage for negative start_day epochs (the regime where
    the floor-vs-int daily-bucket semantics matter).  NOTE: this test
    deliberately does NOT pin the floor semantics themselves — straight
    and chained runs bucket identically under either floor or int, so
    chain consistency holds even with wrong bucketing (verified: the
    test passes with int() reverted).  The semantics are pinned by the
    unit test ``test_daily_forcing_bucket_floor_semantics`` in
    tests/unit/test_time_utils.py, which genuinely discriminates.
    """
    start = -0.01  # 4 steps at dt=300 (0.00347/step) cross day 0
    dA = _build_mpas_driver(str(tmp_path / "straight"), FOUR_STEPS_DAYS,
                            start_day=start)
    assert dA.run() == "COMPLETED"

    outB = str(tmp_path / "chain1")
    dB = _build_mpas_driver(outB, TWO_STEPS_DAYS, start_day=start)
    assert dB.run() == "COMPLETED"
    ckpt = sorted(glob.glob(os.path.join(outB, "checkpoint_day_*.npz")))[-1]
    dB2 = _build_mpas_driver(str(tmp_path / "chain2"), TWO_STEPS_DAYS,
                             start_day=start)
    step, day = dB2.load_checkpoint(ckpt)
    assert day < 0.0, f"restart day should still be negative: {day}"
    assert dB2.run(start_step=step, start_day=day) == "COMPLETED"

    for name, a, b in (
        ("T", dA.state.T.data, dB2.state.T.data),
        ("u", dA.state.u.data, dB2.state.u.data),
        ("p_s", dA.state.p_s.data, dB2.state.p_s.data),
    ):
        np.testing.assert_array_equal(
            np.asarray(a), np.asarray(b),
            err_msg=f"{name}: negative-epoch chain != straight (bitwise)",
        )


# ======================================================================
# Cube compiled loop: 1+1-day chain == 2-day straight run, bitwise
# ======================================================================

def _build_cube_driver(tmpdir: str, days: float) -> ModelDriver:
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=8, nlev=10),
        dycore=DycoreConfig(dt=300.0),
        # Uniform cadence on every run (see the scheduler pins): 1-day
        # checkpoints ⇒ 288-step segments; the restart lands exactly on
        # the segment boundary so per-segment forcing aligns.
        output=OutputConfig(output_dir="", diag_days=1, checkpoint_days=1),
        days=days, dataset="analytical", radiation="gray",
        rad_update_steps=2,
        convection="none", turbulence="tke", precision="fp64",
    )
    d = ModelDriver(cfg, output_dir=tmpdir)
    d.setup()
    return d


@pytest.mark.slow
def test_cube_compiled_bitexact_restart_continuation(tmp_path):
    """Compiled cube chain (1+1 day) == straight 2-day run, BITWISE.

    Fails pre-port with ~4e-3 K divergence: the restarted run's
    segment forcing ran a full day ahead (START_DAY double-count) while
    the tke carry itself round-tripped 0.0.
    """
    dA = _build_cube_driver(str(tmp_path / "straight"), days=2.0)
    assert dA.run() == "COMPLETED"

    outB = str(tmp_path / "chain1")
    dB = _build_cube_driver(outB, days=1.0)
    assert dB.run() == "COMPLETED"
    ckpt = sorted(glob.glob(os.path.join(outB, "checkpoint_day_*.npz")))[-1]
    dB2 = _build_cube_driver(str(tmp_path / "chain2"), days=2.0)
    step, day = dB2.load_checkpoint(ckpt)
    assert step == 288, f"checkpoint not on the segment boundary: {step}"
    assert dB2.run(start_step=step, start_day=day) == "COMPLETED"

    comparisons = [
        ("T", dA.state.T.data, dB2.state.T.data),
        ("u", dA.state.u.data, dB2.state.u.data),
        ("v", dA.state.v.data, dB2.state.v.data),
        ("p_s", dA.state.p_s.data, dB2.state.p_s.data),
        ("q_v", dA.q_v, dB2.q_v),
        ("tke", dA._carry_aux["tke"], dB2._carry_aux["tke"]),
    ]
    for name, a, b in comparisons:
        np.testing.assert_array_equal(
            np.asarray(a), np.asarray(b),
            err_msg=f"{name}: chained compiled restart != straight run "
                    "(bitwise)",
        )


# ======================================================================
# Resumed segment 0 must refresh external forcing (codex iteration-1
# P2): the prepare-context ozone/aerosol/GHG/solar are sampled at the
# epoch START_DAY, but the straight run refreshed the corresponding
# (absolute) segment at its end day — without a seg-0 refresh on resume,
# a restarted AMIP/CMIP run's first segment runs on epoch-day forcing.
# ======================================================================

def _spy_forcing_days(driver: ModelDriver) -> list:
    """Record every day `_precompute_external_forcing` is called with."""
    days: list = []
    orig = driver._precompute_external_forcing

    def spy(day, p_s, lat):
        days.append(float(day))
        return orig(day, p_s, lat)

    driver._precompute_external_forcing = spy
    return days


def _build_small_cube_driver(tmpdir: str, days: float) -> ModelDriver:
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=5),
        dycore=DycoreConfig(dt=600.0),
        output=OutputConfig(output_dir="", diag_days=1, checkpoint_days=1),
        days=days, dataset="analytical", radiation="gray",
        rad_update_steps=2,
        convection="none", turbulence="none", precision="fp64",
    )
    d = ModelDriver(cfg, output_dir=tmpdir)
    d.setup()
    return d


def test_resumed_segment_zero_refreshes_external_forcing(tmp_path):
    """The resumed link's first segment samples external forcing at the
    SAME days the straight run did — not the prepare-time epoch only.

    The bit-exact cube test above cannot catch this: with constant TSI
    and no CMIP datasets the stale epoch values are numerically identical
    to the refreshed ones.  This spy pins the sampling DAYS themselves.
    """
    dA = _build_small_cube_driver(str(tmp_path / "straight"), days=2.0)
    days_straight = _spy_forcing_days(dA)
    assert dA.run() == "COMPLETED"

    outB = str(tmp_path / "chain1")
    dB = _build_small_cube_driver(outB, days=1.0)
    assert dB.run() == "COMPLETED"
    ckpt = sorted(glob.glob(os.path.join(outB, "checkpoint_day_*.npz")))[-1]
    dB2 = _build_small_cube_driver(str(tmp_path / "chain2"), days=2.0)
    days_resumed = _spy_forcing_days(dB2)
    step, day = dB2.load_checkpoint(ckpt)
    assert dB2.run(start_step=step, start_day=day) == "COMPLETED"

    # Straight run: prepare at epoch 0.0, then segment-boundary
    # refreshes.  The resumed run must hit every refresh day the
    # straight run hit AFTER the restart point — in particular its
    # seg_idx == 0 segment day, which only a start_step-aware gate
    # samples.
    # ``>=``: segments sample their START day (gridaudit 2026-10-09), so the
    # straight run refreshes the restart day itself.
    refreshes_after_restart = [d for d in days_straight if d >= day]
    assert refreshes_after_restart, (
        "test setup: the straight run should refresh external forcing "
        f"after day {day}; got calls at {days_straight}"
    )
    for d_refresh in refreshes_after_restart:
        assert d_refresh in days_resumed, (
            f"resumed run never sampled external forcing at day "
            f"{d_refresh} (straight: {days_straight}, "
            f"resumed: {days_resumed}) — its first segment ran on "
            "stale epoch-day forcing"
        )


# ======================================================================
# Spectral path: checkpointing exists at all + bitwise continuation
# ======================================================================

def test_spectral_bitexact_restart_continuation(tmp_path):
    """Spectral 1+1-day chain == 2-day straight run, BITWISE on the
    spectral coefficients.

    Until FIX_RESTART_TIME iteration 4 the spectral loop wrote NO
    checkpoints at all — a --spectral AMIP run silently ignored
    --checkpoint-days and --restart-from was impossible.  This pins
    both the new spectral save/load branch and the (now reachable)
    epoch normalization in _run_spectral.
    """

    def build(tmpdir: str, days: float) -> ModelDriver:
        cfg = ExperimentConfig(
            grid=GridConfig(grid_type="gaussian", resolution=8, nlev=5),
            dycore=DycoreConfig(discretization="spectral", dt=600.0),
            output=OutputConfig(output_dir="", diag_days=1,
                                checkpoint_days=1),
            days=days, dataset="analytical", radiation="gray",
            convection="none", turbulence="none", precision="fp64",
        )
        d = ModelDriver(cfg, output_dir=tmpdir)
        d.setup()
        return d

    dA = build(str(tmp_path / "straight"), days=2.0)
    assert dA.run() == "COMPLETED"

    outB = str(tmp_path / "chain1")
    dB = build(outB, days=1.0)
    assert dB.run() == "COMPLETED"
    ckpts = sorted(glob.glob(os.path.join(outB, "checkpoint_day_*.npz")))
    assert ckpts, "spectral run wrote no checkpoints (iteration-4 gap)"
    dB2 = build(str(tmp_path / "chain2"), days=2.0)
    step, day = dB2.load_checkpoint(ckpts[-1])
    assert (step, day) == (144, 1.0)
    assert dB2.run(start_step=step, start_day=day) == "COMPLETED"

    for name in ("vor_hat", "div_hat", "T_hat", "lnps_hat", "phis_hat"):
        np.testing.assert_array_equal(
            np.asarray(getattr(dA.state, name).data),
            np.asarray(getattr(dB2.state, name).data),
            err_msg=f"{name}: spectral chained restart != straight run "
                    "(bitwise)",
        )


# ======================================================================
# Per-step (compiled=False) path: warmup radiation cadence
# ======================================================================

def test_per_step_bitexact_restart_off_radiation_boundary(tmp_path):
    """Per-step chain == straight run, BITWISE, when the checkpoint step
    is NOT a radiation boundary (codex iteration-2 finding, high).

    dt=600 ⇒ 144 steps/day; checkpoint at step 144 with
    rad_update_steps=3 ⇒ (144+1) % 3 == 1, so the straight run's step
    144 reuses the held radiation tendencies.  The resumed warmup used
    to call ``step_unified(jnp.bool_(True), ...)`` unconditionally —
    recomputing radiation at the wrong step, overwriting the restored
    held tendencies, and sampling forcing at the prepare-context epoch
    day.  The warmup now honors the same need_rad predicate as the main
    loop and refreshes forcing only when it radiates.
    """

    def build(tmpdir: str, days: float) -> ModelDriver:
        cfg = ExperimentConfig(
            grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=5),
            dycore=DycoreConfig(dt=600.0),
            output=OutputConfig(output_dir="", diag_days=1,
                                checkpoint_days=1),
            days=days, dataset="analytical", radiation="gray",
            rad_update_steps=3,
            convection="none", turbulence="none", precision="fp64",
        )
        d = ModelDriver(cfg, output_dir=tmpdir)
        d.setup()
        return d

    dA = build(str(tmp_path / "straight"), days=2.0)
    assert dA.run(compiled=False) == "COMPLETED"

    outB = str(tmp_path / "chain1")
    dB = build(outB, days=1.0)
    assert dB.run(compiled=False) == "COMPLETED"
    ckpt = sorted(glob.glob(os.path.join(outB, "checkpoint_day_*.npz")))[-1]
    dB2 = build(str(tmp_path / "chain2"), days=2.0)
    step, day = dB2.load_checkpoint(ckpt)
    assert (step + 1) % 3 != 0, (
        f"test setup: step {step} must NOT be a radiation boundary"
    )
    assert dB2.run(start_step=step, start_day=day,
                   compiled=False) == "COMPLETED"

    comparisons = [
        ("T", dA.state.T.data, dB2.state.T.data),
        ("u", dA.state.u.data, dB2.state.u.data),
        ("p_s", dA.state.p_s.data, dB2.state.p_s.data),
        ("q_v", dA.q_v, dB2.q_v),
        ("held_dT_rad", dA._carry_aux["held_dT_rad"],
         dB2._carry_aux["held_dT_rad"]),
    ]
    for name, a, b in comparisons:
        np.testing.assert_array_equal(
            np.asarray(a), np.asarray(b),
            err_msg=f"{name}: per-step chained restart != straight run "
                    "(bitwise) — warmup radiation cadence",
        )


# ======================================================================
# start_day convention disambiguation
# ======================================================================

def test_start_day_convention_disambiguation(tmp_path):
    """Only the load_checkpoint convention is epoch-normalized.

    A caller passing back EXACTLY what load_checkpoint returned gets the
    checkpoint-day → epoch normalization (production restart paths); a
    caller passing its own (e.g. epoch) start_day with start_step>0
    keeps the legacy epoch semantics unchanged — never silently shift
    epoch-passing callers (codex finding from the original review on
    fix/persist-physics).
    """
    d = _build_cube_driver(str(tmp_path / "conv"), days=2.0)
    # No checkpoint loaded: epoch semantics preserved verbatim.
    ctx = d._prepare_run_context(288, 0.0, restore_carry=False)
    assert ctx["START_DAY"] == 0.0
    ctx = d._prepare_run_context(288, 5.0, restore_carry=False)
    assert ctx["START_DAY"] == 5.0
    # Simulate the load_checkpoint convention: matching (step, day) hint.
    d._loaded_checkpoint_step_day = (288, 1.0)
    ctx = d._prepare_run_context(288, 1.0, restore_carry=False)
    assert ctx["START_DAY"] == 0.0  # 1.0 - 288*300/86400
    # A non-matching day still means epoch.
    ctx = d._prepare_run_context(288, 2.0, restore_carry=False)
    assert ctx["START_DAY"] == 2.0


def _build_sigma_mpas_driver(tmpdir: str, days: float,
                             tropopause_refine: float = 1.0) -> ModelDriver:
    """Same MPAS driver on the SIGMA coordinate, refinement selectable."""
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=MPAS_RES,
                        nlev=MPAS_NLEV, vertical_coord="sigma",
                        tropopause_refine=tropopause_refine),
        dycore=DycoreConfig(discretization="mpas", dt=DT),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=1),
        days=days, start_day=0.0,
        dataset="analytical", radiation="gray",
        convection="none", turbulence="tke", precision="fp64",
        distributed=False,
    )
    d = ModelDriver(cfg, output_dir=tmpdir)
    d.setup()
    return d


def test_mpas_restart_rejects_a_different_vertical_grid(tmp_path):
    """A checkpoint written on uniform sigma must NOT silently restart on a
    tropopause-refined grid of the SAME nlev.

    ``grid.tropopause_refine`` redistributes levels at FIXED nlev, so every
    array shape is identical and the existing shape guards cannot see the
    difference — the profiles would just be reinterpreted at the wrong
    pressures (a silent physics error, never a crash).  ``meta_vgrid`` (the (A, B) half-level pair)
    in the checkpoint is what closes that hole.
    """
    out = str(tmp_path / "uniform")
    d_uni = _build_sigma_mpas_driver(out, TWO_STEPS_DAYS)
    assert d_uni.run() == "COMPLETED"
    ckpt = sorted(glob.glob(os.path.join(out, "checkpoint_day_*.npz")))[-1]
    assert "meta_vgrid" in np.load(ckpt).files

    # Same nlev, different level POSITIONS -> must raise, not reinterpret.
    d_ref = _build_sigma_mpas_driver(str(tmp_path / "refined"),
                                     TWO_STEPS_DAYS, tropopause_refine=3.0)
    assert d_ref.state.T.data.shape == d_uni.state.T.data.shape
    with pytest.raises(ValueError, match="DIFFERENT vertical grid"):
        d_ref.load_checkpoint(ckpt)

    # The matching grid still loads (the guard is not just "always raise").
    d_same = _build_sigma_mpas_driver(str(tmp_path / "same"), TWO_STEPS_DAYS)
    step, day = d_same.load_checkpoint(ckpt)
    assert step > 0 and day > 0.0
