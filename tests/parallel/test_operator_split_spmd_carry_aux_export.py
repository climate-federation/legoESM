"""The lat-band-SPMD OPERATOR-SPLIT lane must export the surface fields the
coupled drivers consume — and export them with the SAME producer-side units and
per-segment reseed semantics as the serial lane.

``CoupledESMDriver._build_atm_forcing`` / ``EarthSystemDriver._build_atm_forcing``
read ``held_sw_net_sfc`` / ``held_lw_net_sfc`` / ``seg_precip`` out of
``ModelDriver._carry_aux``; ``coupling_fields.require_surface_radiation_aux``
raises when an active radiation/precip config stashed none.  This lane THREADS a
single ``SegmentCarry`` across every segment (unlike ``_run_compiled``, which
re-packs with ``precip_accum=zeros`` each segment), so the accumulator reseed is
explicit in the driver and is exactly what this test pins: a dropped reseed makes
the exported precip RATE grow like the segment index while every other field
stays right.

Runs a MULTI-segment case for that reason — a single-segment run cannot tell a
reseeded accumulator from a run-total one.

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
RES = 8          # n_lat = 8 (divisible by N_DEV=2 -> 4 rows/band), n_lon = 16
NLEV = 4
DT = 2880.0      # seg_len = 86400/2880 = 30 steps/day
DAYS = 1.05      # > 1 day -> a 2nd segment, so the reseed is observable

EXPORT_KEYS = ("held_sw_net_sfc", "held_lw_net_sfc", "seg_precip")


def _make_cfg(out_dir, *, spmd, rad_update_steps=1):
    return ExperimentConfig(
        rad_update_steps=rad_update_steps,
        grid=GridConfig(grid_type="latlon", resolution=RES, nlev=NLEV),
        dycore=DycoreConfig(
            model_type="hydrostatic", discretization="latlon_cgrid", dt=DT,
            fix_mass=True),
        output=OutputConfig(output_dir=out_dir, diag_days=0, checkpoint_days=0),
        days=DAYS,
        dataset="analytical",
        radiation="gray",          # -> held_sw/lw_net_sfc != 0 (~390 W/m2)
        turbulence="tke",
        # SBM is the precip source: measured 2.3e-7 kg/m2/s on this deck, where
        # kessler stays identically 0 (the analytical IC never supersaturates at
        # C8/4-level), which would make the reseed gate below vacuous.
        convection="sbm",
        microphysics="none",
        gravity_wave_drag="none",
        cloud_scheme="none",
        # OFF deliberately: fix_moisture rescales only q_v, and the driver itself
        # warns it is INCORRECT with a prognostic-condensate scheme.  Irrelevant
        # to what this test pins, and identical on both arms either way.
        fix_moisture=False,
        enable_latlon_spmd=spmd,
        distributed=False,
        n_devices=(N_DEV if spmd else 1),
    )


def _run(out_dir, *, spmd, rad_update_steps=1):
    """Run one arm; returns (driver, per_segment_precip_maxima).

    The per-segment list is the RESEED gate's non-vacuity check.  A single
    segment cannot distinguish a reseeded accumulator from a run-total one, and
    neither can a run whose precipitation all falls in the FINAL segment (there
    is nothing earlier for a missed reseed to carry forward) -- so the caller
    checks both the length and the earlier entries.  Note the driver clamps dt
    to the pole-cell CFL (2880 -> 600 s), so seg_len is 144 steps and DAYS=1.05
    yields 2 segments; this measures that rather than assuming it.
    """
    driver = ModelDriver(
        _make_cfg(out_dir, spmd=spmd, rad_update_steps=rad_update_steps),
        output_dir=out_dir)
    driver.setup()
    per_seg = []

    def _sample(drv, _day, _dt_seg):
        _p = drv._carry_aux.get("seg_precip")
        per_seg.append(0.0 if _p is None
                       else float(np.abs(np.asarray(_p)).max()))

    # An uncoupled sampler callback: does NOT set _requires_surface_flux_export,
    # so it cannot trip the lane refusal (tests/coupler/unit/
    # test_coupled_lane_carry_aux.py::test_uncoupled_segment_callback_...).
    status = driver.run(compiled=True, segment_callback=_sample)
    assert status == "COMPLETED", f"{'spmd' if spmd else 'serial'}: {status}"
    return driver, per_seg


def test_operator_split_spmd_exports_surface_fields_like_serial(tmp_path):
    """Serial and SPMD must export the SAME three fields, to the same
    decomposition-invariance tolerance the column-local physics guarantees.

    This is the reseed gate: the serial lane re-packs ``precip_accum=zeros``
    every segment, so if the SPMD lane failed to reseed, its ``seg_precip``
    after 2 segments would be the run accumulation divided by ONE segment's
    duration -- roughly double -- and the parity assertion below goes red while
    the held radiation fields (not accumulators) stay green.
    """
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")

    ser, ser_seg_precip = _run(str(tmp_path / "serial"), spmd=False)
    spmd, spmd_seg_precip = _run(str(tmp_path / "spmd"), spmd=True)
    ser_segs, spmd_segs = len(ser_seg_precip), len(spmd_seg_precip)

    # (1) present and non-None -- exactly what require_surface_radiation_aux tests.
    for key in EXPORT_KEYS:
        assert spmd._carry_aux.get(key) is not None, (
            f"operator-split SPMD lane did not stash {key!r} into _carry_aux; a "
            "coupled run on this lane would force the surface with sw_down=0 / "
            "precip=0, silently.")

    # (2) NON-VACUITY. Three separate ways this test could pass while proving
    # nothing, all closed here: no radiation, no precip, and a single segment
    # (which a missing reseed cannot corrupt).
    ser_sw = np.asarray(ser._carry_aux["held_sw_net_sfc"], dtype=np.float64)
    ser_precip = np.asarray(ser._carry_aux["seg_precip"], dtype=np.float64)
    assert np.abs(ser_sw).max() > 0.0, (
        "gray radiation produced an all-zero held_sw_net_sfc -- the radiation "
        "parity check below would be vacuous")
    assert np.abs(ser_precip).max() > 0.0, (
        "SBM produced an all-zero seg_precip over this window -- the RESEED "
        "check below would be vacuous; lengthen DAYS or moisten the IC")
    assert ser_segs >= 2 and spmd_segs >= 2, (
        f"need >=2 segments for the reseed gate to bite (serial={ser_segs}, "
        f"spmd={spmd_segs}) -- with one segment a run-total accumulator and a "
        f"reseeded one are numerically identical; raise DAYS")
    assert ser_segs == spmd_segs, (
        f"arms disagree on segment count ({ser_segs} vs {spmd_segs}); the "
        f"seg_precip comparison would divide by different durations")
    assert max(spmd_seg_precip[:-1]) > 0.0, (
        f"no precipitation before the FINAL segment (per-segment maxima "
        f"{spmd_seg_precip}) -- a missed reseed carries EARLIER accumulation "
        f"forward, so with all the precip in the last segment the reseed gate "
        f"below cannot fail and is vacuous")

    # (3) Parity, at FIELD-SPECIFIC bounds.  The physics is column-local, but
    # over this window (151 steps after the CFL clamp) the state it reads has
    # absorbed the limited-FV-PPM band-cut residual the sharded DYNAMICS carries
    # at the two band interfaces -- measured here as u/v ~2e-4, T ~9e-7
    # relative.  So "decomposition-invariant" bounds each export by how strongly
    # it reads that drifted state, NOT by round-off:
    #   held_sw_net_sfc  EXACT.  Gray SW at the surface is solar geometry x
    #                    albedo -- it never reads the drifted column.  Measured
    #                    drift is identically 0.0, so this one is pinned hard,
    #                    and it is what rules out a mis-wired stash.
    #   held_lw_net_sfc  A NET flux: a small difference of two large gross
    #                    terms, so T's 9e-7 is amplified by the cancellation
    #                    ratio (measured 1.6e-5).
    #   seg_precip       SBM responds to column RH, amplifying the q_v/T drift
    #                    (measured 4.2e-5).
    # 1e-3 leaves ~25x headroom over the measured values while staying orders
    # BELOW every failure mode this test exists to catch: deleting the
    # accumulator reseed was MEASURED to put seg_precip drift at 8.67 (see the
    # ratio note below), and any wrong-carry-field / wrong-units wiring bug is
    # O(1).
    tol = {"held_sw_net_sfc": 1e-12, "held_lw_net_sfc": 1e-3,
           "seg_precip": 1e-3}
    for key in EXPORT_KEYS:
        a = np.asarray(ser._carry_aux[key], dtype=np.float64)
        b = np.asarray(spmd._carry_aux[key], dtype=np.float64)
        assert a.shape == b.shape, f"{key}: {a.shape} vs {b.shape}"
        rel = float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-30))
        assert rel < tol[key], (
            f"{key} relative drift {rel:.3e} exceeds {tol[key]:.0e} between the "
            f"serial and lat-band SPMD lanes -- far above the band-cut residual "
            f"the sharded dynamics carries. Wiring regression?")

    # (4) The RESEED, stated directly rather than inferred from a tolerance.
    # segment_accum_to_rate divides by the LAST segment's duration; if this lane
    # stopped reseeding precip_accum it would divide the WHOLE-RUN accumulation
    # by that duration.  The inflation is NOT n_segments -- it is
    # (total accumulated duration / final segment duration), which blows up when
    # the run ends on a short tail segment: deleting the reseed here was
    # measured at ratio ~9.7 for a 144-step + 7-step pair.  A ratio test states
    # the invariant directly, independent of how tight (3) happens to be.
    ratio = (float(np.abs(np.asarray(spmd._carry_aux["seg_precip"])).sum())
             / float(np.abs(ser_precip).sum()))
    assert 0.99 < ratio < 1.01, (
        f"SPMD/serial seg_precip magnitude ratio {ratio:.4f} (expected ~1.0; "
        f">>1 means precip_accum is never reseeded across the threaded carry). "
        f"The serial lane re-packs precip_accum=zeros every segment; this lane "
        f"threads one carry and must reseed it explicitly.")


def test_rad_update_steps_above_one_is_refused_not_silently_ignored(tmp_path):
    """The lane builds ``step_unified`` with ``static_need_rad=True``, which
    DELETES the cadence predicate the sharded step computes -- so radiation runs
    every step no matter what ``rad_update_steps`` says.  That was invisible
    while coupled runs were refused here; now that they reach this lane it would
    be a coupled surface flux produced under a different radiation cadence than
    the config requested.  Must refuse, not silently ignore.

    Non-vacuity: rad_update_steps=1 (below) runs to COMPLETED on the identical
    deck, so the refusal is keyed on the knob and not on the config being
    unrunnable.  The measured cost of NOT refusing is recorded at the raise
    site: honouring the pipeline cond instead leaves serial-vs-SPMD held LW at
    1.97e-3 versus 1.63e-5 at rad_update_steps=1.
    """
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")

    with pytest.raises(NotImplementedError, match="rad_update_steps"):
        _run(str(tmp_path / "rus2"), spmd=True, rad_update_steps=2)

    # The same deck at rad_update_steps=1 must still run -- otherwise the check
    # above would pass for the wrong reason.
    driver, per_seg = _run(str(tmp_path / "rus1"), spmd=True, rad_update_steps=1)
    assert driver._carry_aux.get("held_lw_net_sfc") is not None
    assert len(per_seg) >= 2


def test_stateless_spmd_sublane_still_refuses_a_coupled_run(tmp_path):
    """The refusal moved from the run() dispatch into
    ``_run_compiled_latlon_spmd``, so it now fires ONLY on the stateless
    sub-lane.  Pin that it still fires there: dynamics-only SPMD stashes no
    surface fields at all, and a coupled consumer on it is the original silent
    sw_down=0 bug.
    """
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")

    cfg = _make_cfg(str(tmp_path / "dry"), spmd=True)
    cfg = cfg._replace(radiation="none", turbulence="none",
                       convection="none", microphysics="none")
    driver = ModelDriver(cfg, output_dir=str(tmp_path / "dry"))
    driver.setup()
    driver._requires_surface_flux_export = True     # the coupled-driver marker
    with pytest.raises(NotImplementedError, match="sw_down=0"):
        driver.run(compiled=True)
