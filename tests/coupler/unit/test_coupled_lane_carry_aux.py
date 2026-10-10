"""C2 regression: coupled lanes that never populate ``_carry_aux`` must FAIL
LOUD instead of silently forcing the surface with zero radiation / precip.

The lat-lon SPMD and sub-face-tiled cube lanes fire the atmosphere's segment
callback but never stash ``held_sw_net_sfc`` / ``held_lw_net_sfc`` /
``seg_precip``.  Both coupled drivers' ``_build_atm_forcing`` then read them
out of an EMPTY dict via ``aux.get(key, zeros)``, so the ocean/land/ice tiles
were forced with ``sw_down=0`` and ``precip=0``.

That failure is silent: no NaN, no exception, and the reconstructed
``lw_down = (0 + eps*sigma*T^4)/max(eps, 0.01)`` collapses to exactly
``sigma*T_sfc**4`` (the emissivity cancels), a perfectly plausible
~300-400 W/m^2.  Only shortwave and precipitation are lost, and the resulting
cold/salty drift reads as a spin-up transient.

Pure unit test: no GPU, no MPI, no model run, no driver construction.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from legoesm.core.coupling_fields import require_surface_radiation_aux
from legoesm.driver.model_driver import ModelDriver

REPO_ROOT = Path(__file__).resolve().parents[3]
DRIVER_DIR = REPO_ROOT / "packages" / "coupler" / "legoesm" / "driver"
MODEL_DRIVER_SRC = DRIVER_DIR / "model_driver.py"

_FULL_AUX = {
    "held_sw_net_sfc": 1.0,
    "held_lw_net_sfc": 2.0,
    "seg_precip": 3.0,
}


# --------------------------------------------------------------------------
# Consumer-side strict check
# --------------------------------------------------------------------------
def test_empty_carry_aux_raises_when_radiation_active():
    """The exact C2 signature: a lane advanced a segment, stashed nothing,
    and radiation is on.  Pre-fix this silently returned zeros."""
    with pytest.raises(RuntimeError) as exc:
        require_surface_radiation_aux(
            {}, radiation_active=True, precip_active=True, lane="ModelDriver")
    msg = str(exc.value)
    assert "held_sw_net_sfc" in msg
    assert "held_lw_net_sfc" in msg
    assert "seg_precip" in msg


def test_missing_seg_precip_alone_raises():
    """The spectral lane's signature: sw/lw stashed at model_driver.py:6511-
    6512, seg_precip never written."""
    aux = {"held_sw_net_sfc": 1.0, "held_lw_net_sfc": 2.0}
    with pytest.raises(RuntimeError, match="seg_precip"):
        require_surface_radiation_aux(
            aux, radiation_active=True, precip_active=True, lane="spectral")


def test_present_but_none_value_is_treated_as_missing():
    """_run_per_step writes the key UNCONDITIONALLY with a None value when its
    source is inactive (model_driver.py:9762).  A key-presence test would pass
    that through and the consumer would then do `None / jnp.maximum(...)`."""
    aux = {"held_sw_net_sfc": None, "held_lw_net_sfc": 2.0,
           "seg_precip": 3.0}
    with pytest.raises(RuntimeError, match="held_sw_net_sfc"):
        require_surface_radiation_aux(
            aux, radiation_active=True, precip_active=True, lane="per_step")


def test_inactive_sources_tolerate_empty_aux():
    """NO FALSE POSITIVES: zero IS the correct forcing for a dry,
    radiation='none' run, and _run_column legitimately stashes nothing there
    (model_driver.py:5795-5802).  A blanket 'require all three keys' check
    would break this."""
    require_surface_radiation_aux(
        {}, radiation_active=False, precip_active=False, lane="ModelDriver")


def test_dry_run_with_radiation_requires_only_fluxes():
    """radiation on + dry: fluxes demanded, precip correctly not demanded."""
    aux = {"held_sw_net_sfc": 1.0, "held_lw_net_sfc": 2.0}
    require_surface_radiation_aux(
        aux, radiation_active=True, precip_active=False, lane="mpas-dry")


def test_fully_populated_aux_passes():
    require_surface_radiation_aux(
        dict(_FULL_AUX), radiation_active=True, precip_active=True,
        lane="ModelDriver")


# --------------------------------------------------------------------------
# Lane-side refusal -- and its NON-regression on uncoupled runs
# --------------------------------------------------------------------------
def test_reject_coupled_lane_raises_when_a_consumer_is_attached():
    drv = object.__new__(ModelDriver)      # no __init__: no model, no JAX
    drv._requires_surface_flux_export = True
    with pytest.raises(NotImplementedError) as exc:
        drv._reject_coupled_lane("test lane", "a test envelope", "Do X.")
    msg = str(exc.value)
    assert "sw_down=0" in msg
    assert "Do X." in msg


def test_reject_coupled_lane_is_noop_without_the_marker():
    """Default (attribute absent) and explicit-False must both pass."""
    drv = object.__new__(ModelDriver)
    drv._reject_coupled_lane("test lane", "a test envelope", "Do X.")
    drv._requires_surface_flux_export = False
    drv._reject_coupled_lane("test lane", "a test envelope", "Do X.")


def test_uncoupled_segment_callback_does_not_trigger_refusal():
    """THE FALSE-POSITIVE GATE.  ``run(segment_callback=...)`` is a GENERAL
    per-segment hook used by uncoupled diagnostic samplers -- the two
    operator-split fold-back parity tests
    (tests/parallel/test_operator_split_{spmd,tiled_cube}_driver_parity.py)
    call it on real drivers configured for exactly the two guarded lanes, as
    do training/run_to_column_mean.py and ml/physics/data.py.  Gating the
    refusal on ``_segment_callback is not None`` instead of on the coupling
    marker would turn every one of those red."""
    drv = object.__new__(ModelDriver)
    drv._segment_callback = lambda *a, **k: None   # uncoupled sampler
    drv._reject_coupled_lane("test lane", "a test envelope", "Do X.")


# --------------------------------------------------------------------------
# Ratchets: neither dispatch branch nor either consumer may lose its guard
# --------------------------------------------------------------------------
@pytest.mark.parametrize("lane_call", [
    # Anchored at the point the UNGUARDED lane is entered, which is not the
    # same place for the two lanes.  The lat-lon SPMD dispatch branch fans out
    # into TWO sub-lanes -- operator-split (which DOES stash the held surface
    # fields; tests/parallel/test_operator_split_spmd_carry_aux_export.py) and
    # stateless dynamics-only/Held-Suarez (which stashes nothing) -- so the
    # guard sits on the stateless sub-lane inside _run_compiled_latlon_spmd,
    # not at the dispatch, which cannot tell them apart.  The cube lane has no
    # such split and stays guarded at the dispatch.
    "physics_fn = self._latlon_spmd_physics_fn()",
    "status = self._run_tiled_cube_spmd(start_step, start_day)",
])
def test_dispatch_branches_are_guarded(lane_call):
    """Mirrors the project's dispatch-hardening ratchets -- an existing
    refusal must not be deletable without a test going red."""
    src = MODEL_DRIVER_SRC.read_text()
    assert src.count(lane_call) == 1, f"anchor no longer unique: {lane_call}"
    idx = src.index(lane_call)
    window = src[max(0, idx - 900):idx]
    assert "_reject_coupled_lane" in window, (
        f"lane entry {lane_call!r} is not preceded by a "
        "_reject_coupled_lane guard: a coupled run on this lane would force "
        "the surface with sw_down=0 and precip=0.")


def test_operator_split_spmd_sublane_stashes_the_export_keys():
    """The counterpart of the ratchet above: the lat-lon SPMD dispatch lost its
    blanket refusal ONLY because the operator-split sub-lane exports the three
    keys.  Inspect the symbol that actually RUNS (``_run_operator_split_spmd``,
    not a delegating wrapper), so deleting the export cannot leave the dispatch
    silently unguarded."""
    import inspect
    from legoesm.driver.model_driver import ModelDriver

    src = inspect.getsource(ModelDriver._run_operator_split_spmd)
    for key in ("held_sw_net_sfc", "held_lw_net_sfc", "seg_precip"):
        assert f'self._carry_aux["{key}"]' in src, (
            f"_run_operator_split_spmd no longer stashes {key!r} into "
            "_carry_aux, but the run() dispatch no longer refuses coupled runs "
            "on the lat-lon SPMD lane -- a coupled run would force the surface "
            "with zero shortwave / precip, silently.")
    assert '_nm.endswith("_accum")' in src and "carry._replace(**_reseed)" in src, (
        "_run_operator_split_spmd no longer reseeds the segment accumulators. "
        "This lane threads ONE carry across every segment, so without the "
        "reseed segment_accum_to_rate divides a RUN-total accumulation by a "
        "single segment's duration and the exported precip rate inflates.")
    assert 'static_need_rad=(True if ctx["RAD_UPDATE_STEPS"] <= 1 else None)' in src, (
        "_run_operator_split_spmd pinned static_need_rad back to True. That "
        "DELETES the need_rad predicate the sharded step computes "
        "(physics_pipeline.py), making rad_update_steps>1 a SILENT no-op on "
        "this lane while the serial twin honours it -- and coupled runs reach "
        "this lane. Behavioural gate: tests/parallel/"
        "test_operator_split_spmd_carry_aux_export.py::"
        "test_radiation_cadence_matches_serial.")


@pytest.mark.parametrize("driver_file", [
    "coupled_esm_driver.py",
    "earth_system_driver.py",
])
def test_both_coupled_consumers_are_strict(driver_file):
    """BOTH coupled drivers read held_* out of _carry_aux; hardening only one
    leaves the identical silent-zero bug live on the other."""
    src = (DRIVER_DIR / driver_file).read_text()
    assert "held_sw_net_sfc" in src, f"{driver_file}: consumer moved?"
    assert "require_surface_radiation_aux" in src, (
        f"{driver_file} reads held_sw_net_sfc from _carry_aux but does not "
        "call require_surface_radiation_aux: a lane that stashed nothing "
        "would silently force the surface with zero shortwave / precip.")


@pytest.mark.parametrize("driver_file,ctor", [
    ("coupled_esm_driver.py",
     "self._atm = ModelDriver(atm_config, output_dir=output_dir)"),
    ("earth_system_driver.py",
     "self._atm = ModelDriver(config, output_dir=output_dir)"),
])
def test_coupled_drivers_set_the_marker(driver_file, ctor):
    """Without the marker the lane guards are dead code."""
    src = (DRIVER_DIR / driver_file).read_text()
    assert src.count(ctor) == 1, f"{driver_file}: ctor anchor drifted"
    idx = src.index(ctor)
    assert "_requires_surface_flux_export" in src[idx:idx + 900], (
        f"{driver_file} builds its atmosphere but never sets "
        "_requires_surface_flux_export, so ModelDriver._reject_coupled_lane "
        "would never fire for it.")
