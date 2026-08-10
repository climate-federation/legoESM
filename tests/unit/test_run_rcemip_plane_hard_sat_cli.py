"""CLI round-trip: the super-saturation guard is reachable in the RCE/CRM lane.

``run_rcemip_plane.py`` is the driver for the configuration that motivated the
whole uniform-guard change -- an RCEMIP initial sounding at ~140 % relative
humidity, which produced a persistent column-water drift. It offered
``--microphysics {kessler, sundqvist, ..., ml_emulator}`` but exposed NO flag
for ``hard_saturation_adjustment`` at all, for ANY scheme, so the guard could
only be enabled by editing code in the one lane where it is most needed.

These tests pin that the flags exist, thread onto the ACTIVE sub-config only,
are byte-identical when absent, and fail LOUDLY rather than sitting inert.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_SRC = (pathlib.Path(__file__).resolve().parents[2]
        / "scripts/run/run_rcemip_plane.py")


def _load():
    """Import the driver by path (scripts/ is not an importable package)."""
    if "run_rcemip_plane" in sys.modules:
        return sys.modules["run_rcemip_plane"]
    spec = importlib.util.spec_from_file_location("run_rcemip_plane", _SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_rcemip_plane"] = mod
    spec.loader.exec_module(mod)
    return mod


GUARDED = ("kessler", "sundqvist", "seifert_beheng", "morrison", "thompson",
           "p3", "ml_emulator")


@pytest.mark.parametrize("scheme", GUARDED)
def test_flag_threads_onto_the_active_subconfig(scheme):
    rcp = _load()
    cfg = rcp._build_microphysics_config(
        scheme, hard_saturation_adjustment=True)
    assert getattr(cfg, scheme).hard_saturation_adjustment is True
    # ... and ONLY the active scheme is touched: an inactive sub-config that
    # silently flipped would change behaviour on a later --microphysics switch.
    for other in GUARDED:
        if other != scheme:
            assert getattr(cfg, other).hard_saturation_adjustment is False


@pytest.mark.parametrize("scheme", GUARDED)
def test_default_is_byte_identical_to_the_historical_builder(scheme):
    """No flags => the builder returns exactly what it always returned."""
    rcp = _load()
    assert (rcp._build_microphysics_config(scheme)
            == rcp._build_microphysics_config(
                scheme, hard_saturation_adjustment=False))
    assert getattr(
        rcp._build_microphysics_config(scheme),
        scheme).hard_saturation_adjustment is False


def test_float_overrides_round_trip():
    rcp = _load()
    cfg = rcp._build_microphysics_config(
        "sundqvist", hard_saturation_adjustment=True,
        hard_sat_adjust_threshold=1.25, hard_sat_max_heating_K=8.0)
    assert cfg.sundqvist.hard_sat_adjust_threshold == pytest.approx(1.25)
    assert cfg.sundqvist.hard_sat_max_heating_K == pytest.approx(8.0)


def test_float_override_without_the_gate_is_refused():
    """A threshold set without the boolean would be SILENTLY INERT (the schemes
    branch on a static ``if``), so it must raise, not no-op."""
    rcp = _load()
    with pytest.raises(ValueError, match="requires --hard-saturation-adjustment"):
        rcp._build_microphysics_config(
            "kessler", hard_sat_adjust_threshold=1.2)
    with pytest.raises(ValueError, match="requires --hard-saturation-adjustment"):
        rcp._build_microphysics_config(
            "kessler", hard_sat_max_heating_K=8.0)


@pytest.mark.parametrize("scheme", ["sdm", "fast_sbm"])
def test_exempt_schemes_reject_the_flag_loudly(scheme):
    """sdm/fast_sbm resolve super-saturation explicitly and carry no guard
    fields; asking for it must raise via the SHARED helper, never silently do
    nothing."""
    rcp = _load()
    with pytest.raises(ValueError, match="hard_saturation_adjustment"):
        rcp._build_microphysics_config(
            scheme, hard_saturation_adjustment=True)


def test_morrison_flavor_survives_the_guard_threading():
    """The morrison branch builds a bespoke sub-config (SAM flavor for the gSAM
    oracle); the guard must be layered ON TOP of it, not replace it."""
    rcp = _load()
    cfg = rcp._build_microphysics_config(
        "morrison", homogeneous_ice_nucleation=True,
        hard_saturation_adjustment=True)
    assert cfg.morrison.morrison_flavor == "sam"
    assert cfg.morrison.homogeneous_ice_nucleation is True
    assert cfg.morrison.hard_saturation_adjustment is True


def test_flags_round_trip_through_real_argv(monkeypatch):
    """End-to-end through ``parse_args()`` -- the parser is built inline, so
    this is the only way to prove the flags are actually WIRED to argv and not
    merely accepted as keyword arguments by the builder."""
    rcp = _load()
    monkeypatch.setattr(sys, "argv", [
        "run_rcemip_plane.py", "--microphysics", "sundqvist",
        "--hard-saturation-adjustment",
        "--hard-sat-adjust-threshold", "1.3",
        "--hard-sat-max-heating-k", "7.5",
    ])
    args = rcp.parse_args()
    assert args.hard_saturation_adjustment is True
    assert args.hard_sat_adjust_threshold == pytest.approx(1.3)
    assert args.hard_sat_max_heating_k == pytest.approx(7.5)

    # ... and the parsed values reach the sub-config the scheme reads.
    cfg = rcp._build_microphysics_config(
        args.microphysics, args.homogeneous_ice_nucleation,
        hard_saturation_adjustment=args.hard_saturation_adjustment,
        hard_sat_adjust_threshold=args.hard_sat_adjust_threshold,
        hard_sat_max_heating_K=args.hard_sat_max_heating_k)
    assert cfg.sundqvist.hard_saturation_adjustment is True
    assert cfg.sundqvist.hard_sat_adjust_threshold == pytest.approx(1.3)
    assert cfg.sundqvist.hard_sat_max_heating_K == pytest.approx(7.5)


def test_defaults_from_real_argv_leave_the_guard_off(monkeypatch):
    """Non-vacuity for the test above: with no flags the parsed defaults are
    off/None, so the round-trip is not passing on hardcoded values."""
    rcp = _load()
    monkeypatch.setattr(sys, "argv", ["run_rcemip_plane.py"])
    args = rcp.parse_args()
    assert args.hard_saturation_adjustment is False
    assert args.hard_sat_adjust_threshold is None
    assert args.hard_sat_max_heating_k is None


def test_none_scheme_still_returns_none():
    rcp = _load()
    assert rcp._build_microphysics_config("none") is None
