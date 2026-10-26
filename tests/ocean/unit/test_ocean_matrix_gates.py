"""Gates in the ocean test matrix that were rebuilt from measurement.

Two gates were wrong in ways that inverted their verdicts, and both are
rebuilt here against numbers measured on 2026-08-11 rather than chosen.

1. TRACER DRIFT was one absolute number (1e-8) applied to every arm. That
   is the z-star figure; an arm integrating a LINEAR free surface carries
   the free-surface volume change as a surface concentration/dilution term
   and cannot meet it. The FESOM arm was failing geostrophic adjustment on
   a drift of 3.41e-6 over 10 days -- 3.4e-7 per day, entirely expected for
   linfs. The tolerance is now keyed on the MODELLING MODE, never on the
   arm's name, so every model in a mode is held to the same number.

2. The INERTIA-GRAVITY-WAVE L2 gate rewarded an arm for doing nothing.
   ``omega * t_final`` = 2.997 periods, so the analytic solution at the end
   time is within 0.3% of the initial condition, and "close to analytic"
   collapses into "close to your own IC". The cubed-sphere arm -- whose
   free surface was frozen, correlation 1.00 with its IC at EVERY output
   while |u| grew from 0.246 to 0.675 m/s -- scored the BEST L2 of any arm.
   L2 is now reported and ungated; the case gates on stability and on
   having actually propagated.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]


def _matrix():
    if "_rm_gates" in sys.modules:
        return sys.modules["_rm_gates"]
    p = _REPO / "scripts" / "matrix" / "run_ocean_test_matrix.py"
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location("_rm_gates", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_rm_gates"] = mod
    spec.loader.exec_module(mod)
    return mod


class _Cfg:
    def __init__(self, vc):
        self.vertical_coordinate = vc


# ---------------------------------------------------------------------------
# Tracer drift: keyed on the MODE, never on the arm
# ---------------------------------------------------------------------------

def test_mode_is_read_from_the_coordinate_not_the_grid_name():
    M = _matrix()
    assert M._modelling_mode("fesom", _Cfg("linfs")) == "fixed_thickness"
    # The SAME arm in z-star is held to the tight number -- proof the
    # tolerance follows the physics and is not a FESOM exemption.
    assert M._modelling_mode("fesom", _Cfg("zstar")) == "moving_thickness"
    for g in ("latlon", "mpas", "tripole"):
        assert M._modelling_mode(g, None) == "moving_thickness"


def test_unknown_vertical_coordinate_raises():
    """Defaulting would either hold a linfs arm to a tolerance it cannot
    meet, or hide a real leak on a z-star one."""
    M = _matrix()
    with pytest.raises(ValueError, match="vertical_coordinate"):
        M._modelling_mode("fesom", _Cfg("sigma"))


def test_tolerance_is_per_day_and_mode_dependent():
    M = _matrix()
    tight = M._tracer_drift_tolerance("latlon", 10.0, None)
    loose = M._tracer_drift_tolerance("fesom", 10.0, _Cfg("linfs"))
    assert loose > tight, "linfs must get the looser bound"
    # Per DAY: an absolute tolerance would let a short run hide a leak a
    # long one fails on.
    one = M._tracer_drift_tolerance("latlon", 1.0, None)
    assert M._tracer_drift_tolerance("latlon", 10.0, None) == pytest.approx(
        10.0 * one)
    # SUB-day too: max(days, 1.0) used to hand a 0.1-day quick run a full
    # day's budget, a 10x weakening of the shortest runs (codex).
    assert M._tracer_drift_tolerance("latlon", 0.1, None) == pytest.approx(
        0.1 * one)
    with pytest.raises(ValueError):
        M._tracer_drift_tolerance("latlon", 0.0, None)


def test_the_measured_arms_land_on_the_right_side():
    """The numbers that motivated the change, pinned.

    FESOM linfs geostrophic adjustment drifted 3.41e-6 over 10 days and was
    being failed; lat-lon z-star drifts ~1e-15 and must still be held
    tight.
    """
    M = _matrix()
    fesom_tol = M._tracer_drift_tolerance("fesom", 10.0, _Cfg("linfs"))
    assert 3.41e-6 < fesom_tol, "the measured linfs drift must now pass"
    latlon_tol = M._tracer_drift_tolerance("latlon", 10.0, None)
    assert 3.41e-6 > latlon_tol, (
        "a z-star arm drifting like the linfs one must still FAIL, or the "
        "change is an exemption rather than a mode split")


# ---------------------------------------------------------------------------
# IGW: stability + "did it actually propagate"
# ---------------------------------------------------------------------------

def test_igw_metrics_helper_on_synthetic_fields():
    """Drive the REAL helper, not a comparison of two literals.

    Three synthetic fields whose verdict is known by construction:
    propagating (phase-shifted), frozen (identical), extinguished (tiny).
    """
    M = _matrix()
    n = 2000
    x = np.linspace(0, 4 * np.pi, n)
    ic = np.cos(x)
    wet = np.ones(n, bool)

    amp, corr, _ = M._igw_gate_metrics(np.cos(x + 1.7), ic, wet)   # propagated
    assert amp == pytest.approx(1.0, rel=1e-6)
    assert abs(corr) < 0.95 and amp <= M._IGW_AMP_UPPER \
        and amp >= M._IGW_AMP_FLOOR

    amp, corr, _ = M._igw_gate_metrics(0.81 * ic, ic, wet)         # frozen
    assert corr == pytest.approx(1.0, rel=1e-9)
    assert not (corr <= M._IGW_IC_CORR_UPPER), "a frozen field must FAIL"

    amp, corr, _ = M._igw_gate_metrics(1e-6 * np.cos(x + 0.3), ic, wet)
    assert not (amp >= M._IGW_AMP_FLOOR), (
        "an extinguished wave decorrelates just like a propagating one, so "
        "the amplitude FLOOR is the only thing that catches it")

    amp, corr, _ = M._igw_gate_metrics(np.zeros(n), ic, wet)       # dead
    assert corr == float("inf") and not (corr <= M._IGW_IC_CORR_UPPER), (
        "a flat field must fail; NaN would silently pass a '<=' gate")


def test_igw_metrics_helper_ignores_land():
    """Land holds the same constant in both fields. With a non-zero ocean
    mean that identical block drags the correlation toward 1 -- toward the
    'frozen' verdict -- and it also corrupts the peak amplitude."""
    M = _matrix()
    rng = np.random.default_rng(0)
    n, nland = 1000, 800
    ocean_f = 5.0 + rng.normal(size=n - nland)
    ocean_0 = 5.0 + rng.normal(size=n - nland)
    f = np.concatenate([ocean_f, np.zeros(nland)])
    i0 = np.concatenate([ocean_0, np.zeros(nland)])
    wet = np.concatenate([np.ones(n - nland, bool), np.zeros(nland, bool)])
    _a_m, corr_masked, _ = M._igw_gate_metrics(f, i0, wet)
    _a_u, corr_unmasked, _ = M._igw_gate_metrics(f, i0, None)
    assert abs(corr_unmasked) > 0.9, "fixture must show the bias"
    assert abs(corr_masked) < 0.3, (
        f"masking must remove it (masked {corr_masked:.2f} vs unmasked "
        f"{corr_unmasked:.2f})")


def test_measured_arms_pass_the_rebuilt_igw_gates():
    """The four ocean arms, as MEASURED 2026-08-11 after masking."""
    M = _matrix()
    for arm, amp, corr in (("latlon", 0.379, 0.487), ("mpas", 0.435, 0.281),
                           ("fesom", 2.386, 0.227), ("tripole", 2.647, 0.324)):
        assert M._IGW_AMP_FLOOR <= amp <= M._IGW_AMP_UPPER, f"{arm} amplitude"
        assert corr <= M._IGW_IC_CORR_UPPER, f"{arm} correlation"


def test_igw_l2_is_no_longer_gated():
    """L2-vs-analytic must be reported, not gated: it is an f-plane plane
    wave compared against full-sphere integrations, and at this end time it
    is within 0.3% of the IC."""
    M = _matrix()
    src = (_REPO / "scripts" / "matrix" / "run_ocean_test_matrix.py").read_text()
    # EXACT signature, not a name prefix. "def run_inertia_gravity_wave"
    # also matches run_inertia_gravity_wave_CHANNEL, which is defined
    # earlier in the file, so the prefix form silently inspected the wrong
    # function and this test asserted about a body it had never meant to
    # read (found 2026-08-13, the same class as asserting against a
    # delegating wrapper).
    anchor = "def run_inertia_gravity_wave(tc: TestCase"
    assert src.count(anchor) == 1, "the global IGW runner's signature moved"
    body = src[src.index(anchor):]
    body = body[:body.index("\ndef ")]
    assert "L2_UNGATED" in body
    assert 'label="IGW L2 vs analytical"' not in body, (
        "the L2 gate is back; it ranks a frozen dycore best")


def test_igw_end_time_really_is_near_an_integer_period():
    """The fact that makes the old gate degenerate, pinned so nobody
    'fixes' the ungating without re-deriving it."""
    omega = 1.09e-4          # coeff-ok: the case's own dispersion relation
    t_final = 2.0 * 86400.0
    cycles = omega * t_final / (2.0 * np.pi)
    assert abs(cycles - round(cycles)) < 0.01, (
        f"end time is {cycles:.3f} periods; if the case's duration or depth "
        f"changed this is no longer resonant and L2 could be gated again")


# ---------------------------------------------------------------------------
# barotropic_wave: gated on ENERGY, not on the global peak
# ---------------------------------------------------------------------------

def test_barotropic_energy_thresholds_sit_outside_the_measured_spread():
    """The failure mode of the OLD gates: both thresholds fell inside the
    arms' natural scatter (eta_cons 0.454-0.554 against a 0.5 gate;
    min_final_amp 0.098-0.202 against a 0.1 m gate), so which arm failed
    was decided by sampling phase."""
    M = _matrix()
    measured = {"latlon": 0.206, "mpas": 0.215, "fesom": 0.187,
                "tripole": 0.368}       # final/initial <eta^2>, 2026-08-11
    for arm, r in measured.items():
        assert M._BWAVE_ENERGY_FLOOR < r < M._BWAVE_ENERGY_CEILING, arm
    assert M._BWAVE_ENERGY_FLOOR < min(measured.values()) / 5, (
        "the floor must sit well clear of the spread, not through it")


def test_barotropic_energy_gate_still_catches_the_real_failures():
    M = _matrix()
    assert not (0.001 >= M._BWAVE_ENERGY_FLOOR), "extinction must fail"
    assert not (3.0 <= M._BWAVE_ENERGY_CEILING), "spurious growth must fail"


def test_eta_variance_is_area_weighted_and_ocean_only():
    """A plain mean over a lat-lon array over-counts the poles, and land
    must not enter a PE proxy at all."""
    M = _matrix()

    class _F:
        def __init__(self, d): self.data = d

    class _S:
        pass

    nlat, nlon = 10, 20
    eta = np.ones((nlat, nlon))
    eta[:2, :] = 100.0                       # a huge value, on LAND
    mask = np.ones((nlat, nlon)); mask[:2, :] = 0.0
    st = _S(); st.eta = _F(eta); st.land_mask = _F(mask)

    class _G:
        area = np.ones(nlat * nlon)
    val = M._area_weighted_eta_var(st, "latlon", _G())
    assert val == pytest.approx(1.0), (
        f"land leaked into the PE proxy (got {val})")


# ---------------------------------------------------------------------------
# phillips: gated only where the IC is a steady state
# ---------------------------------------------------------------------------

def test_phillips_growth_gated_only_on_aquaplanet_arms():
    M = _matrix()
    assert set(M._AQUAPLANET_OCEAN_GRIDS) == {"latlon", "mpas"}
    src = (_REPO / "scripts" / "matrix" / "run_ocean_test_matrix.py").read_text()
    body = src[src.index("def run_phillips_two_layer"):]
    body = body[:body.index("\ndef ")]
    assert "if tc.grid_type in _AQUAPLANET_OCEAN_GRIDS:" in body
    assert "eta_growth UNGATED on a real-geometry arm" in body
    # Non-vacuity: the measured real-geometry values must be ones the gate
    # WOULD have failed, or making it asymmetric changed nothing.
    for growth in (21.052, 33.928):
        assert growth > 10.0
    # ... and the aquaplanet values must still be inside the band it keeps.
    for growth in (2.793, 2.366):
        assert 0.8 <= growth <= 10.0
