"""Direct test for ``scripts/validate/cloud_reff_incloud_vs_gridmean.py`` (#1521).

The probe compares the effective radius the model computes (droplet number and
cloud water both grid-mean) against the one it would compute with the in-cloud
droplet number fed to the PSD shape parameter.  Two properties decide whether
its number means anything, and both are pinned here.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PROBE = _ROOT / "scripts" / "validate" / "cloud_reff_incloud_vs_gridmean.py"


def _load():
    spec = importlib.util.spec_from_file_location("_reff_probe", _PROBE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_reff_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_overcast_column_is_the_built_in_control():
    """At cf = 1 the two arms are the SAME inputs, so they must agree exactly.

    Codex is right that this is close to tautological — it is a wiring check,
    not a physics one: it catches an arm that accidentally differs by
    something OTHER than the dilution (a stray config, a transposed argument).
    The load-bearing controls are the sign test and the box bound below.
    """
    mod = _load()
    for qc, nc in ((3.0e-4, 100.0e6), (5.0e-4, 300.0e6)):
        a, b = mod.arms(qc, nc, 1.0)
        assert a == pytest.approx(b, rel=1e-12), (
            "the two arms disagree on an overcast column")


def test_broken_cloud_biases_the_radius_small_not_large():
    """Non-vacuity + sign: at broken cover the model's radius must be SMALLER.

    The Martin fit gives a larger shape parameter at lower droplet
    concentration, and a larger shape parameter gives a smaller effective
    radius — so diluting the number by cloud fraction can only shrink it.  A
    zero or positive bias would mean the effect under study is absent or
    reversed, and the issue's attribution would need rethinking.
    """
    mod = _load()
    a, b = mod.arms(5.0e-4, 300.0e6, 0.3)
    assert a < b, "grid-mean-number radius is not smaller at broken cover"
    assert (b - a) > 0.1, (
        f"bias is only {b - a:.3g} um at cf = 0.3 — too small for this probe "
        "to resolve anything")


def test_bias_cannot_close_the_issue_gap_on_its_own():
    """The measured verdict, pinned: this cannot BE the deficit on its own.

    #1521 reports 7.84 um against 11-14 um observed, a gap of 3-6 um.

    The bound is over the whole REACHABLE (q_c, N_c, cf) box, not over two
    hand-picked columns: a bound from two invented states says nothing about
    the states the model visits (codex review, HIGH -- and it was right, the
    two-archetype bound of "under 1 um" was WRONG; the box reaches 1.80 um).

    1.80 um is a third to a half of the low end of the deficit, at an extreme
    corner (2.0 g/kg in-cloud water under 5% cover).  So this mechanism is not
    negligible, and it is not the whole story either.  The threshold below is
    set from the MEASURED worst case with headroom: it goes red if the effect
    grows enough to be the deficit on its own, which would reverse the
    conclusion.

    It is still not the radiatively weighted bias in a run -- that needs the
    model's own joint distribution of (q_c, N_c, cf), which only output can
    supply.
    """
    mod = _load()
    worst, where = mod.worst_over_reachable_box()
    assert worst < 2.5, (
        f"the in-cloud/grid-mean shape-parameter bias reaches {worst:.3g} um "
        f"at {where[:3]} — large enough to be the #1521 deficit on its own, "
        "which reverses the conclusion recorded here")


def test_the_reachable_box_sweep_is_not_trivially_flat():
    """Non-vacuity for the bound: the box must contain a real bias somewhere.

    A sweep that found ~0 everywhere would satisfy the bound above while
    proving nothing.
    """
    mod = _load()
    worst, _ = mod.worst_over_reachable_box()
    assert worst > 0.2, (
        f"the whole reachable box produced a worst bias of {worst:.3g} um — "
        "too small for the bound above to be a meaningful statement")


def test_probe_runs_end_to_end():
    assert _load().main() == 0
