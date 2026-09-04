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

    This is the probe's own control: if it disagrees here the two arms differ
    by something other than the cloud-fraction dilution and every other row is
    uninterpretable.
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


def test_bias_is_far_too_small_to_explain_the_issue_gap():
    """The measured verdict, pinned: this is NOT the missing microns.

    #1521 reports 7.84 um against 11-14 um observed, a gap of 3-6 um.  If this
    mechanism ever grew to that size the negative result below would be wrong
    and the issue would need re-opening on this axis.
    """
    mod = _load()
    worst = max(b - a for cf in (0.9, 0.7, 0.5, 0.3, 0.1)
                for a, b in [mod.arms(5.0e-4, 300.0e6, cf)])
    assert worst < 1.0, (
        f"the in-cloud/grid-mean shape-parameter bias reached {worst:.3g} um, "
        "which is no longer negligible against the 3-6 um deficit in #1521")


def test_probe_runs_end_to_end():
    assert _load().main() == 0
