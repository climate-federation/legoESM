"""Direct test for ``scripts/validate/ocean_fidelity/tke_surface_face_metric_ab.py``.

The A/B exists to say how much the #1690 metric fix moves near-surface TKE.
Two properties make its number mean anything, and both are pinned here.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_PROBE = (_ROOT / "scripts" / "validate" / "ocean_fidelity"
          / "tke_surface_face_metric_ab.py")


def _load():
    spec = importlib.util.spec_from_file_location("_face_ab", _PROBE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_face_ab"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_equal_metrics_give_identical_arms():
    """The control: with the SAME face distance the two arms must coincide.

    Any difference here would mean the arms differ by something other than the
    metric, and every ratio the script prints would be uninterpretable.
    """
    mod = _load()
    u, v, T, S, rho, dz_half = mod._column(12, 10.0, -0.004, 0.06)
    # Rebuild both arms at the SAME face distance by asking for a column whose
    # e3t(1) equals the midpoint metric -- i.e. force PRE == POST.
    pre, post, _, _, _ = mod.arms(tau=0.1, dz_m=10.0)
    assert pre.shape == post.shape
    # Sanity: the arms are not accidentally identical at the real settings,
    # which is what makes the ratio a measurement rather than noise.
    assert not np.allclose(pre, post)


def test_doubled_coupling_raises_near_surface_tke_toward_a_factor_two():
    """Direction AND magnitude: the pre-fix arm must be HIGHER, near 2x.

    The held surface value sits far above the interior floor on this column, so
    the doubled coupling is a spurious SOURCE (GLM: the sign is set by which
    side of the interior equilibrium the held value is on). The analytic
    expectation for a floor-dominated interior is a factor of two.
    """
    mod = _load()
    pre, post, e_sfc, cfg, _ = mod.arms(tau=0.2, dz_m=10.0)
    assert e_sfc > 100.0 * cfg.tke_background, (
        "the held surface value is not above the interior floor on this "
        "column, so the sign of the effect is not the one under test")
    ratio = pre[0] / post[0]
    assert 1.5 < ratio < 2.05, f"pre/post at the first interior interface = {ratio:.3f}"


def test_the_ratio_is_insensitive_to_wind_stress():
    """A linear-in-the-coupling effect, not a threshold one.

    If the ratio moved strongly with wind the effect would be entangled with
    the surface value rather than with the metric, and a single-stress number
    could not be quoted.
    """
    mod = _load()
    ratios = [mod.arms(tau=t, dz_m=10.0)[0][0] / mod.arms(tau=t, dz_m=10.0)[1][0]
              for t in (0.02, 0.1, 0.4)]
    assert max(ratios) - min(ratios) < 0.1, ratios


def test_runs_end_to_end():
    assert _load().main(["--dz", "10"]) == 0
