"""Direct unit tests for the reference-free SCM-RCE ice-budget tuning driver
(``scripts/run/run_scm_rce_ice_tuning.py``).

Covers the host-side, non-JAX-heavy contracts (the actual rrtmgp+morrison column
integration is exercised by the sbatch smoke, not here):
  * the morrison __param_spec__ lookup finds the ice-budget levers;
  * overrides are bounds-checked against the spec (dispatch-hardening: bogus
    field or out-of-bounds value raises);
  * make_cfg applies overrides via MorrisonConfig._replace WITHOUT mutating the
    production MorrisonConfig() defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_SCRIPTS = str(Path(_ROOT) / "scripts" / "run")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import run_scm_rce_ice_tuning as rt  # noqa: E402


def test_morrison_spec_has_levers():
    spec = rt._morrison_spec()
    for lever in rt.ICE_LEVERS:
        assert lever in spec, lever
        assert "bounds" in spec[lever]


def test_ice_levers_within_spec_bounds():
    spec = rt._morrison_spec()
    for name, (default, values) in rt.ICE_LEVERS.items():
        lo, hi = spec[name]["bounds"]
        for v in [default, *values]:
            assert lo <= v <= hi, f"{name}={v} outside [{lo},{hi}]"


def test_assert_in_bounds_rejects_bogus_field():
    with pytest.raises(ValueError):
        rt._assert_in_bounds({"not_a_morrison_field": 1.0})


def test_assert_in_bounds_rejects_out_of_range():
    with pytest.raises(ValueError):
        rt._assert_in_bounds({"homogeneous_freeze_T": 1000.0})


def test_make_cfg_applies_override_without_mutating_default():
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig

    default_T = MorrisonConfig().homogeneous_freeze_T
    cfg = rt.make_cfg({"homogeneous_freeze_T": 240.0}, rad_interval=12)
    assert cfg.microphysics.scheme == "morrison"
    assert cfg.microphysics.morrison.homogeneous_freeze_T == pytest.approx(240.0)
    assert cfg.radiation.scheme == "rrtmgp"
    assert cfg.radiation.cloud_scheme == "resolved"
    # Production default is untouched (no in-place mutation).
    assert MorrisonConfig().homogeneous_freeze_T == pytest.approx(default_T)
    assert default_T != pytest.approx(240.0)


def test_build_initial_profiles_shape_and_monotone():
    import numpy as np
    T, qv = rt.build_initial_profiles(40, 300.0)
    assert T.shape == (40,) and qv.shape == (40,)
    T = np.asarray(T); qv = np.asarray(qv)
    assert np.all(np.isfinite(T)) and np.all(np.isfinite(qv))
    assert float(T[-1]) > float(T[0])   # warmer at surface (bottom index)
    assert float(qv[-1]) > float(qv[0])  # moister near surface


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
