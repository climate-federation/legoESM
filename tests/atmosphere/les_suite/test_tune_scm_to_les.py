"""Tests for the derivative-free SCM→LES tuner (scripts/run/tune_scm_to_les.py).

Runs a small real derivative-free search (builds + runs the single-column SCM
several times), so it is a slower integration test kept intentionally tiny.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from legoesm.atmosphere.les_suite.bridge import LESReferenceArtifact

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "run"))

NZ, NT = 20, 3


def _cbl_artifact() -> LESReferenceArtifact:
    z = np.linspace(10.0, 1500.0, NZ)
    theta_col = np.where(z > 800.0, 300.0 + 0.006 * (z - 800.0), 300.0)
    theta = np.broadcast_to(theta_col, (NT, NZ)).copy()
    return LESReferenceArtifact(
        case_name="cbl_tune", sgs="lasd", heights_m=z,
        times_s=np.linspace(0.0, 600.0, NT), theta=theta,
        u=np.zeros((NT, NZ)), v=np.zeros((NT, NZ)),
        wtheta_resolved=np.zeros((NT, NZ)), wtheta_sgs=np.zeros((NT, NZ)),
        prescribe="fluxes", w_theta_s=np.full(NT, 0.06), f_c=0.0,
    )


def _tuner():
    import tune_scm_to_les  # noqa: PLC0415
    return tune_scm_to_les


def test_candidate_overrides_include_default_and_probes():
    m = _tuner()
    from legoesm.training.param_collector import build_registry
    metas = [x for x in build_registry()
             if x.scheme_key == "atm.turb.MYNN25Config" and x.shape_key is None
             and x.tunable_tier == 1]
    cands = m._candidate_overrides(metas, n_random=3, seed=0)
    # default (empty) + 2 probes per param + 3 random
    assert cands[0] == {}
    assert len(cands) == 1 + 2 * len(metas) + 3
    # every non-default candidate value is within its bound
    bounds = {x.field: x.bounds for x in metas}
    for c in cands[1:]:
        for k, v in c.items():
            lo, hi = bounds[k]
            assert lo - 1e-9 <= v <= hi + 1e-9


def test_tune_returns_best_no_worse_than_default():
    m = _tuner()
    base, scheme_key = m._base_turbulence("mynn25")
    result = m.tune_closure_derivative_free(
        _cbl_artifact(), base, scheme_key,
        tiers=(1,), n_random=2, nlev=NZ, dt=20.0, seed=0,
    )
    assert result.scheme == "mynn25"
    assert np.isfinite(result.best_loss)
    # the default is always among the candidates, so best <= default
    assert result.best_loss <= result.default_loss + 1e-9
    assert result.n_evaluated >= 3
    # best_overrides is a dict of mynn25 fields (or empty if default won)
    assert isinstance(result.best_overrides, dict)


def test_unknown_scheme_rejected():
    m = _tuner()
    with pytest.raises(SystemExit):
        m._base_turbulence("not_a_scheme")


def test_all_wired_schemes_build_with_registry_key():
    # every CBL-wired closure builds a TurbulenceConfig with the right scheme and a
    # registry scheme_key that has scalar tunable params.
    m = _tuner()
    for scheme in ("smagorinsky", "louis", "holtslag_boville", "ysu", "mynn25"):
        base, key = m._base_turbulence(scheme)
        assert base.scheme == scheme
        assert getattr(base, scheme) is not None
        metas = m._scheme_tunable_metas(key, tiers=(1,))
        assert metas  # tier-1 params exist for the ranking


def test_no_tunable_params_raises():
    m = _tuner()
    with pytest.raises(SystemExit):
        m._scheme_tunable_metas("atm.turb.MYNN25Config", tiers=(99,))
