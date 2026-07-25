"""Tests for the σ_LES aggregator (D7 — LES's own uncertainty in loss units)."""
from __future__ import annotations

import numpy as np
import pytest
from legoesm.atmosphere.les_suite.bridge import LESReferenceArtifact
from legoesm.atmosphere.les_suite.sigma_les import (
    SigmaLESError,
    sigma_les_prognostic,
)

NZ, NT = 24, 3


def _artifact(sgs="lasd", *, case="cbl_x", dtheta=0.0, du=0.0, times=None, moist=False):
    """A minimal dry CBL artifact; dtheta/du perturb θ/u so pairs have a known spread."""
    z = np.linspace(10.0, 1500.0, NZ)
    theta = np.broadcast_to(300.0 + 0.004 * z + dtheta, (NT, NZ)).copy()
    u = np.broadcast_to(1.0 + du, (NT, NZ)).astype(float).copy()
    kw = {}
    if moist:
        kw["qt"] = np.full((NT, NZ), 1e-3)
        kw["wqt_resolved"] = np.zeros((NT, NZ))
        kw["wqt_sgs"] = np.zeros((NT, NZ))
    return LESReferenceArtifact(
        case_name=case, sgs=sgs, heights_m=z,
        times_s=(np.linspace(0.0, 600.0, NT) if times is None else times),
        theta=theta, u=u, v=np.zeros((NT, NZ)),
        wtheta_resolved=np.zeros((NT, NZ)), wtheta_sgs=np.zeros((NT, NZ)),
        prescribe="fluxes", w_theta_s=np.full(NT, 0.06), f_c=0.0, **kw)


def test_identical_variants_have_zero_sigma():
    s = sigma_les_prognostic([_artifact("lasd"), _artifact("smagorinsky")])
    assert s.sigma_combined == pytest.approx(0.0, abs=1e-9)
    assert s.n_pairs == 2  # ordered pairs
    assert s.variants == ("lasd", "smagorinsky")


def test_theta_spread_shows_in_sigma_theta_not_u():
    # two variants differ only in θ → sigma_theta > 0 but sigma_u == 0
    s = sigma_les_prognostic([_artifact("lasd"), _artifact("vreman", dtheta=0.5)])
    assert s.sigma_theta > 0.0
    assert s.sigma_u == pytest.approx(0.0, abs=1e-9)
    assert s.sigma_combined > 0.0


def test_more_variants_more_pairs():
    s = sigma_les_prognostic(
        [_artifact("lasd"), _artifact("smagorinsky", dtheta=0.3),
         _artifact("vreman", dtheta=-0.2)])
    assert s.n_pairs == 6  # 3*(3-1)
    assert len(s.per_pair) == 6


def test_needs_at_least_two():
    with pytest.raises(SigmaLESError):
        sigma_les_prognostic([_artifact("lasd")])


def test_mixed_cases_rejected():
    with pytest.raises(SigmaLESError):
        sigma_les_prognostic([_artifact("lasd", case="a"), _artifact("vreman", case="b")])


def test_mismatched_times_rejected():
    a = _artifact("lasd")
    b = _artifact("vreman", times=np.linspace(0.0, 999.0, NT))
    with pytest.raises(SigmaLESError):
        sigma_les_prognostic([a, b])


def test_moist_scored_with_qt():
    # moist artifacts are now handled: q_t is scored (sigma_qt populated), and a q_t-only
    # spread shows there but not in θ. dry sigma_qt stays None.
    a = _artifact("lasd", moist=True)
    b = _artifact("vreman", moist=True)
    # perturb only q_t on b so the spread lands in sigma_qt
    from dataclasses import replace
    b = replace(b, qt=b.qt + 5.0e-4)
    s = sigma_les_prognostic([a, b])
    assert s.sigma_qt is not None and s.sigma_qt > 0.0
    assert s.sigma_theta == pytest.approx(0.0, abs=1e-9)  # θ_l identical → no θ spread
    assert s.sigma_combined > 0.0
    # a dry set leaves sigma_qt None
    dry = sigma_les_prognostic([_artifact("lasd"), _artifact("vreman", dtheta=0.3)])
    assert dry.sigma_qt is None


def test_mixed_moist_and_dry_rejected():
    with pytest.raises(SigmaLESError):
        sigma_les_prognostic([_artifact("lasd", moist=True), _artifact("vreman")])


def test_non_finite_artifact_rejected():
    # a corrupt (NaN) truth would silently LOWER σ_LES via safe_sqrt → reject loudly
    from dataclasses import replace
    a = _artifact("lasd")
    theta = np.asarray(_artifact("vreman").theta).copy()
    theta[0, 0] = np.nan
    bad = replace(_artifact("vreman"), theta=theta)
    with pytest.raises(SigmaLESError):
        sigma_les_prognostic([a, bad])


def test_mismatched_vertical_extent_rejected():
    from dataclasses import replace
    a = _artifact("lasd")
    # a run over a different domain top → extrapolation risk → reject
    z2 = np.linspace(10.0, 3000.0, NZ)
    b = replace(_artifact("vreman"), heights_m=z2)
    with pytest.raises(SigmaLESError):
        sigma_les_prognostic([a, b])


def test_per_pair_is_truth_then_candidate():
    s = sigma_les_prognostic([_artifact("lasd"), _artifact("vreman", dtheta=0.4)])
    # ordered pairs: (lasd→vreman) and (vreman→lasd); first element is the TRUTH label
    truths = {p[0] for p in s.per_pair}
    assert truths == {"lasd", "vreman"}
    assert all(p[0] != p[1] for p in s.per_pair)
