"""#1455 basin-budget lane: the wall balance probe's own arithmetic and gates.

``southern_wall_balance.py`` imports its row reducer, geometry and state loaders
from the recorded harness.  What it OWNS is the meridional sea-surface slope,
its dry-row mask, and the sign of the geostrophic relation — and the sign and
the mask each inverted a verdict once during development, so both are pinned
here.  The oracle's ``mesh_mask.nc`` is required, so the module is skipped where
it is absent rather than mocked.
"""
import os
import sys
from pathlib import Path

import numpy as np

from legoesm import constants
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE_DIR = REPO_ROOT / "scripts" / "validate" / "ocean_fidelity" / "dino_1226"


@pytest.fixture(scope="module")
def W():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(PROBE_DIR))
    sys.path.insert(0, str(PROBE_DIR.parent))
    try:
        try:
            import southern_wall_balance as S
        except (OSError, SystemExit, FileNotFoundError) as exc:
            pytest.skip(f"oracle geometry not available on this host: {exc}")
        return S
    finally:
        for p in (str(PROBE_DIR), str(PROBE_DIR.parent)):
            try:
                sys.path.remove(p)
            except ValueError:
                pass


def _linear_eta(W):
    y = np.concatenate([[0.0], np.cumsum(W._e2v[:-1, 0])])
    return np.repeat((3.7e-7 * y)[:, None], W._e2v.shape[1], axis=1)


def test_probe_gates_pass(W):
    assert W.self_checks(verbose=False) is True


def test_slope_is_exact_on_a_linear_surface(W):
    """The stretched grid makes e2t the wrong denominator — it left a 0.8%
    error.  e2v is the T-to-T distance and must be exact."""
    ok_u = np.zeros_like(W._STENCIL_OK)
    ok_u[:, :-1] = W._STENCIL_OK[:, :-1] & W._STENCIL_OK[:, 1:]
    got = W.deta_dy_at_u(_linear_eta(W))
    assert np.allclose(got[ok_u], 3.7e-7, rtol=0, atol=1e-15)


def test_the_dry_row_cannot_reach_any_scored_row(W):
    """Row 0 is entirely dry and both models store eta = 0.0 there against a wet
    surface near -0.97 m.  Differencing across it produced a wrong-sign
    circulation in row 1 that inverted the branch verdict.  A plant on the dry
    row must move nothing."""
    import southern_circulation_budget as B
    base = np.zeros_like(_linear_eta(W))
    planted = base.copy()
    planted[0, :] = 5.0
    assert np.array_equal(W.row_circulation_ssh(planted)[B.ROWS],
                          W.row_circulation_ssh(base)[B.ROWS])
    assert 1 not in W.VALID_ROWS          # the row the mask must exclude
    assert 2 in W.VALID_ROWS


def test_geostrophic_sign_matches_the_southern_ocean(W):
    """f<0 in the southern hemisphere, so a sea surface rising northward drives
    an EASTWARD flow — the real ACC.  The gate was first written asserting the
    opposite and failed, which is why this is pinned."""
    import southern_circulation_budget as B
    r = W.row_circulation_ssh(_linear_eta(W))
    assert float(np.mean(r[B.ROWS])) > 0
    assert np.allclose(r[B.ROWS], -W.row_circulation_ssh(-_linear_eta(W))[B.ROWS],
                       rtol=0, atol=1e-6)


def test_an_exactly_geostrophic_flow_is_explained_in_full(W):
    """The end-to-end property the verdict rests on: if the velocity IS the
    geostrophic velocity of the sea surface, the fraction explained must be 1.
    This fails if the two reducers ever stop being the same functional."""
    import southern_circulation_budget as B
    eta = _linear_eta(W)
    ug = -W.constants.g * W._INV_F * W.deta_dy_at_u(eta)
    u3 = np.repeat(ug[:, :, None], B.umask.shape[2], axis=2)
    rows = [j for j in B.ROWS if j in W.VALID_ROWS]
    frac = (W.row_circulation_ssh(eta)[rows] / B.row_circulation(u3)[rows])
    assert np.allclose(frac, 1.0, rtol=1e-9, atol=0)


def test_a_non_geostrophic_flow_is_not_explained(W):
    """The complement: a velocity unrelated to the sea surface must NOT be
    explained by it, or the test above is vacuous."""
    import southern_circulation_budget as B
    eta = _linear_eta(W)
    u3 = np.where(B.umask, 0.01, 0.0)
    rows = [j for j in B.ROWS if j in W.VALID_ROWS]
    frac = (W.row_circulation_ssh(eta)[rows] / B.row_circulation(u3)[rows])
    assert np.max(np.abs(frac - 1.0)) > 0.5


def test_the_spacing_gate_is_not_self_referential(W):
    """The analytic slope gate builds y as cumsum(e2v) and divides by e2v, so it
    passes with e2t substituted or the array rolled — a reviewer demonstrated
    both.  S1c pins the spacing against an independent ground truth (the mesh's
    own T-point latitudes) and must fail on either mutation."""
    import acc_thermal_wind as A
    R = constants.R_earth
    gt = np.deg2rad(np.diff(np.asarray(A.gphit, float)[:, 25])) * R
    good = float(np.max(np.abs(W._e2v[:-1, 25] - gt) / gt))
    assert good < 1e-3
    rolled = np.roll(W._e2v, 1, axis=0)
    bad = float(np.max(np.abs(rolled[:-1, 25] - gt) / gt))
    e2t = np.asarray(A.mm["e2t"][0]).squeeze()
    bad2 = float(np.max(np.abs(e2t[:-1, 25] - gt) / gt))
    assert bad > 1e-3 and bad2 > 1e-3        # the gate can reject both


def test_the_one_sided_slope_keeps_the_wall_row_and_touches_no_land(W):
    """Row 1's centred stencil reaches the dry row 0, but rows 1 and 2 are both
    wet, so a forward difference is available and keeps the row with the largest
    single-row difference in the scored set."""
    import southern_circulation_budget as B
    eta = _linear_eta(W)
    centred = W.row_circulation_ssh(eta)
    one_sided = W.row_circulation_ssh(eta, one_sided=True)
    assert centred[1] == 0.0                 # excluded by the centred mask
    assert one_sided[1] != 0.0               # recovered by the forward stencil
    # and it must still be blind to the dry row
    planted = eta.copy()
    planted[0, :] += 5.0
    assert np.array_equal(W.row_circulation_ssh(planted, one_sided=True)[B.ROWS],
                          one_sided[B.ROWS])
