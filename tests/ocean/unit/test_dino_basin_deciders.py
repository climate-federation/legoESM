"""#1455 basin-budget lane: the two-decider probe's own arithmetic and gates.

``southern_basin_deciders.py`` imports its drag transcription, row reducers and
state loaders from the recorded harness; what it OWNS is the pass/fail bins, the
bottom-velocity reduction whose SIGN decides how the drag result is read, and
the gates that decide whether either number may be printed at all.  Those are
tested here, together with the property the whole D1 reading rests on: that the
recorded drag functional is a sink that responds non-linearly to the flow.

The oracle's ``mesh_mask.nc`` is required (the probe's geometry is the
oracle's), so the module is skipped where it is absent rather than mocked.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE_DIR = REPO_ROOT / "scripts" / "validate" / "ocean_fidelity" / "dino_1226"


@pytest.fixture(scope="module")
def P():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(PROBE_DIR))
    sys.path.insert(0, str(PROBE_DIR.parent))
    try:
        try:
            import southern_basin_deciders as S
        except (OSError, SystemExit, FileNotFoundError) as exc:
            pytest.skip(f"oracle geometry not available on this host: {exc}")
        return S
    finally:
        for p in (str(PROBE_DIR), str(PROBE_DIR.parent)):
            try:
                sys.path.remove(p)
            except ValueError:
                pass


def test_probe_gates_pass(P):
    assert P.self_checks(verbose=False) is True


def test_the_drag_gate_would_fail_on_a_linear_law(P, monkeypatch):
    """The S1 gate brackets the doubling response strictly inside (2, 4) and
    within 5% of the non-linear prediction.  A LINEAR drag law gives exactly 2
    and must be rejected — otherwise the gate is decoration."""
    import southern_circulation_budget as B
    orig = B.phi_drag
    monkeypatch.setattr(B, "phi_drag", lambda u, v: orig(u, v) * 0 - _linear(B, u))
    assert P.self_checks(verbose=False) is False


def _linear(B, u):
    """A linear-drag row torque: r = Cd0*sqrt(ke0), independent of the speed."""
    r = float(B.DRAG_KW["cd0"]) * np.sqrt(float(B.DRAG_KW["ke0"]))
    us = np.where(B.umask, np.asarray(u, np.float64), 0.0)
    u_bot = np.take_along_axis(us, B.KBOT_U[:, :, None], axis=2)[:, :, 0]
    col = B.umask.any(axis=2)
    return np.sum(np.where(col, r * u_bot * B.e1u, 0.0), axis=1)


def test_bottom_velocity_reduction_has_the_sign_of_the_flow(P):
    """The whole D1 reading turns on this sign; it is measured, not assumed."""
    import southern_circulation_budget as B
    east = np.where(B.umask, 0.02, 0.0)
    assert P._ubot_band(east) == pytest.approx(0.02, rel=1e-12)
    assert P._ubot_band(-east) == pytest.approx(-0.02, rel=1e-12)


def test_drag_is_a_sink_whichever_way_the_flow_goes(P):
    """Eastward bottom flow -> negative (westward) torque; westward bottom flow
    -> positive torque.  D1's sign argument is invalid if this is not true."""
    import southern_circulation_budget as B
    v = np.zeros_like(B.vmask, dtype=np.float64)
    east = B.phi_drag(np.where(B.umask, 0.02, 0.0), v)
    west = B.phi_drag(np.where(B.umask, -0.02, 0.0), v)
    assert float(np.mean(east[B.ROWS])) < 0
    assert float(np.mean(west[B.ROWS])) > 0
    assert float(np.mean(east[B.ROWS])) == pytest.approx(
        -float(np.mean(west[B.ROWS])), rel=1e-10)


def test_registered_bins_are_fractions_of_the_measured_shortfall(P):
    """The thresholds must be tied to the quantity they judge, not to loose
    constants that could drift away from it."""
    assert P.D1_CONFIRM == pytest.approx(0.5 * P.DEFICIT)
    assert P.D1_REFUTE == pytest.approx(0.2 * P.DEFICIT)
    assert P.D1_REFUTE < P.D1_CONFIRM
    assert P.D2_CONFIRM < P.D2_REFUTE


def test_the_wall_rows_are_inside_the_scored_band(P):
    import southern_circulation_budget as B
    assert set(P.WALL_ROWS).issubset(set(B.ROWS))


def test_the_year_mean_is_time_weighted_not_a_plain_mean(P):
    """The 19 sample days are 30 d apart to day 270 and 10 d apart after it, so
    ten of them sit in the last quarter.  A plain mean over that list overweights
    the end of the year — it did, by 17%, in the first revision.  This pins the
    weighting on a series whose true average is known exactly."""
    days = np.asarray((0,) + P.DAYS, float)
    ramp = days[1:] / days[-1]                  # exact time-average 0.5
    # the probe's own helper, so reverting d1() to a plain mean fails this
    assert P.time_weighted_mean(ramp) == pytest.approx(0.5, abs=1e-12)
    assert float(np.mean(ramp)) > 0.55          # a plain mean is biased high
    # and it must weight by the day spacing, not by sample count
    assert P.time_weighted_mean(np.ones(len(P.DAYS))) == pytest.approx(
        1.0 - 0.5 * P.DAYS[0] / P.DAYS[-1], abs=1e-12)


def test_node_finder_locates_a_known_crossing(P):
    """D0's amplitude-vs-position reading turns on the node, so the interpolation
    is pinned to a profile whose crossing is known by construction."""
    import southern_circulation_budget as B
    rows = list(B.ROWS)
    prof = np.zeros(max(rows) + 2)
    for k, j in enumerate(rows):
        prof[j] = -1.0 + 2.0 * k / (len(rows) - 1)   # crosses mid-way
    node = P._node_of(prof, rows)
    assert node == pytest.approx(rows[0] + (len(rows) - 1) / 2.0, abs=0.02)


def test_a_shifted_profile_moves_the_node(P):
    """A position error must move the node; an amplitude error must not.  If this
    fails, D0 cannot tell the two apart and its reading is worthless."""
    import southern_circulation_budget as B
    rows = list(B.ROWS)
    base = np.zeros(max(rows) + 2)
    for k, j in enumerate(rows):
        base[j] = -1.0 + 2.0 * k / (len(rows) - 1)
    n0 = P._node_of(base, rows)
    amp = base.copy()
    amp[rows[0]:rows[0] + 4] *= 1.5             # amplitude only, wall lobe
    assert P._node_of(amp, rows) == pytest.approx(n0, abs=1e-9)
    # a genuine POSITION error: the same shape, its crossing moved two rows
    shifted = np.zeros(max(rows) + 2)
    for k, j in enumerate(rows):
        shifted[j] = -1.0 + 2.0 * (k + 2) / (len(rows) - 1)
    assert abs(P._node_of(shifted, rows) - n0) > 1.5



def test_node_finder_refuses_a_profile_it_cannot_read(P):
    """np.argmax on an all-False array returns 0, which would report a node at
    the first row for a profile that never crosses.  Both the no-crossing and
    the multiple-crossing cases must be fatal, not guessed."""
    import southern_circulation_budget as B
    rows = list(B.ROWS)
    nocross = np.zeros(max(rows) + 2)
    for j in rows:
        nocross[j] = 1.0
    with pytest.raises(SystemExit):
        P._node_of(nocross, rows)
    zig = np.zeros(max(rows) + 2)
    for k, j in enumerate(rows):
        zig[j] = (-1.0) ** k
    with pytest.raises(SystemExit):
        P._node_of(zig, rows)
