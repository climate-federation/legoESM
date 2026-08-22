"""#1455 verdict360: the 360-day two-model verdict probe's own arithmetic.

The probe owns exactly two pieces of arithmetic that are not imported from a
recorded harness -- the RSS two-sided noise floor and the 2x
INDISTINGUISHABLE rule -- plus the day grids the two sides are sampled on.
Those are what is tested here.  The probe's ``--self-check`` asserts the same
things at run time and is invoked directly so a regression fails in CI rather
than in the middle of an eight-run campaign.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE_DIR = REPO_ROOT / "scripts" / "validate" / "ocean_fidelity" / "dino_1226"


@pytest.fixture(scope="module")
def V():
    sys.path.insert(0, str(PROBE_DIR))
    try:
        import verdict360
        return verdict360
    finally:
        try:
            sys.path.remove(str(PROBE_DIR))
        except ValueError:
            pass


def test_probe_self_check_passes(V):
    assert V._self_check() == 0


def test_two_sided_floor_is_the_rss_of_the_two_measured_spreads(V):
    rows = {"lego": {i: {90: {"x": v}} for i, v in enumerate([1.0, 1.1, 0.9, 1.0])},
            "nemo": {i: {90: {"x": v}} for i, v in enumerate([2.0, 2.6, 1.4, 2.0])}}
    fl, ls, ns = V.two_sided_floor(rows, "x", 90)
    assert ls == pytest.approx(np.std([1.0, 1.1, 0.9, 1.0], ddof=1))
    assert ns == pytest.approx(np.std([2.0, 2.6, 1.4, 2.0], ddof=1))
    assert fl == pytest.approx(np.hypot(ls, ns))
    # a two-sided floor is strictly larger than either side alone -- the
    # failure mode this replaces is quoting one model's floor for a gap
    assert fl > ls and fl > ns


def test_equal_wobble_reduces_the_rss_to_the_sqrt2_difference_factor(V):
    """The 90-day lane's sqrt(2) rule is the equal-spread special case; if this
    stops holding, the two lanes are no longer using the same arithmetic."""
    a = [0.0, 1.0, 2.0, 3.0]
    rows = {"lego": {i: {90: {"x": v}} for i, v in enumerate(a)},
            "nemo": {i: {90: {"x": v + 17.0}} for i, v in enumerate(a)}}
    fl, ls, ns = V.two_sided_floor(rows, "x", 90)
    assert ls == pytest.approx(ns)
    assert fl == pytest.approx(np.sqrt(2.0) * ls)


def test_verdict_rule_accepts_inside_and_rejects_outside_both_signs(V):
    """Non-vacuity: the rule must be able to say `no`, in both directions."""
    assert V.verdict(1.99, 1.0) == "YES"
    assert V.verdict(-1.99, 1.0) == "YES"
    assert V.verdict(2.01, 1.0) == "no"
    assert V.verdict(-2.01, 1.0) == "no"
    assert V.verdict(2.0, 1.0) == "YES"          # the boundary is inclusive


def test_a_zero_floor_is_refused_not_divided_by(V):
    """An ensemble whose members never separated has a floor of exactly zero by
    construction; calling that `indistinguishable` would pass every gap."""
    assert V.verdict(1e9, 0.0) == "NO-FLOOR"
    assert V.verdict(0.0, 0.0) == "NO-FLOOR"


def test_window_mean_uses_the_pre_registered_final_90_days_on_both_sides(V):
    assert V.WINDOW_DAYS == tuple(range(280, 361, 10))
    assert len(V.WINDOW_DAYS) == 9
    rows = {"lego": {0: {d: {"x": float(d)} for d in V.WINDOW_DAYS}}}
    assert V.window_mean(rows, "lego", 0, "x") == pytest.approx(np.mean(V.WINDOW_DAYS))


def test_every_scored_day_has_a_snapshot_on_the_legoesm_side(V):
    """A scored day with no legoESM 3-D snapshot would raise deep inside the
    scoring loop, after hours of integration."""
    assert set(V.SCORE_DAYS) <= set(V.SNAP_GRID)
    assert set(V.HORIZONS) <= set(V.SCORE_DAYS)
    assert set(V.WINDOW_DAYS) <= set(V.SCORE_DAYS)


def test_scored_days_land_on_nemo_restart_dumps(V):
    """NEMO writes a restart every nn_stock=320 steps (10 days).  A scored day
    that is not a multiple of 10 has no NEMO state to compare against."""
    assert all(d % 10 == 0 for d in V.SCORE_DAYS)
    assert V.kt_of(360) == V.KT_END == 17280
    assert V.kt_of(90) == 8640


def test_the_full_section_metric_is_flagged_as_sign_mixing(V):
    """The three latitude groups carry opposite-signed gaps, so the summed
    full-section number must never be read as a per-group verdict."""
    assert "acc" in V.SIGN_MIXING
    assert set(V.SIGN_MIXING) <= set(V.KEYS)
    for k in ("g_south", "g_band", "g_north", "band", "band_c"):
        assert k in V.KEYS and k not in V.SIGN_MIXING


def test_no_two_concurrent_members_share_a_gpu(V, tmp_path, monkeypatch):
    """The scheduler pins a member to a GPU slot and refills the slot only
    after the run occupying it exits.  Picking the GPU from the queue length
    and draining afterwards hands member 3 the device member 1 is still on --
    two 30 GB JAX processes on one 32 GB card.
    """
    live = {}          # gpu -> member currently holding it
    order = []

    class FakePopen:
        def __init__(self, cmd, env=None, stdout=None, stderr=None, cwd=None):
            self.gpu = env["CUDA_VISIBLE_DEVICES"]
            self.member = cmd[cmd.index("--perturb-seed") + 1] \
                if "--perturb-seed" in cmd else "control"
            assert live.get(self.gpu) is None, (
                f"member {self.member} launched on GPU {self.gpu} while "
                f"{live[self.gpu]} still holds it")
            live[self.gpu] = self.member
            order.append((self.member, self.gpu))

        def wait(self):
            live[self.gpu] = None
            return 0

    monkeypatch.setattr(V.subprocess, "Popen", FakePopen)
    V.run_lego(str(tmp_path), ["0", "1"])
    assert len(order) == V.N_MEM
    assert all(v is None for v in live.values())
    # both devices were actually used -- a scheduler that serialises onto one
    # GPU would also pass the no-collision assert
    assert {g for _, g in order} == {"0", "1"}


# ---------------------------------------------------------------------------
# review round 1: the statistics the two reviews required
# ---------------------------------------------------------------------------
def test_permutation_test_floor_is_two_in_seventy(V):
    """A perfectly separated 4-vs-4 can only reach 2/70 = 0.029. Printing that
    as `p = 0.029` without saying it is the design's floor reads like a result
    and is partly a sample-size artifact."""
    sep = {"lego": {i: {90: {"x": float(v)}} for i, v in enumerate([1, 2, 3, 4])},
           "nemo": {i: {90: {"x": float(v)}} for i, v in enumerate([11, 12, 13, 14])}}
    p, obs = V.permutation_p(sep, "x", 90)
    assert p == pytest.approx(2.0 / 70.0)
    assert obs == pytest.approx(10.0)


def test_permutation_test_cannot_manufacture_significance(V):
    ident = {s: {i: {90: {"x": float(v)}} for i, v in enumerate([1, 2, 3, 4])}
             for s in ("lego", "nemo")}
    assert V.permutation_p(ident, "x", 90)[0] == pytest.approx(1.0)


def test_permutation_test_takes_intermediate_values(V):
    """Otherwise it is a two-valued indicator dressed as a p-value."""
    mid = {"lego": {i: {90: {"x": float(v)}} for i, v in enumerate([1, 2, 3, 9])},
           "nemo": {i: {90: {"x": float(v)}} for i, v in enumerate([2, 4, 5, 6])}}
    p = V.permutation_p(mid, "x", 90)[0]
    assert 2.0 / 70.0 < p < 1.0


def test_the_unresolved_band_sits_above_the_registered_rule(V):
    """The registered 2x rule decides; the Welch constant only marks where that
    rule is over-confident about an ESTIMATED floor. If the order ever flipped,
    the advisory would start overriding the pre-registration."""
    assert V.K_PREREG == 2.0
    assert V.K_WELCH > V.K_PREREG
    # and the advisory must never turn a registered YES into anything else
    assert V.verdict(1.5, 1.0) == "YES"


def test_latitude_groups_partition_the_mean_reduced_full_section(V):
    """acc is a MEDIAN over longitudes and the groups are MEANS, so the groups
    can only be checked against the MEAN-reduced full section. all_metrics
    asserts it at score time; this pins that the row exists to check against."""
    assert "acc_mean" in V.KEYS
    assert "acc_mean" not in V.SIGN_MIXING
    assert "acc" in V.SIGN_MIXING


def test_namelist_edit_hits_exactly_one_line_and_never_the_commented_one(
        V, tmp_path, monkeypatch):
    """The single most consequential text edit in the file. The donor namelist
    carries a commented `!nn_itend` directly above the live one; matching it
    would produce a run of the wrong length that still looks correct."""
    donor = tmp_path / "src"
    donor.mkdir()
    (donor / "namelist_cfg").write_text(
        "&namrun\n"
        "   nn_it000    =       5761\n"
        "   !nn_itend    =       32     ! commented decoy\n"
        "   nn_itend    =       8640   ! the live one\n"
        "   nn_stock    =        320\n"
        "/\n")
    (donor / "namelist_ref").write_text("ref\n")
    monkeypatch.setattr(V, "NEMO_SRC", str(donor))
    monkeypatch.setattr(V, "NEMO_CERT", str(tmp_path / "fake_nemo"))
    (tmp_path / "fake_nemo").write_text("binary")
    monkeypatch.setattr(V, "nemo_dir", lambda i: str(tmp_path / f"M{i}"))
    monkeypatch.setattr(V.P, "SRC", tmp_path / "restart.nc")
    (tmp_path / "restart.nc").write_text("restart")
    monkeypatch.setattr(V.P, "perturb", lambda seed, d: {"max_abs_tn_diff": 1e-12})
    V.setup_nemo()
    out = (tmp_path / "M0" / "namelist_cfg").read_text().splitlines()
    live = [ln for ln in out if ln.strip().startswith("nn_itend")]
    decoy = [ln for ln in out if ln.strip().startswith("!nn_itend")]
    assert len(live) == 1 and str(V.KT_END) in live[0]
    assert len(decoy) == 1 and "32" in decoy[0], "the decoy must be untouched"
    # every other line survives byte-for-byte
    src = (donor / "namelist_cfg").read_text().splitlines()
    assert [l for l in out if not l.strip().startswith("nn_itend")] == \
           [l for l in src if not l.strip().startswith("nn_itend")]


def test_setup_refuses_a_directory_that_already_carries_output(
        V, tmp_path, monkeypatch):
    """Scoring a mix of an old run's states and a new run's is a silent
    confound; the pre-registration's `nothing existing is overwritten` was
    false of these four directories."""
    donor = tmp_path / "src"
    donor.mkdir()
    (donor / "namelist_cfg").write_text("   nn_itend    =       8640\n")
    (donor / "namelist_ref").write_text("ref\n")
    monkeypatch.setattr(V, "NEMO_SRC", str(donor))
    monkeypatch.setattr(V, "nemo_dir", lambda i: str(tmp_path / "M0"))
    (tmp_path / "M0").mkdir()
    (tmp_path / "M0" / "time.step").write_text("9999\n")
    with pytest.raises(SystemExit, match="already carries"):
        V.setup_nemo()


def test_scored_day_cadence_matches_the_donor_namelist(V):
    """The restart cadence is nn_stock in the donor namelist, not a constant
    this test is free to invent -- if the source changes, this must go red."""
    import re
    src = f"{V.NEMO_SRC}/namelist_cfg"
    txt = open(src).read()
    m = [ln for ln in txt.splitlines()
         if ln.strip().startswith("nn_stock") and "=" in ln]
    assert len(m) == 1, m
    stock = int(re.search(r"=\s*(\d+)", m[0]).group(1))
    cadence_days = stock // V.G.STEPS_PER_DAY
    assert all(d % cadence_days == 0 for d in V.SCORE_DAYS), cadence_days
