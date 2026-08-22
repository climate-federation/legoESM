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


# ---------------------------------------------------------------------------
# review round 2: the scoring layer driven END TO END on synthetic members
# ---------------------------------------------------------------------------
def _synth(V, gap=0.0, tie_day=None, tie_metric="smax", grow=False):
    """Four members a side on every scored day.

    Member i gets a deterministic offset so the ensemble has a real spread; the
    legoESM side is additionally displaced by `gap`. `tie_day`/`tie_metric`
    plant a two-member tie so the tie handling can be exercised; `grow` makes
    the spread keep growing so the saturation test must say NO.
    """
    rows = {"lego": {i: {} for i in range(V.N_MEM)},
            "nemo": {i: {} for i in range(V.N_MEM)}}
    for day in V.SCORE_DAYS:
        scale = (day / 360.0) ** 3 if grow else min(day, 180) / 180.0
        for side in ("lego", "nemo"):
            for i in range(V.N_MEM):
                for k in V.KEYS:
                    base = 10.0 + V.KEYS.index(k)
                    off = (i + 1) * 1e-3 * scale * (1.0 if side == "lego" else 0.5)
                    val = base + off + (gap if side == "lego" else 0.0)
                    if (tie_day is not None and day == tie_day
                            and k == tie_metric and side == "lego" and i in (1, 2)):
                        val = base + 1e-3 * scale + (gap if side == "lego" else 0.0)
                    rows[side][i][day] = rows[side][i].get(day, {})
                    rows[side][i][day][k] = val
    # P5 reads the day-90 ACC on both sides; plant the recorded pair so the
    # gate passes, since P5 is exercised separately below.
    rows["lego"][0][90]["acc"] = 64.986934
    rows["nemo"][0][90]["acc"] = V.R.NEMO_D90_ACC_SV
    return rows


def _quantum(V, tiny=True):
    q = {k: (1e-12 if tiny else 1e6) for k in V.KEYS}
    return {d: dict(q) for d in V.HORIZONS}


def test_scoring_layer_runs_end_to_end_without_aborting(V, capsys):
    """The whole reporting path, driven once. Both reviews traced their
    blocking findings to the fact that this had never been run: a control that
    raises, or a table that divides by zero, would otherwise surface only after
    eight multi-hour integrations."""
    rows = _synth(V)
    thin = V.separation_control(rows, _quantum(V))
    unsat = V.saturation_table(rows)
    V.spread_curves(rows, {k: 1e-12 for k in V.KEYS})
    V.empirical_rule_controls(rows)
    V.verdict_table(rows, thin, unsat)
    V.window_table(rows, unsat)
    out = capsys.readouterr().out
    assert "THE VERDICT TABLE" in out and "FINAL-90-DAY WINDOW MEANS" in out
    for day in V.HORIZONS:
        assert f"--- day {day} ---" in out
    for k in V.KEYS:
        assert V.LABELS[k] in out


def test_an_early_tie_is_a_flag_and_a_final_tie_is_fatal(V, capsys):
    """The exact abort both reviews measured on the real ensemble: the
    southern-band sigma MAX ties 2-of-4 at day 90 because it is a single-cell
    maximum of a float32-stored field. That must NOT kill the report."""
    early = _synth(V, tie_day=90)
    V.separation_control(early, _quantum(V))          # must not raise
    assert "SEPARATION / TIES" in capsys.readouterr().out
    final = _synth(V, tie_day=V.N_DAYS)
    with pytest.raises(SystemExit, match="distinct member values"):
        V.separation_control(final, _quantum(V))


def test_the_verdict_table_says_no_on_a_planted_gap(V, capsys):
    """Non-vacuity of the whole printed path, not just of verdict()."""
    rows = _synth(V, gap=10.0)
    V.verdict_table(rows, {d: set() for d in V.HORIZONS}, set())
    # Slice the day-360 ROWS only: the legend below the table contains the
    # word YES as part of its explanation, so asserting on the whole tail
    # tests the legend rather than the verdicts (the expectation was wrong
    # here, not the code).
    body = capsys.readouterr().out
    day360 = body.split("--- day 360 ---")[1].split("--- DOES")[0]
    assert " no" in day360 and "YES" not in day360


def test_the_verdict_table_says_yes_when_the_models_agree(V, capsys):
    rows = _synth(V, gap=0.0)
    V.verdict_table(rows, {d: set() for d in V.HORIZONS}, set())
    day360 = capsys.readouterr().out.split("--- day 360 ---")[1] \
        .split("--- DOES")[0]
    assert "YES" in day360 and " no" not in day360


def test_saturation_needs_two_consecutive_quarters(V, capsys):
    """A single quarter under the ratio has a measured false positive (NEMO's
    ACC spread fell 30->60 then grew 300x). A still-growing ensemble must be
    flagged unsaturated."""
    assert V.SATURATION_QUARTERS == ((180, 270), (270, 360))
    flat = V.saturation_table(_synth(V))
    assert flat == set(), "a flat ensemble must read as saturated"
    growing = V.saturation_table(_synth(V, grow=True))
    assert growing == set(V.KEYS), "a growing ensemble must read as unsaturated"


def test_one_sided_floor_is_flagged(V):
    """The pre-registration justified the RSS as sqrt(2) x one side when the
    two sides wobble equally. Measured, they differ by 9x-16600x, so the flag
    must fire whenever the RSS is really one model's dispersion."""
    rows = _synth(V)
    lopsided = {"lego": {i: {360: {"x": float(v)}} for i, v in enumerate([0, 1, 2, 3])},
                "nemo": {i: {360: {"x": float(v) * 1e-4} for _ in [0]}
                         for i, v in enumerate([0, 1, 2, 3])}}
    saved_keys = V.KEYS
    try:
        V.KEYS = ("x",)
        assert V.one_sided_flags(lopsided, 360) == {"x"}
        even = {s: {i: {360: {"x": float(v)}} for i, v in enumerate([0, 1, 2, 3])}
                for s in ("lego", "nemo")}
        assert V.one_sided_flags(even, 360) == set()
    finally:
        V.KEYS = saved_keys


def test_p5_gates_the_registered_gap_not_the_two_absolutes(V, capsys):
    """The registered quantity is the GAP. Two absolutes can each drift the
    same way and leave the gap perfect (must PASS), or drift opposite ways and
    blow a gap the per-side test would accept (must FAIL)."""
    rows = _synth(V)
    V.p5_provenance_gate(rows)                       # exact -> passes
    shifted = _synth(V)
    shifted["lego"][0][90]["acc"] += 0.5             # both sides move together
    shifted["nemo"][0][90]["acc"] += 0.5
    V.p5_provenance_gate(shifted)                    # gap unchanged -> passes
    broken = _synth(V)
    broken["lego"][0][90]["acc"] += 0.01             # gap moves
    with pytest.raises(SystemExit, match="P5 REFUTED"):
        V.p5_provenance_gate(broken)


def test_unres_band_helper_is_shared_by_both_printers(V):
    """It was spelled inline in two printers and tested in neither."""
    assert V._label(1.5, 1.0) == "YES"
    assert V._label(2.2, 1.0) == "unres"
    assert V._label(9.0, 1.0) == "no"
    assert V._label(-2.2, 1.0) == "unres"
    assert V._label(1.0, 0.0) == "NO-FLOOR"


def test_launch_sha_is_recorded_and_reused(V, tmp_path):
    """Gating a partially-complete ensemble against LIVE HEAD makes it
    unresumable the moment anything else is committed -- which happened twice
    during this campaign."""
    d = str(tmp_path)
    first = V.launch_sha(d)
    assert (tmp_path / ".launch_sha").exists()
    (tmp_path / ".launch_sha").write_text("deadbeef\n")
    assert V.launch_sha(d) == "deadbeef"
    assert first != "deadbeef"


def test_the_q_flag_print_path_is_exercised(V, capsys):
    """No test drove separation_control with a NON-EMPTY thin set, so the
    branch that tells a reader a number is dtype-limited had never printed."""
    rows = _synth(V)
    thin = V.separation_control(rows, _quantum(V, tiny=False))   # huge quantum
    out = capsys.readouterr().out
    assert all(thin[d] == set(V.KEYS) for d in V.HORIZONS)
    assert "under 10x the float32 storage quantum" in out
    V.verdict_table(rows, thin, set())
    body = capsys.readouterr().out.split("--- day 360 ---")[1].split("--- DOES")[0]
    assert "q" in body


def test_positive_control_null_matches_the_folded_cauchy(V):
    """The leave-two-out fix CHANGED the null, and the control briefly printed
    a fraction against no expectation at all. gap = sqrt(2) sigma Z over a
    two-member std sigma|Z'| gives sqrt(2)|Z/Z'|, a scaled folded Cauchy."""
    import numpy as np
    assert V.positive_control_null(np.sqrt(2.0)) == pytest.approx(0.5)
    assert V.positive_control_null(2.0) == pytest.approx(0.6082, abs=1e-4)
    assert V.positive_control_null(2.45) == pytest.approx(0.6667, abs=1e-4)
    # monotone, and it never promises certainty
    assert V.positive_control_null(0.0) == pytest.approx(0.0)
    assert V.positive_control_null(1e9) < 1.0


def test_unsaturated_flag_only_voids_a_no_it_could_actually_overturn(V):
    """10 of 11 metrics fail two-quarter saturation, so an unqualified `u`
    would void every `no` in the table for free -- including gaps sitting 15x
    their floor whose floors grew ~1.0x last quarter."""
    rows = _synth(V)
    # A huge gap: x2YES is enormous, the floor is flat, so `u` must NOT attach.
    far = _synth(V, gap=1000.0)
    material, table = V.unsaturated_materiality(far, set(V.KEYS), V.N_DAYS)
    assert material == set(), "a flat floor cannot close a 1000x gap"
    ratios = {k: r for k, r, *_ in table}
    assert all(r > 100 for r in ratios.values())
    # A gap just outside the band with a fast-growing floor: `u` SHOULD attach.
    grow = _synth(V, grow=True)
    m2, _ = V.unsaturated_materiality(grow, set(V.KEYS), V.N_DAYS)
    assert m2 == set(V.KEYS)
    # And a metric that is SATURATED never gets the flag, however small the gap.
    m3, _ = V.unsaturated_materiality(grow, set(), V.N_DAYS)
    assert m3 == set()
