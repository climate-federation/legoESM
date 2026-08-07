"""Direct tests for the calibration campaign tracker.

The point of this file is the PATHOLOGICAL cases. A dashboard that has only
ever been shown a healthy synthetic campaign proves nothing, so each of the
failure modes the page exists to catch gets its own test that asserts the alarm
actually fires:

* ``test_parameter_pinned_at_a_bound_*`` — a parameter jammed at its limit;
* ``test_half_the_ensemble_blew_up_*`` — an iteration that lost half its members;
* ``test_a_loss_that_increases_*`` — a campaign going the wrong way;
* ``test_single_iteration_*`` / ``test_zero_iterations_*`` — the degenerate
  campaigns that would divide by ``n - 1``;

plus the two non-negotiables: a tracking failure must never propagate, and the
three outputs must not be able to disagree.
"""

from __future__ import annotations

import json
import math
import os
import re
import xml.etree.ElementTree as ET

import pytest
from legoesm.training.calibration_tracking import (
    DASHBOARD_FILENAME,
    DEFAULT_PIN_THRESHOLD,
    MEMBERS_FILENAME,
    TENSORBOARD_DIRNAME,
    CalibrationTracker,
    IterationSummary,
    MemberRecord,
    StatisticRecord,
    append_member_record,
    coerce_record,
    pinned_parameters,
    read_member_records,
    record_to_json,
    render_dashboard,
    render_html,
    summarize,
    summarize_iteration,
    tensorboard_points,
)

BOUNDS = {"rh_crit": (0.5, 0.99), "alpha_xr": (10.0, 1000.0)}

# Observed global-mean 2 m air temperature [K], the reference value quoted in
# ``observation_error``. Named so the scorecard fixtures read like the real
# thing.
TAS_OBS = 287.0  # const-ok: observed tas reference [K], not the gas constant


def make_record(iteration, member, *, rh_crit=0.7, alpha_xr=500.0, loss=10.0,
                blew_up=False, stats=True):
    statistics = ()
    if stats:
        statistics = (
            StatisticRecord("rsut", 105.0, 98.856, 0.75 * loss),
            StatisticRecord("tas", TAS_OBS + 1.0, TAS_OBS, 0.25 * loss),
            StatisticRecord("net_toa", -4.0, 0.9, float("nan"), fitted=False),
        )
    return MemberRecord(
        iteration=iteration, member=member,
        parameters={"rh_crit": rh_crit, "alpha_xr": alpha_xr},
        loss=loss, statistics=statistics, blew_up=blew_up,
        timestamp=1000.0 + iteration)


def campaign(losses_by_iteration, **kwargs):
    """One record per (iteration, member) from a list of per-member loss lists."""
    return [make_record(i, m, loss=loss, **kwargs)
            for i, losses in enumerate(losses_by_iteration)
            for m, loss in enumerate(losses)]


# ---------------------------------------------------------------------------
# The record and the append-only log
# ---------------------------------------------------------------------------


def test_a_record_round_trips_through_the_jsonl(tmp_path):
    path = tmp_path / MEMBERS_FILENAME
    append_member_record(path, make_record(0, 3, loss=12.5))
    (restored,) = read_member_records(path)
    assert restored.iteration == 0 and restored.member == 3
    assert restored.loss == pytest.approx(12.5)
    assert restored.parameters == {"rh_crit": 0.7, "alpha_xr": 500.0}
    assert [s.name for s in restored.statistics] == ["rsut", "tas", "net_toa"]
    assert restored.statistics[2].fitted is False


def test_the_log_is_append_only(tmp_path):
    path = tmp_path / MEMBERS_FILENAME
    for member in range(3):
        append_member_record(path, make_record(0, member))
    assert len(path.read_text().splitlines()) == 3
    append_member_record(path, make_record(1, 0))
    lines = path.read_text().splitlines()
    assert len(lines) == 4
    # The earlier lines are untouched: nothing rewrites history.
    assert json.loads(lines[0])["member"] == 0
    assert json.loads(lines[3])["iteration"] == 1


def test_every_line_is_strict_json_with_no_nan_tokens(tmp_path):
    """``json.dumps`` emits a bare ``NaN`` by default, which no other language
    can parse; the permanent record must survive any reader."""
    path = tmp_path / MEMBERS_FILENAME
    append_member_record(path, make_record(0, 0, loss=float("nan"),
                                           blew_up=True))
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    payload = json.loads(text)
    assert payload["loss"] is None and payload["blew_up"] is True


def test_a_nonfinite_loss_marks_the_member_as_blown_up_even_if_unflagged():
    record = make_record(0, 0, loss=float("inf"), blew_up=False)
    assert record.failed is True
    assert json.loads(record_to_json(record))["blew_up"] is True


def test_a_truncated_final_line_is_skipped_not_fatal(tmp_path, capsys):
    """A SLURM kill mid-write must not make the surviving history unreadable."""
    path = tmp_path / MEMBERS_FILENAME
    append_member_record(path, make_record(0, 0))
    append_member_record(path, make_record(0, 1))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"schema": 1, "iteration": 0, "mem')   # killed here
    records = read_member_records(path)
    assert [r.member for r in records] == [0, 1]
    assert "unparseable" in capsys.readouterr().err


def test_a_future_schema_raises_rather_than_being_mis_parsed(tmp_path):
    path = tmp_path / MEMBERS_FILENAME
    path.write_text(json.dumps({"schema": 99, "iteration": 0, "member": 0}))
    with pytest.raises(ValueError, match="newer than this reader"):
        read_member_records(path)


def test_a_plain_dict_is_accepted_so_the_tracker_is_optimiser_agnostic():
    """The decoupling requirement: a driver need not import our types."""
    record = coerce_record({
        "iteration": 2, "member": 5, "loss": 3.5,
        "parameters": {"rh_crit": 0.8},
        "statistics": {"rsut": {"model": 100.0, "observed": 98.0,
                                "contribution": 3.5}},
    })
    assert isinstance(record, MemberRecord)
    assert record.iteration == 2 and record.member == 5
    assert record.statistics[0].name == "rsut"
    assert record.statistics[0].fitted is True


def test_a_record_without_the_required_keys_is_rejected():
    with pytest.raises(ValueError, match="iteration"):
        coerce_record({"member": 0})


def test_read_of_a_missing_log_is_empty_not_an_error(tmp_path):
    assert read_member_records(tmp_path / "nope.jsonl") == []


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_loss_statistics_use_the_surviving_members_only():
    """Mirrors ``etki_update``: non-finite members are dropped, not counted."""
    records = [make_record(0, 0, loss=10.0), make_record(0, 1, loss=20.0),
               make_record(0, 2, loss=float("nan"), blew_up=True)]
    summary = summarize_iteration(records, BOUNDS)
    assert summary.n_members == 3 and summary.n_valid == 2
    assert summary.blown_up == (2,)
    assert summary.loss_mean == pytest.approx(15.0)
    assert summary.loss_best == pytest.approx(10.0)
    assert summary.loss_spread == pytest.approx(10.0)


def test_bound_fraction_is_the_linear_position_in_the_interval():
    summary = summarize_iteration(
        [make_record(0, 0, rh_crit=0.745)], {"rh_crit": (0.5, 0.99)})
    (param,) = [p for p in summary.parameters if p.name == "rh_crit"]
    assert param.bound_fraction == pytest.approx((0.745 - 0.5) / 0.49)
    assert param.pinned == ""


@pytest.mark.parametrize("value,expected", [(0.5, "lower"), (0.505, "lower"),
                                            (0.99, "upper"), (0.985, "upper"),
                                            (0.75, "")])
def test_pinning_is_detected_at_both_ends(value, expected):
    summary = summarize_iteration(
        [make_record(0, 0, rh_crit=value)], {"rh_crit": (0.5, 0.99)})
    (param,) = [p for p in summary.parameters if p.name == "rh_crit"]
    assert param.pinned == expected


def test_a_value_outside_its_bounds_is_reported_not_clipped():
    """Clipping to 1.0 would disguise a mis-registered bound as mere pinning."""
    summary = summarize_iteration(
        [make_record(0, 0, rh_crit=1.2)], {"rh_crit": (0.5, 0.99)})
    (param,) = [p for p in summary.parameters if p.name == "rh_crit"]
    assert param.bound_fraction > 1.0
    assert param.pinned == "upper"


def test_a_parameter_with_no_registered_bounds_is_tracked_but_not_judged():
    summary = summarize_iteration([make_record(0, 0)], {})
    param = summary.parameters[0]
    assert param.lo is None and param.bound_fraction is None
    assert param.pinned == ""
    assert param.mean == pytest.approx(0.7)


def test_a_degenerate_bound_pair_yields_no_fraction_not_a_zero_division():
    summary = summarize_iteration(
        [make_record(0, 0, rh_crit=0.7)], {"rh_crit": (0.7, 0.7)})
    (param,) = [p for p in summary.parameters if p.name == "rh_crit"]
    assert param.bound_fraction is None and param.pinned == ""


def test_diagnostic_statistics_are_excluded_from_the_contribution_shares():
    """``net_toa`` is reported but never fitted; it must not appear to drive
    a fit it takes no part in (``observation_error.OBSERVATION_FLOORS``)."""
    summary = summarize_iteration([make_record(0, 0, loss=100.0)], BOUNDS)
    by_name = {s.name: s for s in summary.statistics}
    assert by_name["net_toa"].fitted is False
    assert by_name["net_toa"].share == 0.0
    assert by_name["rsut"].share == pytest.approx(0.75)
    assert by_name["tas"].share == pytest.approx(0.25)
    assert sum(s.share for s in summary.statistics) == pytest.approx(1.0)


def test_an_iteration_where_every_member_blew_up_has_no_loss_not_a_zero():
    records = [make_record(0, m, loss=float("nan"), blew_up=True)
               for m in range(4)]
    summary = summarize_iteration(records, BOUNDS)
    assert summary.n_valid == 0
    assert summary.loss_mean is None and summary.loss_spread is None
    assert summary.blown_up == (0, 1, 2, 3)


def test_summarize_orders_iterations_and_rejects_an_empty_one():
    records = campaign([[10.0], [8.0]])[::-1]      # shuffled input
    summaries = summarize(records, BOUNDS)
    assert [s.iteration for s in summaries] == [0, 1]
    with pytest.raises(ValueError, match="no records"):
        summarize_iteration([], BOUNDS)


def test_an_out_of_range_pin_threshold_is_rejected():
    with pytest.raises(ValueError, match="pin_threshold"):
        summarize_iteration([make_record(0, 0)], BOUNDS, pin_threshold=0.9)


# ---------------------------------------------------------------------------
# PATHOLOGICAL CASE 1 — a parameter pinned at its bound
# ---------------------------------------------------------------------------


def test_parameter_pinned_at_a_bound_is_reported_by_the_summary():
    summaries = summarize(
        [make_record(0, 0, rh_crit=0.70), make_record(1, 0, rh_crit=0.988)],
        BOUNDS)
    pinned = pinned_parameters(summaries)
    assert [(p.name, p.pinned) for p in pinned] == [("rh_crit", "upper")]


def test_parameter_pinned_at_a_bound_is_visually_alarming_on_the_page():
    """Not a number in a table someone has to notice: an alarm banner, a badge,
    and the row carrying the ``pinned`` class that paints it red."""
    summaries = summarize(
        [make_record(0, 0, rh_crit=0.70), make_record(1, 0, rh_crit=0.988)],
        BOUNDS)
    page = render_html(summaries)
    assert "PINNED AT UPPER BOUND" in page
    assert "pinned at a bound" in page
    assert 'class="banner alarm"' in page
    assert 'class="prow pinned"' in page
    assert "rh_crit" in page          # it names the offender, not just a count


def test_a_healthy_campaign_does_not_cry_wolf():
    summaries = summarize(campaign([[10.0, 11.0], [8.0, 8.5]]), BOUNDS)
    page = render_html(summaries)
    assert 'class="banner alarm"' not in page
    assert "PINNED AT" not in page
    assert "No parameter is at a bound" in page


def test_only_a_parameter_pinned_now_raises_the_alarm():
    """A parameter that touched a bound early and moved off it is not the
    failure mode; one sitting there at the latest iteration is."""
    summaries = summarize(
        [make_record(0, 0, rh_crit=0.988), make_record(1, 0, rh_crit=0.70)],
        BOUNDS)
    assert pinned_parameters(summaries) == ()
    assert 'class="banner alarm"' not in render_html(summaries)


def test_the_pin_threshold_is_configurable_end_to_end():
    strict = summarize([make_record(0, 0, rh_crit=0.95)], BOUNDS,
                       pin_threshold=0.2)
    assert [p.pinned for p in strict[0].parameters
            if p.name == "rh_crit"] == ["upper"]
    relaxed = summarize([make_record(0, 0, rh_crit=0.95)], BOUNDS,
                        pin_threshold=DEFAULT_PIN_THRESHOLD)
    assert [p.pinned for p in relaxed[0].parameters
            if p.name == "rh_crit"] == [""]


# ---------------------------------------------------------------------------
# PATHOLOGICAL CASE 2 — half the ensemble blew up
# ---------------------------------------------------------------------------


def test_half_the_ensemble_blew_up_is_counted_and_named():
    records = [make_record(0, m, loss=10.0 + m) for m in range(5)] + [
        make_record(0, m, loss=float("nan"), blew_up=True) for m in range(5, 10)]
    summaries = summarize(records, BOUNDS)
    assert summaries[0].n_valid == 5 and summaries[0].n_members == 10
    assert summaries[0].blown_up == (5, 6, 7, 8, 9)
    page = render_html(summaries)
    assert "5/10" in page                      # the count
    assert "5, 6, 7, 8, 9" in page             # WHICH members
    assert "Blown-up members" in page


def test_a_blown_up_member_does_not_pollute_the_parameter_means():
    """ETKI resets a blown-up member to the ensemble mean, so including its
    parameters would smear the position being tracked."""
    records = [make_record(0, 0, rh_crit=0.60, loss=10.0),
               make_record(0, 1, rh_crit=0.62, loss=10.0),
               make_record(0, 2, rh_crit=0.98, loss=float("nan"),
                           blew_up=True)]
    summary = summarize_iteration(records, BOUNDS)
    (param,) = [p for p in summary.parameters if p.name == "rh_crit"]
    assert param.mean == pytest.approx(0.61)
    assert param.maximum == pytest.approx(0.62)


def test_a_rising_blow_up_count_is_called_out():
    records = (campaign([[10.0, 11.0]])
               + [make_record(1, 0, loss=10.0),
                  make_record(1, 1, loss=float("nan"), blew_up=True)])
    page = render_html(summarize(records, BOUNDS))
    assert "highest in the latest iteration" in page


def test_a_mid_campaign_blow_up_spike_is_still_reported():
    """counts = [0, 5, 0]: not rising at the end, but the reader still needs to
    know the iteration-1 update was computed from half an ensemble."""
    records = (campaign([[10.0] * 6])
               + [make_record(1, m, loss=10.0) for m in range(3)]
               + [make_record(1, m, loss=float("nan"), blew_up=True)
                  for m in range(3, 6)]
               + [make_record(2, m, loss=9.0) for m in range(6)])
    page = render_html(summarize(records, BOUNDS))
    assert "3 of 6 members blew" in page
    assert "iteration 1" in page


def test_a_campaign_with_no_blow_ups_says_so():
    page = render_html(summarize(campaign([[10.0, 11.0]]), BOUNDS))
    assert "No member has blown up" in page


# ---------------------------------------------------------------------------
# PATHOLOGICAL CASE 3 — a loss that gets worse
# ---------------------------------------------------------------------------


def test_a_loss_that_increases_renders_and_is_called_out():
    summaries = summarize(
        campaign([[10.0, 12.0], [6.0, 7.0], [15.0, 18.0]]), BOUNDS)
    assert [round(s.loss_mean, 1) for s in summaries] == [11.0, 6.5, 16.5]
    page = render_html(summaries)
    assert "loss went UP at iteration 2" in page
    assert "best iteration so far is 1" in page


def test_a_spread_that_re_widens_is_called_out():
    """Compared against its own MINIMUM, not against iteration 0: the prior
    spread is always large, so "smaller than at the start" is not convergence."""
    summaries = summarize(campaign([
        [1.0, 21.0],        # spread 20
        [9.0, 11.0],        # spread 2   <- the minimum
        [2.0, 20.0],        # spread 18  <- re-widened
    ]), BOUNDS)
    assert "re-widened" in render_html(summaries)


def test_a_converging_campaign_is_not_flagged_as_thrashing():
    summaries = summarize(campaign([
        [1.0, 21.0], [8.0, 14.0], [10.0, 11.0]]), BOUNDS)
    page = render_html(summaries)
    assert "re-widened" not in page
    assert "still contracting" in page


def test_one_dominant_statistic_is_named_as_a_single_variable_fit():
    record = MemberRecord(
        iteration=0, member=0, parameters={"rh_crit": 0.7}, loss=100.0,
        statistics=(StatisticRecord("rsut", 130.0, 98.856, 92.0),
                    StatisticRecord("tas", TAS_OBS + 0.2, TAS_OBS, 8.0)))
    page = render_html(summarize([record], BOUNDS))
    assert "single-variable fit" in page
    assert "rsut" in page


def test_a_balanced_objective_is_not_flagged():
    record = MemberRecord(
        iteration=0, member=0, parameters={"rh_crit": 0.7}, loss=100.0,
        statistics=(StatisticRecord("rsut", 130.0, 98.856, 40.0),
                    StatisticRecord("tas", TAS_OBS + 0.2, TAS_OBS, 35.0),
                    StatisticRecord("pr", 3.2, 3.086, 25.0)))
    assert "single-variable fit" not in render_html(summarize([record], BOUNDS))


# ---------------------------------------------------------------------------
# PATHOLOGICAL CASE 4 — degenerate campaigns
# ---------------------------------------------------------------------------


def test_zero_iterations_renders_an_awaiting_page_rather_than_crashing():
    page = render_html(())
    assert "Awaiting the first iteration" in page
    assert "<html" in page and "</html>" in page
    assert "Nothing logged yet" in page


def test_single_iteration_renders_without_dividing_by_n_minus_one():
    summaries = summarize(campaign([[10.0, 12.0, 11.0]]), BOUNDS)
    assert len(summaries) == 1
    page = render_html(summaries)
    assert "<svg" in page
    assert page.count("<circle") >= 2      # the single point is drawn, not lost


def test_a_single_member_iteration_has_zero_spread_not_an_error():
    summary = summarize_iteration([make_record(0, 0, loss=5.0)], BOUNDS)
    assert summary.loss_std == 0.0
    assert summary.loss_spread == pytest.approx(0.0)
    assert "<svg" in render_html((summary,))


def test_an_all_blown_up_campaign_still_renders():
    records = [make_record(0, m, loss=float("nan"), blew_up=True)
               for m in range(3)]
    page = render_html(summarize(records, BOUNDS))
    assert "No surviving members yet" in page
    assert "</html>" in page


# ---------------------------------------------------------------------------
# The page itself
# ---------------------------------------------------------------------------


def _svgs(page: str) -> list[str]:
    return re.findall(r"<svg\b.*?</svg>", page, re.S)


def test_the_page_is_self_contained_no_cdn_no_external_anything():
    page = render_html(summarize(campaign([[10.0, 12.0], [8.0, 9.0]]), BOUNDS))
    assert not re.search(r"https?://", page)
    assert "<script src" not in page and "<link" not in page
    assert "@import" not in page
    assert "<style>" in page and "<script>" in page


def test_every_chart_is_well_formed_markup():
    page = render_html(summarize(campaign([[10.0, 12.0], [8.0, 9.0]]), BOUNDS))
    svgs = _svgs(page)
    assert svgs, "expected at least one chart"
    for svg in svgs:
        ET.fromstring(svg)          # raises ParseError on malformed markup


def test_every_chart_ships_a_table_view():
    """The accessibility twin: no value is reachable only through colour."""
    page = render_html(summarize(campaign([[10.0, 12.0], [8.0, 9.0]]), BOUNDS))
    assert page.count("<details>") >= 3
    assert page.count("Table view") >= 3


def test_a_hostile_parameter_name_cannot_inject_markup():
    record = MemberRecord(
        iteration=0, member=0, loss=1.0,
        parameters={'<img src=x onerror="alert(1)">': 0.5})
    page = render_html(summarize([record], {}))
    assert "<img src=x" not in page
    assert "&lt;img" in page


def test_the_page_reports_which_statistics_are_diagnostic_only():
    page = render_html(summarize(campaign([[10.0, 12.0]]), BOUNDS))
    assert "net_toa" in page
    assert "diagnostic only" in page or "NOT fitted" in page


# ---------------------------------------------------------------------------
# The three outputs cannot disagree
# ---------------------------------------------------------------------------


def test_tensorboard_scalars_are_the_same_numbers_as_the_summaries():
    summaries = summarize(campaign([[10.0, 12.0], [6.0, 8.0]]), BOUNDS)
    points = {(p.tag, p.step): p.value for p in tensorboard_points(summaries)}
    for summary in summaries:
        step = summary.iteration
        assert points[("loss/ensemble_mean", step)] == pytest.approx(
            summary.loss_mean)
        assert points[("loss/best_member", step)] == pytest.approx(
            summary.loss_best)
        assert points[("loss/spread_range", step)] == pytest.approx(
            summary.loss_spread)
        assert points[("ensemble/blown_up", step)] == len(summary.blown_up)
        for param in summary.parameters:
            assert points[(f"parameters/{param.name}/mean",
                           step)] == pytest.approx(param.mean)
            assert points[(f"parameters/{param.name}/bound_fraction",
                           step)] == pytest.approx(param.bound_fraction)


def test_tensorboard_omits_a_missing_value_rather_than_writing_a_zero():
    dead = summarize([make_record(0, 0, loss=float("nan"), blew_up=True)],
                     BOUNDS)
    tags = {p.tag for p in tensorboard_points(dead)}
    assert "loss/ensemble_mean" not in tags
    assert "ensemble/blown_up" in tags


def test_render_dashboard_writes_all_three_outputs_from_the_one_log(tmp_path):
    for record in campaign([[10.0, 12.0], [6.0, 8.0]]):
        append_member_record(tmp_path / MEMBERS_FILENAME, record)
    path = render_dashboard(tmp_path, BOUNDS)
    assert os.path.basename(path) == DASHBOARD_FILENAME
    assert (tmp_path / MEMBERS_FILENAME).exists()
    assert (tmp_path / TENSORBOARD_DIRNAME).is_dir()
    assert any("tfevents" in p.name
               for p in (tmp_path / TENSORBOARD_DIRNAME).iterdir())
    # The page agrees with the log it was rendered from.
    page = (tmp_path / DASHBOARD_FILENAME).read_text()
    summaries = summarize(read_member_records(tmp_path / MEMBERS_FILENAME),
                          BOUNDS)
    assert f"{summaries[-1].loss_mean:.4f}" in page


def test_the_dashboard_can_be_rebuilt_from_the_log_alone(tmp_path):
    """A killed job leaves the JSONL; everything else is reproducible."""
    for record in campaign([[10.0, 12.0], [6.0, 8.0]]):
        append_member_record(tmp_path / MEMBERS_FILENAME, record)
    first = render_dashboard(tmp_path, BOUNDS)
    original = open(first, encoding="utf-8").read()
    os.remove(first)
    rebuilt = open(render_dashboard(tmp_path, BOUNDS), encoding="utf-8").read()

    def strip(page):
        return re.sub(r"generated [^<]*", "", page)

    assert strip(rebuilt) == strip(original)


def test_the_page_is_written_atomically(tmp_path):
    append_member_record(tmp_path / MEMBERS_FILENAME, make_record(0, 0))
    render_dashboard(tmp_path, BOUNDS)
    assert not (tmp_path / f"{DASHBOARD_FILENAME}.tmp").exists()


def test_render_dashboard_on_an_empty_directory_produces_a_valid_page(tmp_path):
    path = render_dashboard(tmp_path, BOUNDS)
    assert "Awaiting the first iteration" in open(path, encoding="utf-8").read()


# ---------------------------------------------------------------------------
# A tracking failure must never kill the calibration run
# ---------------------------------------------------------------------------


def test_a_failing_jsonl_write_does_not_propagate(tmp_path, monkeypatch,
                                                  capsys):
    """Losing a log line is cheaper than losing 19 GPU-hours."""
    import legoesm.training.calibration_tracking as module

    tracker = CalibrationTracker(tmp_path, BOUNDS)

    def boom(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(module, "append_member_record", boom)
    tracker.log_member(make_record(0, 0))          # must not raise
    assert tracker.errors == 1
    assert "FAILED" in capsys.readouterr().err
    log = tmp_path / "tracking_errors.log"
    assert log.exists() and "OSError" in log.read_text()


def test_a_failing_render_does_not_propagate(tmp_path, monkeypatch):
    import legoesm.training.calibration_tracking as module

    tracker = CalibrationTracker(tmp_path, BOUNDS)
    tracker.log_member(make_record(0, 0))

    def boom(*args, **kwargs):
        raise RuntimeError("a plotting bug")

    monkeypatch.setattr(module, "render_dashboard", boom)
    assert tracker.end_iteration() is None         # must not raise
    assert tracker.errors == 1


def test_a_malformed_record_does_not_propagate(tmp_path):
    tracker = CalibrationTracker(tmp_path, BOUNDS)
    tracker.log_member({"member": 0})              # missing 'iteration'
    assert tracker.errors == 1
    tracker.log_member(make_record(0, 0))          # and the tracker still works
    assert len(read_member_records(tracker.members_path)) == 1


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permission bits")
def test_an_unwritable_run_directory_does_not_propagate(tmp_path):
    run_dir = tmp_path / "readonly"
    run_dir.mkdir()
    os.chmod(run_dir, 0o500)
    try:
        tracker = CalibrationTracker(run_dir, BOUNDS)
        tracker.log_member(make_record(0, 0))
        assert tracker.end_iteration() is None
        assert tracker.errors >= 1
    finally:
        os.chmod(run_dir, 0o700)


def test_a_keyboard_interrupt_is_still_allowed_through(tmp_path, monkeypatch):
    """Swallowing everything must not break the user's Ctrl-C."""
    import legoesm.training.calibration_tracking as module

    tracker = CalibrationTracker(tmp_path, BOUNDS)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "append_member_record", interrupt)
    with pytest.raises(KeyboardInterrupt):
        tracker.log_member(make_record(0, 0))


def test_summaries_helper_is_guarded_too(tmp_path):
    tracker = CalibrationTracker(tmp_path, BOUNDS)
    assert tracker.summaries() == ()               # empty log, no exception
    tracker.log_member(make_record(0, 0))
    assert len(tracker.summaries()) == 1


def test_the_tracker_holds_no_state_a_kill_could_lose(tmp_path):
    """Each record is durable on return, so a second tracker over the same
    directory sees everything the first one wrote."""
    first = CalibrationTracker(tmp_path, BOUNDS)
    first.log_member(make_record(0, 0))
    first.log_member(make_record(0, 1))
    del first                                      # simulate the job dying
    second = CalibrationTracker(tmp_path, BOUNDS)
    assert len(read_member_records(second.members_path)) == 2
    assert second.end_iteration() is not None


def test_end_to_end_through_the_tracker(tmp_path):
    tracker = CalibrationTracker(tmp_path, BOUNDS, run_label="unit test")
    for iteration, losses in enumerate([[10.0, 12.0], [6.0, 7.0]]):
        for member, loss in enumerate(losses):
            tracker.log_member(make_record(iteration, member, loss=loss))
        tracker.end_iteration()
    assert tracker.errors == 0
    page = open(tracker.dashboard_path, encoding="utf-8").read()
    assert "unit test" in page
    assert len(read_member_records(tracker.members_path)) == 4


def test_summary_types_are_plain_namedtuples_for_downstream_reuse():
    summary = summarize_iteration([make_record(0, 0)], BOUNDS)
    assert isinstance(summary, IterationSummary)
    assert isinstance(summary, tuple)
    assert math.isfinite(summary.wall_time)


def test_a_re_run_member_supersedes_its_earlier_attempt(tmp_path):
    """The append-only log plus a restartable campaign means a member can
    legitimately appear twice; counting both would weight it twice in the
    ensemble mean — a silently wrong number."""
    records = [make_record(0, 0, loss=10.0), make_record(0, 1, loss=20.0),
               make_record(0, 0, loss=30.0)]        # member 0, re-run
    summary = summarize_iteration(records, BOUNDS)
    assert summary.n_members == 2 and summary.n_valid == 2
    assert summary.loss_mean == pytest.approx(25.0)   # (30 + 20) / 2
    assert summary.loss_best == pytest.approx(20.0)


def test_a_member_that_blew_up_then_succeeded_on_retry_is_not_counted_dead():
    records = [make_record(0, 0, loss=float("nan"), blew_up=True),
               make_record(0, 1, loss=20.0),
               make_record(0, 0, loss=12.0)]        # the retry worked
    summary = summarize_iteration(records, BOUNDS)
    assert summary.blown_up == ()
    assert summary.n_valid == 2
    assert summary.loss_mean == pytest.approx(16.0)


def test_the_permanent_log_still_keeps_every_attempt(tmp_path):
    """De-duplication is an ENSEMBLE-VIEW decision; the record keeps both."""
    path = tmp_path / MEMBERS_FILENAME
    append_member_record(path, make_record(0, 0, loss=10.0))
    append_member_record(path, make_record(0, 0, loss=30.0))
    assert len(read_member_records(path)) == 2


def test_resolve_param_bounds_reads_the_real_param_spec_registry():
    """The bounds the page draws against are the REGISTERED ones, not a copy.

    Imports JAX (via ``param_collector``), which is why the import is lazy
    inside the function rather than at module scope.
    """
    from legoesm.training.calibration_tracking import resolve_param_bounds

    bounds = resolve_param_bounds()
    assert bounds["atm.clouds.CloudConfig.rh_crit"] == (0.5, 0.99)
    assert bounds["atm.clouds.CloudConfig.alpha_xr"] == (10.0, 1000.0)
    for low, high in bounds.values():
        assert math.isfinite(low) and math.isfinite(high) and high > low


def test_resolve_param_bounds_reports_an_unregistered_name_rather_than_faking_one(
        capsys):
    from legoesm.training.calibration_tracking import resolve_param_bounds

    subset = resolve_param_bounds(["atm.clouds.CloudConfig.rh_crit",
                                   "atm.clouds.CloudConfig.no_such_field"])
    assert set(subset) == {"atm.clouds.CloudConfig.rh_crit"}
    assert "no registered bounds" in capsys.readouterr().err


def test_the_demo_bounds_are_the_registered_ones_not_a_stale_copy():
    """--demo inlines its bounds so it needs no JAX; this is the drift gate."""
    from legoesm.training.calibration_tracking import resolve_param_bounds

    from scripts.plot.render_calibration_dashboard import DEMO_BOUNDS

    registry = resolve_param_bounds()
    for name, pair in DEMO_BOUNDS.items():
        assert registry[name] == pair, f"{name} drifted from __param_spec__"


def test_the_live_loop_grows_the_event_file_without_rewriting_history(tmp_path):
    """The real usage pattern: render after EVERY iteration.

    Each render rewrites the whole event file, so this asserts the contract
    that makes that safe — the new file must be a byte-PREFIX extension of the
    old one, or a TensorBoard already tailing it would see earlier steps change
    underneath it.
    """
    from legoesm.training.tfevent_writer import EVENT_FILE_NAME

    tracker = CalibrationTracker(tmp_path, BOUNDS)
    events = tmp_path / TENSORBOARD_DIRNAME / EVENT_FILE_NAME
    previous = b""
    for iteration in range(4):
        for member in range(5):
            tracker.log_member(make_record(iteration, member,
                                           loss=20.0 - 3.0 * iteration + member))
        tracker.end_iteration()
        current = events.read_bytes()
        assert current.startswith(previous), (
            f"iteration {iteration} rewrote event-file history")
        assert len(current) > len(previous)
        previous = current
    assert tracker.errors == 0


def test_a_dead_iteration_breaks_the_spread_band_instead_of_spanning_it():
    """Iterations 0 and 2 have members, iteration 1 lost every one of them.

    A single polygon across all three would paint a spread over iteration 1
    that was never measured, so the band must come in two pieces.
    """
    records = (campaign([[1.0, 21.0]])
               + [make_record(1, m, loss=float("nan"), blew_up=True)
                  for m in range(2)]
               + [make_record(2, 0, loss=2.0), make_record(2, 1, loss=20.0)])
    summaries = summarize(records, BOUNDS)
    assert summaries[1].loss_mean is None
    page = render_html(summaries)
    loss_svg = re.findall(r"<svg\b.*?</svg>", page, re.S)[-2]
    assert loss_svg.count("<polygon") == 0, (
        "two 2-point runs each need >= 2 points; here each side has only one "
        "live iteration, so no polygon is drawable")
    # With three live iterations either side of the gap, two bands appear.
    records = (campaign([[1.0, 21.0], [2.0, 18.0]])
               + [make_record(2, m, loss=float("nan"), blew_up=True)
                  for m in range(2)]
               + campaign([[3.0, 17.0], [4.0, 15.0]])[:0]
               + [make_record(3, 0, loss=3.0), make_record(3, 1, loss=17.0),
                  make_record(4, 0, loss=4.0), make_record(4, 1, loss=15.0)])
    page = render_html(summarize(records, BOUNDS))
    loss_svg = re.findall(r"<svg\b.*?</svg>", page, re.S)[-2]
    assert loss_svg.count("<polygon") == 2, "the gap must split the band in two"


def test_no_html_entity_is_double_escaped_into_visible_text():
    """Guard for a real bug: a label containing ``&minus;`` was passed through
    the escaper and rendered as the literal text "&minus;" on the page."""
    page = render_html(summarize(campaign([[10.0, 12.0], [8.0, 9.0]]), BOUNDS))
    leaked = re.findall(r"&amp;(?:[a-zA-Z]+|#\d+);", page)
    assert not leaked, f"double-escaped entities visible on the page: {leaked}"


# ---------------------------------------------------------------------------
# Regressions from the 2026-08-07 adversarial review
# ---------------------------------------------------------------------------


def test_a_huge_but_finite_loss_does_not_take_down_the_dashboard(tmp_path,
                                                                 capsys):
    """A member diverging but not yet NaN is exactly what the page must show.

    ``simple_value`` is a float32, so a squared normalised residual of ~1e41
    (reachable from a small sigma) used to raise ``OverflowError`` out of the
    event writer and — because events were written first — take the page too.
    """
    append_member_record(tmp_path / MEMBERS_FILENAME, make_record(0, 0, loss=10.0))
    append_member_record(tmp_path / MEMBERS_FILENAME,
                         make_record(0, 1, loss=4.2e41))
    path = render_dashboard(tmp_path, BOUNDS)
    assert os.path.exists(path), "the page must survive an unwritable scalar"
    assert "not representable as a float32" in capsys.readouterr().err
    # the event file is still written, just without the offending point
    assert any("tfevents" in p.name
               for p in (tmp_path / TENSORBOARD_DIRNAME).iterdir())


def test_the_page_is_written_before_the_event_files(tmp_path, monkeypatch):
    """Ordering is the structural guarantee behind the test above: a failure in
    the secondary rendering must never be able to cost the primary one."""
    import legoesm.training.calibration_tracking as module

    append_member_record(tmp_path / MEMBERS_FILENAME, make_record(0, 0))

    def boom(*args, **kwargs):
        raise RuntimeError("event writer exploded")

    monkeypatch.setattr(module, "write_tensorboard", boom)
    path = render_dashboard(tmp_path, BOUNDS)     # must not raise
    assert os.path.exists(path)


def test_a_corrupt_schema_value_costs_only_its_own_line(tmp_path, capsys):
    """The docstring promises a bad line is skipped; it used to abort the read
    of the whole history."""
    path = tmp_path / MEMBERS_FILENAME
    append_member_record(path, make_record(0, 0))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"schema": "x", "iteration": 0, "member": 9})
                     + "\n")
        handle.write("5\n")                       # a bare JSON scalar
        handle.write(json.dumps([1, 2, 3]) + "\n")   # a JSON array
    append_member_record(path, make_record(0, 1))
    assert [r.member for r in read_member_records(path)] == [0, 1]
    err = capsys.readouterr().err
    assert "unreadable schema" in err and "must be a JSON object" in err


def test_a_negative_index_is_rejected_at_the_record_boundary():
    """It used to reach the event encoder, where it raised and killed the
    render for every subsequent iteration."""
    with pytest.raises(ValueError, match=">= 0"):
        coerce_record({"iteration": -1, "member": 0, "loss": 1.0})
    with pytest.raises(ValueError, match=">= 0"):
        coerce_record({"iteration": 0, "member": -2, "loss": 1.0})


def test_model_and_observed_are_averaged_over_the_same_members():
    """The page subtracts them to show a bias; means over different subsets of
    members are not a bias of anything."""
    good = MemberRecord(0, 0, {"rh_crit": 0.7}, 1.0,
                        (StatisticRecord("rsut", 100.0, 98.0, 1.0),))
    partial = MemberRecord(0, 1, {"rh_crit": 0.7}, 1.0,
                           (StatisticRecord("rsut", 300.0, float("nan"), 1.0),))
    (stat,) = summarize_iteration([good, partial], BOUNDS).statistics
    assert stat.model_mean == pytest.approx(100.0)   # not (100 + 300) / 2
    assert stat.observed == pytest.approx(98.0)
    assert stat.n_reported == 1


def test_a_disagreeing_observed_reference_is_surfaced_not_averaged(capsys):
    """A reference value is constant by construction; if members disagree that
    is a scorer bug, and averaging it hides one."""
    records = [
        MemberRecord(0, 0, {"rh_crit": 0.7}, 1.0,
                     (StatisticRecord("rsut", 100.0, 98.0, 1.0),)),
        MemberRecord(0, 1, {"rh_crit": 0.7}, 1.0,
                     (StatisticRecord("rsut", 100.0, 50.0, 1.0),))]
    (stat,) = summarize_iteration(records, BOUNDS).statistics
    assert stat.observed == pytest.approx(98.0)      # taken, not 74.0
    assert "DIFFERENT observed values" in capsys.readouterr().err


def test_a_loss_the_statistics_do_not_decompose_is_declared():
    """"share of the loss" is only true when the contributions sum to it."""
    record = MemberRecord(
        0, 0, {"rh_crit": 0.7}, 100.0,
        (StatisticRecord("rsut", 130.0, 98.856, 12.0),
         StatisticRecord("tas", TAS_OBS + 1.0, TAS_OBS, 8.0)))
    (summary,) = summarize([record], BOUNDS)
    assert summary.decomposed_total == pytest.approx(20.0)
    assert summary.undecomposed_loss == pytest.approx(80.0)
    page = render_html((summary,))
    assert "is NOT decomposed by any of them" in page
    assert "of the fitted loss" not in page       # the overclaiming label


def test_a_fully_decomposed_loss_gets_no_residual_note():
    record = MemberRecord(
        0, 0, {"rh_crit": 0.7}, 20.0,
        (StatisticRecord("rsut", 130.0, 98.856, 12.0),
         StatisticRecord("tas", TAS_OBS + 1.0, TAS_OBS, 8.0)))
    assert "NOT decomposed" not in render_html(summarize([record], BOUNDS))


def test_a_negative_contribution_cannot_produce_an_impossible_share():
    record = MemberRecord(
        0, 0, {"rh_crit": 0.7}, 5.0,
        (StatisticRecord("a", 1.0, 0.0, -5.0),
         StatisticRecord("b", 1.0, 0.0, 10.0)))
    (summary,) = summarize([record], BOUNDS)
    shares = {s.name: s.share for s in summary.statistics}
    assert shares["b"] == pytest.approx(1.0)
    assert shares["a"] == 0.0
    assert all(0.0 <= s <= 1.0 for s in shares.values())


def test_a_parameter_reported_by_a_minority_does_not_raise_the_alarm():
    """Pinning is the headline output; one stray member must not fire the same
    red banner as a whole ensemble jammed against its bound."""
    bounds = {**BOUNDS, "stray": (0.0, 1.0)}
    records = [make_record(0, m, rh_crit=0.70) for m in range(3)]
    # Only member 3 carries `stray`, and it sits hard against the upper bound.
    records.append(MemberRecord(0, 3, {"rh_crit": 0.70, "stray": 0.999}, 10.0))
    summary = summarize_iteration(records, bounds)
    stray = next(p for p in summary.parameters if p.name == "stray")
    assert stray.bound_fraction > 0.97       # it IS at the bound
    assert stray.n_reported == 1
    assert stray.pinned == "", "one member out of four is not a quorum"
    consensus = next(p for p in summary.parameters if p.name == "rh_crit")
    assert consensus.n_reported == 4


def test_a_majority_at_the_bound_still_raises_the_alarm():
    records = [make_record(0, m, rh_crit=0.988) for m in range(3)]
    summary = summarize_iteration(records, BOUNDS)
    assert next(p for p in summary.parameters
                if p.name == "rh_crit").pinned == "upper"


def test_a_duplicate_statistic_name_is_refused_not_silently_dropped():
    """The log keys statistics by name, so one of them would be lost and the
    record and the log would disagree."""
    record = MemberRecord(
        0, 0, {"rh_crit": 0.7}, 1.0,
        (StatisticRecord("rsut", 150.0, 98.0, 1.0),
         StatisticRecord("rsut", 200.0, 98.0, 1.0)))
    with pytest.raises(ValueError, match="more than once"):
        record_to_json(record)


def test_table_cells_are_escaped_by_the_table_helper():
    """Headers were escaped and cells were not; the next row added would have
    become an injection point."""
    from legoesm.training.calibration_tracking import _table

    rendered = _table(["h"], [['<script>alert(1)</script>']])
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_summarize_iteration_refuses_records_from_several_iterations():
    """De-duplication keys on the member index, so mixed input would silently
    erase records."""
    with pytest.raises(ValueError, match="ONE iteration"):
        summarize_iteration([make_record(0, 0), make_record(1, 0)], BOUNDS)


def test_two_concurrent_renderers_never_publish_a_spliced_page(tmp_path):
    """--watch is documented to run alongside the live tracker, so the temp
    name must be unique per writer, not a shared f"{path}.tmp"."""
    import threading

    for record in campaign([[10.0, 12.0], [6.0, 8.0]]):
        append_member_record(tmp_path / MEMBERS_FILENAME, record)
    errors: list[BaseException] = []

    def render():
        try:
            for _ in range(12):
                render_dashboard(tmp_path, BOUNDS)
        except BaseException as exc:               # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=render) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, f"concurrent renders raised: {errors}"
    page = (tmp_path / DASHBOARD_FILENAME).read_text()
    assert page.startswith("<!DOCTYPE html>") and page.rstrip().endswith("</html>")
    assert "\x00" not in page, "the published page was spliced from two writers"
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
