"""Direct tests for the dashboard CLI and its synthetic pathological campaign.

The ``--demo`` campaign is the artefact a human looks at before trusting the
real dashboard, so it is asserted to actually CONTAIN each pathology — a demo
that quietly renders a healthy campaign would be worse than no demo.
"""

from __future__ import annotations

import json
import os

import pytest
from legoesm.training.calibration_tracking import (
    DASHBOARD_FILENAME,
    MEMBERS_FILENAME,
    TENSORBOARD_DIRNAME,
    pinned_parameters,
    read_member_records,
    summarize,
)

from scripts.plot.render_calibration_dashboard import (
    DEMO_BOUNDS,
    DEMO_STATISTICS,
    build_arg_parser,
    build_demo_campaign,
    load_bounds,
    main,
)


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    """One demo campaign, reused: it is deterministic, so this is safe."""
    run_dir = tmp_path_factory.mktemp("demo")
    build_demo_campaign(str(run_dir))
    return run_dir


@pytest.fixture(scope="module")
def demo_summaries(demo):
    return summarize(read_member_records(demo / MEMBERS_FILENAME), DEMO_BOUNDS)


# --- argument handling ------------------------------------------------------


def test_the_parser_exposes_every_documented_flag():
    args = build_arg_parser().parse_args(
        ["rundir", "--title", "T", "--run-label", "L", "--pin-threshold",
         "0.05", "--no-events", "--watch", "30", "--demo"])
    assert args.run_dir == "rundir" and args.title == "T"
    assert args.pin_threshold == 0.05 and args.no_events is True
    assert args.watch == 30.0 and args.demo is True


@pytest.mark.parametrize("argv", [["d", "--pin-threshold", "0.9"],
                                  ["d", "--pin-threshold", "-0.1"],
                                  ["d", "--watch", "0"]])
def test_out_of_range_flags_are_a_hard_error_not_a_silent_default(argv, tmp_path):
    with pytest.raises(SystemExit):
        main([str(tmp_path)] + argv[1:])


def test_load_bounds_round_trips_a_json_file(tmp_path):
    path = tmp_path / "b.json"
    path.write_text(json.dumps({"rh_crit": [0.5, 0.99]}))
    assert load_bounds(str(path)) == {"rh_crit": (0.5, 0.99)}


@pytest.mark.parametrize("payload", [{"a": [1.0]},
                                     {"a": [2.0, 1.0]},
                                     {"a": "nope"},
                                     {"a": [0.0, float("inf")]}])
def test_load_bounds_rejects_a_malformed_pair(payload, tmp_path):
    path = tmp_path / "b.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_bounds(str(path))


# --- the demo really is pathological ---------------------------------------


def test_the_demo_writes_all_three_outputs(demo):
    assert (demo / MEMBERS_FILENAME).exists()
    assert (demo / DASHBOARD_FILENAME).exists()
    assert any("tfevents" in p.name
               for p in (demo / TENSORBOARD_DIRNAME).iterdir())


def test_the_demo_is_the_advertised_size(demo_summaries):
    """~120 member evaluations over ~6 iterations, the real campaign's shape."""
    assert len(demo_summaries) == 6
    assert sum(s.n_members for s in demo_summaries) == 120


def test_the_demo_pins_a_parameter_at_its_bound(demo_summaries):
    pinned = pinned_parameters(demo_summaries)
    assert [(p.name, p.pinned) for p in pinned] == [
        ("atm.clouds.CloudConfig.rh_crit", "upper")]
    # and it WALKS there rather than starting there — the trajectory is the
    # thing the track is drawing.
    fractions = [next(p.bound_fraction for p in s.parameters
                      if p.name.endswith("rh_crit")) for s in demo_summaries]
    assert fractions == sorted(fractions), "rh_crit should march monotonically"
    assert fractions[0] < 0.5 and fractions[-1] > 0.97


def test_the_demo_loses_half_an_ensemble_at_one_iteration(demo_summaries):
    blown = [len(s.blown_up) for s in demo_summaries]
    assert blown == [0, 0, 0, 10, 0, 0]
    assert demo_summaries[3].n_valid == 10


def test_the_demo_loss_turns_back_upward(demo_summaries):
    means = [s.loss_mean for s in demo_summaries]
    assert means[0] > means[1] > means[2] > means[4], "it should improve first"
    assert means[5] > means[4], "and then get worse"


def test_the_demo_spread_re_widens(demo_summaries):
    spreads = [s.loss_spread for s in demo_summaries]
    assert spreads[-1] > min(spreads) * 1.2


def test_the_demo_has_one_dominant_statistic(demo_summaries):
    shares = {s.name: s.share for s in demo_summaries[-1].statistics}
    assert shares["rsut"] > 0.5


def test_the_demo_carries_a_diagnostic_only_statistic(demo_summaries):
    by_name = {s.name: s for s in demo_summaries[-1].statistics}
    assert by_name["net_toa"].fitted is False
    assert by_name["net_toa"].share == 0.0
    assert DEMO_STATISTICS["net_toa"][3] is False


def test_the_demo_page_shows_every_alarm(demo):
    page = (demo / DASHBOARD_FILENAME).read_text()
    for needle in ("PINNED AT UPPER BOUND",
                   "pinned at a bound",
                   "single-variable fit",
                   "loss went UP",
                   "re-widened",
                   "members blew",
                   "Blown-up members"):
        assert needle in page, f"the demo page never says {needle!r}"


def test_the_demo_is_deterministic(tmp_path):
    """A demo that changed run to run could not be used as a reference."""
    first = tmp_path / "a"
    second = tmp_path / "b"
    build_demo_campaign(str(first))
    build_demo_campaign(str(second))
    assert ((first / MEMBERS_FILENAME).read_text()
            == (second / MEMBERS_FILENAME).read_text())


def test_the_demo_member_scatter_is_in_physical_units_not_a_fraction():
    """A 2%-of-287 K member spread would be 5.7 K — the exact mistake
    ``observation_error`` exists to forbid."""
    _observed, _sigma, spread, _fitted = DEMO_STATISTICS["tas"]
    assert spread < 1.0, "tas member scatter must be a fraction of a kelvin"


# --- the entry point --------------------------------------------------------


def test_main_renders_an_existing_campaign(tmp_path, capsys):
    build_demo_campaign(str(tmp_path))
    assert main([str(tmp_path), "--title", "Custom"]) == 0
    assert capsys.readouterr().out.strip().endswith(DASHBOARD_FILENAME)
    assert "Custom" in (tmp_path / DASHBOARD_FILENAME).read_text()


def test_main_on_an_empty_directory_renders_rather_than_failing(tmp_path,
                                                                capsys):
    assert main([str(tmp_path)]) == 0
    page = (tmp_path / DASHBOARD_FILENAME).read_text()
    assert "Awaiting the first iteration" in page
    assert "no" in capsys.readouterr().err.lower()


def test_no_events_skips_the_event_files(tmp_path):
    assert main([str(tmp_path), "--demo", "--no-events"]) == 0
    # --demo builds its own tracker (which does write events); the FINAL render
    # is what --no-events controls, so assert the flag reached render_dashboard
    # by re-rendering into a clean directory instead.
    clean = tmp_path / "clean"
    os.makedirs(clean)
    (clean / MEMBERS_FILENAME).write_text(
        (tmp_path / MEMBERS_FILENAME).read_text())
    assert main([str(clean), "--no-events"]) == 0
    assert not (clean / TENSORBOARD_DIRNAME).exists()


def test_bounds_from_a_json_file_reach_the_page(tmp_path):
    build_demo_campaign(str(tmp_path))
    bounds_file = tmp_path / "bounds.json"
    bounds_file.write_text(json.dumps(
        {"atm.clouds.CloudConfig.rh_crit": [0.5, 0.99]}))
    assert main([str(tmp_path), "--bounds-json", str(bounds_file)]) == 0
    assert "PINNED AT UPPER BOUND" in (tmp_path / DASHBOARD_FILENAME).read_text()


def test_without_bounds_nothing_is_declared_pinned(tmp_path):
    """The guard against a false alarm: no registered bounds, no verdict."""
    build_demo_campaign(str(tmp_path))
    assert main([str(tmp_path)]) == 0
    page = (tmp_path / DASHBOARD_FILENAME).read_text()
    assert "PINNED AT" not in page
    assert "no registered bounds" in page


# --- regressions from the 2026-08-07 adversarial review --------------------


def test_demo_refuses_to_write_over_a_real_campaign(tmp_path):
    """The log is append-only and the summary keeps the LAST record per
    (iteration, member), so synthetic members do not pad a real campaign — they
    REPLACE it. One mistyped run directory would falsify a multi-day run."""
    from legoesm.training.calibration_tracking import (
        MemberRecord,
        append_member_record,
        read_member_records,
        summarize,
    )

    real = MemberRecord(0, 0, {"real_param": 0.5}, 3.0)
    append_member_record(tmp_path / MEMBERS_FILENAME, real)
    with pytest.raises(SystemExit, match="already holds a campaign"):
        build_demo_campaign(str(tmp_path))
    with pytest.raises(SystemExit, match="already holds a campaign"):
        main([str(tmp_path), "--demo"])

    # the real campaign is untouched
    records = read_member_records(tmp_path / MEMBERS_FILENAME)
    assert len(records) == 1
    (summary,) = summarize(records, {})
    assert summary.loss_mean == pytest.approx(3.0)
    assert [p.name for p in summary.parameters] == ["real_param"]


def test_force_demo_is_the_explicit_escape_hatch(tmp_path):
    from legoesm.training.calibration_tracking import (
        MemberRecord,
        append_member_record,
        read_member_records,
    )

    append_member_record(tmp_path / MEMBERS_FILENAME,
                         MemberRecord(0, 0, {"real_param": 0.5}, 3.0))
    assert main([str(tmp_path), "--demo", "--force-demo"]) == 0
    assert len(read_member_records(tmp_path / MEMBERS_FILENAME)) > 100


def test_demo_into_a_fresh_directory_still_works(tmp_path):
    assert main([str(tmp_path / "fresh"), "--demo"]) == 0
    assert (tmp_path / "fresh" / DASHBOARD_FILENAME).exists()
