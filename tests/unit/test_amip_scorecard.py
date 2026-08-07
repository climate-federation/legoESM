"""Unit tests for the broad AMIP scorecard objective and its CLI.

The decisive test is :func:`test_reproduces_cc_on_baseline_objective`: the
objective is pinned against the ACTUAL published cc_on numbers
(``/scratch/b/b309178/amip_ccab/cc_on/pattern_eval_ccab_m1-12_metrics.txt``),
computed by hand below.  If the blend ever changes, that test goes red and the
change has to be deliberate.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from legoesm.training.amip_scorecard import (
    BIAS_TOLERANCES,
    DEFAULT_MAX_PATTERN_DROP,
    PATTERN_TOLERANCE,
    SCORED_FIELDS,
    ScorecardWeights,
    bias_term,
    compare_scorecards,
    evaluate_scorecard,
    field_score,
    pattern_regressions,
    pattern_term,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_cli():
    """Import the validator CLI by path (scripts/ is not an installed pkg)."""
    path = _REPO_ROOT / "scripts" / "validate" / "amip_scorecard.py"
    spec = importlib.util.spec_from_file_location("_amip_scorecard_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The published cc_on 365-day baseline, months 1-12 (the numbers this harness
# must reproduce).  Source: amip_ccab/cc_on/pattern_eval_ccab_m1-12_metrics.txt
CC_ON = {
    "tas": {"bias": -1.366, "rmse_centered": 2.955, "pattern_corr": 0.980},
    "pr": {"bias": -1.171, "rmse_centered": 2.148, "pattern_corr": 0.481},
    "rlut": {"bias": -36.552, "rmse_centered": 24.264, "pattern_corr": 0.583},
    "rsut": {"bias": 59.326, "rmse_centered": 40.799, "pattern_corr": -0.120},
    # extra rows the objective must ignore rather than fold in
    "prw": {"bias": 1.847, "rmse_centered": 5.623, "pattern_corr": 0.931},
    "psl": {"bias": -0.939, "rmse_centered": 9.793, "pattern_corr": -0.185},
}


# --------------------------------------------------------------------------
# term arithmetic
# --------------------------------------------------------------------------

def test_bias_term_is_absolute_tolerances():
    assert bias_term(16.0, 8.0) == pytest.approx(2.0)
    # sign-independent: a -16 W/m2 bias is as bad as +16
    assert bias_term(-16.0, 8.0) == pytest.approx(2.0)
    assert bias_term(0.0, 8.0) == 0.0


def test_bias_term_rejects_nonpositive_tolerance():
    with pytest.raises(ValueError, match="positive"):
        bias_term(1.0, 0.0)


def test_pattern_term_scale_and_anticorrelation():
    assert pattern_term(1.0) == pytest.approx(0.0)          # perfect
    assert pattern_term(0.9) == pytest.approx(1.0)          # one tolerance
    assert pattern_term(0.0) == pytest.approx(10.0)         # uncorrelated
    # anticorrelated must be strictly worse than uncorrelated, not clipped
    assert pattern_term(-1.0) == pytest.approx(20.0)
    assert pattern_term(-0.120) > pattern_term(0.0)


def test_pattern_tolerance_is_the_documented_scale():
    assert PATTERN_TOLERANCE == pytest.approx(0.10)


def test_bias_tolerances_match_the_repo_gate():
    # These MUST stay equal to scripts/validate/amip_skill_score.py REFERENCE
    # tolerances; drifting them silently redefines "one tolerance of error".
    assert BIAS_TOLERANCES == {"rsut": 8.0, "rlut": 8.0, "tas": 2.0, "pr": 0.8}


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------

def test_default_weights_sum_to_one_and_split_toa_half():
    w = ScorecardWeights().validate()
    assert sum(w.field_weight(f) for f in SCORED_FIELDS) == pytest.approx(1.0)
    assert w.rsut + w.rlut == pytest.approx(0.5)   # TOA carries half
    assert w.rsut == pytest.approx(w.rlut)         # split evenly, not by error
    assert w.tas == pytest.approx(w.pr)


def test_weights_that_do_not_sum_to_one_raise():
    with pytest.raises(ValueError, match="sum to 1"):
        ScorecardWeights(rsut=0.5, rlut=0.5, tas=0.5, pr=0.5).validate()


def test_negative_weight_raises():
    with pytest.raises(ValueError, match="negative"):
        ScorecardWeights(rsut=-0.25, rlut=0.5, tas=0.5, pr=0.25).validate()


def test_bias_fraction_out_of_range_raises():
    with pytest.raises(ValueError, match="bias_fraction"):
        ScorecardWeights(bias_fraction=1.5).validate()


def test_field_weight_rejects_unknown_field():
    with pytest.raises(ValueError, match="unknown scorecard field"):
        ScorecardWeights().field_weight("clt")


# --------------------------------------------------------------------------
# the pinned baseline objective  (the self-check, in test form)
# --------------------------------------------------------------------------

def test_reproduces_cc_on_baseline_objective():
    """Hand-computed cc_on objective under the default weights.

    rsut: 0.5*(59.326/8) + 0.5*(1.120/0.1) = 3.70788 + 5.600  = 9.30788
    rlut: 0.5*(36.552/8) + 0.5*(0.417/0.1) = 2.28450 + 2.085  = 4.36950
    tas : 0.5*( 1.366/2) + 0.5*(0.020/0.1) = 0.34150 + 0.100  = 0.44150
    pr  : 0.5*( 1.171/.8)+ 0.5*(0.519/0.1) = 0.731875+ 2.595  = 3.326875
    J = 0.25 * (9.30788 + 4.36950 + 0.44150 + 3.326875) = 4.36144...
    """
    res = evaluate_scorecard(CC_ON)
    by = res.by_field()
    assert by["rsut"].score == pytest.approx(9.30788, abs=1e-5)
    assert by["rlut"].score == pytest.approx(4.36950, abs=1e-5)
    assert by["tas"].score == pytest.approx(0.44150, abs=1e-5)
    assert by["pr"].score == pytest.approx(3.326875, abs=1e-5)
    assert res.objective == pytest.approx(4.36144, abs=1e-4)


def test_rsut_dominates_the_baseline_objective():
    """The largest scorecard term must be the one the campaign is chasing."""
    res = evaluate_scorecard(CC_ON)
    ranked = sorted(res.fields, key=lambda f: f.contribution, reverse=True)
    assert ranked[0].field == "rsut"
    # and its pattern half must exceed its bias half: the defect is structural
    assert ranked[0].pattern_term > ranked[0].bias_term


def test_extra_metric_rows_are_ignored_not_folded_in():
    res = evaluate_scorecard(CC_ON)
    assert {f.field for f in res.fields} == set(SCORED_FIELDS)
    trimmed = {k: v for k, v in CC_ON.items() if k in SCORED_FIELDS}
    assert evaluate_scorecard(trimmed).objective == pytest.approx(
        res.objective)


def test_perfect_model_scores_zero():
    perfect = {f: {"bias": 0.0, "pattern_corr": 1.0} for f in SCORED_FIELDS}
    assert evaluate_scorecard(perfect).objective == pytest.approx(0.0)


def test_missing_field_raises_rather_than_scoring_as_perfect():
    partial = {k: v for k, v in CC_ON.items() if k != "rsut"}
    with pytest.raises(ValueError, match="rsut.*missing"):
        evaluate_scorecard(partial)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_metric_is_fatal(bad):
    metrics = {k: dict(v) for k, v in CC_ON.items()}
    metrics["rsut"]["pattern_corr"] = bad
    with pytest.raises(ValueError, match="not finite"):
        evaluate_scorecard(metrics)


def test_bias_fraction_one_ignores_pattern():
    w = ScorecardWeights(bias_fraction=1.0)
    res = evaluate_scorecard(CC_ON, w)
    by = res.by_field()
    assert by["rsut"].score == pytest.approx(59.326 / 8.0)
    assert by["tas"].score == pytest.approx(1.366 / 2.0)


def test_bias_fraction_zero_ignores_bias():
    w = ScorecardWeights(bias_fraction=0.0)
    by = evaluate_scorecard(CC_ON, w).by_field()
    assert by["rsut"].score == pytest.approx(1.120 / 0.1, abs=1e-6)


def test_field_score_contribution_is_weight_times_score():
    fs = field_score("tas", -1.366, 0.980, ScorecardWeights())
    assert fs.contribution == pytest.approx(fs.weight * fs.score)


def test_field_score_rejects_field_without_tolerance():
    with pytest.raises(ValueError, match="no bias tolerance"):
        field_score("clt", 1.0, 0.5, ScorecardWeights())


def test_format_table_reports_every_field_and_total():
    text = evaluate_scorecard(CC_ON).format_table()
    for f in SCORED_FIELDS:
        assert f in text
    assert "TOTAL" in text
    assert "4.361" in text


# --------------------------------------------------------------------------
# the pattern-correlation constraint
# --------------------------------------------------------------------------

def _perturb(base, field, *, bias=None, corr=None):
    out = {k: dict(v) for k, v in base.items()}
    if bias is not None:
        out[field]["bias"] = bias
    if corr is not None:
        out[field]["pattern_corr"] = corr
    return out


def test_no_regression_when_nothing_moves():
    assert pattern_regressions(CC_ON, CC_ON) == ()


def test_pattern_drop_beyond_tolerance_is_flagged():
    cand = _perturb(CC_ON, "pr", corr=0.481 - 0.10)
    regs = pattern_regressions(CC_ON, cand)
    assert [r.field for r in regs] == ["pr"]
    assert regs[0].drop == pytest.approx(0.10)


def test_pattern_drop_inside_tolerance_is_not_flagged():
    cand = _perturb(CC_ON, "pr", corr=0.481 - DEFAULT_MAX_PATTERN_DROP / 2)
    assert pattern_regressions(CC_ON, cand) == ()


def test_pattern_improvement_is_never_a_regression():
    cand = _perturb(CC_ON, "rsut", corr=0.5)
    assert pattern_regressions(CC_ON, cand) == ()


def test_negative_max_drop_raises():
    with pytest.raises(ValueError, match="max_drop"):
        pattern_regressions(CC_ON, CC_ON, max_drop=-0.1)


def test_the_trap_case_lower_objective_but_worse_pattern_is_rejected():
    """The failure mode the constraint exists for.

    Nearly eliminate the rsut bias (a large objective win) while destroying tas
    pattern skill.  J falls, so a scalar-only ranking would ACCEPT this; the
    verdict must reject it.

    Note how much rsut has to move to outweigh the tas damage: dropping tas
    correlation 0.98 -> 0.50 costs 0.25*0.5*(0.48/0.1) = 0.60, so a merely
    halved rsut bias is NOT enough.  That is the weighting working as designed
    — tas is cheap to keep and expensive to break.
    """
    cand = _perturb(CC_ON, "rsut", bias=5.0)
    cand = _perturb(cand, "tas", corr=0.50)
    cmp_ = compare_scorecards(CC_ON, cand)
    assert cmp_.delta_objective < 0.0, "objective did fall"
    assert [r.field for r in cmp_.regressions] == ["tas"]
    assert cmp_.improved is False
    assert "REJECTED" in cmp_.format_report()


def test_genuine_improvement_is_accepted():
    cand = _perturb(CC_ON, "rsut", bias=20.0, corr=0.40)
    cmp_ = compare_scorecards(CC_ON, cand)
    assert cmp_.delta_objective < 0.0
    assert cmp_.regressions == ()
    assert cmp_.improved is True
    assert "IMPROVED" in cmp_.format_report()


def test_worse_objective_is_not_improved_even_without_regression():
    cand = _perturb(CC_ON, "rsut", bias=80.0)
    cmp_ = compare_scorecards(CC_ON, cand)
    assert cmp_.delta_objective > 0.0
    assert cmp_.improved is False


# --------------------------------------------------------------------------
# CLI: text parsing must round-trip the published report
# --------------------------------------------------------------------------

_REAL_REPORT = """# AMIP pattern evaluation — cc_on_m1-12

tas        bias=   -1.366  rmse_c=   2.955  r_pattern= 0.980   vs reanalysis_ERA5 (x)
pr         bias=   -1.171  rmse_c=   2.148  r_pattern= 0.481   vs observation_GPCP (x)
rlut       bias=  -36.552  rmse_c=  24.264  r_pattern= 0.583   vs observation_CERES-EBAF (x)
rsut       bias=  +59.326  rmse_c=  40.799  r_pattern=-0.120   vs observation_CERES-EBAF (x)
prw        bias=   +1.847  rmse_c=   5.623  r_pattern= 0.931   vs reanalysis_ERA5 (x)
psl        bias=   -0.939  rmse_c=   9.793  r_pattern=-0.185   vs reanalysis_ERA5 (x)
zonal_ta   bias=   -8.140  rmse_c=  18.318  r_pattern= 0.815   vs ERA5 (x)

TOA net (rsdt-rsut-rlut) = -24.83 W/m2 (obs ~ +0.9)
E - P (global) = -0.074 mm/day
global Bowen (hfss/hfls) = 0.23 (obs ~ 0.2-0.3)
"""


def test_parse_metrics_txt_reads_the_published_report():
    cli = _load_cli()
    parsed = cli.parse_metrics_txt(_REAL_REPORT)
    assert parsed["rsut"]["bias"] == pytest.approx(59.326)
    assert parsed["rsut"]["pattern_corr"] == pytest.approx(-0.120)
    assert parsed["rlut"]["bias"] == pytest.approx(-36.552)
    assert parsed["tas"]["rmse_centered"] == pytest.approx(2.955)
    assert "zonal_ta" in parsed
    # the trailer lines must not become fields
    assert "TOA" not in parsed and "E" not in parsed


def test_parsed_report_scores_identically_to_the_literal_table():
    cli = _load_cli()
    parsed = cli.parse_metrics_txt(_REAL_REPORT)
    assert evaluate_scorecard(parsed).objective == pytest.approx(
        evaluate_scorecard(CC_ON).objective)


def test_cli_json_and_txt_paths_agree(tmp_path):
    cli = _load_cli()
    txt = tmp_path / "run_metrics.txt"
    txt.write_text(_REAL_REPORT)
    js = tmp_path / "pe.json"
    js.write_text(json.dumps({"fields": CC_ON}))
    from_json = cli.load_metrics(js)
    from_txt = cli.load_metrics(txt)
    for f in SCORED_FIELDS:
        assert from_json[f] == from_txt[f], f
    assert evaluate_scorecard(from_json).objective == pytest.approx(
        evaluate_scorecard(from_txt).objective)


def test_cli_run_dir_mode_finds_the_report(tmp_path):
    cli = _load_cli()
    (tmp_path / "pattern_eval_metrics.txt").write_text(_REAL_REPORT)
    m = cli.load_metrics(tmp_path)
    assert m["rsut"]["bias"] == pytest.approx(59.326)


def test_cli_run_dir_with_two_reports_refuses_to_guess(tmp_path):
    cli = _load_cli()
    (tmp_path / "a_metrics.txt").write_text(_REAL_REPORT)
    (tmp_path / "b_metrics.txt").write_text(_REAL_REPORT)
    with pytest.raises(SystemExit, match="pass the one you mean"):
        cli.load_metrics(tmp_path)


def test_cli_run_dir_without_report_errors(tmp_path):
    cli = _load_cli()
    with pytest.raises(SystemExit, match="no \\*_metrics.txt"):
        cli.load_metrics(tmp_path)


def test_cli_parse_weights_defaults_and_override():
    cli = _load_cli()
    assert cli.parse_weights("", 0.5) == ScorecardWeights()
    w = cli.parse_weights("rsut=0.4,rlut=0.2,tas=0.2,pr=0.2", 0.6)
    assert w.rsut == pytest.approx(0.4)
    assert w.bias_fraction == pytest.approx(0.6)


@pytest.mark.parametrize("spec,match", [
    ("clt=0.5", "unknown"),
    ("rsut", "not 'field=value'"),
    ("rsut=abc", "not a number"),
    ("rsut=0.1,rsut=0.2", "twice"),
])
def test_cli_parse_weights_rejects_bad_specs(spec, match):
    cli = _load_cli()
    with pytest.raises(SystemExit, match=match):
        cli.parse_weights(spec, 0.5)


def test_cli_parse_weights_rejects_non_normalised():
    cli = _load_cli()
    with pytest.raises(ValueError, match="sum to 1"):
        cli.parse_weights("rsut=0.9,rlut=0.9,tas=0.9,pr=0.9", 0.5)


def test_cli_gate_rejects_the_trap_case(tmp_path, capsys):
    """End-to-end: --gate must exit non-zero on lower-J-worse-pattern."""
    cli = _load_cli()
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"fields": CC_ON}))
    trap = _perturb(CC_ON, "rsut", bias=25.0)
    trap = _perturb(trap, "tas", corr=0.50)
    cand = tmp_path / "cand.json"
    cand.write_text(json.dumps({"fields": trap}))
    rc = cli.main(["--metrics-json", str(cand),
                   "--baseline-json", str(base), "--gate"])
    assert rc == 1
    assert "REJECTED" in capsys.readouterr().out


def test_cli_gate_accepts_a_genuine_improvement(tmp_path):
    cli = _load_cli()
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"fields": CC_ON}))
    good = _perturb(CC_ON, "rsut", bias=20.0, corr=0.40)
    cand = tmp_path / "cand.json"
    cand.write_text(json.dumps({"fields": good}))
    out = tmp_path / "score.json"
    rc = cli.main(["--metrics-json", str(cand), "--baseline-json", str(base),
                   "--gate", "--json-out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["improved"] is True
    assert payload["pattern_regressions"] == []
    assert payload["delta_objective"] < 0.0


def test_cli_gate_without_baseline_errors(tmp_path):
    cli = _load_cli()
    cand = tmp_path / "cand.json"
    cand.write_text(json.dumps({"fields": CC_ON}))
    with pytest.raises(SystemExit, match="--gate needs --baseline-json"):
        cli.main(["--metrics-json", str(cand), "--gate"])


def test_cli_scores_the_baseline_to_the_pinned_objective(tmp_path, capsys):
    cli = _load_cli()
    cand = tmp_path / "cand.json"
    cand.write_text(json.dumps({"fields": CC_ON}))
    out = tmp_path / "score.json"
    assert cli.main(["--metrics-json", str(cand),
                     "--json-out", str(out)]) == 0
    assert json.loads(out.read_text())["objective"] == pytest.approx(
        4.36144, abs=1e-4)
    assert "rsut" in capsys.readouterr().out


def test_pattern_eval_scorer_exposes_json_flag():
    """The canonical scorer must still offer the --json this harness consumes."""
    src = (_REPO_ROOT / "scripts" / "plot"
           / "plot_amip_pattern_eval.py").read_text()
    assert '"--json"' in src
    assert "json.dump(payload" in src
    # and the JSON must be built from the SAME metrics dict as the .txt
    assert '"fields": {k: dict(v) for k, v in metrics.items()}' in src


def test_module_has_no_hidden_nan_paths():
    """A NaN metric must never reach the objective as a finite number."""
    assert math.isnan(float("nan"))
    with pytest.raises(ValueError):
        evaluate_scorecard(
            {**{f: {"bias": 0.0, "pattern_corr": 1.0} for f in SCORED_FIELDS},
             "rsut": {"bias": float("nan"), "pattern_corr": 1.0}})
