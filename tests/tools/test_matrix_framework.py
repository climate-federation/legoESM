"""Direct unit tests for the shared test-matrix framework
(``legoesm.experiments.matrix``).

Exercises the leaf functions of ``core`` / ``gates`` / ``report`` / ``registry``
without running any model — the framework is pure plumbing and must be testable
in isolation (CLAUDE.md hygiene: every new leaf module gets a direct test).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from legoesm.experiments.matrix import (
    CONS_THRESH,
    MatrixCase,
    MatrixRunner,
    ResultRecorder,
    RunMaturity,
    RunStatus,
    Tier,
    detect_regressions,
    energy_gate,
    finite_gate,
    heat_gate,
    mass_gate,
    salt_gate,
    write_summary,
)
from legoesm.experiments.matrix.gates import (
    benchmark_error_gate,
    default_tol,
    drift_gate,
)

pytestmark = pytest.mark.tier0


# --- core.Tier --------------------------------------------------------------

def test_tier_parse_accepts_int_name_and_tierN():
    assert Tier.parse(1) is Tier.RESEARCH
    assert Tier.parse("research") is Tier.RESEARCH
    assert Tier.parse("tier1") is Tier.RESEARCH
    assert Tier.parse("2") is Tier.INTERMEDIATE
    assert Tier.parse(Tier.UNIT) is Tier.UNIT


def test_tier_parse_rejects_unknown():
    with pytest.raises(ValueError):
        Tier.parse("operationalish")


def test_tier_is_ordered_for_filtering():
    assert Tier.UNIT < Tier.RESEARCH < Tier.INTERMEDIATE < Tier.OPERATIONAL


# --- core.MatrixCase --------------------------------------------------------

def _case(**kw):
    base = dict(component="ocean", case="rest_state", grid_type="cubed_sphere",
                tier=1, complexity="full_3d", resolution="C24")
    base.update(kw)
    return MatrixCase(**base)


def test_matrixcase_coerces_tier_string():
    c = _case(tier="research")
    assert c.tier is Tier.RESEARCH


def test_matrixcase_output_path_skips_none_vcoord():
    c = _case(vertical_coord="none")
    assert c.output_path == "ocean/full_3d/rest_state/cubed_sphere/C24"
    c2 = _case(vertical_coord="zstar")
    assert c2.output_path.endswith("/C24/zstar")


def test_matrixcase_maturity_default_and_override():
    assert _case().maturity is RunMaturity.RUN_TESTED
    assert _case(maturity=RunMaturity.INIT_ONLY).maturity is RunMaturity.INIT_ONLY


# --- core.ResultRecorder ----------------------------------------------------

def test_recorder_counts_and_ok():
    rec = ResultRecorder(verbose=False)
    rec.record(_case(), RunStatus.PASS, 1.0)
    rec.record(_case(case="gyre"), "SKIP", 0.0)
    assert rec.ok is True
    rec.record(_case(case="wave"), RunStatus.FAIL, 2.0, "drift too big")
    assert rec.ok is False
    c = rec.counts()
    assert c["PASS"] == 1 and c["FAIL"] == 1 and c["SKIP"] == 1


# --- gates ------------------------------------------------------------------

def test_default_tol_known_and_unknown():
    assert default_tol("atmosphere", "mass_rel_drift") == CONS_THRESH["atmosphere"]["mass_rel_drift"]
    with pytest.raises(KeyError):
        default_tol("atmosphere", "no_such_key")
    with pytest.raises(KeyError):
        default_tol("no_such_component", "mass_rel_drift")


def test_drift_gate_requires_tol_or_component_key():
    with pytest.raises(ValueError):
        drift_gate(True, "", [1.0, 1.0], label="x")


def test_mass_gate_passes_on_flat_series():
    ok, notes = mass_gate(True, "", [5e19, 5e19, 5e19], component="atmosphere")
    assert ok is True


def test_mass_gate_fails_on_large_drift():
    ok, notes = mass_gate(True, "", [5e19, 5e19, 6e19], component="atmosphere")
    assert ok is False
    assert "mass" in notes and "FAIL" in notes


def test_mass_gate_ocean_uses_volume_key():
    # ocean has no mass_rel_drift; mass_gate must route to volume_rel_drift.
    ok, _ = mass_gate(True, "", [1.34e18, 1.34e18], component="ocean")
    assert ok is True


def test_energy_gate_fails_on_nan_anywhere():
    ok, notes = energy_gate(True, "", [1.0, float("nan"), 1.0], component="atmosphere")
    assert ok is False


def test_energy_gate_routes_ocean_to_heat_key():
    # codex MEDIUM-1: energy_gate(component="ocean") must NOT KeyError on the
    # missing energy_rel_drift; it routes to the ocean heat_rel_drift tolerance.
    ok, _ = energy_gate(True, "", [1e25, 1e25], component="ocean")
    assert ok is True
    bad, notes = energy_gate(True, "", [1e25, 2e25], component="ocean")
    assert bad is False and "energy" in notes


def test_energy_gate_works_for_every_energy_component():
    # Every CONS_THRESH component that has an energy/heat tolerance must be
    # callable through the shared energy_gate without KeyError.
    for comp in ("atmosphere", "ocean", "land", "coupled"):
        ok, _ = energy_gate(True, "", [1.0e6, 1.0e6], component=comp)
        assert ok is True, comp


def test_heat_gate_and_salt_gate_ocean():
    ok, _ = heat_gate(True, "", [1e25, 1e25])
    assert ok is True
    ok2, _ = salt_gate(True, "", [4.9e16, 4.9e16])
    assert ok2 is True


def test_benchmark_error_gate_le():
    ok, _ = benchmark_error_gate(True, "", 0.01, 0.05, label="l2")
    assert ok is True
    ok2, notes = benchmark_error_gate(True, "", 0.10, 0.05, label="l2")
    assert ok2 is False and "l2" in notes


def test_finite_gate_detects_nan():
    ok, _ = finite_gate(True, "", {"u": np.ones(4), "v": np.zeros(4)})
    assert ok is True
    bad, notes = finite_gate(True, "", {"u": np.array([1.0, np.inf])})
    assert bad is False and "NaN/Inf" in notes


def test_gates_are_idempotent_on_failed_state():
    # once ok=False, a subsequent gate must not flip it back to True
    ok, notes = mass_gate(False, "prior fail", [5e19, 5e19], component="atmosphere")
    assert ok is False and notes == "prior fail"


# --- report -----------------------------------------------------------------

def test_write_summary_emits_three_files(tmp_path):
    rec = ResultRecorder(verbose=False)
    rec.record(_case(), RunStatus.PASS, 1.5, "clean")
    rec.record(_case(case="gyre"), RunStatus.FAIL, 2.0, "drift")
    json_path = write_summary(tmp_path, rec, total_wall=3.5, meta={"quick_mode": True})
    assert json_path.exists()
    assert (tmp_path / "summary.txt").exists()
    assert (tmp_path / "summary.md").exists()
    data = json.loads(json_path.read_text())
    assert data["n_pass"] == 1 and data["n_fail"] == 1
    assert data["quick_mode"] is True
    assert data["results"][0]["component"] == "ocean"


def test_detect_regressions_flags_pass_to_fail(tmp_path):
    prev = ResultRecorder(verbose=False)
    prev.record(_case(), RunStatus.PASS, 1.0)
    write_summary(tmp_path, prev, total_wall=1.0)

    cur = ResultRecorder(verbose=False)
    cur.record(_case(), RunStatus.FAIL, 1.0, "now broken")
    regressions = detect_regressions(cur, tmp_path / "summary.json")
    assert len(regressions) == 1 and "REGRESSION" in regressions[0]


def test_detect_regressions_empty_without_prior(tmp_path):
    cur = ResultRecorder(verbose=False)
    cur.record(_case(), RunStatus.FAIL, 1.0)
    assert detect_regressions(cur, tmp_path / "missing.json") == []


def test_detect_regressions_distinguishes_sibling_cases(tmp_path):
    # codex HIGH: two cases with the SAME component/test/grid/resolution but
    # different complexity (or vertical_coord) must not collide in the
    # prior-status map — a regression in one must not be masked by the other.
    a = MatrixCase("atmosphere", "baroclinic", "cubed_sphere", tier=2,
                   complexity="hydrostatic", resolution="C48", vertical_coord="sigma")
    b = MatrixCase("atmosphere", "baroclinic", "cubed_sphere", tier=2,
                   complexity="nonhydrostatic", resolution="C48", vertical_coord="sigma")
    prev = ResultRecorder(verbose=False)
    prev.record(a, RunStatus.PASS, 1.0)
    prev.record(b, RunStatus.PASS, 1.0)
    write_summary(tmp_path, prev, total_wall=2.0)

    cur = ResultRecorder(verbose=False)
    cur.record(a, RunStatus.PASS, 1.0)          # hydrostatic still passes
    cur.record(b, RunStatus.FAIL, 1.0, "blew up")  # nonhydrostatic regressed
    regressions = detect_regressions(cur, tmp_path / "summary.json")
    # exactly the nonhydrostatic sibling must be flagged (not masked by the
    # passing hydrostatic sibling sharing the partial key).
    assert len(regressions) == 1 and "nonhydrostatic" in regressions[0]


# --- registry.MatrixRunner --------------------------------------------------

class _DummyRunner(MatrixRunner):
    component = "ocean"

    def build_cases(self):
        return [
            MatrixCase("ocean", "rest_state", "cubed_sphere", tier=1,
                       complexity="full_3d", resolution="C24"),
            MatrixCase("ocean", "baroclinic", "latlon", tier=2,
                       complexity="full_3d", resolution="36x72"),
            MatrixCase("ocean", "omip", "mpas", tier=3,
                       complexity="full_3d", resolution="mpas"),
        ]

    def run_case(self, case, *, quick, output_dir):
        return RunStatus.PASS, "ok", {}


def test_runner_requires_component():
    class NoComp(MatrixRunner):
        pass
    with pytest.raises(ValueError):
        NoComp()


def test_runner_filter_by_tier_and_grid():
    r = _DummyRunner()
    assert len(r.filter(r.cases, tier=Tier.RESEARCH)) == 1
    assert len(r.filter(r.cases, max_tier=Tier.INTERMEDIATE)) == 2
    assert len(r.filter(r.cases, grid="mpas")) == 1


def test_runner_main_list_returns_zero(capsys):
    rc = _DummyRunner().main(["--list", "--max-tier", "research"])
    assert rc == 0
    assert "1 case(s) selected" in capsys.readouterr().out


def test_runner_main_runs_and_writes_summary(tmp_path):
    rc = _DummyRunner().main(["--output", str(tmp_path), "--tier", "1"])
    assert rc == 0
    summary = tmp_path / "ocean" / "summary.json"
    assert summary.exists()
    assert json.loads(summary.read_text())["n_pass"] == 1


def test_runner_main_nonzero_on_failure(tmp_path):
    class FailRunner(_DummyRunner):
        def run_case(self, case, *, quick, output_dir):
            return RunStatus.FAIL, "bad", {}
    rc = FailRunner().main(["--output", str(tmp_path), "--tier", "1"])
    assert rc == 1


def test_runner_main_captures_exception_as_error(tmp_path):
    class BoomRunner(_DummyRunner):
        def run_case(self, case, *, quick, output_dir):
            raise RuntimeError("boom")
    rc = BoomRunner().main(["--output", str(tmp_path), "--tier", "1"])
    assert rc == 1
    rec = json.loads((tmp_path / "ocean" / "summary.json").read_text())
    assert rec["n_error"] == 1
    assert "boom" in rec["results"][0]["notes"]


def test_runner_main_empty_selection_fails(tmp_path, capsys):
    # codex HIGH-1: a typo'd --grid selects zero cases; must FAIL (rc!=0), not
    # silently write a zero-test summary and return 0.
    rc = _DummyRunner().main(["--output", str(tmp_path), "--grid", "cubedsphere"])
    assert rc == 2
    assert "0 of 3 cases selected" in capsys.readouterr().out


def test_runner_main_empty_selection_allow_empty_ok(tmp_path):
    rc = _DummyRunner().main(
        ["--output", str(tmp_path), "--grid", "nonesuch", "--allow-empty"]
    )
    assert rc == 0


def test_runner_main_list_empty_does_not_fail(tmp_path):
    # --list short-circuits before the empty-selection guard (listing nothing
    # is legitimate); only the run path fails fast.
    rc = _DummyRunner().main(["--list", "--grid", "nonesuch"])
    assert rc == 0
