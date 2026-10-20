"""Direct test of scripts/validate/failure_ledger.py."""
import importlib.util
import pathlib

_P = pathlib.Path(__file__).resolve().parents[2] / "scripts/validate/failure_ledger.py"
_spec = importlib.util.spec_from_file_location("failure_ledger", _P)
tfl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tfl)

_XML = """<?xml version="1.0"?>
<testsuites><testsuite name="pytest">
  <testcase classname="tests.unit.test_a.TestX" name="test_ok"/>
  <testcase classname="tests.unit.test_a.TestX" name="test_slow">
    <failure message="Failed: Timeout &gt;300.0s">stack</failure></testcase>
  <testcase classname="tests.test_no_hardcoded_constants" name="test_ratchet">
    <failure message="AssertionError: 3 new literals">x</failure></testcase>
  <testcase classname="tests.unit.test_b" name="test_fake[p1]">
    <error message="AttributeError: '_FakeDriver' object has no attribute 'v'">e</error></testcase>
  <testcase classname="tests.unit.test_c" name="test_ref">
    <failure message="AssertionError: &#10;Not equal to tolerance rtol=1e-07">t</failure></testcase>
  <testcase classname="tests.unit.test_d" name="test_mystery">
    <failure message="AssertionError: assert 1 == 2">t</failure></testcase>
  <testcase classname="tests.unit.test_d" name="test_mystery">
    <error message="failed on teardown">t</error></testcase>
  <testcase classname="" name="tests.unit.test_broken">
    <error message="collection failure">ImportError: cannot import name 'x'</error></testcase>
  <testcase classname="tests.unit.test_e" name="test_skip"><skipped message="s"/></testcase>
</testsuite></testsuites>
"""


def test_rows_ids_and_categories(tmp_path):
    xml = tmp_path / "u.xml"
    xml.write_text(_XML)
    got = tfl.rows_from_junit("unit", str(xml))
    rows = {r["id"]: r for r in got}
    assert len(got) == len(rows) == 6  # fail + teardown error on one test = one row
    assert set(rows) == {
        "tests/unit/test_a.py::TestX::test_slow",
        "tests/test_no_hardcoded_constants.py::test_ratchet",
        "tests/unit/test_b.py::test_fake[p1]",
        "tests/unit/test_c.py::test_ref",
        "tests/unit/test_d.py::test_mystery",
        "tests/unit/test_broken.py",
    }  # passing test excluded
    cat = {k: r["category"] for k, r in rows.items()}
    assert cat["tests/unit/test_a.py::TestX::test_slow"] == "timeout"
    assert cat["tests/test_no_hardcoded_constants.py::test_ratchet"] == "policy drift"
    assert cat["tests/unit/test_b.py::test_fake[p1]"] == "stale fake driver"
    assert rows["tests/unit/test_b.py::test_fake[p1]"]["kind"] == "error"
    assert cat["tests/unit/test_c.py::test_ref"] == "pinned reference changed"
    assert cat["tests/unit/test_d.py::test_mystery"] == "other"
    assert rows["tests/unit/test_c.py::test_ref"]["reason"] == "AssertionError: Not equal to tolerance rtol=1e-07"
    assert rows["tests/unit/test_d.py::test_mystery"]["kind"] == "failure+error"
    assert cat["tests/unit/test_broken.py"] == "stale-after-rename"


def test_main_writes_markdown_with_counts(tmp_path, capsys):
    xml = tmp_path / "u.xml"
    xml.write_text(_XML)
    md = tmp_path / "ledger.md"
    assert tfl.main([f"unit={xml}", "--md", str(md)]) == 0
    text = md.read_text()
    assert "# Failure ledger (6 rows)" in text
    assert "| timeout | 1 |" in text and "## other" in text
    assert "Failure ledger" in capsys.readouterr().out


def test_without_md_prints_rows(tmp_path, capsys):
    xml = tmp_path / "u.xml"
    xml.write_text(_XML)
    assert tfl.main([f"unit={xml}"]) == 0
    assert "`tests/unit/test_b.py::test_fake[p1]`" in capsys.readouterr().out


def test_approx_text_alone_is_not_a_pinned_reference():
    assert tfl.classify("tests/unit/t.py::f", "assert x == pytest.approx(1.0)") == "other"
    assert tfl.classify("tests/unit/t.py::f", "assert 1.2 == 1.0 \u00b1 1e-6") == "pinned reference changed"


def test_same_job_twice_is_one_row_per_test(tmp_path, capsys):
    xml = tmp_path / "u.xml"
    xml.write_text(_XML)
    tfl.main([f"unit={xml}", f"unit={xml}"])
    assert "# Failure ledger (6 rows)" in capsys.readouterr().out


def test_safety_rule_needs_a_raised_guard():
    assert tfl.classify("tests/unit/t.py::f", "temperature must be positive is required") == "other"
    assert tfl.classify("tests/unit/t.py::f",
                        "ValueError: the tiled dycore refuses the energy fixer") == "new safety check vs old test"


def test_policy_drift_is_any_top_level_ratchet_not_subdir():
    assert tfl.classify("tests/test_pe_dycore_inline_imports.py::t[x]", "AssertionError: 24 inline") == "policy drift"
    assert tfl.classify("tests/unit/test_pe.py::t", "AssertionError: 24 inline") == "other"
