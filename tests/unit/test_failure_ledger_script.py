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
</testsuite></testsuites>
"""


def test_rows_ids_and_categories(tmp_path):
    xml = tmp_path / "u.xml"
    xml.write_text(_XML)
    rows = {r["id"]: r for r in tfl.rows_from_junit("unit", str(xml))}
    assert set(rows) == {
        "tests/unit/test_a.py::TestX::test_slow",
        "tests/test_no_hardcoded_constants.py::test_ratchet",
        "tests/unit/test_b.py::test_fake[p1]",
        "tests/unit/test_c.py::test_ref",
        "tests/unit/test_d.py::test_mystery",
    }  # passing test excluded
    cat = {k: r["category"] for k, r in rows.items()}
    assert cat["tests/unit/test_a.py::TestX::test_slow"] == "timeout"
    assert cat["tests/test_no_hardcoded_constants.py::test_ratchet"] == "policy drift"
    assert cat["tests/unit/test_b.py::test_fake[p1]"] == "stale fake driver"
    assert rows["tests/unit/test_b.py::test_fake[p1]"]["kind"] == "error"
    assert cat["tests/unit/test_c.py::test_ref"] == "pinned reference changed"
    assert cat["tests/unit/test_d.py::test_mystery"] == "other"
    assert rows["tests/unit/test_c.py::test_ref"]["reason"] == "AssertionError:"


def test_main_writes_markdown_with_counts(tmp_path, capsys):
    xml = tmp_path / "u.xml"
    xml.write_text(_XML)
    md = tmp_path / "ledger.md"
    assert tfl.main([f"unit={xml}", "--md", str(md)]) == 0
    text = md.read_text()
    assert "# Failure ledger (5 rows)" in text
    assert "| timeout | 1 |" in text and "## other" in text
    assert "Failure ledger" in capsys.readouterr().out
