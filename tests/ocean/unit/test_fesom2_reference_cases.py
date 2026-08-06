"""Direct tests for the FESOM2 reference-case extractor.

The extractor's job is to keep FESOM2's committed `fcheck` truth values
citable WITHOUT transcribing them into prose that goes stale. These tests pin
the two things a wrong extraction would silently break:

1. truth values are read from the `fcheck:` block only, and a setup with no
   `fcheck:` is SKIPPED rather than reported as having zero truth values;
2. the config discriminants travel WITH the truth values -- a truth value
   without its geometry/EOS/closure context invites exactly the confounded
   comparison this module's docstring warns about.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (Path(__file__).resolve().parents[3]
           / "scripts" / "validate" / "ocean_fidelity"
           / "fesom2_reference_cases.py")


def _load():
    spec = importlib.util.spec_from_file_location("fesom2_cases", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load()

_SETUP = """\
mesh: test_thing
ntasks: 8
namelist.config:
    timestep:
        run_length: 1
        run_length_unit: "d"
    geometry:
        cyclic_length: 4.5
    run_config:
        use_ice: False
        toy_ocean: True
        which_toy: "soufflet"
namelist.oce:
    oce_dyn:
        state_equation: 0
        Fer_GM: False
        mix_scheme: "PP"

fcheck:
    salt: 35.0
    temp: 14.32970888593529
    u: 0.027590123469330336
"""


def _write(tmp_path, name, text):
    d = tmp_path / "setups" / name
    d.mkdir(parents=True)
    (d / "setup.yml").write_text(text)
    return d


def test_parse_setup_extracts_truth_values_and_types(tmp_path):
    d = _write(tmp_path, "test_thing", _SETUP)
    rec = ref.parse_setup(d / "setup.yml")
    assert rec["fcheck"]["salt"] == pytest.approx(35.0)
    assert rec["fcheck"]["temp"] == pytest.approx(14.32970888593529)
    assert rec["fcheck"]["u"] == pytest.approx(0.027590123469330336)
    # full float precision must survive -- these are exact reference values
    assert repr(rec["fcheck"]["temp"]) == "14.32970888593529"


def test_config_discriminants_travel_with_the_truth_values(tmp_path):
    """A truth value without its geometry is an invitation to a confounded
    comparison, which is the whole hazard this extractor exists to surface."""
    d = _write(tmp_path, "test_thing", _SETUP)
    cfg = ref.parse_setup(d / "setup.yml")["config"]
    assert cfg["which_toy"] == "soufflet"
    assert cfg["cyclic_length"] == pytest.approx(4.5)
    assert cfg["use_ice"] is False
    assert cfg["toy_ocean"] is True
    assert cfg["state_equation"] == 0
    assert cfg["Fer_GM"] is False
    assert cfg["mix_scheme"] == "PP"


def test_fcheck_block_ends_at_dedent(tmp_path):
    """Keys after the fcheck block must NOT be swallowed as truth values."""
    text = _SETUP + "\ntrailing_key: 99\n"
    d = _write(tmp_path, "test_thing", text)
    rec = ref.parse_setup(d / "setup.yml")
    assert "trailing_key" not in rec["fcheck"]


def test_nested_block_under_fcheck_is_not_flattened(tmp_path):
    """Codex review finding on the first (hand-parsed) version: it treated ANY
    indented mapping beneath ``fcheck:`` as truth values, so a nested block
    would have been flattened into bogus reference numbers. Only SCALARS
    directly under fcheck may count."""
    text = _SETUP + """    nested:
        bogus: 123.456
        also_bogus: 7.0
"""
    d = _write(tmp_path, "test_thing", text)
    fc = ref.parse_setup(d / "setup.yml")["fcheck"]
    assert "bogus" not in fc and "also_bogus" not in fc
    assert "nested" not in fc
    assert set(fc) == {"salt", "temp", "u"}


def test_comments_and_quoting_do_not_corrupt_values(tmp_path):
    """The other half of the same finding: real YAML handles comments and
    quotes, a line-splitter does not."""
    text = """\
fcheck:
    temp: 14.5   # trailing comment must not become part of the value
    label: "PP"
"""
    d = _write(tmp_path, "test_thing", text)
    fc = ref.parse_setup(d / "setup.yml")["fcheck"]
    assert fc["temp"] == pytest.approx(14.5)
    assert fc["label"] == "PP"


def test_non_mapping_fcheck_raises(tmp_path):
    d = _write(tmp_path, "test_thing", "fcheck: 5\n")
    with pytest.raises(ValueError, match="not a mapping"):
        ref.parse_setup(d / "setup.yml")


def test_collect_skips_setups_without_truth_values(tmp_path):
    _write(tmp_path, "test_with", _SETUP)
    _write(tmp_path, "test_without", "mesh: x\nntasks: 2\n")
    out = ref.collect(tmp_path)
    assert set(out["cases"]) == {"test_with"}
    assert "_provenance" in out
    assert out["_provenance"]["fesom2_root"] == str(tmp_path)


def test_collect_raises_when_there_is_no_setups_dir(tmp_path):
    with pytest.raises(SystemExit, match="no setups/"):
        ref.collect(tmp_path / "nowhere")
