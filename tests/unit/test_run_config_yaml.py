"""Direct unit tests for the shared --config YAML loader
(``legoesm.driver.run_config_yaml``) used by run_amip.py and run_coupled.py.

Covers: include-merge precedence, cycle/missing/non-mapping errors, unknown-key
rejection (dispatch-hardening), and quoted-scalar type coercion. The driver-level
round-trips against the real production configs live in the per-driver CLI tests
(``test_run_amip_cli.py`` / ``test_run_coupled_config_yaml.py``).
"""

from __future__ import annotations

import argparse

import pytest
from legoesm.driver.run_config_yaml import (
    load_yaml_config,
    read_yaml_with_includes,
)


def _parser():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--name", type=str, default="x")
    p.add_argument("--flag", action="store_true")
    return p


def test_read_includes_base_then_override(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("alpha: 1.0\nname: base\n")
    child = tmp_path / "child.yaml"
    child.write_text("include: base.yaml\nname: child\n")
    merged = read_yaml_with_includes(str(child))
    # base supplies alpha; child overrides name (file > base)
    assert merged == {"alpha": 1.0, "name": "child"}


def test_include_cycle_raises(tmp_path):
    (tmp_path / "a.yaml").write_text("include: b.yaml\n")
    (tmp_path / "b.yaml").write_text("include: a.yaml\n")
    with pytest.raises(SystemExit, match="cycle"):
        read_yaml_with_includes(str(tmp_path / "a.yaml"))


def test_missing_file_raises(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        read_yaml_with_includes(str(tmp_path / "nope.yaml"))


def test_non_mapping_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n")
    with pytest.raises(SystemExit, match="mapping"):
        read_yaml_with_includes(str(bad))


def test_empty_file_is_empty_dict(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    assert read_yaml_with_includes(str(empty)) == {}


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("alpha: 2.0\nbogus_key: 5\n")
    with pytest.raises(SystemExit, match="unknown key"):
        load_yaml_config(str(p), _parser())


def test_unknown_key_hint_included(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("bogus_key: 5\n")
    with pytest.raises(SystemExit, match="my_example_dest"):
        load_yaml_config(str(p), _parser(), example_keys="'my_example_dest'")


def test_quoted_scalar_coerced_through_argtype(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text('alpha: "2.5"\nname: "hello"\n')
    out = load_yaml_config(str(p), _parser())
    assert out["alpha"] == 2.5 and isinstance(out["alpha"], float)
    assert out["name"] == "hello"


def test_bad_scalar_value_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text('alpha: "not-a-float"\n')
    with pytest.raises(SystemExit, match="not a valid"):
        load_yaml_config(str(p), _parser())


def test_native_types_passthrough(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("alpha: 3.0\nflag: true\n")
    out = load_yaml_config(str(p), _parser())
    assert out["alpha"] == 3.0 and out["flag"] is True
