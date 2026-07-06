"""Unit tests for the per-test-case namelist writer (issue #682)."""
from __future__ import annotations

import dataclasses
from typing import NamedTuple

import pytest

from legoesm.experiments.matrix.namelist import (
    build_namelist,
    write_case_namelist,
)


@dataclasses.dataclass
class _FakeCase:
    equation_set: str = "shallow_water"
    case: str = "colliding_modons"
    grid_type: str = "cubed_sphere"
    resolution: str = "C48"
    vertical_coord: str = "none"
    duration_days: float = 100.0
    quick_days: float = 1.0
    run_kwargs: dict = dataclasses.field(default_factory=lambda: {"test_num": 8})
    family: str = ""


class _FakeConfig(NamedTuple):
    scheme: str = "kpp"
    Ri_crit: float = 0.3


def test_build_namelist_flattens_dataclass_and_run_kwargs():
    nl = build_namelist(_FakeCase())
    assert nl["equation_set"] == "shallow_water"
    assert nl["case"] == "colliding_modons"
    assert nl["duration_days"] == 100.0
    # nested run_kwargs are flattened with dotted keys.
    assert nl["run_kwargs.test_num"] == 8


def test_build_namelist_empty_dict_marked():
    nl = build_namelist(_FakeCase(run_kwargs={}))
    assert nl["run_kwargs"] == "{}"


def test_build_namelist_merges_extra_and_flattens_namedtuple():
    nl = build_namelist(
        _FakeCase(), extra={"radiation": "gray", "vmix": _FakeConfig()}
    )
    assert nl["radiation"] == "gray"
    # a namedtuple config in extra is expanded leaf-by-leaf.
    assert nl["vmix.scheme"] == "kpp"
    assert nl["vmix.Ri_crit"] == 0.3


def test_extra_overrides_case_field():
    nl = build_namelist(_FakeCase(), extra={"duration_days": 2.0})
    assert nl["duration_days"] == 2.0


def test_write_case_namelist_writes_sorted_file(tmp_path):
    out = write_case_namelist(
        tmp_path / "sw" / "colliding_modons",
        _FakeCase(),
        title="atmosphere test-case namelist",
        extra={"radiation": "gray"},
    )
    assert out.name == "namelist.txt"
    assert out.exists()
    text = out.read_text()
    assert "# atmosphere test-case namelist" in text
    assert "issue #682" in text
    assert "case: colliding_modons" in text
    assert "run_kwargs.test_num: 8" in text
    assert "radiation: gray" in text
    # data lines (non-comment) are sorted.
    data_lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    assert data_lines == sorted(data_lines)


def test_write_case_namelist_creates_missing_dirs(tmp_path):
    out = write_case_namelist(tmp_path / "a" / "b" / "c", _FakeCase())
    assert out.exists()


def test_build_namelist_accepts_plain_dict():
    nl = build_namelist({"grid": "latlon", "nested": {"x": 1}})
    assert nl["grid"] == "latlon"
    assert nl["nested.x"] == 1
