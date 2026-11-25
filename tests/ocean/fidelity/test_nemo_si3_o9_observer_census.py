"""Pins the FP-instruction passivity census of the O9 observers (nemo_si3_o9_observer_census.py)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o9_observer_census.py"
REPORT = TOOL.with_suffix(".json")

_spec = importlib.util.spec_from_file_location("nemo_si3_o9_observer_census", TOOL)
census_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(census_mod)


def _variants():
    return json.loads(REPORT.read_text())["variants"]


@pytest.mark.parametrize("name", ["picard_after_do_while", "frames_f1_f4"])
def test_registered_observers_are_passive_and_the_plant_fires(name):
    v = _variants()[name]
    routine = v["functions"][v["routine"]]
    assert routine["plain_total"] > 0
    assert all(not f["differing_mnemonics"] for f in v["functions"].values())
    assert v["passive"] and v["plant_fired"]


def test_observer_inside_the_do_while_is_not_passive():
    """The rejected placement: a call inside NEMO's outer loop moves FP conversions/compares."""
    v = _variants()["picard_inside_do_while"]
    assert not v["passive"]
    assert v["functions"][v["routine"]]["differing_mnemonics"]


def test_report_flags_are_the_record_build_flags():
    rep = json.loads(REPORT.read_text())
    for flag in ("-O3", "-funroll-all-loops", "-fno-tree-vectorize", "-fdefault-real-8"):
        assert flag in rep["flags"]


def _fake(n=800):
    return [f"line {i}\n" for i in range(1, n + 1)]


def test_instrument_refuses_a_moved_anchor():
    v = census_mod.VARIANTS["picard_after_do_while"]
    lines = _fake()
    for ln, text in v["anchors"].items():
        lines[ln - 1] = "      " + text + "\n"
    out = census_mod.instrument(lines, v)
    assert out[772] == census_mod.PICARD
    lines.insert(0, "! shifted\n")  # misplacement plant
    with pytest.raises(SystemExit):
        census_mod.instrument(lines, v)


def test_fp_regex_counts_memory_operand_conversions():
    for m in ("cvtsi2sdl", "cvtsi2sd", "addsd", "vfmadd231sd", "maxsd", "comisd", "ucomisd",
              "andpd", "xorpd"):
        assert census_mod.FP.match(m), m
    for m in ("movsd", "movapd", "pxor", "movq", "leaq"):
        assert not census_mod.FP.match(m), m


@pytest.mark.skipif(not (census_mod.BUILD / "obj/icethd_zdf_bl99.o").exists()
                    or not (census_mod.TOOLCHAIN / "bin/mpif90").exists(),
                    reason="record build or toolchain absent")
def test_report_regenerates_from_the_record_build():
    assert census_mod.census() == json.loads(REPORT.read_text())
