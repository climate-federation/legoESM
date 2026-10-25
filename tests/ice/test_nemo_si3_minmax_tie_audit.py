"""Direct tests for the MIN/MAX tie-order audit of the record build's object code."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
TOOL = REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_minmax_tie_audit.py"
TABLE = REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o6_minmax_sites.json"
SPEC = importlib.util.spec_from_file_location("nemo_si3_minmax_tie_audit_test", TOOL)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

BUILD = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_SAS_ICE_O7_REC/BLD")
GFORTRAN = "/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran"
HAVE_BUILD = (
    (BUILD / "obj" / "icecor.o").is_file() and Path(GFORTRAN).is_file() and shutil.which("objdump")
)

ASM = """\
__m_MOD_f:
# /x/ppsrc/nemo/m.f90:7:   y = MAX( a, 0._wp )
\tmaxsd\t%xmm6, %xmm9\t# tmp12, M.3_8
\tmaxsd\t.LC1(%rip), %xmm2\t# , _9
\tminsd\t(%rsi), %xmm7\t# MEM[(real(kind=8) *)_5 + ivtmp.1_4 * 8], M.4_10
"""
OPT = """\
;; Function f (__m_MOD_f, funcdef_no=0, decl_uid=1, cgraph_uid=1, symbol_order=0)
  [/x/m.f90:7:3] M.3_8 = MAX_EXPR <_7, 0.0>;
  [/x/m.f90:7:3] M.5_11 = MAX_EXPR <_9, 1.0e+0>;
  [/x/m.f90:7:3] _2 = MEM[(real(kind=8) *)_5 + ivtmp.1_4 * 8];
  [/x/m.f90:7:3] M.4_10 = MIN_EXPR <_1, _2>;
  [/x/m.f90:8:3] y_3 = M.4_10 + 1.0e+0;
"""
CUN = """\
;; Function f (__m_MOD_f, funcdef_no=0, decl_uid=1, cgraph_uid=1, symbol_order=0)
  [/x/m.f90:7:3] _20 = v_i.data;
  [/x/m.f90:7:3] _2 = (*_20)[_6];
  [/x/m.f90:7:3] _21 = a_i.data;
  [/x/m.f90:7:3] _1 = (*_21)[_6];
"""


def test_src_operand_is_the_tie_and_nan_survivor(tmp_path):
    (tmp_path / "m.s").write_text(ASM)
    (tmp_path / "o").write_text(OPT)
    (tmp_path / "c").write_text(CUN)
    rows = audit.asm_minmax(tmp_path / "m.s", {7})
    opt = audit.gimple_functions(tmp_path / "o")
    cun = audit.gimple_functions(tmp_path / "c")
    got = [audit.classify(r, opt, cun) for r in rows]
    assert [g["status"] for g in got] == ["MATCHED"] * 3
    assert got[0]["keeps"] == "0.0"  # RTL constant pseudo
    assert got[1]["keeps"] == "1.0"  # constant-pool memory operand
    assert (got[2]["keeps"], got[2]["other"]) == ("v_i[]", "a_i[]")  # named memory operand
    assert got[2]["consumer_lines"] == [8]


def test_expectation_rules_bind():
    row = {
        "line": 7,
        "function": "__m_MOD_f",
        "status": "MATCHED",
        "kind": "MIN",
        "keeps": "v_i[]",
        "insn": "minsd (%rsi),%xmm7",
        "consumer_lines": [8],
    }
    report = {"units": {"m": {"rows": [row]}}}
    ok = {"m:7": {"copies": 1, "rules": [{"kind": "MIN", "keeps": "v_i[]"}]}}
    assert audit.check_expectations(report, ok) == []
    wrong = {"m:7": {"copies": 1, "rules": [{"kind": "MIN", "keeps": "a_i[]"}]}}
    assert audit.check_expectations(report, wrong)
    assert audit.check_expectations(report, {"m:7": {"copies": 2, "rules": ok["m:7"]["rules"]}})
    copydep = {"m:7": {"copies": 1, "rules": [{"kind": "MIN", "keeps_any": ["v_i[]", "a_i[]"]}]}}
    assert audit.check_expectations(report, copydep)  # both survivors must be observed


def _sites():
    sites = json.loads(TABLE.read_text())["sites"]
    units: dict[str, list[int]] = {}
    for key in sites:
        unit, line = key.split(":")
        units.setdefault(unit, []).append(int(line))
    return sites, units


@pytest.mark.skipif(not HAVE_BUILD, reason="record build or its gfortran not on this host")
def test_record_object_code_matches_committed_table_and_plants_fire(tmp_path):
    sites, units = _sites()
    report = audit.audit(BUILD, GFORTRAN, units, tmp_path)
    assert all(u["recompiled_instructions_equal_record"] for u in report["units"].values())
    assert [
        r for u in report["units"].values() for r in u["rows"] if r["status"] != "MATCHED"
    ] == []
    assert audit.check_expectations(report, sites) == []
    assert sum(len(u["rows"]) for u in report["units"].values()) == sum(
        s["copies"] for s in sites.values()
    )
    planted = copy.deepcopy(sites)
    planted["icedyn_rdgrft:857"]["rules"][0]["keeps"] = "phrmax"  # the other operand
    assert audit.check_expectations(report, planted)
    planted = copy.deepcopy(sites)
    planted["icecor:118"]["rules"][1]["keeps"] = planted["icecor:118"]["rules"][2][
        "keeps"
    ]  # kn=0 copy
    assert audit.check_expectations(report, planted)
    planted = copy.deepcopy(sites)
    planted["icedyn_rdgrft:856"]["rules"][0]["keeps_any"] = ["phrmin"]
    assert audit.check_expectations(report, planted)


@pytest.mark.skipif(not HAVE_BUILD, reason="record build or its gfortran not on this host")
def test_refuses_an_object_that_is_not_the_records(tmp_path, monkeypatch):
    flags = audit.fcflags(BUILD)
    monkeypatch.setattr(audit, "fcflags", lambda build: [f if f != "-O3" else "-O2" for f in flags])
    with pytest.raises(audit.AuditError, match="differ from the record"):
        audit.compile_unit(BUILD, GFORTRAN, "icecor", tmp_path)


def _one(asm_line, gimple_lines):
    asm = f"__m_MOD_f:\n# /x/m.f90:7:  y = MAX(a, b)\n{asm_line}\n"
    opt = (
        ";; Function f (__m_MOD_f, funcdef_no=0, decl_uid=1, cgraph_uid=1, symbol_order=0)\n"
        + "".join(f"  [/x/m.f90:7:3] {g};\n" for g in gimple_lines)
    )
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "m.s").write_text(asm)
        (Path(d) / "o").write_text(opt)
        row = audit.asm_minmax(Path(d) / "m.s", {7})[0]
        return audit.classify(row, audit.gimple_functions(Path(d) / "o"), {})


def test_input_match_needs_both_operands_and_a_unique_pair():
    ok = _one("\tmaxsd\t%xmm1, %xmm2\t# a_1, b_2", ["r_3 = MAX_EXPR <a_1, b_2>"])
    assert (ok["status"], ok["matched_by"], ok["keeps"]) == ("MATCHED", "input", "a")
    # destination names the survivor itself, not the other operand: refuse
    assert (
        _one("\tmaxsd\t%xmm1, %xmm2\t# a_1, a_1", ["r_3 = MAX_EXPR <a_1, b_2>"])["status"]
        == "UNMATCHED"
    )
    # the same operand pair in two statements: ambiguous, refuse
    dup = ["r_3 = MAX_EXPR <a_1, b_2>", "r_4 = MAX_EXPR <b_2, a_1>"]
    assert _one("\tmaxsd\t%xmm1, %xmm2\t# a_1, b_2", dup)["status"] == "UNMATCHED"
    # a result-named destination still wins over an input reading
    res = _one("\tmaxsd\t%xmm1, %xmm2\t# a_1, r_3", ["r_3 = MAX_EXPR <a_1, b_2>"])
    assert (res["status"], res["matched_by"]) == ("MATCHED", "result")
