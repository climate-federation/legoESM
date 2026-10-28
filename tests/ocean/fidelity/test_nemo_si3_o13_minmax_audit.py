"""Pin every floating MIN/MAX/SIGN site and operand role in O13's object."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o13_minmax_audit.py"
REPORT = TOOL.with_suffix(".json")
_spec = importlib.util.spec_from_file_location("nemo_si3_o13_minmax_audit", TOOL)
audit_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit_mod)

SITE_SIGNATURES = {
    "222": "271046fd915d45af7f8fb53fad366438e0ca1f270d7524ac7ed810737ac5d52b",
    "282": "368b7b63b3053b92e5f528e532b9aa6058cddad95320de2537d70ce24430cb01",
    "289": "5ef238990dcc75435f8db9da888bf4d393028c2b0f7aca87c71317fae7dcf623",
    "298": "917cb66b98a646a621127ac29f0c22ee1d44aa9e95e640fe09131d2adbe1e1c1",
    "326": "1bf5da05ec900335acebe49550661c2f9657f7beddd468a108e5ad8de9368157",
    "327": "13b5566c825ad21a2792ff8dadb365a767155319355d1172c7eba7439fe1bf48",
    "331": "a0e6f1cdd7bd3fd010c355a98ca3f9ad4c2d58c5a038f69f5f68c97cf18f96e5",
    "334": "6a2502f3a06f1f03a19a91f056cdd2297dff696deebdf51db7b51600b69097d3",
}


def _report():
    return json.loads(REPORT.read_text())


def _signature(site):
    compact = json.dumps(site, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(compact.encode()).hexdigest()


def check_report(report):
    bad = []
    obj = report["objects"]["iceitd"]
    if obj["record_object_sha256"] != "2642ba5ec85fc283fc288d25ade003a837a3fe6f8cccc0f32b7f3bb3d89caa71":
        bad.append("record object changed")
    if obj["ppsrc_sha256"] != "22baac67260037d4cde180d48ab1938e730b57f492c7d01eb6ade93c9403e7ba":
        bad.append("ppsrc changed")
    if set(obj["sites"]) != set(SITE_SIGNATURES):
        bad.append("floating site set changed")
    for line, expected in SITE_SIGNATURES.items():
        if line in obj["sites"] and _signature(obj["sites"][line]) != expected:
            bad.append(f"site {line} changed")
    if obj["instructions_without_token"]:
        bad.append("instruction without token appeared")
    if set(obj["token_statements_without_instruction"]) != {"280"}:
        bad.append("fused line-280 mapping changed")
    if obj["sign_tokens"]:
        bad.append("SIGN site appeared")
    verdict_sites = {key.split(":", 1)[1] for key in report["tie_verdicts"]}
    if verdict_sites != set(SITE_SIGNATURES) | {"280"}:
        bad.append("tie-verdict coverage changed")
    return bad


def test_committed_o13_object_audit_and_tie_verdicts() -> None:
    assert check_report(_report()) == []


def test_record_build_regenerates_o13_object_audit() -> None:
    if not audit_mod.BUILD.exists():
        pytest.skip("record build is not mounted")
    assert audit_mod.audit() == _report()


@pytest.mark.parametrize("plant", ["drop", "operand", "sign", "fused", "object"])
def test_o13_object_audit_plants_turn_gate_red(plant: str) -> None:
    report = copy.deepcopy(_report())
    obj = report["objects"]["iceitd"]
    if plant == "drop":
        del obj["sites"]["327"]
    elif plant == "operand":
        obj["sites"]["331"]["instructions"][0]["src_defs"] = "load@331"
    elif plant == "sign":
        obj["sign_tokens"] = ["298"]
    elif plant == "fused":
        del obj["token_statements_without_instruction"]["280"]
    else:
        obj["record_object_sha256"] = "0" * 64
    assert check_report(report), plant


def test_o13_minsd_maxsd_operand_roles_match_call_sites() -> None:
    report = _report()
    sites = report["objects"]["iceitd"]["sites"]
    assert (sites["222"]["instructions"][0]["src_defs"], sites["222"]["instructions"][0]["dst_defs"]) == (
        "load@222", "subsd@222"
    )
    assert (sites["298"]["instructions"][0]["src_defs"], sites["298"]["instructions"][0]["dst_defs"]) == (
        "subsd@278", "load@280"
    )
    assert (sites["331"]["instructions"][0]["src_defs"], sites["331"]["instructions"][0]["dst_defs"]) == (
        "load@325", "load@331"
    )
    assert len(sites["282"]["instructions"]) == 2  # source lines 280+282 fused

    census = report["objects"]["iceitd"]["census"]
    for routine, names in audit_mod.CENSUS.items():
        assert set(census[routine]) == set(names)
        assert all(census[routine][name] for name in names)
