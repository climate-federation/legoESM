"""Pins every MIN/MAX/SIGN/SQRT site in O12's compiled production routines."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o12_minmax_audit.py"
REPORT = TOOL.with_suffix(".json")

_spec = importlib.util.spec_from_file_location("nemo_si3_o12_minmax_audit", TOOL)
audit_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit_mod)

SITE_SIGNATURES = {
    "1398": "f8a9e62d221129e4f0bf0f7f18ca0892ffb6b5cd65d4c3cf82cd18025c647872",
    "456": "b275c534a625d7913935396f2c32997c7a998c89a57fb6cffe197d5cf5b6fa8a",
    "465": "0cb11bffba68873078ddac8ea3b4fb56fd17ee1c9b442ca40703b3e5ae460937",
    "472": "abecb0ea8bd9b687a64f164446daa2a70bb6afbe89fc8e367a0ca2007a41a3be",
    "478": "bff21b22c5829f98472fff1e96e58d7eee86fc3f77fc3eba1d5c299726e05d4c",
    "484": "7f6db18a6d66dd163ededf1e34fd320579ecc902178c942b2a4b3693d981ca89",
    "490": "f8a15f6ab3052d7200b90c7d1fd2e7472d82e4e08ca22e6b49f141c42d450ce8",
    "494": "ed2cc3f0526920e4dea189446c3641818116e01c117480b76d7c67abb1eccfa6",
    "497": "db5ede4272cbf3f7296e8946259d9f18c26b405bf9a5b7f1e1be1e7569fcfdaf",
    "502": "415738b0be9419a1e6a71955f2c033bec9bf8dffda73ae28ba0a4434d19d84dc",
    "508": "d808b5bb26f4971472c0944b7e23a8777d015519c426f5dbf073cea47635495d",
    "522": "5973ac126dbc93428e6fa3d9f222ef191fbe73ac1572b54d3a68b552a16311c6",
    "527": "e310c299796ca60c2391efb1aac974aafe515c1440d396faf953f9c118c103a9",
}
NOT_ON_PRODUCTION_PATH = {"502"}
PRODUCTION_SITES = set(SITE_SIGNATURES) - NOT_ON_PRODUCTION_PATH


def _report():
    return json.loads(REPORT.read_text())


def _signature(site):
    compact = json.dumps(site, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(compact.encode()).hexdigest()


def check_report(report):
    """Return all violations; each plant below must make this nonempty."""
    bad = []
    obj = report["objects"]["icethd_pnd"]
    if obj["record_object_sha256"] != "d1a09537f9cb55c6da4cbf44c73729f1ddd3ee107ed3602ef628f9547fa8fec0":
        bad.append("record object changed")
    if obj["ppsrc_sha256"] != "11e5de4e7fdd2c3ef6cf6ef75247833c306b9cc75c79a0dc1684f3a4e7bb5c23":
        bad.append("ppsrc changed")
    if set(obj["sites"]) != set(SITE_SIGNATURES):
        bad.append("floating site set changed")
    for line, expected in SITE_SIGNATURES.items():
        if line in obj["sites"] and _signature(obj["sites"][line]) != expected:
            bad.append(f"site {line} changed")
    if obj["instructions_without_token"] != ["522"]:
        bad.append("fused 522/523 instruction mapping changed")
    if set(obj["token_statements_without_instruction"]) != {"523"}:
        bad.append("fused 523 token mapping changed")
    if obj["sign_tokens"]:
        bad.append("SIGN site appeared")
    verdict_sites = {key.split(":", 1)[1] for key in report["tie_verdicts"]}
    if verdict_sites != set(SITE_SIGNATURES):
        bad.append("tie-verdict coverage changed")
    return bad


def test_committed_object_audit_pins_every_site_and_tie_verdict():
    assert check_report(_report()) == []


def test_record_build_regenerates_committed_object_audit():
    if not audit_mod.BUILD.exists():
        pytest.skip("record build is not mounted")
    assert audit_mod.audit() == _report()


@pytest.mark.parametrize(
    "plant",
    ["drop", "operand", "sign", "fused_522", "fused_523", "object"],
)
def test_object_audit_plants_turn_the_gate_red(plant):
    report = copy.deepcopy(_report())
    obj = report["objects"]["icethd_pnd"]
    if plant == "drop":
        del obj["sites"]["465"]
    elif plant == "operand":
        obj["sites"]["456"]["instructions"][0]["src"] = "VAR"
    elif plant == "sign":
        obj["sign_tokens"] = ["456"]
    elif plant == "fused_522":
        obj["instructions_without_token"] = []
    elif plant == "fused_523":
        del obj["token_statements_without_instruction"]["523"]
    else:
        obj["record_object_sha256"] = "0" * 64
    assert check_report(report), plant


def test_only_no_lid_else_arm_is_excluded_from_production_deck():
    report = _report()
    sites = report["objects"]["icethd_pnd"]["sites"]
    assert "ELSE of IF( ln_pnd_lids )" in sites["502"]["enclosing"]
    assert report["tie_verdicts"]["pnd:502"]["verdict"] == "NOT_ON_DECK"
    for line in PRODUCTION_SITES:
        assert report["tie_verdicts"][f"pnd:{line}"]["verdict"] in ("NO_TIE", "MASKED")


def test_census_and_opposite_minsd_operand_roles_at_465_472_and_522():
    report = _report()
    obj = report["objects"]["icethd_pnd"]
    census = obj["census"]
    for routine, names in audit_mod.CENSUS.items():
        assert set(census[routine]) == set(names)
        assert all(census[routine][name] for name in names), routine

    sites = obj["sites"]
    min_465 = [i for i in sites["465"]["instructions"] if i["op"] == "minsd"][0]
    min_472 = [i for i in sites["472"]["instructions"] if i["op"] == "minsd"][0]
    assert (min_465["dst_defs"], min_465["src_defs"]) == ("subsd@465", "maxsd@456")
    assert (min_472["dst_defs"], min_472["src_defs"]) == ("maxsd@465", "subsd@472")

    min_522 = sites["522"]["instructions"][0]
    assert min_522["op"] == "minsd"
    assert (min_522["dst_defs"], min_522["src_defs"]) == ("mulsd@522", "load@519")
