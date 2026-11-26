"""Pins every MIN/MAX/SIGN site in O10's compiled production routine."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o10_minmax_audit.py"
REPORT = TOOL.with_suffix(".json")

_spec = importlib.util.spec_from_file_location("nemo_si3_o10_minmax_audit", TOOL)
audit_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit_mod)

SITE_SIGNATURES = {
    "277": "66af4ec73e26c1f5ecf77c6bf795d0caee571650d5ca22e51dadf38283a37121",
    "279": "6f8179817191debe4cac8974198742a1cf81259174aacc87a4fc6a14fb2d346b",
    "292": "c0ec05cfa69263a616d95052172c3ba89a9d2fbd25c5507050b88b9f86e3ce41",
    "334": "eaa0485fd7bce083623c32788b2822b3f0c767f5f2a312da57ae535ca765746d",
    "350": "bc914399eb55787cb48e3b5a37ec54abdad8b35f4305eb3853495cfd863af907",
    "367": "aef706e6d385e5ce45e12645e635dfa3db4f6e5be893b6bdeab6df62f30baf50",
    "397": "2f14e682b5a7964cf82ba285b9da91ecb7b12d3256650689ae22cf8b12c34f69",
    "412": "b700d3547730f6df7fcac4a30834fc3c12edb9ad6d76b2ec3a2f9ef553fe8895",
    "428": "d48cabb15ad0cc9d6c8de5e45dad0532f697abd8aa05c487f731bb4a0bfa91a3",
    "434": "609d86c1b7c8b41f7aa120c96421f0e423a45ce8fe7d21d097c633e0fbae8b0c",
    "444": "6263890739b418d7c400df6a418735a74045eeab234927c825c858c2fdd75953",
    "482": "10fabd93cdd4b62490f6197d5a7416e8a070849748b5a43081b6ea5f9a524e89",
    "497": "321452d78bcf0bd6586d2530ff30bd231b2d406b1d6556c5af7f51aa851d3f75",
    "513": "13c58fb4b22c5f27cf1a1cf2b40b2458725b8fe3a133c26a2b165188dec4f62a",
    "520": "0c3fab11501040cf397dbd6657a550eea5d164503a9c8f845303a51215369da9",
    "534": "a883fd9fecfbd1cbb87988a840035b3c54c825265478732c0f0942cba7d890dd",
    "535": "4eb87cf95aefc7835cc230d7a06bdec57d083251c6f9c03929a4e20bf7e655c1",
    "538": "b05bfe2c40c3ee41096ad7121802c74181ced3a228a4dcfed924f5707eec351d",
    "569": "7d31aa70304e586546d43f77ec4ebec9e88f858859b4a2d36f316c2f34077faa",
    "605": "6fb9d36d1551ec9aaca079a931de8410ccbfbe642828a28c33911566a605fb47",
    "619": "6468f352761b602b458c03370fbe5c28ae9eec0c10fbad63d40e78c26786c86b",
    "637": "a484fe12079d2fa5f68607df753497800c1bec7256e2222e7b4d2c5773dd3c0c",
    "641": "bc68a6f21cd92f96d05990de9882b9a94405f3f909682598133b4bbda859946a",
    "644": "57507ad24ecbba0c836e755650d8915b97e1c5e54e3d8a5f1e3052516bffc889",
    "677": "234d26e2bcb6ae6ab565bcc2921d9e9e06d3fe78cbf94565e00c876b59de1a73",
}
NOT_ON_PRODUCTION_PATH = {"277", "279", "292", "428", "434", "513", "520"}
PRODUCTION_SITES = set(SITE_SIGNATURES) - NOT_ON_PRODUCTION_PATH
INTEGER_SITES = {"447", "448", "607", "608"}


def _report():
    return json.loads(REPORT.read_text())


def _signature(site):
    compact = json.dumps(site, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(compact.encode()).hexdigest()


def check_report(report):
    """Return all violations; each plant below must make this nonempty."""
    bad = []
    obj = report["objects"]["icethd_sal"]
    if obj["record_object_sha256"] != "a99e605bbf4b454393a2d2b499d6779c25dfd2037ddd7e78fc22a757f8a0f72b":
        bad.append("record object changed")
    if obj["ppsrc_sha256"] != "db71f3961ffd52fdce8092f9930d69a4abd2247b5b9cc744e9a00f5f9135c216":
        bad.append("ppsrc changed")
    if set(obj["sites"]) != set(SITE_SIGNATURES):
        bad.append("floating site set changed")
    for line, expected in SITE_SIGNATURES.items():
        if line in obj["sites"] and _signature(obj["sites"][line]) != expected:
            bad.append(f"site {line} changed")
    if set(obj["token_statements_without_instruction"]) != INTEGER_SITES:
        bad.append("integer MIN/MAX site set changed")
    if obj["instructions_without_token"]:
        bad.append("instruction without source token")
    if obj["sign_tokens"]:
        bad.append("SIGN site appeared")
    verdict_sites = {key.split(":", 1)[1] for key in report["tie_verdicts"]}
    if verdict_sites != PRODUCTION_SITES:
        bad.append("production tie-verdict coverage changed")
    return bad


def test_committed_object_audit_pins_every_site_and_tie_verdict():
    assert check_report(_report()) == []


def test_record_build_regenerates_committed_object_audit():
    if not audit_mod.BUILD.exists():
        pytest.skip("record build is not mounted")
    assert audit_mod.audit() == _report()


@pytest.mark.parametrize("plant", ["drop", "operand", "sign", "integer", "object"])
def test_object_audit_plants_turn_the_gate_red(plant):
    report = copy.deepcopy(_report())
    obj = report["objects"]["icethd_sal"]
    if plant == "drop":
        del obj["sites"]["397"]
    elif plant == "operand":
        obj["sites"]["397"]["instructions"][0]["src"] = "VAR"
    elif plant == "sign":
        obj["sign_tokens"] = ["397"]
    elif plant == "integer":
        del obj["token_statements_without_instruction"]["607"]
    else:
        obj["record_object_sha256"] = "0" * 64
    assert check_report(report), plant


def test_only_dead_selector_arms_are_excluded_from_production_verdicts():
    report = _report()
    sites = report["objects"]["icethd_sal"]["sites"]
    assert "CASE( 2 )" in sites["277"]["enclosing"]
    assert "CASE( 2 )" in sites["279"]["enclosing"]
    assert "CASE( 2 )" in sites["292"]["enclosing"]
    for line in ("428", "513"):
        assert "ELSEIF ( nn_sal_scheme == 2 )" in sites[line]["enclosing"]
    for line in ("434", "520"):
        assert "ELSEIF ( nn_sal_scheme == 3 )" in sites[line]["enclosing"]


def test_census_names_are_real_and_tie_labels_match_checked_object_facts():
    report = _report()
    census = report["objects"]["icethd_sal"]["census"]["ice_thd_sal"]
    assert set(census) == set(audit_mod.CENSUS["ice_thd_sal"])
    assert all(census[name] for name in census), census
    assert not ({"zv_b", "zv", "zvold", "zvold2", "zdh"} & set(census))

    verdicts = report["tie_verdicts"]
    assert verdicts["sal:334"] == {
        "verdict": "NO_TIE", "proof": "Other operand is nonzero epsi10 (L0)."
    }
    assert verdicts["sal:535"]["verdict"] == "MASKED"
    assert verdicts["sal:538"]["verdict"] == "NO_TIE"
    assert verdicts["sal:641"]["verdict"] == "NO_TIE"
    assert verdicts["sal:644"]["verdict"] == "FOUND"
    assert set(census["zs_brmax"]) == {"340", "350", "357"}
    assert "> sss_1d" in census["zs_brmax"]["357"][1]
    assert set(census["zs_min"]) == {"356", "535", "554", "575", "641", "663"}
    assert all(
        "IF( ln_sal_chk" in " ".join(census["zs_min"][line][2])
        for line in ("554", "663")
    )


def test_line_644_object_has_both_pinned_operand_orders():
    instructions = _report()["objects"]["icethd_sal"]["sites"]["644"]["instructions"]
    min_zero_dst = sum(
        row["count"] for row in instructions
        if row["op"] == "minsd" and row["dst"] == "ZERO"
    )
    min_zero_src = sum(
        row["count"] for row in instructions
        if row["op"] == "minsd" and row["src"] == "ZERO"
    )
    max_floor_dst = sum(
        row["count"] for row in instructions
        if row["op"] == "maxsd" and row.get("dst_defs") == "subsd@644"
    )
    max_min_dst = sum(
        row["count"] for row in instructions
        if row["op"] == "maxsd" and row.get("dst_defs") == "minsd@644"
    )
    assert (min_zero_dst, min_zero_src) == (1, 3)
    assert (max_floor_dst, max_min_dst) == (2, 2)
