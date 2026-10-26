"""Pins the exhaustive MIN/MAX/SQRT object-code audit of the SI3 O9 routines.

The committed report (``nemo_si3_o9_minmax_audit.json``) lists every
``minsd``/``maxsd``/``sqrtsd`` of the record build's O9 routines mapped to its
ppsrc statement.  These tests pin that inventory site by site, the operand
role of every zero-operand site, the census mismatches with their reasons,
and (where the build exists) that the report regenerates byte-for-byte.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o9_minmax_audit.py"
REPORT = TOOL.with_suffix(".json")

_spec = importlib.util.spec_from_file_location("nemo_si3_o9_minmax_audit", TOOL)
audit_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit_mod)

# object -> statement line -> executes on the production ORCA2_SAS_ICE deck
# (np_cnd_OFF, ln_cndi_P07, ln_virtual_itd = F, nn_icesal = 4).
NOT_ON_DECK = {
    "icethd_zdf_bl99": {261: "ln_cndi_U64 arm", 262: "ln_cndi_U64 arm", 266: "ln_cndi_U64 arm",
                        309: "ln_virtual_itd arm",
                        747: "np_cnd_ON iteration", 748: "np_cnd_ON iteration",
                        754: "np_cnd_ON iteration", 755: "np_cnd_ON iteration",
                        800: "np_cnd_ON exit", 803: "np_cnd_ON exit"},
    "icethd_dh": {525: "nn_icesal == 2 arm"},
}
SITES = {
    "icethd": [285],
    "icethd_dh": [128, 130, 170, 180, 194, 200, 206, 207, 211, 221, 228, 229, 230, 231, 251,
                  272, 274, 289, 290, 295, 310, 311, 348, 355, 388, 409, 411, 426, 427, 456,
                  490, 491, 494, 525, 619],
    "icethd_zdf": [],
    "icethd_zdf_bl99": [173, 181, 202, 219, 229, 261, 262, 266, 274, 276, 281, 292, 309, 353,
                        577, 582, 583, 589, 590, 747, 748, 754, 755, 800, 803],
    "icevar": [955, 957, 1045],
}
# Census mismatches, each with the reason it is not a missing site.
TOKEN_WITHOUT_INSTRUCTION = {
    "icethd_dh": {
        "222": "MAX(zdum, -zh_s) emitted as a negated minsd that the line table assigns to :228",
        "300": "MAX(-zh_i, -zevap_rema*r1_rhoi) emitted as a negated minsd assigned to :295",
        "526": "nn_icesal == 2 arm; MAX(epsi10, h_i) of :525 reused (PLAUSIBLE: common subexpression)",
    },
    "icethd_zdf_bl99": {
        "578": "MAX(zdti_max = 0, ABS(.)) folded to ABS(.): no maxsd at this statement",
    },
}
INSTRUCTION_WITHOUT_TOKEN = {"icethd_dh": ["228", "295"]}


def _report():
    return json.loads(REPORT.read_text())


def check_report(rep) -> list[str]:
    """Every invariant the receipt rests on; returns the violations."""
    bad = []
    for obj, sites in SITES.items():
        got = rep["objects"][obj]
        if sorted(map(int, got["sites"])) != sites:
            bad.append(f"{obj}: site list changed")
        if got["sign_tokens"]:
            bad.append(f"{obj}: SIGN executes at {got['sign_tokens']}")
        if dict((k, v["tokens"]) for k, v in got["token_statements_without_instruction"].items()
                ).keys() != TOKEN_WITHOUT_INSTRUCTION.get(obj, {}).keys():
            bad.append(f"{obj}: token-without-instruction set changed")
        if got["instructions_without_token"] != INSTRUCTION_WITHOUT_TOKEN.get(obj, []):
            bad.append(f"{obj}: instruction-without-token set changed")
        for line, site in got["sites"].items():
            for ins in site["instructions"]:
                if ins["op"] == "sqrtsd":
                    if (obj, line) != ("icethd", "285"):
                        bad.append(f"{obj}:{line} unexpected sqrtsd")
                    continue
                kinds = (ins["src"], ins["dst"])
                if any(k.startswith("MIXED") for k in kinds):
                    if int(line) not in NOT_ON_DECK.get(obj, {}):
                        bad.append(f"{obj}:{line} mixed operand kinds on the deck's path")
                elif ins["dst"] == "ZERO":
                    if int(line) not in NOT_ON_DECK.get(obj, {}):
                        bad.append(f"{obj}:{line} zero is the destination (tie returns the other)")
                elif ins["src"] == "ZERO" and ins["dst"] != "VAR":
                    bad.append(f"{obj}:{line} zero against a non-variable")
    return bad


def test_report_invariants_hold():
    assert check_report(_report()) == []


def test_every_zero_site_on_the_deck_keeps_the_zero():
    """maxsd/minsd return SRC on a tie or NaN; every deck zero operand is the source."""
    rep = _report()
    zero_sites = []
    for obj, v in rep["objects"].items():
        for line, site in v["sites"].items():
            if int(line) in NOT_ON_DECK.get(obj, {}):
                continue
            for ins in site["instructions"]:
                if "ZERO" in (ins["src"], ins["dst"]):
                    assert ins["src"] == "ZERO", (obj, line, ins)
                    zero_sites.append((obj, line))
    assert ("icethd_dh", "180") in zero_sites  # MAX(0, -qprec): the tie the model pins
    assert ("icethd_dh", "211") in zero_sites  # MIN(x, 0): zero second in source, still src
    assert len(set(zero_sites)) == 33


@pytest.mark.parametrize("plant", ["swap_roles", "drop_site", "add_sign", "new_mismatch"])
def test_plants_turn_the_check_red(plant):
    rep = copy.deepcopy(_report())
    dh = rep["objects"]["icethd_dh"]
    if plant == "swap_roles":
        ins = dh["sites"]["180"]["instructions"][0]
        ins["src"], ins["dst"] = ins["dst"], ins["src"]
    elif plant == "drop_site":
        del dh["sites"]["490"]
    elif plant == "add_sign":
        dh["sign_tokens"] = ["133"]
    else:
        dh["instructions_without_token"] = ["228", "295", "400"]
    assert check_report(rep) != []


def test_mismatch_reasons_cover_the_report():
    rep = _report()
    for obj, reasons in TOKEN_WITHOUT_INSTRUCTION.items():
        assert set(rep["objects"][obj]["token_statements_without_instruction"]) == set(reasons)


def test_reaching_definitions_on_a_synthetic_cfg():
    """Zero on one path, a load on the other -> MIXED; jmpq is unconditional."""
    ins = lambda addr, op, *args: dict(addr=addr, op=op, args=list(args), reloc=None, line=1)
    prog = [
        ins(0, "test", "%eax", "%eax"),
        ins(2, "je", "a <f+0xa>"),
        ins(4, "pxor", "%xmm1", "%xmm1"),
        ins(8, "jmpq", "e <f+0xe>"),
        ins(10, "movsd", "(%rdi)", "%xmm1"),
        ins(14, "maxsd", "%xmm1", "%xmm0"),
    ]
    inn, effects = audit_mod._reaching(prog, {}, {})
    assert audit_mod._kind(5, "%xmm1", inn, effects) == "MIXED(VAR|ZERO)"
    straight = prog[2:3] + prog[5:]
    straight[1] = dict(straight[1], addr=8)
    inn, effects = audit_mod._reaching(straight, {}, {})
    assert audit_mod._kind(1, "%xmm1", inn, effects) == "ZERO"
    assert audit_mod._cfg(prog)[3] == [5]  # jmpq: no fall-through edge


def test_constant_resolution_through_local_symbol():
    import struct
    ro = {".rodata.cst8": struct.pack("<dd", 1.0, 1e-20)}
    syms = {".LC3": (".rodata.cst8", 8)}
    assert audit_mod._const(".LC3-0x4", ro, syms) == float.hex(1e-20)
    assert audit_mod._const(".rodata.cst8-0x4", ro, syms) == float.hex(1.0)


@pytest.mark.skipif(not (audit_mod.BUILD / "obj/icethd_dh.o").exists()
                    or not (audit_mod.TOOLCHAIN / "bin/mpif90").exists(),
                    reason="record build or toolchain absent")
def test_report_regenerates_from_the_record_build():
    assert audit_mod.audit() == _report()
