"""O9 MIN/MAX tie survivors, one verdict per executing site of the record object code.

Every ``minsd``/``maxsd`` statement the production deck executes in the O9 routines
(``nemo_si3_o9_minmax_audit.json``) falls in exactly one class below.  Sites with a
reachable signed-zero tie get a cell (``data/si3_o9_tie_cells.json``, from
``nemo_si3_o9_tie_search.py``) on which the independent oracle sees the tie, swapping
that one site's emitted operand roles changes output bits, and the model equals the
unswapped oracle in every bit.  The other classes state why no bit-changing tie exists.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_si3_o9_zdf_dh_temp as o9

ROOT = Path(__file__).resolve().parents[3]
CELLS = json.loads((Path(__file__).parent / "data/si3_o9_tie_cells.json").read_text())
AUDIT = json.loads((ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o9_minmax_audit.json").read_text())
OBJ = {"dh": "icethd_dh", "zdf": "icethd_zdf_bl99", "var": "icevar", "thd": "icethd"}

libm_policy = o9.libm_policy  # same autouse precision policy as the O9 suite

# Signed-zero tie reached, and the surviving operand changes output bits (oracle swap).
FOUND = ("dh:180", "dh:194", "dh:200", "dh:251", "dh:272.min", "dh:388", "dh:409.min")
# Signed-zero tie reached, output bits unchanged by the swap: the result only feeds a
# MAX(0,.)/MIN(.,0) whose zero is the source (:272 max, :409 max), or a branch that the
# tie value never enters (:130 zq_bot only used when zf_tt > 0; :295 zh_i re-clamped).
MASKED = ("dh:130", "dh:272.max", "dh:295")
CLASSES = {
    "nonzero_constant": {  # equal-value ties have identical bits; only a NaN tells operands apart
        "zdf:173", "zdf:181", "zdf:274", "zdf:276", "zdf:281", "zdf:292", "zdf:353", "dh:221",
        "dh:348.min", "var:957.min", "dh:619.max2", "var:1045.max2"},
    "kelvin": {  # both operands are temperatures >= rt0 - 100 K: never zero
        "zdf:202", "zdf:577", "zdf:582", "zdf:589", "var:955"},
    "abs": {"zdf:583", "zdf:590"},  # both operands are ABS(.) or a MAX of them: never -0
    # No differently-signed tie in the search's entry cells (PLAUSIBLE, search evidence only).
    "unreached": {
        "dh:128", "dh:170", "dh:206", "dh:207", "dh:211", "dh:228", "dh:229", "dh:230", "dh:231", "dh:274",
        "dh:289", "dh:290", "dh:310", "dh:311", "dh:348.max", "dh:355", "dh:409.max", "dh:411", "dh:426",
        "dh:427", "dh:456", "dh:490", "dh:491", "dh:494", "dh:619.max1", "zdf:219", "zdf:229", "thd:285",
        "var:957.max", "var:1045.max1"},
}
NOT_ON_DECK = {"zdf": {261, 262, 266, 309, 747, 748, 754, 755, 800, 803}, "dh": {525}}
# Variable/variable deck sites: (src defs, dst defs) of each emitted copy, from the audit's
# reaching definitions, and the oracle's (dst, src) argument text that follows them.
VAR_ROLES = {
    "dh:194": ({("addsd@186", "mulsd@194"), ("load@186", "mulsd@194")}, ("c['evap_ice'] * R1_RHOS * RDT", "h_s")),
    "dh:200": ({("minsd@211", "xorpd@200"), ("minsd@211|xorpd@194", "xorpd@200"), ("xorpd@194", "xorpd@200")},
               ("-zhs[jk]", "zdeltah")),
    "dh:228": ({("load@219", "divsd@221")}, ("zq_top / MAXSD(EPSI20, zes[jk], 'dh:221')", "zhs[jk]")),
    "dh:272": ({("xorpd@251", "mulsd@270")}, ("zdum", "-zh[jk]")),
    "dh:295": ({("maxsd@289", "mulsd@300")}, ("zevap * R1_RHOI", "zh[jk]")),
    # One of four emitted copies has the roles reversed; masked by the enclosing MIN(., 0).
    "dh:409": ({("xorpd@388", "mulsd@407"), ("mulsd@407", "xorpd@388")}, ("zdum", "-zh[jk]")),
    "dh:490": ({("maxsd@456|maxsd@494", "load@490"), ("maxsd@494", "load@490")}, ("zhs[jk]", "zdeltah")),
}


def _executing():
    out = set()
    for short, obj in OBJ.items():
        for line in AUDIT["objects"][obj]["sites"]:
            if int(line) not in NOT_ON_DECK.get(short, ()):
                out.add(f"{short}:{line}")
    return out


def _statement(site):
    return site.split(".")[0]


def test_every_executing_site_has_exactly_one_verdict():
    tagged = list(FOUND) + list(MASKED) + [s for v in CLASSES.values() for s in v]
    assert len(tagged) == len(set(tagged))
    assert {_statement(s) for s in tagged} == _executing()


def test_nonzero_constant_sites_have_a_nonzero_constant_operand():
    for site in CLASSES["nonzero_constant"]:
        short, line = _statement(site).split(":")
        ins = AUDIT["objects"][OBJ[short]]["sites"][line]["instructions"]
        consts = [k for i in ins for k in (i["src"], i["dst"]) if k.startswith("CONST:")]
        assert consts and all(float.fromhex(k[6:]) != 0.0 for k in consts), site


def test_abs_sites_compare_only_abs_results():
    for site in CLASSES["abs"]:
        short, line = site.split(":")
        for i in AUDIT["objects"][OBJ[short]]["sites"][line]["instructions"]:
            for defs in (i["src_defs"], i["dst_defs"]):
                assert all(d.startswith(("andpd@", "maxsd@")) for d in defs.split("|")), (site, defs)


def _oracle_calls():
    """site tag -> (dst text, src text) of every tagged MINSD/MAXSD call in the oracle."""
    tree = ast.parse(Path(o9.__file__).read_text())
    dynamic = {"_SITE[kind][1] + '.max'": ("dh:272.max", "dh:409.max")}
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("MAXSD", "MINSD") \
                and len(node.args) == 3:
            tag = node.args[2]
            tags = (tag.value,) if isinstance(tag, ast.Constant) else dynamic.get(ast.unparse(tag), ())
            for t in tags:
                out[t] = (ast.unparse(node.args[0]), ast.unparse(node.args[1]))
    return out


def test_variable_roles_follow_the_object_code():
    calls = _oracle_calls()
    for site, (copies, (dst, src)) in VAR_ROLES.items():
        short, line = site.split(":")
        got = {(i["src_defs"], i["dst_defs"]) for i in AUDIT["objects"][OBJ[short]]["sites"][line]["instructions"]
               if "src_defs" in i}
        assert got == copies, site
        tag = site + ".max" if site in ("dh:272", "dh:409") else site
        assert calls[tag] == (dst, src), site


def _cells(group):
    return {s: o9_cell(v["cell"]) for s, v in CELLS[group].items()}


def o9_cell(hexcell):
    return {k: ([float.fromhex(x) for x in v] if isinstance(v, list) else float.fromhex(v))
            for k, v in hexcell.items()}


def _oracle(cell, swap=()):
    o9.SWAP.clear()
    o9.SWAP.update(swap)
    o9.TIES.clear()
    try:
        leaves = o9._oracle_leaves(o9._oracle_rows([cell], o9._deck()))
    finally:
        o9.SWAP.clear()
    return {k: o9._bits(v) for k, v in leaves.items()}, dict(o9.TIES)


def test_fixture_covers_the_found_and_masked_sites():
    assert sorted(CELLS["found"]) == sorted(FOUND)
    assert sorted(CELLS["masked"]) == sorted(MASKED)
    assert CELLS["candidates"] == 303564 and set(CELLS["tied"]) == set(FOUND) | set(MASKED)


@pytest.mark.parametrize("site", FOUND)
def test_tie_survivor_changes_output_bits(site):
    cell = _cells("found")[site]
    ref, ties = _oracle(cell)
    assert ties.get(site, 0) > 0
    alt, _ = _oracle(cell, {site})
    changed = sorted(k for k in ref if not np.array_equal(ref[k], alt[k]))
    assert changed == CELLS["found"][site]["leaves"]


@pytest.mark.parametrize("site", MASKED)
def test_masked_tie_leaves_output_bits(site):
    cell = _cells("masked")[site]
    ref, ties = _oracle(cell)
    assert ties.get(site, 0) > 0
    alt, _ = _oracle(cell, {site})
    assert all(np.array_equal(ref[k], alt[k]) for k in ref)


def _model_mismatches():
    sites = list(FOUND) + list(MASKED)
    cells = [_cells("found" if s in FOUND else "masked")[s] for s in sites]
    oracle = o9._oracle_leaves(o9._oracle_rows(cells, o9._deck()))
    model = o9._leaves(o9._model(cells, o9._deck()))
    return {k: [sites[i] for i in np.nonzero(np.atleast_2d(o9._bits(model[k]) != o9._bits(v))
                                             .reshape(len(cells), -1).any(axis=1))[0]]
            for k, v in oracle.items() if not np.array_equal(o9._bits(model[k]), o9._bits(v))}


def test_model_equals_the_object_code_survivor_on_every_tie_cell():
    assert _model_mismatches() == {}


def test_plant_reversed_model_roles_fires(monkeypatch):
    monkeypatch.setattr(o9.bl, "_maxsd", lambda d, s: jnp.where(s > d, s, d))
    monkeypatch.setattr(o9.bl, "_minsd", lambda d, s: jnp.where(s < d, s, d))
    jax.clear_caches()
    try:
        assert _model_mismatches() != {}
    finally:
        monkeypatch.undo()
        jax.clear_caches()


def test_search_tool_reproduces_a_found_site():
    spec = importlib.util.spec_from_file_location(
        "tie_search", ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o9_tie_search.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    rep = tool.search(0)  # structured candidates only
    assert rep["candidates"] == 3564 and "dh:180" in rep["found"]
    cell = tool.from_hex(rep["found"]["dh:180"]["cell"])
    assert tool.from_hex(tool.to_hex(cell)) == cell
    jax.clear_caches()
