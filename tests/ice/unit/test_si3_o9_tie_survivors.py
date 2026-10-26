"""O9 MIN/MAX tie survivors, one verdict per executing site of the record object code.

Every ``minsd``/``maxsd`` of the O9 routines (``nemo_si3_o9_minmax_audit.json``) carries one
verdict in that report's ``tie_verdicts``: FOUND sites have a cell
(``data/si3_o9_tie_cells.json``, from ``nemo_si3_o9_tie_search.py``) on which the independent
oracle sees the tie, swapping that one site's emitted operand roles changes output bits, and
the model equals the unswapped oracle in every bit; NO_TIE and MASKED sites carry a short
proof whose premises (operand kinds, census lines, enclosing blocks, deck values) are checked
here against the report, and every MASKED site with zero results is also checked by negating
those zeros in the oracle; NOT_ON_DECK sites name the arm and the deck value that skips it.
"""

from __future__ import annotations

import ast
import importlib.util
import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_si3_o9_zdf_dh_temp as o9

from legoesm.core.nemo_namelist import overlay, parse_namelist

ROOT = Path(__file__).resolve().parents[3]
CELLS = json.loads((Path(__file__).parent / "data/si3_o9_tie_cells.json").read_text())
AUDIT = json.loads((ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o9_minmax_audit.json").read_text())
VERDICTS = AUDIT["tie_verdicts"]
OBJ = {"dh": "icethd_dh", "zdf": "icethd_zdf_bl99", "var": "icevar", "thd": "icethd"}
SEARCH = ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o9_tie_search.py"

libm_policy = o9.libm_policy  # same autouse precision policy as the O9 suite

FOUND = tuple(sorted(t for t, v in VERDICTS.items() if v["verdict"] == "FOUND"))
MASKED = tuple(sorted(t for t, v in VERDICTS.items() if ["tie_masked"] in v["facts"]))  # tie cell, swap invisible
FLIPPED = tuple(sorted(t for t, v in VERDICTS.items() if ["flip"] in v["facts"]))
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


def _site(tag):
    short, rest = tag.split(":")
    line, _, part = rest.partition(".")
    return short, line, part


def _ins(tag, rep=AUDIT):
    """The instructions a verdict tag covers (``.min``/``.max`` by op; ``.max1``/``.max2`` ZERO/CONST src)."""
    short, line, part = _site(tag)
    out = []
    for i in rep["objects"][OBJ[short]]["sites"][line]["instructions"]:
        if i["op"] == "sqrtsd":
            continue
        if (part in ("", "min", "max") and (not part or i["op"] == part + "sd")) \
                or (part == "max1" and i["op"] == "maxsd" and i["src"] == "ZERO") \
                or (part == "max2" and i["op"] == "maxsd" and i["src"].startswith("CONST:")):
            out.append(i)
    return out


def test_every_site_instruction_has_exactly_one_verdict():
    covered = {}
    for tag in VERDICTS:
        short, line, _ = _site(tag)
        assert _ins(tag), tag
        for i in _ins(tag):
            covered.setdefault(id(i), []).append(tag)
    n = 0
    for obj in OBJ.values():
        for line, site in AUDIT["objects"][obj]["sites"].items():
            for i in site["instructions"]:
                if i["op"] != "sqrtsd":
                    n += 1
                    assert len(covered.get(id(i), [])) == 1, (obj, line, i, covered.get(id(i)))
    assert n == len(covered)
    assert {v["verdict"] for v in VERDICTS.values()} == {"FOUND", "NO_TIE", "MASKED", "NOT_ON_DECK"}
    assert all(v["proof"] and v["facts"] for v in VERDICTS.values())


def _deck_value(name):
    v = overlay(parse_namelist(o9.DATA / "namelist_ice_ref", "ice"), parse_namelist(o9.DATA / "namelist_ice_cfg", "ice"))
    hits = [val[0] for key, val in v.items() if key[2] == name]
    assert len(hits) == 1, name
    return hits[0]


def _census(routine, var, rep):
    for obj in rep["objects"].values():
        if routine in obj["census"]:
            return obj["census"][routine][var]
    raise KeyError(routine)


def _sitedict(tag, rep):
    short, line, _ = _site(tag)
    return rep["objects"][OBJ[short]]["sites"][line]


def check_fact(tag, fact, rep=AUDIT):
    """True when report ``rep`` satisfies one premise of ``tag``'s verdict."""
    kind, *a = fact
    if kind == "src_zero":
        return all(i["src"] == "ZERO" for i in _ins(a[0], rep))
    if kind == "const":
        return all(any(k.startswith("CONST:") and float.fromhex(k[6:]) != 0.0 for k in (i["src"], i["dst"]))
                   for i in _ins(a[0], rep))
    if kind == "abs":
        return all(all(d.startswith(("andpd@", "maxsd@")) for d in i[f].split("|"))
                   for i in _ins(a[0], rep) for f in ("src_defs", "dst_defs"))
    if kind == "defs":
        return all(i.get(a[1] + "_defs") == a[2] for i in _ins(a[0], rep))
    if kind in ("roles", "roles_of"):
        short, line, _ = _site(a[0] if a else tag)
        return f"{short}:{line}" in VAR_ROLES
    if kind == "encl":
        return a[1] in _sitedict(a[0], rep)["enclosing"]
    if kind == "encl_only_do":
        return all(h.startswith("DO ") for h in _sitedict(a[0], rep)["enclosing"])
    if kind == "writes":
        return sorted(int(k) for k, u in _census(a[0], a[1], rep).items() if "w" in u[0]) == a[2]
    if kind == "uses":
        return sorted(int(k) for k in _census(a[0], a[1], rep)) == a[2]
    if kind == "stmt":
        return _census(a[0], a[1], rep)[str(a[2])][1] == a[3]
    if kind == "census_encl":
        return a[3] in _census(a[0], a[1], rep)[str(a[2])][2]
    if kind == "text":
        return a[1] in " ".join(_sitedict(a[0], rep)["statement"].split())
    if kind == "deck":
        return _deck_value(a[0]) == a[1]
    if kind == "found":
        return tag in CELLS["found"]
    if kind == "tie_masked":
        return tag in CELLS["masked"]
    if kind == "flip":
        return tag in FLIPPED
    raise ValueError(f"unknown fact {kind}")


@pytest.mark.parametrize("tag", sorted(VERDICTS))
def test_verdict_premises_hold(tag):
    v = VERDICTS[tag]
    bad = [f for f in v["facts"] if not check_fact(tag, f)]
    assert bad == [], (tag, bad)
    if v["verdict"] == "FOUND":
        assert ["found"] in v["facts"]
    if v["verdict"] == "NOT_ON_DECK":
        assert {"deck", "encl"} <= {f[0] for f in v["facts"]}
    assert all(lemma in AUDIT["tie_lemmas"] for lemma in ("L0", "L1", "L2", "L3") if lemma in v["proof"])


@pytest.mark.parametrize("plant", ["kind", "uses", "encl", "deck", "text", "census_encl"])
def test_fact_plants_fail(plant):
    """Each premise kind can fail: perturb a copy of the report under one verdict per kind."""
    import copy
    rep = copy.deepcopy(AUDIT)
    dh = rep["objects"]["icethd_dh"]
    if plant == "kind":
        dh["sites"]["170"]["instructions"][0]["src"] = "VAR"
        tag, fact = "dh:170", ["src_zero", "dh:170"]
    elif plant == "uses":
        dh["census"]["ice_thd_dh"]["zq_top"]["300"] = ["rw", "zq_top = zq_top * 2", []]
        tag, fact = "dh:128", VERDICTS["dh:128"]["facts"][1]
    elif plant == "encl":
        dh["sites"]["229"]["enclosing"] = ["DO ji = 1, npti"]
        tag, fact = "dh:229", VERDICTS["dh:229"]["facts"][1]
    elif plant == "deck":
        assert check_fact("dh:525", ["deck", "nn_icesal", 4]) and not check_fact("dh:525", ["deck", "nn_icesal", 2])
        return
    elif plant == "text":
        rep["objects"]["icethd_zdf_bl99"]["sites"]["219"]["statement"] = "x = MAX( 0._wp, y )"
        tag, fact = "zdf:219", VERDICTS["zdf:219"]["facts"][2]
    else:
        dh["census"]["snw_ent"]["zeh_cum1"]["608"][2] = ["DO jk0 = 1, nlay_s+1"]
        tag, fact = "dh:619.max1", VERDICTS["dh:619.max1"]["facts"][3]
    assert check_fact(tag, fact) and not check_fact(tag, fact, rep)


def _flip_cells():
    spec = importlib.util.spec_from_file_location("tie_search_flip", SEARCH)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    fixture = [o9_cell(v["cell"]) for g in ("found", "masked") for v in CELLS[g].values()]
    return fixture + list(itertools.islice(tool.candidates(0), 0, None, 6))


def _flipping(fn, sites, hits):
    """The oracle's MAXSD/MINSD with a zero result negated at ``sites`` (counted in ``hits``)."""
    def wrapped(dst, src, site=None):
        r = fn(dst, src, site)
        if site in sites and r == 0.0:
            hits.append(site)
            return -r
        return r
    return wrapped


def _flip_report(tags):
    """tag -> (cells where the site's zero result was negated, cells whose output bits changed)."""
    nml = o9._deck()
    out = {t: [0, 0] for t in tags}
    maxsd, minsd = o9.MAXSD, o9.MINSD
    for cell in _flip_cells():
        try:
            ref = o9._oracle_leaves(o9._oracle_rows([cell], nml))
        except (ZeroDivisionError, ValueError, OverflowError):
            continue
        for t in tags:
            sites, hits = ({t} if "." in t else {t, t + ".max1", t + ".max2"}), []
            o9.MAXSD, o9.MINSD = _flipping(maxsd, sites, hits), _flipping(minsd, sites, hits)
            try:
                alt = o9._oracle_leaves(o9._oracle_rows([cell], nml))
            finally:
                o9.MAXSD, o9.MINSD = maxsd, minsd
            if hits:
                out[t][0] += 1
                out[t][1] += any(not np.array_equal(o9._bits(ref[k]), o9._bits(alt[k])) for k in ref)
    return out


def test_masked_zero_signs_never_reach_the_output():
    rep = _flip_report(FLIPPED + ("dh:251",))
    assert rep.pop("dh:251")[1] > 0  # plant: a FOUND site's zero sign does reach the output
    for t, (negated, changed) in rep.items():
        assert negated > 0 and changed == 0, (t, negated, changed)


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
    assert CELLS["candidates"] == 303726 and set(CELLS["tied"]) == set(FOUND) | set(MASKED)


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
    assert rep["candidates"] == 3726 and {"dh:180", "dh:295"} <= set(rep["found"])  # 3564 structured + 162 melt-outs
    cell = tool.from_hex(rep["found"]["dh:180"]["cell"])
    assert tool.from_hex(tool.to_hex(cell)) == cell
    jax.clear_caches()
