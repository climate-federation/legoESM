"""RULE 5 switch inventory of the WRF KF-Eta port (``_kf_eta.py``, issue #2280).

Every statement in the traced column code that can switch on model state
carries a ``# kfsw: <id>[,<id>]`` tag, on one of its lines or on a comment
line directly above it.  Switching nodes: a comparison or conditional
expression, ``//`` or ``%``, any use of an integer dtype, a subscript or slice
that is not a literal, one per ``if``/``while`` header, and every call except
to a known-smooth function (``_SMOOTH_CALLS``) or to a function defined in the
module (its body is scanned instead; the masking helpers ``_HELPERS`` count at
their calls and their bodies are tagged ``H``).  The whole module is scanned
except ``_UNTRACED``.  Each id is an
entry of ``_kf_eta.KF_SWITCHES``.  The gate:

* every such statement is tagged, with registered ids only;
* every registered id tags at least one statement;
* switches are counted per id as the switching nodes of the statements it
  tags; ``BASELINE`` records every id's count exactly (static ids included),
  and a ``"hard"`` id may never exceed its row, so a new switch, a status flip
  or a freed switch is always an edit of this file.  A PR that smooths an id
  sets its status to ``"smooth"`` and updates its row.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

import pytest

_SRC = (Path(__file__).resolve().parents[2] / "packages" / "atmosphere" / "legoesm" / "atmosphere"
        / "physics" / "convection" / "_kf_eta.py")

# A call is a switch node unless it is to a function known to be smooth or to
# a function defined in the scanned module itself (whose body is scanned).
# The port's masking helpers are the exception: their calls count, and their
# own bodies are tagged H.
_SMOOTH_CALLS = frozenset({
    "asarray", "array", "exp", "log", "dict", "zeros", "zeros_like", "ones", "ones_like",
    "full", "full_like", "set", "arange", "concatenate", "scan", "tree_map", "items"})
_HELPERS = frozenset({"_w", "_put", "_sd", "_safe_sqrt", "_nint", "_ftrunc_i"})
# A library function referenced without being called (passed to scan/tree_map/
# _scan_levels, or aliased) is a switch node; these uncalled library names are
# values, not functions.
_LIB_ROOTS = frozenset({"jnp", "jax", "lax", "np", "math"})
_LIB_VALUES = frozenset({"inf", "nan", "pi", "float32", "float64", "nn", "numpy", "tree_util",
                         "lax", "core", "Array"})
# Integer dtypes: any use (cast, constructor dtype, alias, dtype string) is a
# switch node.
_INT_DTYPES = frozenset({
    "int", "int_", "intp", "intc", "uint", "long", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64", "i32"})
# Not traced column code: the numpy table build on constants and the batch
# driver (Python-level validation and dispatch).  Everything else in the
# module, module-level statements included, is scanned.
_UNTRACED = frozenset({"kf_lookup_tables", "kf_eta_para"})
_BODIES = ("body", "orelse", "finalbody", "handlers", "cases")
_TAG = re.compile(r"kfsw:\s*([A-Za-z0-9\-]+(?:\s*,\s*[A-Za-z0-9\-]+)*)")

# (status, switch count) per id, exact; the count is the switching nodes of the
# statements the id tags.  Hard counts may only shrink; a PR that smooths an id
# or frees switches edits its row here.
BASELINE = {
    "A0": ("hard", 42), "A1": ("hard", 7), "A2": ("hard", 2), "A3": ("hard", 2),
    "A4": ("hard", 5), "A5": ("hard", 3), "A6": ("hard", 10), "B1": ("hard", 16),
    "B2": ("hard", 29), "B3": ("hard", 75), "B4": ("hard", 33), "B5": ("hard", 4),
    "C-KCHECK": ("hard", 39), "C-KLCL": ("hard", 89), "C-KLCL2": ("hard", 34),
    "C-KPBL": ("hard", 60), "C-L5": ("hard", 6), "C-LDB": ("hard", 33), "C-LET": ("hard", 35),
    "C-LFS": ("hard", 37), "C-LTOP": ("hard", 106), "C-ML": ("hard", 6), "C-NIC": ("hard", 7),
    "C-NSTEP": ("hard", 4), "D1": ("hard", 3), "D2": ("hard", 5), "D3": ("hard", 4),
    "D4": ("hard", 14), "D5": ("hard", 2), "D6": ("hard", 4), "D7a": ("smooth", 18),
    "D7b": ("smooth", 2), "D7c": ("hard", 2), "D7d": ("smooth", 5), "D7e": ("smooth", 7),
    "D7f": ("hard", 8), "D7g": ("hard", 16), "D7h": ("hard", 6), "D7i": ("hard", 15),
    "D7j": ("hard", 4), "D7k": ("hard", 3), "E1": ("hard", 2), "E2": ("hard", 2),
    "E3": ("hard", 2), "E4": ("hard", 3), "E5": ("hard", 4), "E6": ("hard", 6),
    "E7": ("hard", 2), "E8": ("hard", 1), "E9": ("hard", 4), "E10": ("hard", 5),
    "E11": ("hard", 13), "E11c": ("smooth", 4), "E12": ("hard", 5), "E13": ("hard", 2),
    "E14": ("hard", 1), "E15": ("hard", 9), "E16": ("hard", 5), "E17": ("hard", 1),
    "E18": ("hard", 1), "E19": ("hard", 1), "F1": ("smooth", 19), "F2": ("smooth", 40),
    "S": ("static", 137), "H": ("static", 17),
}


def _call_name(node):
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)


def _literal(e):
    if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub):
        e = e.operand
    return e is None or isinstance(e, ast.Constant)


def _is_switch(n, smooth, callee):
    if isinstance(n, (ast.Compare, ast.IfExp)):
        return True
    if isinstance(n, (ast.BinOp, ast.AugAssign)) and isinstance(n.op, (ast.FloorDiv, ast.Mod)):
        return True
    if isinstance(n, ast.Name) and n.id in _INT_DTYPES:
        return True
    if isinstance(n, ast.Attribute) and n.attr in _INT_DTYPES:
        return True
    if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in _INT_DTYPES:
        return True
    if isinstance(n, ast.Call) and _call_name(n) not in smooth:
        return True
    if id(n) not in callee:
        # an uncalled reference to a library function, a helper or an exempt function
        if isinstance(n, ast.Name) and n.id in _HELPERS | _UNTRACED:
            return True
        if isinstance(n, ast.Attribute) and n.attr not in _LIB_VALUES | _SMOOTH_CALLS:
            root = n
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in _LIB_ROOTS:
                return True
    # a non-literal index or slice: a gather at a traced level is a level
    # switch; a static loop index is tagged S
    if isinstance(n, ast.Subscript):
        sl = n.slice
        if isinstance(sl, ast.Slice):
            return not all(_literal(e) for e in (sl.lower, sl.upper, sl.step))
        return not _literal(sl)
    return False


def switch_statements(src):
    """[(first line, last line, column, n switch nodes)] in the scanned code.

    A simple statement is one entry.  A compound statement (def, class, if,
    while, for, with, try, except, match/case) contributes its header (every
    non-body field, plus one switch for an if/while branch) as one entry
    spanning the header's own lines, and its bodies are scanned statement by
    statement."""
    tree = ast.parse(src)
    own = {n.name for n in ast.walk(tree)
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    smooth = (_SMOOTH_CALLS | own) - _HELPERS

    # the function position of a call and the inner links of a dotted name
    callee = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    callee |= {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}

    def count(node):
        return sum(_is_switch(n, smooth, callee) for n in ast.walk(node))

    found = []

    def visit(body):
        for s in body:
            bodies = [f for f in _BODIES if getattr(s, f, None)]
            if not bodies:
                if n := count(s):
                    found.append((s.lineno, s.end_lineno, s.col_offset, n))
                continue
            heads = [v for f, v in ast.iter_fields(s) if f not in _BODIES
                     for v in (v if isinstance(v, list) else [v]) if isinstance(v, ast.AST)]
            n = sum(count(h) for h in heads) + isinstance(s, (ast.If, ast.While))
            first = getattr(s, "lineno", None) or s.pattern.lineno
            last = max([first] + [h.end_lineno for h in heads if hasattr(h, "end_lineno")])
            if n:
                found.append((first, last, getattr(s, "col_offset", 0), n))
            for f in bodies:
                visit(getattr(s, f))

    visit([s for s in tree.body
           if not (isinstance(s, ast.FunctionDef) and s.name in _UNTRACED)])
    return found


def _tag_comments(src):
    return {tok.start[0]: tok.string
            for tok in tokenize.generate_tokens(io.StringIO(src).readline)
            if tok.type == tokenize.COMMENT and _TAG.search(tok.string)}


def _attached(src):
    """{(first, last, col): ((ids, n), comment lines read)} per switching statement."""
    comments = _tag_comments(src)
    lines = src.splitlines()
    out = {}
    for a, b, col, n in switch_statements(src):
        above = a - 1 if lines[a - 2].lstrip().startswith("#") else None
        read = [ln for ln in (above, *range(a, b + 1)) if ln in comments]
        ids = [x.strip() for ln in read for m in _TAG.findall(comments[ln])
               for x in m.split(",")]
        out[(a, b, col)] = ((list(dict.fromkeys(ids)), n), read)
    return out


def statement_tags(src):
    """{(first, last, col): (ids, n switches)} per switching statement."""
    return {k: v for k, (v, _) in _attached(src).items()}


def switch_counts(src):
    counts = {}
    for ids, n in statement_tags(src).values():
        for i in ids:
            counts[i] = counts.get(i, 0) + n
    return counts


def check_inventory(src, registry, baseline):
    """Return the list of gate violations (empty = pass)."""
    errs = []
    att = _attached(src)
    used = {ln for _, read in att.values() for ln in read}
    errs += [f"line {ln}: kfsw tag attached to no switching statement"
             for ln in sorted(set(_tag_comments(src)) - used)]
    for (a, _, _), ((ids, _), _) in sorted(att.items()):
        if not ids:
            errs.append(f"line {a}: switching statement has no kfsw tag")
        errs += [f"line {a}: unknown kfsw id {i!r}" for i in ids if i not in registry]
    counts = switch_counts(src)
    for i, entry in registry.items():
        status = entry[1]
        if status not in ("hard", "smooth", "static"):
            errs.append(f"{i}: bad status {status!r}")
        if counts.get(i, 0) == 0:
            errs.append(f"{i}: registered but tags no statement")
        if status == "hard":
            if i not in baseline or baseline[i][0] != "hard":
                errs.append(f"{i}: new hard switch (not hard in BASELINE)")
            elif counts.get(i, 0) > baseline[i][1]:
                errs.append(f"{i}: {counts[i]} hard switches > baseline {baseline[i][1]}")
    return errs


@pytest.fixture(scope="module")
def src():
    return _SRC.read_text()


@pytest.fixture(scope="module")
def registry():
    from legoesm.atmosphere.physics.convection._kf_eta import KF_SWITCHES
    return KF_SWITCHES


def test_inventory_is_complete(src, registry):
    assert check_inventory(src, registry, BASELINE) == []


def test_baseline_is_tight(src, registry):
    """Every id's switch count is recorded exactly (static ids included), so any
    status flip, new static tag or freed switch is an edit of this file."""
    counts = switch_counts(src)
    assert {i: (e[1], counts.get(i, 0)) for i, e in registry.items()} == BASELINE


def test_helper_tag_is_confined_to_the_counted_helpers(src):
    """H (switch attributed at the call site) is legal only inside the helpers
    whose calls are counted, and every switching statement in them is H."""
    defs = [d for d in ast.parse(src).body if isinstance(d, ast.FunctionDef)]
    for (a, _, _), (ids, _) in statement_tags(src).items():
        owner = next((d.name for d in defs if d.lineno <= a <= d.end_lineno), None)
        assert ("H" in ids) == (owner in _HELPERS), (a, owner, ids)
        assert owner not in _HELPERS or ids == ["H"], (a, owner, ids)
    assert _HELPERS <= {d.name for d in defs}


def test_registry_entries_are_well_formed(registry):
    for i, (family, status, desc) in registry.items():
        assert family in "ABCDEFS" and len(family) == 1, i
        assert desc.strip(), i


# --- planted violations: each gate condition fires ---
_REG = {"X1": ("D", "hard", "x"), "S": ("S", "static", "s")}


def _planted(body):
    return f"def _col(a, b):\n{body}\n    return a\n"


def test_gate_flags_untagged_switch():
    errs = check_inventory(_planted("    a = jnp.where(a > b, a, b)"), _REG, {"X1": ("hard", 1)})
    assert any("no kfsw tag" in e for e in errs)


def test_gate_flags_untagged_integer_cast():
    errs = check_inventory(_planted("    a = b.astype(i32)  # kfsw: X1\n    a = a.astype(i32)"),
                           _REG, {"X1": ("hard", 2)})
    assert any("no kfsw tag" in e for e in errs)


def test_gate_flags_unknown_id():
    errs = check_inventory(_planted("    a = jnp.minimum(a, b)  # kfsw: X1,NOPE\n"
                                    "    c = a == 1  # kfsw: S"), _REG, {"X1": ("hard", 1)})
    assert errs == ["line 2: unknown kfsw id 'NOPE'"]


def test_gate_flags_unused_registry_id():
    body = "    a = jnp.minimum(a, b)  # kfsw: S"
    errs = check_inventory(_planted(body), _REG, {"X1": ("hard", 1)})
    assert errs == ["X1: registered but tags no statement"]


def test_gate_flags_new_hard_id_and_hard_growth():
    body = ("    a = jnp.minimum(a, b)  # kfsw: X1\n    b = a > 0  # kfsw: X1\n"
            "    c = a == 1  # kfsw: S")
    assert check_inventory(_planted(body), _REG, {"X1": ("hard", 2)}) == []
    assert check_inventory(_planted(body), _REG, {"X1": ("hard", 1)}) == [
        "X1: 2 hard switches > baseline 1"]
    assert check_inventory(_planted(body), _REG, {}) == [
        "X1: new hard switch (not hard in BASELINE)"]
    assert check_inventory(_planted(body), _REG, {"X1": ("smooth", 2)}) == [
        "X1: new hard switch (not hard in BASELINE)"]


def test_gate_flags_traced_index_gather():
    body = ("    a = a[b]\n    f = lambda x: a[x]\n"
            "    c = a[k]  # kfsw: S\n    d = a > 0  # kfsw: X1")
    errs = check_inventory(_planted(body), _REG, {"X1": ("hard", 1)})
    assert errs == ["line 2: switching statement has no kfsw tag",
                    "line 3: switching statement has no kfsw tag"]


def test_gate_flags_untagged_branch_header_and_lax_select():
    body = ("    if b:\n        c = a == 1  # kfsw: S\n    while not a:\n        a = 2\n"
            "    a = lax.select(b, a, b)\n    d = a > 0  # kfsw: X1")
    errs = check_inventory(_planted(body), _REG, {"X1": ("hard", 1)})
    assert errs == [f"line {n}: switching statement has no kfsw tag" for n in (2, 4, 6)]


def test_gate_flags_sqrt_ifexp_floordiv_and_slices():
    body = ("    a = jnp.sqrt(a)\n    a = a if b else b\n    a = a // 2\n    a = a[b:3]\n"
            "    a = a[1:3]\n    d = a > 0  # kfsw: X1\n    e = a[k]  # kfsw: S")
    errs = check_inventory(_planted(body), _REG, {"X1": ("hard", 1)})
    assert errs == [f"line {n}: switching statement has no kfsw tag" for n in (2, 3, 4, 5)]


def test_gate_counts_switches_not_statements():
    body = ("    a = jnp.minimum(a, b); b = jnp.maximum(a, b)  # kfsw: X1\n"
            "    c = jnp.where(a > b, a, b)  # kfsw: X1\n    e = a[k]  # kfsw: S")
    assert switch_counts(_planted(body)) == {"X1": 4, "S": 1}
    assert check_inventory(_planted(body), _REG, {"X1": ("hard", 3)}) == [
        "X1: 4 hard switches > baseline 3"]


def test_gate_reads_multiline_branch_header_and_casts():
    body = ("    if (a\n            > b):  # kfsw: S\n        a = jnp.int32(b)\n"
            "        a = lax.convert_element_type(a, i32)  # kfsw: X1")
    errs = check_inventory(_planted(body), _REG, {"X1": ("hard", 2)})
    assert errs == ["line 4: switching statement has no kfsw tag"]


def test_gate_flags_bad_status():
    reg = {**_REG, "X2": ("D", "bogus", "x")}
    body = "    a = a > 0  # kfsw: X1,S,X2"
    assert check_inventory(_planted(body), reg, {"X1": ("hard", 1)}) == ["X2: bad status 'bogus'"]


def test_gate_reads_multiline_statements_and_comment_above():
    body = ("    a = jnp.where(\n        a > b,  # kfsw: X1\n        a, b)\n"
            "    # kfsw: S\n    c = jnp.clip(a, 0, 1)")
    assert check_inventory(_planted(body), _REG, {"X1": ("hard", 2)}) == []
    body_gap = "    # kfsw: S\n\n    c = jnp.clip(a, 0, 1)\n    d = b < 0  # kfsw: X1"
    errs = check_inventory(_planted(body_gap), _REG, {"X1": ("hard", 1)})
    assert any("no kfsw tag" in e for e in errs)


def test_gate_flags_augassign_int_dtype_with_and_try():
    body = ("    a //= b\n    a = jnp.asarray(a, dtype=jnp.int32)\n    with ctx(a > b):\n"
            "        try:\n            c = a == 1  # kfsw: S\n            d = a > 0\n"
            "        except E:\n            e = a < 0  # kfsw: X1\n")
    errs = check_inventory(_planted(body), _REG, {"X1": ("hard", 1)})
    assert errs == [f"line {n}: switching statement has no kfsw tag" for n in (2, 3, 4, 7)]


def test_gate_tag_parsing_dedupes_and_reads_comma_space():
    body = ("    # kfsw: X1\n    a = jnp.minimum(a, b)  # kfsw: X1, S\n"
            "    b = a > 0  # note; kfsw: X1 ; kfsw: S")
    assert switch_counts(_planted(body)) == {"X1": 2, "S": 2}


def test_gate_scans_helper_bodies():
    src = "def _w(m, new, old):\n    return jnp.where(m, new, old)\n"
    assert check_inventory(src, {"S": ("S", "static", "s")}, {}) == [
        "line 2: switching statement has no kfsw tag", "S: registered but tags no statement"]


def test_gate_counts_unknown_calls_aliases_and_dtype_strings():
    body = ("    a = jnp.greater(a, b)\n    a = jax.nn.relu(a)\n    choose = jnp.where\n"
            "    a = choose(b, a, b)\n    a = jnp.asarray(a, dtype=\"int32\")\n"
            "    a = a[k] + jnp.exp(a) + _col(a, b)  # kfsw: S\n    d = a > 0  # kfsw: X1")
    errs = check_inventory(_planted(body), _REG, {"X1": ("hard", 1)})
    assert errs == [f"line {n}: switching statement has no kfsw tag" for n in (2, 3, 4, 5, 6)]


def test_gate_scans_module_level_and_except_headers():
    src = ("f = lambda x: jnp.where(x > 0, x, 0)\n"
           "def _col(a):\n    try:\n        a = a > 0  # kfsw: X1\n"
           "    except ValueError if a else TypeError:\n        pass\n    return a\n")
    errs = check_inventory(src, _REG, {"X1": ("hard", 1)})
    assert errs == ["line 1: switching statement has no kfsw tag",
                    "line 5: switching statement has no kfsw tag",
                    "S: registered but tags no statement"]


def test_gate_flags_tag_on_no_switching_statement():
    body = "    a = a + 1  # kfsw: X1\n    d = a > 0  # kfsw: X1\n    e = a[k]  # kfsw: S"
    errs = check_inventory(_planted(body), _REG, {"X1": ("hard", 1)})
    assert errs == ["line 2: kfsw tag attached to no switching statement"]


def test_gate_flags_switching_function_passed_to_higher_order_call():
    body = ("    a = jax.tree_util.tree_map(jnp.minimum, a, b)\n"
            "    a = jax.tree_util.tree_map(lambda x: x, a)\n"
            "    a = _scan_levels(jnp.maximum, a, 1, 3)\n"
            "    a = jax.lax.scan(f=lax.select, init=a)\n"
            "    a = jax.lax.scan(kf_lookup_tables, a)\n    a = jax.lax.scan(zbody, a) * jnp.inf\n"
            "    d = a > 0  # kfsw: X1\n    e = a[k]  # kfsw: S")
    errs = check_inventory(_planted(body), _REG, {"X1": ("hard", 1)})
    assert errs == [f"line {n}: switching statement has no kfsw tag" for n in (2, 4, 5, 6)]
