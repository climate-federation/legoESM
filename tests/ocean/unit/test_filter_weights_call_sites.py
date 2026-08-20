"""Every caller of ``compute_filter_weights`` must unpack all four values.

#1609 widened the barotropic averaging window to 2n-1 substeps so its first
moment lands on t+dt, and added the loop length ``n_loop`` as a fourth return
value. Two call sites were updated; a THIRD -- the lat-lon C-grid solver's
non-NEMO branch -- was missed and shipped to main unpacking three, which

  * raised ``ValueError: too many values to unpack`` the moment that branch
    ran (it took the ocean benchmark suite down on the lat-lon rest-state
    cases), and
  * had ``n_loop = n_substeps`` hardcoded underneath, so "fixing" the unpack
    by dropping the fourth value would silently restore the HALF window #1609
    removed -- the one that ran gravity waves at about half speed.

Two adversarial reviews of #1609 both checked "the production callers" and
both found only two. This test is the mechanical version of that check, so
the next change to the signature cannot rely on anyone enumerating callers
correctly.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_ROOTS = (_REPO / "packages", _REPO / "scripts", _REPO / "src")
_FN = "compute_filter_weights"


def _call_sites():
    """(path, lineno, n_targets) for every ``... = compute_filter_weights(...)``."""
    out = []
    for root in _ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text())
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                call = node.value
                if not isinstance(call, ast.Call):
                    continue
                fn = call.func
                name = (fn.id if isinstance(fn, ast.Name)
                        else fn.attr if isinstance(fn, ast.Attribute) else None)
                if name != _FN:
                    continue
                tgt = node.targets[0]
                n = len(tgt.elts) if isinstance(tgt, (ast.Tuple, ast.List)) else 1
                out.append((path.relative_to(_REPO), node.lineno, n))
    return out


def test_every_call_site_unpacks_four():
    """The check that would have caught the shipped defect."""
    sites = _call_sites()
    assert sites, "found no call sites -- this test has stopped checking anything"
    bad = [(p, ln, n) for p, ln, n in sites if n != 4]
    assert not bad, (
        "these unpack the wrong number of values from compute_filter_weights "
        f"(it returns 4: w_filter, w_total, w_transport, n_loop): {bad}")


def test_the_function_really_returns_four():
    """Non-vacuity: pin the arity the test above is asserting against.

    If the signature ever changes deliberately, this goes red first and says
    so, rather than the caller test failing mysteriously."""
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.barotropic_common import compute_filter_weights
    got = compute_filter_weights(8, jnp.float64, use_cosine=True)
    assert len(got) == 4


def test_no_caller_hardcodes_the_loop_length():
    """``n_loop = n_substeps`` right after a call is the half window #1609 removed.

    The missed call site had exactly that line underneath it, so a careless fix
    to the unpack -- drop the fourth value, keep the old assignment -- would
    have turned the error green while restoring the defect.

    Scoped to the few lines FOLLOWING a call, not the whole file: the same
    module has a legitimate ``n_loop = n_substeps`` in its MOM6/AM4 branch,
    which builds its own weights of length n_substeps and never calls this
    function. Flagging that one would be a false positive, and a guard that
    cries wolf gets deleted."""
    for path, lineno, _n in _call_sites():
        lines = (_REPO / path).read_text().splitlines()
        window = lines[lineno - 1:lineno + 6]
        offenders = [w.strip() for w in window
                     if w.strip() == "n_loop = n_substeps"]
        assert not offenders, (
            f"{path}:{lineno} hardcodes n_loop = n_substeps just after calling "
            f"compute_filter_weights; that is the HALF window. Take the value "
            f"the function returns.")


def test_the_legitimate_hardcode_is_still_there():
    """Non-vacuity for the scoping above.

    If the MOM6/AM4 branch ever stops setting its own loop length, the test
    above has quietly narrowed to nothing and should be re-examined rather
    than trusted."""
    src = (_REPO / "packages/ocean/legoesm/ocean/dynamics"
           / "barotropic_latlon_cgrid.py").read_text()
    assert "n_loop = n_substeps" in src, (
        "the branch this test was scoped around has gone; re-check that the "
        "scoping in test_no_caller_hardcodes_the_loop_length still makes sense")
