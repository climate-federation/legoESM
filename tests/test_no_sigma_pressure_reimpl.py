"""Ratchet: physics bridges and drivers take pressure and layer thickness
from the vertical coordinate, never from ``p_s * sigma_full`` /
``p_s * dsigma`` / ``pressure_from_sigma``.

On the hybrid lanes (``vertical_coord='cam_l32'``) ``sigma_full`` is the
compatibility view ``A_full + B_full``, so ``p_s * sigma_full`` is the WRONG
pressure wherever ``p_s != p_ref`` (codex deck review 2026-09-21, P1).  The
only correct spelling is ``coord.pressure_at_full(p_s)`` /
``pressure_at_half``, which is exact ``p_s * sigma`` on ``SigmaCoordinate``.

Token-based (not line-regex): docstrings and comments may still describe the
old convention; only executable code is scanned.

KNOWN BLIND SPOTS (the rule is name-based, one logical line at a time): an
alias (``sig = coord.sigma_full; p = p_s * sig``), a differently named
surface pressure (``ps``, ``p_sfc``), ``jnp.multiply`` / ``einsum`` /
``outer`` spellings, and a sigma array handed to a consumer that multiplies
deeper down (``column_water_vapor(q, p_s, dsigma)`` without ``dp=``).  The
functional test tests/unit/test_hybrid_pressure_in_physics_bridges.py covers
the turbulence / microphysics bridges by value, not by spelling.
"""
from __future__ import annotations

import io
import tokenize
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCANNED = (
    REPO / "packages/atmosphere/legoesm/atmosphere/physics",
    REPO / "packages/coupler/legoesm/driver",
    REPO / "packages/coupler/legoesm/coupler",
)
SIGMA_NAMES = {"sigma_full", "sigma_half", "dsigma"}


def sigma_pressure_violations(source: str) -> list[int]:
    """Logical lines that multiply ``p_s`` by ``sigma_full``/``sigma_half``
    or ``dsigma`` (one operand of a binary ``*`` / ``*=`` names ``p_s``, the
    other a sigma array; attribute / subscript / call chains are walked) or call
    ``pressure_from_sigma``.  Returns their starting line numbers."""
    out: list[int] = []
    toks: list[tokenize.TokenInfo] = []
    start = None
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if start is None and tok.type not in (tokenize.NL, tokenize.NEWLINE,
                                              tokenize.COMMENT, tokenize.INDENT,
                                              tokenize.DEDENT):
            start = tok.start[0]
        if tok.type in (tokenize.NAME, tokenize.OP):
            toks.append(tok)
        if tok.type == tokenize.NEWLINE:
            if _is_violation(toks):
                out.append(start)
            toks, start = [], None
    return out


_OPEN, _CLOSE = "([{", ")]}"


def _left_operand(toks, i):
    """NAME tokens of the operand ending at ``toks[i-1]`` (a NAME / attribute
    / subscript / call chain, brackets balanced)."""
    names, depth, j = set(), 0, i - 1
    while j >= 0:
        t = toks[j]
        if t.type == tokenize.OP:
            if t.string in _CLOSE:
                depth += 1
            elif t.string in _OPEN:
                if depth == 0:
                    break
                depth -= 1
            elif depth == 0 and t.string != ".":
                break
        elif depth == 0 and t.type == tokenize.NAME and j > 0 and (
                toks[j - 1].type == tokenize.NAME):
            names.add(t.string)
            break  # ``return x`` / ``not x``: keyword before the operand
        if t.type == tokenize.NAME:
            names.add(t.string)
        j -= 1
    return names


def _right_operand(toks, i):
    """NAME tokens of the operand starting at ``toks[i+1]``."""
    names, depth, j = set(), 0, i + 1
    while j < len(toks):
        t = toks[j]
        if t.type == tokenize.OP:
            if t.string in _OPEN:
                depth += 1
            elif t.string in _CLOSE:
                if depth == 0:
                    break
                depth -= 1
            elif depth == 0 and t.string != ".":
                break
        if t.type == tokenize.NAME:
            names.add(t.string)
        j += 1
    return names


def _is_violation(toks) -> bool:
    names = {t.string for t in toks if t.type == tokenize.NAME}
    if "pressure_from_sigma" in names:
        return True
    if "p_s" not in names:
        return False
    for i, t in enumerate(toks):
        if t.type == tokenize.OP and t.string in ("*", "*=") and i > 0 and (
                toks[i - 1].type == tokenize.NAME
                or toks[i - 1].string in (")", "]")):  # binary *, not *args
            left, right = _left_operand(toks, i), _right_operand(toks, i)
            if ("p_s" in left and right & SIGMA_NAMES) or (
                    "p_s" in right and left & SIGMA_NAMES):
                return True
    return False


# Verified sigma fallbacks (shrink-only).  Each entry: file -> reason that
# was checked in code when it was written.
ALLOWLIST = {
    # DiagnosticCollector._p_full (661) / _dp (668) keep a pure-sigma fallback for
    # vcoord=None; the only production construction (model_driver
    # ``self.diagnostics = DiagnosticCollector(... vcoord=self.sigma``)
    # always passes the coordinate, and
    # tests/unit/test_diagnostics_hybrid_pressure.py pins the fallback.
    "packages/coupler/legoesm/driver/diagnostics.py": [661, 668],
}


def _scanned_files():
    for root in SCANNED:
        yield from sorted(root.rglob("*.py"))


@pytest.mark.parametrize("path", list(_scanned_files()),
                         ids=lambda p: str(p.relative_to(REPO)))
def test_no_sigma_pressure_reimpl(path):
    hits = sigma_pressure_violations(path.read_text())
    allowed = ALLOWLIST.get(str(path.relative_to(REPO)), [])
    stale = [a for a in allowed if a not in hits]
    assert not stale, f"ALLOWLIST entries no longer match a hit: {stale}"
    hits = [h for h in hits if h not in allowed]
    assert not hits, (
        f"{path.relative_to(REPO)} lines {hits}: pressure built from p_s * "
        "sigma_full/sigma_half (or pressure_from_sigma).  Use "
        "sigma_coord.pressure_at_full(p_s) / pressure_at_half(p_s); on the "
        "hybrid (cam_l32) lane sigma_full is A_full+B_full and the product is "
        "the wrong pressure.")


def test_ratchet_is_not_vacuous():
    bad = (
        "def f(p_s, sigma_coord):\n"
        "    # p = p_s * sigma_full is fine in a comment\n"
        "    '''and p_s * sigma_full in a docstring'''\n"
        "    p_full = p_s[..., None] * sigma_coord.sigma_full\n"
        "    p_half = pressure_from_sigma(sigma_coord.sigma_half, p_s)\n"
        "    return p_full, p_half\n"
    )
    assert sigma_pressure_violations(bad) == [4, 5]
    dp = "dp = p_s[..., None] * (sigma_half[1:] - sigma_half[:-1])\n"
    assert sigma_pressure_violations(dp) == [1]
    assert sigma_pressure_violations("dp = p_s * dsigma[-1]\n") == [1]
    star_args = "snap(p_s, dsigma, T, *(getattr(c, n) for n in names))\n"
    assert sigma_pressure_violations(star_args) == []
    wrapped = ("_pf = self.state.p_s.data[:, None] * jnp.asarray(\n"
               "    self.sigma.sigma_full)[None, :]\n")
    assert sigma_pressure_violations(wrapped) == [1]
    assert sigma_pressure_violations("f(p_s, dsigma, weight * 2)\n") == []
    assert sigma_pressure_violations(
        "dp = (sigma_half[1:] - sigma_half[:-1])[None, :] * p_s[:, None]\n") == [1]
    assert sigma_pressure_violations("f(x=p_s * sigma_full)\n") == [1]
    assert sigma_pressure_violations("return p_s[..., None] * sigma_full\n") == [1]
    assert sigma_pressure_violations("p = p_s[..., None]\np *= dsigma\n") == []  # p_s not on the line: blind spot
    good = (
        "def f(p_s, sigma_coord):\n"
        "    p_full = sigma_coord.pressure_at_full(p_s)\n"
        "    p_low = sigma_coord.pressure_at_full(p_s)[..., -1]\n"
        "    return p_full * 2.0, sigma_coord.sigma_full * 0.5\n"
        "    tracker.update(p_s, sigma_full, a, b, c, d, e, f, g, t=day * 86400.0)\n"
    )
    assert sigma_pressure_violations(good) == []
