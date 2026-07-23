#!/usr/bin/env python3
"""PreToolUse hook: block an Edit/Write/MultiEdit/NotebookEdit that introduces a
hardcoded physical constant or a re-derived saturation prefactor.

Real-time, edit-time counterpart of ``tests/test_no_hardcoded_constants.py`` and
``tests/test_no_saturation_reimpl.py`` — it catches the banned literal *before* it
lands, instead of at CI. Deliberately a fast, line-oriented pre-filter; the AST
ratchet tests remain the AUTHORITATIVE check, so two known gaps are acceptable:
  * non-canonical spellings (``273.150``, ``2.7315e2``, ``6371*1000``) are not
    flagged here (the CI tests fold/normalise and catch them);
  * a file written via a raw ``Bash`` heredoc/sed bypasses this hook entirely
    (it only sees tool-based edits) — again, CI is the backstop.

Honored escape: a line carrying ``# const-ok`` / ``# satcurve-ok`` is skipped.
Exempt paths: the constants/thermo definition sites, the guard tests, this
``.claude/hooks/`` directory (it necessarily lists the literals), and
``scripts/tmp`` probes.

Fail-open: ANY error or unexpected input → exit 0 (allow). It must never wedge a
session. Block protocol: exit 2 with a message on stderr (fed back to the model).

Runs under whatever ``python3`` the harness invokes it with, including the
Python 3.6 shipped as the system interpreter on some HPC login nodes — so it
avoids 3.7+-only syntax (``from __future__ import annotations``) and 3.9+-only
subscripted-generic annotations in evaluated (signature) positions.
"""
import json
import re
import sys

_BANNED_LITERALS = [
    "6.371229e6", "6.371e6", "9.80616", "7.292e-5", "1004.64", "2.501e6",
    "461.51", "287.05", "287.0", "273.15", "611.2", "0.622",
]
_SYMBOL = {
    "6.371229e6": "R_earth", "6.371e6": "R_earth", "9.80616": "g",
    "7.292e-5": "Omega", "1004.64": "c_pd", "2.501e6": "L_v", "461.51": "R_v",
    "287.05": "R_d", "287.0": "R_d", "273.15": "T_freeze",
    "611.2": "thermo.saturation_vapor_pressure", "0.622": "epsilon",
}
# A banned literal as a whole numeric token: not preceded or followed by another
# number/word/dot char (so ``273.15e2``, ``287.0j``, ``x273.15`` do NOT match).
_PAT = re.compile(
    r"(?<![0-9A-Za-z_.])(?:%s)(?![0-9A-Za-z_.])" % "|".join(re.escape(v) for v in _BANNED_LITERALS)
)
_EXEMPT_COMMENT = ("const-ok", "satcurve-ok")
_WHITELIST_SUFFIX = (
    "/constants.py", "/thermo.py",
    "/test_no_hardcoded_constants.py", "/test_no_saturation_reimpl.py",
)
_WHITELIST_SUBSTR = ("/scripts/tmp/", "/.claude/hooks/")


def _proposed_text(tool: str, ti: dict) -> str:
    if tool == "Write":
        return str(ti.get("content", "") or "")
    if tool == "Edit":
        return str(ti.get("new_string", "") or "")
    if tool == "MultiEdit":
        return "\n".join(
            str((e or {}).get("new_string", "") or "") for e in ti.get("edits", []) or []
        )
    if tool == "NotebookEdit":
        src = ti.get("new_source", "")
        return "\n".join(map(str, src)) if isinstance(src, list) else str(src or "")
    return ""


def _hits(text):  # -> list[str]
    out = []
    for line in text.splitlines():
        if any(tag in line for tag in _EXEMPT_COMMENT):
            continue
        for m in _PAT.finditer(line):
            lit = m.group(0)
            out.append(f"{lit!r} (-> legoesm.constants.{_SYMBOL.get(lit, '?')})  in: {line.strip()[:80]}")
    return out


def main() -> None:
    try:
        data = json.load(sys.stdin)
        if not isinstance(data, dict):
            sys.exit(0)
        tool = data.get("tool_name", "")
        ti = data.get("tool_input", {})
        if not isinstance(ti, dict):
            sys.exit(0)
        if tool == "NotebookEdit":
            fp = str(ti.get("notebook_path", "") or "").replace("\\", "/")
        else:
            fp = str(ti.get("file_path", "") or "").replace("\\", "/")
            if not fp.endswith(".py"):
                sys.exit(0)
        # Normalise so relative paths (``.claude/hooks/x.py``) match the same as
        # absolute ones before the whitelist check.
        norm = "/" + fp.lstrip("/")
        if any(norm.endswith(s) for s in _WHITELIST_SUFFIX) or any(s in norm for s in _WHITELIST_SUBSTR):
            sys.exit(0)
        hits = _hits(_proposed_text(tool, ti))
    except Exception:
        sys.exit(0)  # fail-open on anything unexpected
    if not hits:
        sys.exit(0)
    print(
        "Blocked by harness (check_banned_literals): hardcoded physical "
        f"constant / saturation prefactor in the edit to {fp}:\n"
        + "\n".join(f"  - {h}" for h in hits)
        + "\nUse `from legoesm import constants` / `legoesm.thermo` instead. If a "
        "value is genuinely unrelated to the physical constant, add a "
        "`# const-ok: <reason>` (or `# satcurve-ok:`) comment on that line.",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
