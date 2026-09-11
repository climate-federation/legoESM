#!/usr/bin/env python3
"""Every bare symbol an instrument writes must be IN SCOPE where it lands.

This exists because it was not.  The first version of ``run.sh`` inserted
``IF( kt <= nit000 + 1 )`` into ``tra_zdf_imp``, whose dummy arguments are
``(cdtype, p2dt, Kbb, Kmm, Krhs, pt, Kaa, kjpt)`` -- there is no ``kt`` there.
Nothing noticed until gfortran did, sixty seconds into a build the user had
already paid for, with ``Symbol 'kt' at (1) has no IMPLICIT type``.

The check reads the PRISTINE COMPILED source, so it sees the routine's real
argument list and real declaration block rather than what the patch believes
it inserted:

* the enclosing ``SUBROUTINE`` of the insertion anchor, its dummy arguments
  and every name declared between its header and the anchor;
* module-scope declarations of the routine's own file;
* every module-scope declaration in NEMO's COMPILED tree -- i.e. every name
  that could reach the routine through a ``USE``.  A module variable is
  declared BEFORE its module's ``CONTAINS``; ``kt`` never is, because it is
  always a dummy argument, which is exactly why this discriminates.

WHAT IT CANNOT SEE, named rather than left to be found: a name that IS a
module variable somewhere in NEMO but is not reachable through THIS file's
``USE`` list.  Resolving that needs the transitive USE closure; the compiler
still catches it, and it is a much rarer mistake than the one above.
"""
from __future__ import annotations

import pathlib
import re
import sys

#: Fortran keywords and intrinsics the tokenizer must not treat as variables.
_KEYWORDS = {
    "if", "then", "else", "elseif", "endif", "end", "do", "enddo", "while",
    "write", "open", "close", "trim", "unit", "file", "access", "form",
    "status", "and", "or", "not", "true", "false", "call", "return",
}


def _strip(code: str) -> str:
    """Fortran comments and character literals removed, line by line."""
    out = []
    for line in code.splitlines():
        line = re.sub(r"'[^']*'", " ", line)
        line = re.sub(r'"[^"]*"', " ", line)
        cut = line.find("!")
        out.append(line if cut < 0 else line[:cut])
    return "\n".join(out)


def _declared(text: str) -> set[str]:
    names: set[str] = set()
    for m in re.finditer(r"::([^\n]*)", _strip(text)):
        for tok in re.findall(r"[A-Za-z_]\w*", m.group(1)):
            names.add(tok.lower())
    return names


def module_scope_names(ppsrc_dir: pathlib.Path) -> set[str]:
    """Every name declared BEFORE a ``CONTAINS`` in the compiled tree."""
    names: set[str] = set()
    for f in sorted(ppsrc_dir.glob("*.f90")):
        text = f.read_text(errors="replace")
        head = re.split(r"(?im)^\s*CONTAINS\s*$", text, maxsplit=1)[0]
        names |= _declared(head)
    if not names:
        raise SystemExit(f"REFUSED: no module-scope names found under "
                         f"{ppsrc_dir}; the check would pass vacuously")
    return names


def check(ppsrc_file: pathlib.Path, anchor: str, patched: str,
          block_re: str, declared_by_patch: set[str],
          ppsrc_dir: pathlib.Path | None = None) -> str:
    """Return the enclosing routine name, or raise SystemExit naming the gap."""
    src = ppsrc_file.read_text(errors="replace").splitlines()
    hit = [i for i, line in enumerate(src) if anchor in line]
    if len(hit) != 1:
        raise SystemExit(
            f"REFUSED: the anchor {anchor!r} occurs {len(hit)} times in "
            f"{ppsrc_file}; the scope check cannot say which routine it "
            "lands in")
    starts = [i for i in range(hit[0])
              if re.match(r"\s*(?:RECURSIVE\s+)?SUBROUTINE\s", src[i])]
    if not starts:
        raise SystemExit("REFUSED: no enclosing SUBROUTINE above the anchor")
    start = starts[-1]
    head = src[start]
    routine = re.match(r"\s*(?:RECURSIVE\s+)?SUBROUTINE\s+(\w+)",
                       head).group(1)
    arglist = re.search(r"\((.*?)\)", head)
    args = [a.strip() for a in arglist.group(1).split(",")] if arglist else []
    in_scope = {a.lower() for a in args}
    in_scope |= _declared("\n".join(src[start:hit[0]]))       # routine locals
    # THE MODULE HEAD IS WHAT PRECEDES ``CONTAINS``, NOT WHAT PRECEDES THE
    # ROUTINE.  Taking src[:start] swept in every EARLIER routine's own
    # declarations -- in this file that is ``tra_zdf``, which declares ``kt``,
    # so the exact defect this check exists for passed it.  Caught by the
    # test that puts ``kt`` back.
    head_only = re.split(r"(?im)^\s*CONTAINS\s*$",
                         "\n".join(src[:start]), maxsplit=1)[0]
    in_scope |= _declared(head_only)                          # own module head
    in_scope |= module_scope_names(
        ppsrc_dir if ppsrc_dir is not None else ppsrc_file.parent)

    blocks = re.findall(block_re, patched, re.S)
    if len(blocks) != 1:
        raise SystemExit(f"REFUSED: found {len(blocks)} instrument blocks in "
                         "the patched source; the check needs exactly one")
    used = {t.lower() for t in re.findall(r"[A-Za-z_]\w*", _strip(blocks[0]))}
    used -= _KEYWORDS
    missing = sorted(t for t in used
                     if t not in in_scope and t not in declared_by_patch)
    if missing:
        raise SystemExit(
            f"REFUSED: the instrument references {missing}, which is NOT in "
            f"scope in {routine} (its dummy arguments are {args}). Fix the "
            "patch -- otherwise the build fails a minute from now with a "
            "message pointing at generated source.")
    return routine


#: The tracer matrix dump's own parameters, so run.sh and the test share them.
ANCHOR = "zwt(ji,1) = zwd(ji,1)"
BLOCK_RE = r"! ---- #1728 WRITE-ONLY matrix dump.*?\n\s*ENDIF"
DECLARED_BY_PATCH = {"ji2", "jk2", "cl_zm", "nzdfmat_kt"}


def main() -> int:
    nemo, copy = (pathlib.Path(a) for a in sys.argv[1:3])
    routine = check(
        nemo / "cfgs/DINO/BLD/ppsrc/nemo/trazdf.f90", ANCHOR,
        (copy / "MY_SRC" / "trazdf.F90").read_text(), BLOCK_RE,
        DECLARED_BY_PATCH)
    print(f"  scope check: every symbol the writer uses is in scope in "
          f"{routine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
