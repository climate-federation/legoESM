#!/usr/bin/env python
"""GENERATE ``docs/ocean/fidelity/dino_step_wiring_generated.md``.

WHY THIS EXISTS
---------------
``docs/ocean/fidelity/dino_wiring_diagram.md`` answers the right question --
"which NEMO routines run, in what order, each timestep, and what does legoESM
do at each" -- but it is HAND-MAINTAINED.  A hand-written map of a moving
target goes stale silently, and by 2026-08 it had: it still says legoESM has
no leapfrog/Asselin filter (leapfrog is the shipped card's integrator), still
says the Prandtl number is unity (``prandtl_mode="nemo_ri"`` is realized), and
still marks several rows UNTRACED that have since been measured.

Nothing in that document FAILS when it drifts, so nothing made anyone fix it.
This script derives the same map MECHANICALLY, and
``tests/ocean/unit/test_step_wiring_generated.py`` asserts the committed doc
is byte-identical to a fresh regeneration -- so drift in the Fortran, in the
gate, or in the parse turns red in CI instead of quietly misinforming.

WHAT IS DERIVED FROM WHERE (every column has a machine source; see the
"provenance of each column" block emitted into the doc itself)
------------------------------------------------------------------
1. THE CALL CHAIN -- parsed from the oracle's own Fortran,
   ``cfgs/DINO/MY_SRC/stpmlf.F90``.  DINO ships its own MY_SRC override of
   the time-stepping routine and MY_SRC wins over ``src/OCE`` in NEMO's
   build.  ``cfgs/DINO/cpp_DINO.fcm`` is ``key_qco key_vco_3d`` -- ``key_RK3``
   is NOT defined, so the file's ``#if ! defined key_RK3`` /
   ``# if defined key_qco`` branch is the one that compiles and ``stprk3*`` is
   dead code.  Every ``CALL`` in the ``stp_MLF`` body is extracted in source
   order with its line number and argument list.
2. LIVE / DEAD -- resolved two ways.  cpp branches are evaluated against the
   fppkeys actually in ``cpp_DINO.fcm``.  ``IF( ln_* )`` guards (both
   same-line and enclosing ``IF(...) THEN`` blocks) are resolved against the
   RESOLVED namelist NEMO printed for the run, ``RUN_GDB/ocean.output`` --
   not against the namelist files, which do not show what the defaults
   resolved to.  A guard this script cannot resolve is reported
   ``UNRESOLVED``, never guessed.
3. DISPATCH DESCENT (``zdf_phy`` -> ``zdf_tke``/``zdf_evd``/``zdf_drg``, ...)
   -- joined from the committed, human-reviewed ``stpmlf_call_coverage.py``,
   whose ``CALLS`` list already resolves each wrapper one level to the
   concrete routine DINO runs and quotes the deciding namelist line.
4. MEASURED STATUS -- IMPORTED from ``fidelity_bar_gate.py`` (its
   ``MEASUREMENTS``/``PER_ELEMENT`` dicts and its own ``classify()``).  No
   number is transcribed by hand and no bar constant is read or re-derived
   here; this script is strictly read-only on the gate.
5. PROVENANCE -- ``fidelity_bar_gate.PROVENANCE_SCRIPT`` joined against
   ``git ls-files``.  A row whose cited probe is not tracked in git is
   flagged ``UNREPRODUCIBLE`` so nobody cites it as evidence.
6. legoESM COUNTERPART + SEAM -- joined from the ``## 1. The step skeleton``
   table in ``docs/ocean/fidelity/nemo_mlf_step_transcription_spec.md``
   (its "lego kernel", "levels in->out" and "injection point" columns).

HONESTY RULES THIS SCRIPT FOLLOWS
---------------------------------
* Every cell is machine-derived or literally ``UNMAPPED``/``UNRESOLVED``/
  ``UNMEASURED``.  There is no hand-written status text anywhere below --
  the only prose is this docstring and the column-provenance block.
* Joins are by (routine name, ordinal-among-same-name), NOT by line number.
  The line numbers recorded in ``stpmlf_call_coverage.py`` and in the
  transcription spec are ALREADY STALE against the current Fortran (e.g.
  ``tra_npc`` is recorded at 436 and is really at 446, ``mlf_baro_corr`` at
  457 and is really at 467) because the file grew when the #1226 debug dumps
  were added.  Joining on those numbers would silently mis-attribute rows.
  Every count mismatch between the parse and a join source is reported in the
  DRIFT section rather than being papered over.

Usage:
  gen_step_wiring.py                 -- write the doc, print a summary
  gen_step_wiring.py --stdout        -- print the doc, write nothing
  gen_step_wiring.py --check         -- exit 1 if the committed doc is stale
  gen_step_wiring.py --self-test     -- prove the generator is non-vacuous
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))

# The NEMO oracle checkout lives outside the repo.  Overridable so the doc is
# regenerable — and therefore CHECKABLE — on a machine whose checkout is
# elsewhere.  Absolute paths are deliberately kept OUT of the emitted header
# (only the oracle-RELATIVE paths are written), or the byte-comparison gate
# could never pass for anyone but the author, which would defeat its purpose.
ORACLE = os.environ.get(
    "DINO_ORACLE_ROOT",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO")
STPMLF = os.path.join(ORACLE, "MY_SRC/stpmlf.F90")
CPP_FCM = os.path.join(ORACLE, "cpp_DINO.fcm")
OCEAN_OUTPUT = os.path.join(ORACLE, "RUN_GDB/ocean.output")

SPEC_MD = os.path.join(REPO, "docs/ocean/fidelity/"
                              "nemo_mlf_step_transcription_spec.md")
OUT_MD = os.path.join(REPO, "docs/ocean/fidelity/"
                             "dino_step_wiring_generated.md")

# Time-level tokens NEMO passes as actual arguments.  Which level a routine is
# handed is the whole substance of a leapfrog wiring diagram (oracle-fidelity
# skill Rule 1d), so they are extracted rather than summarised.
LEVELS = ("Nbb", "Nnn", "Naa", "Nrhs")

UNMAPPED = "UNMAPPED"
UNRESOLVED = "UNRESOLVED"

# Fortran logicals NEMO derives from a cpp key at COMPILE time, so they ARE
# resolvable from cpp_DINO.fcm even though `ocean.output` never prints them.
# Each entry is READ from the oracle source; do not add one without a citation.
# Without `lk_linssh` here, the three `dom_qco_r3c` calls (the z-star thickness
# ratios -- the core of #1226) were filed UNRESOLVED, and the doc told the
# reader their liveness was "genuinely unknown" when it is pinned by the build.
_KEY_LOGICALS: dict[str, tuple[str, str]] = {
    # src/OCE/DOM/dom_oce.F90:130-134
    #   #if defined key_linssh / lk_linssh = .TRUE. / #else / .FALSE. / #endif
    "lk_linssh": ("key_linssh", "src/OCE/DOM/dom_oce.F90:130-134"),
}


# ---------------------------------------------------------------------------
# 1. Parse the oracle's Fortran.
# ---------------------------------------------------------------------------
@dataclass
class ParsedCall:
    line: int
    name: str
    args: str
    levels: tuple[str, ...]
    cpp: tuple[str, ...]          # enclosing cpp conditions, outermost first
    guards: tuple[str, ...]       # enclosing/inline IF conditions
    live: str                     # LIVE | DEAD | UNRESOLVED
    live_reason: str


_CALL_RE = re.compile(r"\bCALL\s+([A-Za-z_]\w*)\s*(\(.*)?$", re.I)
_IF_THEN_RE = re.compile(r"^\s*IF\s*\((.*)\)\s*THEN\s*$", re.I)
_ELSE_RE = re.compile(r"^\s*ELSE\s*$", re.I)
_ELSEIF_RE = re.compile(r"^\s*ELSE\s*IF\s*\((.*)\)\s*THEN\s*$", re.I)
_ENDIF_RE = re.compile(r"^\s*END\s?IF\s*$", re.I)
_INLINE_IF_RE = re.compile(r"^\s*IF\s*\((.*?)\)\s*(?=CALL\b|&)", re.I)

# An ELSE branch runs when its IF condition is FALSE.  The branch is tagged
# rather than rewritten so the resolver can flip the sense: without this,
# `ldf_slp` (which lives in the ELSE of `IF(ln_traldf_triad)`, and DOES run --
# DINO sets ln_traldf_triad=F) was reported DEAD.  There are no ELSE IF
# branches in this stp_MLF body (checked: 0), so a single flip is exact here;
# an ELSE IF would need the preceding conditions too and is rejected below.
_ELSE_MARK = "ELSE-OF: "


def _strip_comment(raw: str) -> str:
    """Drop a trailing Fortran ``!`` comment.  Quotes are respected so a
    ``'...!...'`` string literal is not truncated."""
    out, quote = [], None
    for ch in raw:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            out.append(ch)
        elif ch == "!":
            break
        else:
            out.append(ch)
    return "".join(out)


def _logical_lines(lines: list[str], start: int, end: int):
    """Yield ``(first_lineno, joined_code, members)`` over ``lines[start-1:end]``,
    joining Fortran ``&`` continuations so an ``IF(...)`` split across lines is
    seen together with the ``CALL`` it guards.

    ``members`` is the list of ``(physical_lineno, text)`` the logical line was
    built from.  It exists so a CALL can be reported at the line the ``CALL``
    TOKEN is on, not at the line the ``IF(...) &`` head is on -- see
    :func:`_call_lineno`.
    """
    buf, first, members = "", None, []
    for no in range(start, end + 1):
        raw = lines[no - 1]
        if raw.lstrip().startswith("#"):
            # Preprocessor line.  It must NOT be comment-stripped: `!` is
            # cpp NEGATION here, and stripping it turns `#if ! defined
            # key_xios` into a bare `#if` with an empty condition.
            if buf:
                yield first, buf, members
                buf, first, members = "", None, []
            yield no, raw.strip(), [(no, raw.strip())]
            continue
        code = _strip_comment(raw)
        if not code.strip():
            if buf:
                yield first, buf, members
                buf, first, members = "", None, []
            continue
        stripped = code.strip()
        if first is None:
            first = no
        members.append((no, stripped))
        cont = stripped.endswith("&")
        # A leading '&' continues the PREVIOUS logical line.
        buf = (buf + " " + stripped.lstrip("&").rstrip("&")) if buf \
            else stripped.rstrip("&")
        if not cont:
            yield first, buf, members
            buf, first, members = "", None, []
    if buf:
        yield first, buf, members


def _call_lineno(name: str, members: list[tuple[int, str]], fallback: int) -> int:
    """The PHYSICAL line the ``CALL <name>`` token sits on.

    A guard and its call are one logical line but often two physical ones::

        IF( ln_dynspg_ts ) &                       ! 314
           &            CALL wzv( ... )            ! 315  <- the CALL

    Reporting the logical line's first number here made the generator claim
    ``stpmlf_call_coverage.py`` was stale at 8 sites where the HAND-MAINTAINED
    table was right and this parser was off by one.  A mechanical tool telling
    a human their correct record is wrong is worse than no tool, so the CALL
    line is located exactly.
    """
    for no, text in members:
        if re.search(rf"\bCALL\s+{re.escape(name)}\b", text, re.I):
            return no
    return fallback


def parse_stpmlf(path: str) -> list[ParsedCall]:
    """Extract every CALL in the ``stp_MLF`` body, in source order, with the
    cpp and namelist conditions that gate it."""
    lines = open(path).read().split("\n")
    start = end = None
    for i, line in enumerate(lines, 1):
        if start is None and re.match(r"\s*SUBROUTINE stp_MLF", line):
            start = i
        if start is not None and re.match(r"\s*END SUBROUTINE stp_MLF", line):
            end = i
            break
    if start is None or end is None:
        raise SystemExit(f"could not locate the stp_MLF body in {path} -- "
                         "the oracle source changed shape, refusing to guess")

    cpp_keys = read_cpp_keys(CPP_FCM)
    switches = read_ocean_output(OCEAN_OUTPUT)

    calls: list[ParsedCall] = []
    # Walk logical (continuation-joined) lines, replaying cpp and IF-block
    # nesting so every CALL carries the stack that gates it.
    cpp_stack: list[tuple[str, bool]] = []
    if_stack: list[str] = []
    for no, code, members in _logical_lines(lines, start, end):
        stripped = code.strip()
        m = re.match(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$",
                     stripped, re.I)
        if m:
            kind, rest = m.group(1).lower(), m.group(2).strip()
            if kind in ("if", "ifdef", "ifndef"):
                cond = rest if kind == "if" else (
                    f"defined {rest}" if kind == "ifdef"
                    else f"! defined {rest}")
                cpp_stack.append((cond, _eval_cpp(cond, cpp_keys)))
            elif kind == "elif" and cpp_stack:
                cpp_stack[-1] = (rest, _eval_cpp(rest, cpp_keys))
            elif kind == "else" and cpp_stack:
                cond, taken = cpp_stack[-1]
                cpp_stack[-1] = (f"! ({cond})", not taken)
            elif kind == "endif" and cpp_stack:
                cpp_stack.pop()
            continue

        cm = _CALL_RE.search(stripped)
        if cm is None:
            # `IF(x) CALL y` opens no block, so block tracking is only
            # attempted when the line is not a CALL.  The end-of-body balance
            # check below is what catches any form this misses.
            if _IF_THEN_RE.match(stripped):
                if_stack.append(_IF_THEN_RE.match(stripped).group(1).strip())
            elif _ELSEIF_RE.match(stripped):
                if if_stack:
                    if_stack[-1] = ("ELSE-IF " +
                                    _ELSEIF_RE.match(stripped).group(1).strip())
            elif _ELSE_RE.match(stripped):
                if if_stack:
                    if_stack[-1] = _ELSE_MARK + if_stack[-1]
            elif _ENDIF_RE.match(stripped):
                if if_stack:
                    if_stack.pop()
            continue

        args = (cm.group(2) or "").strip()
        inline = _INLINE_IF_RE.match(stripped)
        guards = tuple(if_stack)
        if inline:
            guards = guards + (inline.group(1).strip(),)
        cppconds = tuple(c for c, _ in cpp_stack)
        cpp_dead = [c for c, taken in cpp_stack if not taken]
        live, reason = _resolve_live(cpp_dead, guards, switches, cpp_keys)
        calls.append(ParsedCall(
            line=_call_lineno(cm.group(1), members, no), name=cm.group(1),
            args=args,
            levels=tuple(lv for lv in LEVELS
                         if re.search(rf"\b{lv}\b", args)),
            cpp=cppconds, guards=guards, live=live, live_reason=reason))
    # Fail closed on an unbalanced walk.  An `#if`/`IF` nesting form this
    # parser mishandles would silently mis-gate EVERY subsequent call, which
    # is exactly the kind of confident-wrong output this file exists to avoid.
    if if_stack or cpp_stack:
        raise SystemExit(
            f"unbalanced nesting at END SUBROUTINE stp_MLF: "
            f"IF stack={if_stack}, cpp stack={[c for c, _ in cpp_stack]} -- "
            "the guard stack is corrupt, refusing to emit a wiring table")
    return calls


def _eval_cpp(cond: str, keys: set[str]) -> bool:
    """Evaluate a ``#if`` condition made of ``defined key_x``, ``!``, ``&&``,
    ``||`` against the fppkeys DINO actually builds with.  Anything outside
    that grammar raises rather than being assumed true."""
    expr = cond.strip()
    expr = re.sub(r"defined\s*\(\s*(\w+)\s*\)", r"defined \1", expr)
    expr = re.sub(r"defined\s+(\w+)",
                  lambda m: "True" if m.group(1) in keys else "False", expr)
    expr = expr.replace("&&", " and ").replace("||", " or ")
    expr = re.sub(r"!(?!=)", " not ", expr)
    if not re.fullmatch(r"[\s()TruealsFnotdr]*", expr):
        raise SystemExit(f"cpp condition {cond!r} uses grammar this parser "
                         "does not implement -- refusing to guess its value")
    return bool(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307


def _resolve_live(cpp_dead: list[str], guards: tuple[str, ...],
                  switches: dict[str, bool],
                  cpp_keys: set[str]) -> tuple[str, str]:
    """LIVE / DEAD / UNRESOLVED for one CALL, plus the deciding switch.

    A guard is treated as a CONJUNCTION and split on ``.AND.``.  Each conjunct
    must be a bare (optionally ``.NOT.``-ed) logical name this script can
    resolve -- an ``ln_*`` printed in ``ocean.output`` or a cpp-derived
    ``lk_*`` from :data:`_KEY_LOGICALS`.  Anything else (an arithmetic test
    like ``kstp == nit000``, an internal ``l_*``, or a nested ``.OR.`` group)
    makes that conjunct UNRESOLVABLE.

    The verdict lattice, in order:

    * a compiled-out cpp branch      -> DEAD (whatever the guards say)
    * any RESOLVED conjunct FALSE    -> DEAD (sound: the guard is an AND)
    * any conjunct unresolvable      -> UNRESOLVED
    * all resolved conjuncts TRUE    -> LIVE

    Judging a guard only on the ``ln_*`` names it happens to contain, and
    ignoring the rest of the conjunction, produced FALSE LIVE verdicts (e.g.
    ``IF( ln_zdfosm .AND. lrst_oce )`` would have been called live on
    ``ln_zdfosm`` alone, dropping the restart-step condition entirely).
    """
    if cpp_dead:
        return "DEAD", f"cpp: {'; '.join(cpp_dead)} false for cpp_DINO.fcm"
    reasons, unresolved = [], []
    for g in guards:
        if g.startswith("ELSE-IF "):
            raise SystemExit("ELSE IF branch encountered; resolving it needs "
                             "the preceding conditions too -- refusing to "
                             "guess. Extend _resolve_live.")
        in_else = g.startswith(_ELSE_MARK)
        cond = g[len(_ELSE_MARK):] if in_else else g
        if cond.count("(") != cond.count(")"):
            # `_INLINE_IF_RE` stops at the FIRST `)`, so a guard containing a
            # function call (`IF( Agrif_NbStepint() == 0 .AND. ... )`) is
            # captured truncated.  Every such guard in this file sits in a
            # cpp-dead branch, which short-circuits above, so this never fires
            # today -- but a truncated condition must never be read as if it
            # were the whole one.
            unresolved.append(f"{g} [truncated capture, unbalanced parens]")
            continue
        conjuncts = re.split(r"\.AND\.", cond, flags=re.I)
        if in_else and len(conjuncts) > 1:
            # De Morgan: the ELSE of `IF(a .AND. b)` runs on NOT(a AND b),
            # which is NOT(a) OR NOT(b) -- an OR, not the AND fold below.
            # Negating each conjunct inside the fold would compute
            # `NOT a AND NOT b` and report DEAD for a branch that RUNS (e.g.
            # a=T, b=F), with a reason string identical to the correct DEAD
            # case, so right and wrong would be indistinguishable.  The one
            # ELSE in this file is single-conjunct, so this never fires today.
            unresolved.append(f"ELSE of a multi-conjunct guard ({cond})")
            continue
        for conjunct in conjuncts:
            atom = conjunct.strip()
            neg = False
            m = re.match(r"^\.NOT\.\s*(.+)$", atom, re.I)
            if m:
                neg, atom = True, m.group(1).strip()
            atom = atom.strip("() ")
            if not re.fullmatch(r"\w+", atom):
                unresolved.append(conjunct.strip())    # arithmetic / .OR. / etc
                continue
            if atom in switches:
                raw, src = switches[atom], "ocean.output"
            elif atom in _KEY_LOGICALS:
                key, src = _KEY_LOGICALS[atom]
                raw = key in cpp_keys
            else:
                unresolved.append(atom)                # internal l_* / lk_*
                continue
            val = (not raw) if neg else raw
            if in_else:
                val = not val        # the ELSE branch runs on the negation
            # Parenthesise under .NOT. so the printed value is unambiguously
            # the ATOM's, not the branch's: `.NOT.(lk_linssh=F)`, not
            # `.NOT.lk_linssh=F` which reads as "the negation is false".
            shown = f"{atom}={'T' if raw else 'F'}"
            tag = (f"{'ELSE of ' if in_else else ''}"
                   + (f".NOT.({shown})" if neg else shown))
            if not val:
                return "DEAD", f"{tag} ({src})"
            reasons.append(tag)
    if unresolved:
        return UNRESOLVED, ("guard not resolvable: " +
                            "; ".join(sorted(set(unresolved))))
    return "LIVE", ("; ".join(reasons) if reasons else "unconditional")


def read_cpp_keys(path: str) -> set[str]:
    keys = set()
    for line in open(path):
        if "fppkeys" in line:
            keys.update(re.findall(r"\bkey_\w+", line))
    if not keys:
        raise SystemExit(f"no fppkeys found in {path}")
    return keys


def read_ocean_output(path: str) -> dict[str, bool]:
    """The RESOLVED namelist: what NEMO printed for the run.  Namelist FILES
    do not show what unset entries defaulted to, which is why this reads the
    output instead."""
    switches: dict[str, bool] = {}
    for line in open(path):
        for name, val in re.findall(r"\b(ln_\w+)\s*=\s*([TF])\b", line):
            val_b = val == "T"
            if switches.get(name, val_b) != val_b:
                raise SystemExit(
                    f"{name} is printed both T and F in {path}; first-print-"
                    "wins would pick one arbitrarily -- refusing to guess")
            switches[name] = val_b
    if not switches:
        raise SystemExit(f"no ln_* switches parsed from {path}")
    return switches


# ---------------------------------------------------------------------------
# 2. Join sources.
# ---------------------------------------------------------------------------
def load_coverage() -> dict[str, list]:
    """``stpmlf_call_coverage.CALLS`` bucketed by the wrapper name its
    ``routine`` field starts with, preserving order.  That field is where the
    one-level dispatch descent already lives (``"zdf_phy -> zdf_tke"``)."""
    sys.path.insert(0, HERE)
    import stpmlf_call_coverage as cov  # noqa: E402
    buckets: dict[str, list] = {}
    for entry in cov.CALLS:
        # NB: the split class must not be written `[\s(->]` -- that is a
        # character RANGE `(`..`>` which swallows digits, silently bucketing
        # `bn2` under `bn`.
        head = re.split(r"[\s(]|->", entry.routine.strip())[0]
        buckets.setdefault(head, []).append(entry)
    return buckets


_SPEC_ROW = re.compile(r"^\|\s*(\d+)\s*\|(.*)\|\s*$")


def load_spec() -> dict[str, list[dict]]:
    """The ``## 1. The step skeleton`` table, bucketed by routine name.

    Columns: ``# | NEMO line | routine | lego kernel | levels in->out |
    injection point | governs``.  The last three are the SEAM: what array
    changes hands, at which time level, and where it lands.

    Only the ``## 1.`` section is scanned -- the spec file holds other numbered
    tables (the phase/effort table, the P3 results tables) and a bare
    whole-file scan silently injected their rows into this join.

    A routine cell may name SEVERAL routines (row 8 is
    ``` `eos`(Nbb, in-situ) + `ldf_slp` ```), so every identifier in the cell
    gets a bucket; indexing only the first left `ldf_slp` -- the isoneutral
    slope call feeding tra_ldf and GM -- with no spec bucket at all.

    But identifiers inside PARENTHESES are references, not routines: row 11 is
    ``` `ssh_nxt` (incl. `div_hor`) ``` and describes the ssh_nxt call only.
    Indexing the parenthetical made the `div_hor` call at line 302 inherit
    ssh_nxt's seam (``Nbb,Nnn -> Naa ssh``) instead of its own row 20 seam
    (``Naa`` / state-commit) -- two rows printing the same spec cell, one of
    them wrong. So parenthesised groups are stripped before the scan.
    """
    buckets: dict[str, list[dict]] = {}
    in_section = False
    for line in open(SPEC_MD):
        if line.startswith("## "):
            in_section = line.startswith("## 1.")
            continue
        if not in_section:
            continue
        m = _SPEC_ROW.match(line.rstrip("\n"))
        if not m:
            continue
        cells = [c.strip() for c in m.group(2).split("|")]
        if len(cells) < 6:
            continue
        nemo_line, routine, lego, levels, injection = cells[:5]
        row = {"spec_row": m.group(1), "spec_line": nemo_line,
               "routine": routine, "lego": lego, "levels": levels,
               "injection": injection,
               "governs": cells[5] if len(cells) > 5 else ""}
        # Backticked identifiers OUTSIDE parentheses.  "(inline)" rows name no
        # routine and so are skipped -- they are real spec entries for inline
        # code, not drift.
        outer = re.sub(r"\([^()]*\)", " ", routine)
        for head in re.findall(r"`([A-Za-z_]\w*)", outer):
            buckets.setdefault(head, []).append(row)
    if not buckets:
        raise SystemExit(f"no step-skeleton rows parsed from {SPEC_MD}")
    return buckets


_QUOTED = re.compile(r'"([^"]+)"')


def gate_rows_in(note: str, measurements: dict) -> list[str]:
    """Gate row names cited in a coverage note.  Coverage writes them in
    double quotes; only names that really exist in MEASUREMENTS are kept, so
    a prose phrase in quotes cannot masquerade as a measured row."""
    return [q for q in _QUOTED.findall(note) if q in measurements]


def tracked_files(directory: str) -> set[str]:
    # No silent empty set: without git every row would read UNREPRODUCIBLE and
    # the provenance ledger would report a fake 0-of-53 with no diagnostic.
    try:
        out = subprocess.run(["git", "-C", REPO, "ls-files",
                              os.path.relpath(directory, REPO)],
                             capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(
            f"`git ls-files` failed ({exc}); provenance would silently read "
            "UNREPRODUCIBLE for every row -- refusing to emit the doc") from exc
    return {os.path.basename(p) if "/" not in
            os.path.relpath(p, os.path.relpath(directory, REPO)) else
            os.path.relpath(p, os.path.relpath(directory, REPO))
            for p in out.stdout.split()}


# ---------------------------------------------------------------------------
# 3. Build the rows.
# ---------------------------------------------------------------------------
@dataclass
class Row:
    call: ParsedCall
    concrete: str
    disposition: str
    lego: str
    seam_levels: str
    seam_injection: str
    statuses: list[tuple[str, str, str, str]] = field(default_factory=list)
    # (gate row name, classify(), corr/ratio, provenance verdict)


def build_rows(calls: list[ParsedCall]):
    sys.path.insert(0, HERE)
    import fidelity_bar_gate as gate  # read-only: numbers + classify()

    cov = load_coverage()
    spec = load_spec()
    tracked = tracked_files(HERE)

    from collections import Counter
    n_sites = Counter(c.name for c in calls)
    # TWO ordinal counters, because the two join sources enumerate different
    # things.  `stpmlf_call_coverage.py` lists EVERY call site including the
    # dead ones (it has a `dom_qco_r3c` entry for the ln_dynspg_exp branch), so
    # it joins on the all-occurrence ordinal.  The spec's §1 skeleton documents
    # only the LIVE chain, so a dead call must not consume a spec ordinal --
    # letting it do so shifted `dom_qco_r3c` at 303 onto spec row 35 and left
    # 362 UNMAPPED, i.e. two rows carrying the wrong seam.
    seen: dict[str, int] = {}
    seen_live: dict[str, int] = {}
    rows: list[Row] = []
    used_cov: set[int] = set()
    used_spec: set[int] = set()
    for call in calls:
        ordinal = seen.get(call.name, 0)
        seen[call.name] = ordinal + 1
        if call.live == "DEAD":
            spec_ordinal = None
        else:
            spec_ordinal = seen_live.get(call.name, 0)
            seen_live[call.name] = spec_ordinal + 1
        cov_entries = cov.get(call.name, [])
        spec_rows = spec.get(call.name, [])

        # A wrapper resolving to several concrete routines (zdf_phy) yields
        # one output row per concrete routine.  Same-name repeats (eos_rab at
        # two time levels) are matched by ordinal.  Counts that do not line up
        # are reported in DRIFT, never silently zipped.
        if (len(cov_entries) > 1 and _is_wrapper(cov_entries)
                and n_sites[call.name] == 1):
            chosen = cov_entries
        elif cov_entries:
            chosen = [cov_entries[ordinal]] if ordinal < len(cov_entries) \
                else []
        else:
            chosen = []

        if not chosen:
            rows.append(_mk_row(call, None, spec_rows, spec_ordinal, gate,
                                tracked, used_spec))
            continue
        for entry in chosen:
            used_cov.add(id(entry))
            rows.append(_mk_row(call, entry, spec_rows, spec_ordinal, gate,
                                tracked, used_spec))
    return rows, cov, spec, used_cov, used_spec, gate


def _is_wrapper(entries) -> bool:
    """True when coverage resolved ONE call site into several concrete
    routines (its ``routine`` strings carry the ``->`` descent marker)."""
    return all("->" in e.routine for e in entries)


def _per_elem(gate, name):
    """PER_ELEMENT value for `name`, unwrapping the (value, note) tuple form."""
    val = gate.PER_ELEMENT.get(name)
    return val[0] if isinstance(val, tuple) else val


def _covers_all_levels(spec_row) -> bool:
    """True when a spec row's routine cell names MORE THAN ONE time level, i.e.
    that single row documents several same-name calls (``(Nbb, Nnn)``)."""
    return sum(lv in spec_row["routine"] for lv in LEVELS) > 1


def _mk_row(call, entry, spec_rows, ordinal, gate, tracked, used_spec=None):
    concrete = entry.routine if entry else UNMAPPED
    disposition = entry.disposition if entry else UNMAPPED

    match = None
    is_wrapper = entry is not None and "->" in entry.routine
    if spec_rows:
        if is_wrapper:
            # Descend by the CONCRETE routine name.  No positional fallback
            # here: every zdf_phy child shares ordinal 0, so falling back to
            # spec_rows[0] would hand `zdf_mxl_turb` the `zdf_drg` row -- a
            # silent mis-attribution, which is the whole failure mode this
            # generator exists to prevent.  No tail match => UNMAPPED.
            # Match on the first concrete identifier after the descent
            # marker, word-bounded.  Coverage writes "->" and the spec writes
            # the Unicode arrow, and the spec drops spaces around "+", so a
            # literal substring test on the whole tail misses; and a
            # non-bounded test would let `zdf_mxl` claim a `zdf_mxl_turb` row.
            tail = entry.routine.split("->")[-1]
            tok = re.search(r"[A-Za-z_]\w*", tail)
            for sr in spec_rows:
                if tok and re.search(rf"\b{tok.group(0)}\b", sr["routine"]):
                    match = sr
                    break
        elif ordinal is None:
            match = None          # dead call: the spec skeleton has no row
        elif ordinal < len(spec_rows):
            match = spec_rows[ordinal]
        elif len(spec_rows) == 1 and _covers_all_levels(spec_rows[0]):
            # ONE spec row that explicitly names several time levels covers
            # every occurrence: rows 2 and 3 are "`eos_rab` (Nbb, Nnn)" and
            # "`bn2` (Nbb, Nnn)", one row deliberately documenting both calls.
            # The condition is read from the spec's own text, not assumed from
            # the row count -- without it the Nnn-level `eos_rab`/`bn2` calls
            # went UNMAPPED and the drift section told the maintainers their
            # correct spec rows were missing.
            match = spec_rows[0]
    if match is not None and used_spec is not None:
        used_spec.add(id(match))

    statuses = []
    if entry is not None:
        for name in gate_rows_in(entry.note, gate.MEASUREMENTS):
            corr, ratio, _note = gate.MEASUREMENTS[name]
            verdict = gate.classify(corr, ratio, _per_elem(gate, name), name)
            numbers = ("UNMEASURED" if corr is None or ratio is None
                       else f"corr {corr!r} / ratio {ratio!r}")
            script = gate.PROVENANCE_SCRIPT.get(name, "")
            if not script:
                prov = "UNREPRODUCIBLE (no probe recorded)"
            elif script in tracked:
                prov = f"`{script}` (tracked)"
            else:
                prov = f"UNREPRODUCIBLE (`{script}` not in git)"
            statuses.append((name, verdict, numbers, prov))

    lego = UNMAPPED
    if match:
        # The spec's lego cells contain "same kernel as row 11"-style
        # cross-references that point at SPEC row numbers.  This table
        # renumbers independently (one row per concrete routine, live rows
        # only), so the spec row id is printed alongside to keep them
        # resolvable instead of silently pointing at the wrong row.
        lego = f"(spec \u00a71 row {match['spec_row']}) {match['lego']}"
    return Row(call=call, concrete=concrete, disposition=disposition,
               lego=lego,
               seam_levels=(match["levels"] if match else UNMAPPED),
               seam_injection=(match["injection"] if match else UNMAPPED),
               statuses=statuses)


# ---------------------------------------------------------------------------
# 4. Render.
# ---------------------------------------------------------------------------
def _cell(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("|", "\\|")).strip() or "—"


def render(rows, cov, spec, used_cov, used_spec, gate) -> str:
    sha = hashlib.sha256(open(STPMLF, "rb").read()).hexdigest()
    commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                            capture_output=True, text=True,
                            check=True).stdout.strip()
    stamp = datetime.datetime.now(datetime.UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    keys = " ".join(sorted(read_cpp_keys(CPP_FCM)))

    out: list[str] = []
    w = out.append
    w("# DINO stp_MLF step wiring — GENERATED, do not edit by hand")
    w("")
    w("<!-- GENERATED by scripts/validate/ocean_fidelity/dino_1226/"
      "gen_step_wiring.py. Edits here are lost on the next run and will FAIL")
    w("     tests/ocean/unit/test_step_wiring_generated.py. Change the "
      "generator, or the sources it reads, instead. -->")
    w("")
    w(f"- generated-at: {stamp}")
    w(f"- legoesm-commit: {commit}")
    w("- oracle-root: `$DINO_ORACLE_ROOT` (defaults to the NEMO 5.0.2 DINO "
      "config dir; paths below are relative to it, so this header is "
      "identical on any machine with the same checkout)")
    w(f"- oracle-source: `{os.path.relpath(STPMLF, ORACLE)}`")
    w(f"- oracle-source-sha256: {sha}")
    w(f"- oracle-cpp-keys: `{keys}` (from "
      f"`{os.path.relpath(CPP_FCM, ORACLE)}`) — `key_RK3` absent, so the "
      "`#if ! defined key_RK3` MLF branch compiles and `stprk3*` is dead")
    w(f"- resolved-namelist: `{os.path.relpath(OCEAN_OUTPUT, ORACLE)}`")
    w(f"- gate: `fidelity_bar_gate.py` ({len(gate.MEASUREMENTS)} rows, "
      "imported read-only)")
    w(f"- spec-table: `{os.path.relpath(SPEC_MD, REPO)}` "
      "(§1 The step skeleton)")
    w(f"- coverage-table: `stpmlf_call_coverage.py` "
      f"({sum(len(v) for v in cov.values())} entries)")
    w("")
    w("This file SUPERSEDES the hand-maintained "
      "`dino_wiring_diagram.md`.")
    w("")
    w("## Provenance of each column")
    w("")
    w("| column | derived from |")
    w("|---|---|")
    w("| NEMO line, CALL, args, time levels | parsed from the oracle Fortran "
      "named above |")
    w("| LIVE/DEAD | cpp keys from `cpp_DINO.fcm`; `ln_*` guards resolved "
      "against `ocean.output` (the RESOLVED namelist, not the namelist "
      "files). Unresolvable guards say `UNRESOLVED`, never assumed live |")
    w("| concrete routine | `stpmlf_call_coverage.py` `CALLS[].routine` "
      "(one-level dispatch descent) |")
    w("| legoESM counterpart, seam | `nemo_mlf_step_transcription_spec.md` "
      "§1 columns *lego kernel* / *levels in→out* / *injection point* |")
    w("| gate row, status, numbers | `fidelity_bar_gate.MEASUREMENTS` + "
      "`PER_ELEMENT` classified by that module's own `classify()` |")
    w("| provenance | `fidelity_bar_gate.PROVENANCE_SCRIPT` joined against "
      "`git ls-files` |")
    w("")
    w("`UNMAPPED` = no join source records a counterpart. `UNRESOLVED` = the "
      "guard is an internal logical (`l_*`, `lk_*`) that `ocean.output` does "
      "not print, so liveness is genuinely unknown to this parser. Neither "
      "is a guess.")
    w("")

    # THREE categories, not two.  A debug dump or an I/O call that this build
    # really executes is LIVE -- it is not a "dead branch", and filing it under
    # that heading is a false statement about the oracle.  Splitting on
    # coverage's own WAIVED disposition keeps the physics chain readable
    # without hiding anything: every parsed call appears in exactly one of the
    # three tables, asserted below.
    chain = [r for r in rows if r.call.live == "LIVE"
             and r.disposition != "WAIVED"]
    waived_live = [r for r in rows if r.call.live == "LIVE"
                   and r.disposition == "WAIVED"]
    not_live = [r for r in rows if r.call.live != "LIVE"]
    assert len(chain) + len(waived_live) + len(not_live) == len(rows)

    w("## 1. Live physics chain, in execution order")
    w("")
    w("Calls this build executes that `stpmlf_call_coverage.py` has NOT "
      "waived. The waived-but-live calls are in §2 and the non-live ones in "
      "§3 — every parsed call appears in exactly one of the three.")
    w("")
    w("| # | line | CALL | levels | gate | concrete routine | legoESM "
      "counterpart | seam (levels in→out) | seam (injection) | status |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(chain, 1):
        w(f"| {i} | {r.call.line} | `{r.call.name}` | "
          f"{_cell(', '.join(r.call.levels))} | {_cell(r.call.live_reason)} | "
          f"{_cell(r.concrete)} | {_cell(r.lego)} | {_cell(r.seam_levels)} | "
          f"{_cell(r.seam_injection)} | {_cell(_status_cell(r))} |")
    w("")

    w("## 2. Live, but waived: #1226 instrumentation and I/O")
    w("")
    w("These DO run in this build. They are separated from the physics chain "
      "only because coverage waives them, never because they are inactive.")
    w("")
    w("| line | CALL | verdict | why | disposition |")
    w("|---|---|---|---|---|")
    for r in waived_live:
        w(f"| {r.call.line} | `{r.call.name}` | {r.call.live} | "
          f"{_cell(r.call.live_reason)} | {_cell(r.disposition)} |")
    w("")

    w("## 3. Not live: dead branches and unresolved guards")
    w("")
    w("| line | CALL | verdict | why | disposition |")
    w("|---|---|---|---|---|")
    for r in not_live:
        w(f"| {r.call.line} | `{r.call.name}` | {r.call.live} | "
          f"{_cell(r.call.live_reason)} | {_cell(r.disposition)} |")
    w("")

    w("## 4. Measured status per gate row reached from this chain")
    w("")
    w("| gate row | reached from | classify() | numbers | provenance |")
    w("|---|---|---|---|---|")
    emitted: dict[str, tuple] = {}
    sites: dict[str, list[str]] = {}
    for r in rows:
        for name, verdict, numbers, prov in r.statuses:
            emitted.setdefault(name, (verdict, numbers, prov))
            if r.call.name not in sites.setdefault(name, []):
                sites[name].append(r.call.name)
    for name, (verdict, numbers, prov) in emitted.items():
        w(f"| {_cell(name)} | "
          f"{', '.join('`' + s + '`' for s in sites[name])} | {verdict} | "
          f"{_cell(numbers)} | {_cell(prov)} |")
    w("")
    unreached = sorted(set(gate.MEASUREMENTS) - set(emitted))
    w(f"{len(emitted)} of {len(gate.MEASUREMENTS)} gate rows are reachable "
      "from a parsed `stp_MLF` call. The rest are not cited by any coverage "
      "note and so cannot be attached to a step:")
    w("")
    for name in unreached:
        corr, ratio, _note = gate.MEASUREMENTS[name]
        w(f"- `{name}` — "
          f"{gate.classify(corr, ratio, _per_elem(gate, name), name)}")
    w("")

    w("## 5. Provenance ledger")
    w("")
    tracked = tracked_files(HERE)
    bad = [(n, s) for n, s in gate.PROVENANCE_SCRIPT.items()
           if not s or s not in tracked]
    w(f"{len(gate.PROVENANCE_SCRIPT) - len(bad)} of "
      f"{len(gate.PROVENANCE_SCRIPT)} gate rows cite a probe that is tracked "
      "in git and can be re-run. The rest are UNREPRODUCIBLE — their numbers "
      "must not be cited as evidence without rebuilding the probe:")
    w("")
    w("| gate row | cited probe | why unreproducible |")
    w("|---|---|---|")
    for name, script in sorted(bad):
        why = ("no probe recorded" if not script
               else "cited file is not tracked in git")
        w(f"| {_cell(name)} | {_cell(script) if script else '—'} | {why} |")
    w("")

    w("## 6. Drift between the parse and the hand-maintained join sources")
    w("")
    w("These are the reasons this document is generated. Each line is a place "
      "where a hand-maintained table no longer lines up with the Fortran.")
    w("")
    drift: list[str] = []
    parsed_names = {r.call.name for r in rows}
    spec_rows_all = {id(sr): sr for srows in spec.values() for sr in srows}
    for head, entries in sorted(cov.items()):
        if head not in parsed_names:
            drift.append(f"`stpmlf_call_coverage.py` records `{head}` "
                         "but no such CALL was parsed from the current "
                         "Fortran")
        for e in entries:
            if id(e) not in used_cov:
                drift.append(f"`stpmlf_call_coverage.py` entry for "
                             f"`{_cell(e.routine)}` (recorded at line "
                             f"{e.line}) was not matched to any parsed CALL")
    for head, srows in sorted(spec.items()):
        if head not in parsed_names:
            drift.append(f"spec §1 records routine `{head}` but no such CALL "
                         "was parsed from the current Fortran")
    for sid, sr in spec_rows_all.items():
        if sid not in used_spec:
            drift.append(f"spec §1 row {sr['spec_row']} "
                         f"({_cell(sr['routine'])}) was not matched to any "
                         "parsed CALL -- its legoESM counterpart and seam "
                         "reach no row of this table")
    for r in rows:
        if r.disposition == UNMAPPED and r.call.live == "LIVE":
            drift.append(f"live CALL `{r.call.name}` at line {r.call.line} "
                         "has NO entry in `stpmlf_call_coverage.py`")
        # A missing legoESM counterpart is only drift for a call coverage has
        # NOT waived.  The #1226 debug dumps and the I/O calls are WAIVED in
        # `stpmlf_call_coverage.py` and legitimately have no lego kernel;
        # listing them here would bury the real gaps in noise.  The filter is
        # coverage's own disposition, not a judgment made here.
        if (r.lego == UNMAPPED and r.call.live == "LIVE"
                and r.disposition not in ("WAIVED", UNMAPPED)):
            drift.append(f"live CALL `{r.call.name}` at line {r.call.line} "
                         f"(concrete: {_cell(r.concrete)}) has NO legoESM "
                         "counterpart in the spec \u00a71 table")
    # Line-number staleness: the join sources record line numbers that have
    # moved.  Reported, and deliberately NOT used as a join key.
    by_name: dict[str, list[int]] = {}
    for r in rows:
        by_name.setdefault(r.call.name, []).append(r.call.line)
    for head, entries in sorted(cov.items()):
        for e in entries:
            if head in by_name and e.line not in by_name[head]:
                drift.append(f"`stpmlf_call_coverage.py` records `{head}` at "
                             f"line {e.line}; the current Fortran has it at "
                             f"{'/'.join(str(x) for x in sorted(set(by_name[head])))}")
    for line in sorted(set(drift)):
        w(f"- {line}")
    if not drift:
        w("- none")
    w("")
    return "\n".join(out) + "\n"


def _status_cell(row: Row) -> str:
    if not row.statuses:
        return f"{row.disposition} / no gate row cited"
    return "; ".join(f"{n}: {v}" for n, v, _num, _p in row.statuses)


# ---------------------------------------------------------------------------
# 5. Entry point.
# ---------------------------------------------------------------------------
VOLATILE = re.compile(r"^- (generated-at|legoesm-commit):.*$", re.M)


def normalise(text: str) -> str:
    """Blank the two intentionally volatile header fields so the CI test can
    compare content.  Everything else -- including the oracle source sha256 --
    is compared exactly."""
    return VOLATILE.sub(lambda m: f"- {m.group(1)}: <normalised>", text)


def generate() -> str:
    calls = parse_stpmlf(STPMLF)
    return render(*build_rows(calls))


def _self_test() -> int:
    """Synthetic-violation check: mutating one parsed row MUST change the
    document.  A generator whose output is insensitive to its input would
    make the byte-identical CI test vacuous."""
    baseline = generate()
    calls = parse_stpmlf(STPMLF)
    live = [c for c in calls if c.live == "LIVE"]
    assert live, "no LIVE calls parsed -- the gating resolver is broken"
    victim = next(c for c in live if c.name == "dyn_vor")
    victim.name = "dyn_vor_MUTATED"
    mutated = render(*build_rows(calls))
    assert normalise(mutated) != normalise(baseline), (
        "mutating a parsed CALL did not change the generated document -- "
        "the byte-identical test would be VACUOUS")
    assert "dyn_vor_MUTATED" in mutated, "mutation did not reach the output"
    # And the parse must be non-trivial in the ways the doc depends on.
    assert any(c.live == "DEAD" for c in calls), "no DEAD branch resolved"
    assert any(c.levels for c in calls), "no time levels extracted"
    print(f"self-test OK: {len(calls)} calls parsed, "
          f"{len(live)} live; mutation is visible in the output")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stdout", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return _self_test()
    text = generate()
    if args.stdout:
        sys.stdout.write(text)
        return 0
    if args.check:
        if not os.path.exists(OUT_MD):
            print(f"MISSING: {OUT_MD}")
            return 1
        if normalise(open(OUT_MD).read()) != normalise(text):
            print(f"STALE: {OUT_MD} differs from a fresh generation -- "
                  "re-run gen_step_wiring.py")
            return 1
        print(f"up to date: {OUT_MD}")
        return 0
    with open(OUT_MD, "w") as fh:
        fh.write(text)
    calls = parse_stpmlf(STPMLF)
    print(f"wrote {OUT_MD}")
    print(f"  {len(calls)} CALLs parsed, "
          f"{sum(c.live == 'LIVE' for c in calls)} LIVE, "
          f"{sum(c.live == 'DEAD' for c in calls)} DEAD, "
          f"{sum(c.live == UNRESOLVED for c in calls)} UNRESOLVED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
