#!/usr/bin/env python3
"""Codemod PR-2: consolidate atmosphere forcing modules under ``atmosphere/forcing/``.

Moves 10 forcing modules out of the flat ``atmosphere/`` top level and out of
``atmosphere/dynamics/`` into ``atmosphere/forcing/{idealized,scm}/`` (and the
``forcing/`` root), and rewrites every import site across the repo. See
docs/production_reorg.md Part B.

Two disjoint source anchors, so the rewrites never collide:
  * TOP movers live at ``legoesm.atmosphere.<mod>`` (8 modules).
  * DYN movers live at ``legoesm.atmosphere.dynamics.<mod>`` (2 forcing files
    that were parked in dynamics/ — moved to ``forcing/`` root here, NOT in the
    PR-1 dynamics bucket reorg).

Safety anchors (mirrors the hardened reorg_dynamics.py):
  * Every dotted rewrite is anchored on the FULL package path and ``\\b``-ended,
    longest-name-first, so ``scm`` never shadows ``scm_forcing`` and ocean/other
    subpackages are never touched.
  * A ``from legoesm.atmosphere.<mod> import`` (form B) contains the dotted
    substring, so the single dotted rule rewrites it too.
  * Idempotent: after a move the new path (``...forcing.idealized.held_suarez``)
    no longer has ``legoesm.atmosphere.held_suarez`` at a matchable position.
  * Slash-path string literals (``atmosphere/<mod>.py`` in budget dicts /
    legoesm_source_path / AST-guard readers) are rewritten too.
  * The verify gate REPORTS (does not auto-fix) any surviving flat reference,
    multi-line grouped ``from <pkg> import ( <mover> )``, or module-form
    ``from legoesm.atmosphere import <mover>`` the single-line rewriter can't
    safely split.

Default is a DRY RUN. Pass --apply to move files (git mv) and edit imports.

    python scripts/reorg/reorg_forcing.py --self-test    # regex checks
    python scripts/reorg/reorg_forcing.py                # dry run report
    python scripts/reorg/reorg_forcing.py --apply        # do it
    python scripts/reorg/reorg_forcing.py --verify       # gate: 0 stragglers
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ATM = "legoesm.atmosphere"

# --- TOP movers: source ``legoesm.atmosphere.<mod>`` -> target subpackage -----
# value = dotted subpackage under legoesm.atmosphere (also the slash dir).
TOP: dict[str, str] = {
    "kessler_forcing": "forcing.idealized",
    "large_scale_forcing": "forcing.idealized",
    "column_forcing": "forcing.idealized",
    "held_suarez": "forcing.idealized",
    "scm": "forcing.scm",
    "scm_forcing": "forcing.scm",
    "dephy_scm": "forcing.scm",
    "sam_case_forcing": "forcing",
}
# --- DYN movers: source ``legoesm.atmosphere.dynamics.<mod>`` -> forcing/ ------
DYN: dict[str, str] = {
    "plane_large_scale_forcing": "forcing",
    "column_large_scale_extract": "forcing",
}

# Physical destination dir (relative to the atmosphere package) per module, and
# the source file location. Buckets whose __init__.py must exist.
_ATM_REL = Path("packages/atmosphere/legoesm/atmosphere")
BUCKET_DIRS = ["forcing", "forcing/idealized", "forcing/scm"]

SKIP_DIRS = {"__pycache__", ".git", ".claude", "build", "dist",
             "node_modules", ".mypy_cache", ".ruff_cache"}


def _skip(p: Path) -> bool:
    parts = set(p.parts)
    return bool(SKIP_DIRS & parts) or any(
        part == ".venv" or part.startswith(".venv-") for part in parts
    )


def _alt(mods) -> str:
    return "|".join(re.escape(m) for m in sorted(mods, key=len, reverse=True))


_ESC_ATM = re.escape(ATM)
_ESC_DYN = re.escape(f"{ATM}.dynamics")
_TOP_ALT = _alt(TOP)
_DYN_ALT = _alt(DYN)

# DYN must be tried before TOP would ever see these names, but the mover sets
# are disjoint and the DYN anchor includes ``.dynamics.``, so order is safe.
RE_DYN_DOTTED = re.compile(rf"\b{_ESC_DYN}\.({_DYN_ALT})\b")
RE_TOP_DOTTED = re.compile(rf"\b{_ESC_ATM}\.({_TOP_ALT})\b")
# Module-form ``from <pkg> import <mover>`` (single name on the line).
RE_TOP_FROM = re.compile(rf"\bfrom {_ESC_ATM} import ({_TOP_ALT})\b")
RE_DYN_FROM = re.compile(rf"\bfrom {_ESC_DYN} import ({_DYN_ALT})\b")
# Slash-path literals. TOP: ``atmosphere/<mod>.py``; DYN: ``atmosphere/dynamics/<mod>.py``.
RE_TOP_PATH = re.compile(rf"(?<!dynamics/)atmosphere/({_TOP_ALT})\.py")
RE_DYN_PATH = re.compile(rf"atmosphere/dynamics/({_DYN_ALT})\.py")
# Verify-only: grouped ``from <pkg> import ( ... )`` the single-line rules miss.
RE_TOP_GROUP = re.compile(rf"from {_ESC_ATM} import \(([^)]*)\)", re.S)
RE_DYN_GROUP = re.compile(rf"from {_ESC_DYN} import \(([^)]*)\)", re.S)


def _rewrite(text: str) -> tuple[str, int]:
    n = 0

    def dyn_dotted(m):
        nonlocal n
        n += 1
        mod = m.group(1)
        return f"{ATM}.{DYN[mod]}.{mod}"

    def top_dotted(m):
        nonlocal n
        n += 1
        mod = m.group(1)
        return f"{ATM}.{TOP[mod]}.{mod}"

    def dyn_from(m):
        nonlocal n
        n += 1
        mod = m.group(1)
        return f"from {ATM}.{DYN[mod]} import {mod}"

    def top_from(m):
        nonlocal n
        n += 1
        mod = m.group(1)
        return f"from {ATM}.{TOP[mod]} import {mod}"

    def dyn_path(m):
        nonlocal n
        n += 1
        mod = m.group(1)
        return f"atmosphere/{DYN[mod].replace('.', '/')}/{mod}.py"

    def top_path(m):
        nonlocal n
        n += 1
        mod = m.group(1)
        return f"atmosphere/{TOP[mod].replace('.', '/')}/{mod}.py"

    text = RE_DYN_DOTTED.sub(dyn_dotted, text)
    text = RE_TOP_DOTTED.sub(top_dotted, text)
    text = RE_DYN_FROM.sub(dyn_from, text)
    text = RE_TOP_FROM.sub(top_from, text)
    text = RE_DYN_PATH.sub(dyn_path, text)
    text = RE_TOP_PATH.sub(top_path, text)
    return text, n


def _iter_py(repo: Path, self_path: Path):
    for p in repo.rglob("*.py"):
        if p == self_path or _skip(p):
            continue
        yield p


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True)


def do_moves(repo: Path, apply: bool) -> None:
    atm = repo / _ATM_REL
    if not atm.is_dir():
        sys.exit(f"atmosphere dir not found: {atm}")
    for b in BUCKET_DIRS:
        bdir = atm / b
        init = bdir / "__init__.py"
        if apply:
            bdir.mkdir(parents=True, exist_ok=True)
            if not init.exists():
                init.write_text(
                    f'"""{b} forcing modules (see docs/production_reorg.md)."""\n'
                )
                _git(repo, "add", str(init.relative_to(repo)))
        else:
            print(f"  mkdir  {bdir.relative_to(repo)}/  + __init__.py")
    plan = [(m, atm / f"{m}.py", atm / TOP[m].replace('.', '/') / f"{m}.py") for m in TOP]
    plan += [(m, atm / "dynamics" / f"{m}.py", atm / DYN[m].replace('.', '/') / f"{m}.py")
             for m in DYN]
    for mod, src, dst in sorted(plan, key=lambda t: str(t[2])):
        if not src.exists():
            print(f"  !! missing (already moved?): {src.relative_to(repo)}")
            continue
        if apply:
            _git(repo, "mv", str(src.relative_to(repo)), str(dst.relative_to(repo)))
        else:
            print(f"  git mv {src.relative_to(atm)} -> {dst.relative_to(atm)}")


def do_rewrite(repo: Path, apply: bool) -> None:
    self_path = Path(__file__).resolve()
    files_changed = edits = 0
    for p in _iter_py(repo, self_path):
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new, n = _rewrite(text)
        if n:
            files_changed += 1
            edits += n
            if apply:
                p.write_text(new, encoding="utf-8")
    verb = "rewrote" if apply else "would rewrite"
    print(f"  {verb} {edits} import(s) across {files_changed} file(s)")


def _grouped_movers(text: str, rx: re.Pattern, movers: set[str]):
    for m in rx.finditer(text):
        names = {n.strip().split(" as ")[0].strip()
                 for n in m.group(1).replace("\n", " ").split(",")}
        yield from sorted(names & movers)


def do_verify(repo: Path) -> int:
    self_path = Path(__file__).resolve()
    stale = 0
    for p in _iter_py(repo, self_path):
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = p.relative_to(repo)
        for rx, kind in ((RE_DYN_DOTTED, "dyn-dotted"), (RE_TOP_DOTTED, "top-dotted"),
                         (RE_DYN_FROM, "dyn-from"), (RE_TOP_FROM, "top-from"),
                         (RE_DYN_PATH, "dyn-path"), (RE_TOP_PATH, "top-path")):
            for m in rx.finditer(text):
                print(f"  STALE {kind:10} {rel}: {m.group(0)}")
                stale += 1
        for mod in _grouped_movers(text, RE_TOP_GROUP, set(TOP)):
            print(f"  STALE group      {rel}: from {ATM} import (... {mod} ...) -> .{TOP[mod]}")
            stale += 1
        for mod in _grouped_movers(text, RE_DYN_GROUP, set(DYN)):
            print(f"  STALE group      {rel}: from {ATM}.dynamics import (... {mod} ...) -> .{DYN[mod]}")
            stale += 1
    print(f"  {stale} straggler(s)")
    return stale


def self_test() -> None:
    cases = [
        # TOP dotted -> idealized
        (f"import {ATM}.held_suarez as hs",
         f"import {ATM}.forcing.idealized.held_suarez as hs"),
        # TOP form B (contains dotted substring)
        (f"from {ATM}.scm import make_scm",
         f"from {ATM}.forcing.scm.scm import make_scm"),
        # scm must NOT shadow scm_forcing
        (f"from {ATM}.scm_forcing import X",
         f"from {ATM}.forcing.scm.scm_forcing import X"),
        # TOP -> forcing root
        (f"import {ATM}.sam_case_forcing",
         f"import {ATM}.forcing.sam_case_forcing"),
        # DYN dotted -> forcing root
        (f"from {ATM}.dynamics.plane_large_scale_forcing import lsf",
         f"from {ATM}.forcing.plane_large_scale_forcing import lsf"),
        # TOP module-form
        (f"from {ATM} import held_suarez",
         f"from {ATM}.forcing.idealized import held_suarez"),
        # slash path TOP
        ('"packages/atmosphere/legoesm/atmosphere/held_suarez.py"',
         '"packages/atmosphere/legoesm/atmosphere/forcing/idealized/held_suarez.py"'),
        # slash path TOP root
        ("atmosphere/sam_case_forcing.py",
         "atmosphere/forcing/sam_case_forcing.py"),
        # slash path DYN
        ("atmosphere/dynamics/plane_large_scale_forcing.py",
         "atmosphere/forcing/plane_large_scale_forcing.py"),
        # a non-mover atmosphere submodule: untouched
        (f"from {ATM}.physics.microphysics import kessler",
         f"from {ATM}.physics.microphysics import kessler"),
        # ocean: untouched
        ("from legoesm.ocean.dynamics.foo import X",
         "from legoesm.ocean.dynamics.foo import X"),
        # idempotent: already-moved dotted
        (f"import {ATM}.forcing.idealized.held_suarez",
         f"import {ATM}.forcing.idealized.held_suarez"),
        # idempotent: already-moved slash
        ("atmosphere/forcing/idealized/held_suarez.py",
         "atmosphere/forcing/idealized/held_suarez.py"),
    ]
    ok = True
    for src, want in cases:
        got, _ = _rewrite(src)
        flag = "ok " if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  [{flag}] {src!r}\n         -> {got!r}")
    assert ok, "self-test failed"
    print("  all self-tests passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--apply", action="store_true", help="move files + edit imports")
    ap.add_argument("--verify", action="store_true", help="gate only: report stragglers")
    ap.add_argument("--self-test", action="store_true", help="regex unit checks, no repo")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    repo = Path(args.repo).resolve()
    if not (repo / _ATM_REL).is_dir():
        sys.exit(f"not a legoESM repo: {repo}")

    if args.verify:
        sys.exit(1 if do_verify(repo) else 0)

    mode = "APPLY" if args.apply else "DRY RUN (use --apply to execute)"
    print(f"== reorg atmosphere forcing/ [{mode}] ==")
    print(f"repo: {repo}")
    print(f"moving {len(TOP) + len(DYN)} modules into forcing/{{idealized,scm,.}}")
    print("-- moves --")
    do_moves(repo, args.apply)
    print("-- import rewrite --")
    do_rewrite(repo, args.apply)
    if args.apply:
        print("-- verify --")
        n = do_verify(repo)
        if n:
            sys.exit(f"FAILED: {n} stale flat path(s) remain")
        print("OK. Next: run pytest, then smoke-import:")
        print(f"  python -c \"import {ATM}.forcing, {ATM}.forcing.idealized, "
              f"{ATM}.forcing.scm\"")


if __name__ == "__main__":
    main()
