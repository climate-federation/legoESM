#!/usr/bin/env python3
"""Codemod PR-1: reorganize ``atmosphere/dynamics/`` into category subpackages.

Moves 52 dycore modules into ``dynamics/{gcm,les,crm,shared,neural}/`` and
rewrites every import site across the repo.

Safety anchors (see docs/production_reorg.md):
  * Every rewrite is anchored on the FULL ``legoesm.atmosphere.dynamics.``
    path — never bare ``dynamics.`` — so OCEAN's ``dynamics/`` package
    (``legoesm.ocean.dynamics.*``, 260+ sites) is never touched.
  * Only modules in the explicit BUCKETS allowlist move; non-moved modules
    (``latlon_cgrid_operators``, ...) are left alone.
  * The dotted-path rewrite also fixes the string literals in the
    ``dynamics/__init__.py`` lazy-import table, so package-attribute imports
    (``from ...dynamics import CDGridShallowWaterModel``) keep resolving with
    no change.
  * Alternation is sorted longest-first and ``\\b``-anchored, so prefix pairs
    resolve correctly and to different buckets:
    ``tracer_transport`` (shared) vs ``tracer_transport_latlon`` (gcm).

Default is a DRY RUN. Pass --apply to move files (git mv) and edit imports.

    python scripts/reorg/reorg_dynamics.py --self-test         # regex checks
    python scripts/reorg/reorg_dynamics.py                     # dry run report
    python scripts/reorg/reorg_dynamics.py --apply             # do it
    python scripts/reorg/reorg_dynamics.py --verify            # gate: 0 stragglers

The 2 forcing files still in dynamics/ (plane_large_scale_forcing,
column_large_scale_extract) are intentionally NOT moved here — they belong to
the atmosphere/forcing/ consolidation (PR-2), not a dynamics bucket.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# --- module -> bucket (52 movers; the source of truth) ----------------------
BUCKETS: dict[str, str] = {
    # gcm/ — global dycores (lat-lon / cubed-sphere / spectral / MPAS)
    "compressible_euler": "gcm",
    "compressible_euler_cdgrid": "gcm",
    "compressible_euler_latlon_cgrid": "gcm",
    "compressible_euler_mpas": "gcm",
    "primitive_eq_cdgrid": "gcm",
    "primitive_eq_latlon_cgrid": "gcm",
    "primitive_eq_mpas": "gcm",
    "semi_implicit_cdgrid": "gcm",
    "shallow_water_fv3_cdgrid": "gcm",
    "shallow_water_latlon_cgrid": "gcm",
    "shallow_water_mpas": "gcm",
    "shallow_water_nesting": "gcm",
    "spectral_pe": "gcm",
    "spectral_sw": "gcm",
    "spectral_nh": "gcm",
    "sharded_atm_latlon_step": "gcm",
    "tiled_step_adapter": "gcm",
    "tracer_transport_latlon": "gcm",
    "tracer_transport_mpas": "gcm",
    "_fv3_lin_pgf": "gcm",
    "dcmip2025_ic": "gcm",
    # les/ — large-eddy (plane / pseudo-incompressible)
    "spectral_les_plane": "les",
    "spectral_les_moist": "les",
    "spectral_plane": "les",
    "column_les": "les",
    "column_les_diagnosis": "les",
    "les_closure_diagnosis": "les",
    "les_regime": "les",
    "les_vertical_mapping": "les",
    "tke_sgs_plane": "les",
    "pseudo_incompressible_plane": "les",
    "pseudo_incompressible_plane_mpi": "les",
    "pseudo_incompressible_poisson": "les",
    "pseudo_incompressible_poisson_mpi": "les",
    "compressible_euler_plane": "les",
    "compressible_euler_plane_halo": "les",
    "plane_fd_advection": "les",
    "plane_operators": "les",
    "plane_operators_halo": "les",
    # crm/ — cloud-resolving (RCE / SAM)
    "rce_diagnostics": "crm",
    "rce_mpi": "crm",
    "rce_surface_flux": "crm",
    "sam_case_setup": "crm",
    "moist_mass_fixer": "crm",
    # shared/ — grid-agnostic numerics
    "tracer_transport": "shared",
    "flux_form_tracer_transport": "shared",
    "tracer_positivity": "shared",
    "cfl_diagnostic": "shared",
    "mean_wind_filter": "shared",
    # neural/ — learned dycores (surrogates)
    "sfno_pe": "neural",
    "sfno_sw": "neural",
    "ucast_pe": "neural",
}

PKG = "legoesm.atmosphere.dynamics"
DYN_REL = Path("packages/atmosphere/legoesm/atmosphere/dynamics")
# NB: skip ALL virtualenvs (.venv*) and .claude agent-worktree copies — the
# codemod must only touch the live source tree, never installed pkgs or the
# isolated worktree checkouts under .claude/worktrees/ (each its own git tree).
SKIP_DIRS = {"__pycache__", ".git", ".claude", "build", "dist",
             "node_modules", ".mypy_cache", ".ruff_cache"}


def _skip(p: Path) -> bool:
    parts = set(p.parts)
    return bool(SKIP_DIRS & parts) or any(
        part == ".venv" or part.startswith(".venv-") for part in parts
    )

# Longest-first so prefixes (tracer_transport) never shadow the longer name
# (tracer_transport_latlon); \b then rejects a prefix match on the longer one.
_ALT = "|".join(re.escape(m) for m in sorted(BUCKETS, key=len, reverse=True))
_ESC_PKG = re.escape(PKG)
# Rule 1: any dotted reference — imports AND lazy-table string literals.
RE_DOTTED = re.compile(rf"\b{_ESC_PKG}\.({_ALT})\b")
# Rule 2: `from <pkg> import <module>` (module-form; verified: no mixed lines).
RE_FROM = re.compile(rf"\bfrom {_ESC_PKG} import ({_ALT})\b")
# Rule 3: slash-path string literals to moved source files, anchored on the
# ``atmosphere/dynamics/<mod>.py`` suffix (any prefix: bare, ``src/legoesm/…``,
# ``packages/atmosphere/legoesm/…``). Ocean is ``ocean/dynamics/`` so never
# matches. Used by legoesm_source_path(), AST-guard readers, and baseline
# dicts. Idempotent: a bucketed path has ``<bucket>/`` after ``dynamics/`` so
# the module token no longer sits directly before ``.py``.
RE_PATH = re.compile(rf"atmosphere/dynamics/({_ALT})\.py")
# Verify-only: multi-line grouped `from <pkg> import ( ... mover ... )`. The
# auto-rewriter (RE_FROM) is single-line; a parenthesized group mixing movers
# and non-movers needs a hand edit, so the gate REPORTS it loudly rather than
# silently leaving a broken top-package attribute import. ponytail: report,
# not auto-fix — one occurrence repo-wide, a group-splitter isn't worth it.
RE_GROUP = re.compile(rf"from {_ESC_PKG} import \(([^)]*)\)", re.S)
_MOVERS = frozenset(BUCKETS)


def _rewrite(text: str) -> tuple[str, int]:
    """Return (new_text, num_substitutions)."""
    n = 0

    def d(m):
        nonlocal n
        n += 1
        mod = m.group(1)
        return f"{PKG}.{BUCKETS[mod]}.{mod}"

    def f(m):
        nonlocal n
        n += 1
        mod = m.group(1)
        return f"from {PKG}.{BUCKETS[mod]} import {mod}"

    def s(m):
        nonlocal n
        n += 1
        mod = m.group(1)
        return f"atmosphere/dynamics/{BUCKETS[mod]}/{mod}.py"

    text = RE_DOTTED.sub(d, text)
    text = RE_FROM.sub(f, text)
    text = RE_PATH.sub(s, text)
    return text, n


def _iter_py(repo: Path, self_path: Path):
    for p in repo.rglob("*.py"):
        if p == self_path:
            continue
        if _skip(p):
            continue
        yield p


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True)


def do_moves(repo: Path, apply: bool) -> None:
    dyn = repo / DYN_REL
    if not dyn.is_dir():
        sys.exit(f"dynamics dir not found: {dyn}")
    buckets = sorted(set(BUCKETS.values()))
    for b in buckets:
        bdir = dyn / b
        init = bdir / "__init__.py"
        if apply:
            bdir.mkdir(exist_ok=True)
            if not init.exists():
                init.write_text(f'"""{b} dycores (see docs/production_reorg.md)."""\n')
                _git(repo, "add", str(init.relative_to(repo)))
        else:
            print(f"  mkdir  {bdir.relative_to(repo)}/  + __init__.py")
    for mod, b in sorted(BUCKETS.items(), key=lambda kv: (kv[1], kv[0])):
        src = dyn / f"{mod}.py"
        dst = dyn / b / f"{mod}.py"
        if not src.exists():
            print(f"  !! missing (already moved?): {src.relative_to(repo)}")
            continue
        if apply:
            _git(repo, "mv", str(src.relative_to(repo)), str(dst.relative_to(repo)))
        else:
            print(f"  git mv {mod}.py -> {b}/{mod}.py")


def do_rewrite(repo: Path, apply: bool) -> None:
    self_path = Path(__file__).resolve()
    files_changed = 0
    edits = 0
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


def do_verify(repo: Path) -> int:
    """Gate: no stale flat paths remain. Exit code = number of stragglers."""
    self_path = Path(__file__).resolve()
    stale = 0
    for p in _iter_py(repo, self_path):
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for m in RE_DOTTED.finditer(text):
            print(f"  STALE dotted  {p.relative_to(repo)}: {m.group(0)}")
            stale += 1
        for m in RE_FROM.finditer(text):
            print(f"  STALE from    {p.relative_to(repo)}: {m.group(0)}")
            stale += 1
        for m in RE_PATH.finditer(text):
            print(f"  STALE path    {p.relative_to(repo)}: {m.group(0)}")
            stale += 1
        for m in RE_GROUP.finditer(text):
            names = {n.strip().split(" as ")[0].strip()
                     for n in m.group(1).replace("\n", " ").split(",")}
            for mod in sorted(names & _MOVERS):
                print(f"  STALE group   {p.relative_to(repo)}: "
                      f"from {PKG} import (... {mod} ...) -> hand-fix to "
                      f".{BUCKETS[mod]} submodule")
                stale += 1
    print(f"  {stale} straggler(s)")
    return stale


def self_test() -> None:
    cases = [
        # (input, expected)
        (f"import {PKG}.tracer_transport as tt",
         f"import {PKG}.shared.tracer_transport as tt"),
        (f"import {PKG}.tracer_transport_latlon",          # prefix not mis-hit
         f"import {PKG}.gcm.tracer_transport_latlon"),
        (f"from {PKG}.compressible_euler_plane import X",  # plane -> les
         f"from {PKG}.les.compressible_euler_plane import X"),
        (f"from {PKG} import spectral_les_plane as sl",    # form B module
         f"from {PKG}.les import spectral_les_plane as sl"),
        (f"from {PKG} import get_solver_class",            # __init__ attr: untouched
         f"from {PKG} import get_solver_class"),
        ("from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import X",  # ocean: untouched
         "from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import X"),
        (f'    "spectral_pe": ("{PKG}.spectral_pe", "SpectralPrimitiveEquationModel"),',  # lazy string
         f'    "spectral_pe": ("{PKG}.gcm.spectral_pe", "SpectralPrimitiveEquationModel"),'),
        (f"import {PKG}._fv3_lin_pgf",                     # leading-underscore module
         f"import {PKG}.gcm._fv3_lin_pgf"),
        (f"import {PKG}.gcm.spectral_pe",                  # idempotent: already bucketed
         f"import {PKG}.gcm.spectral_pe"),
        ('legoesm_source_path("atmosphere/dynamics/primitive_eq_cdgrid.py")',  # slash path -> bucket
         'legoesm_source_path("atmosphere/dynamics/gcm/primitive_eq_cdgrid.py")'),
        ('"packages/atmosphere/legoesm/atmosphere/dynamics/compressible_euler_plane.py"',  # pkg-prefixed
         '"packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py"'),
        ("src/legoesm/atmosphere/dynamics/plane_operators.py",  # src-prefixed docstring
         "src/legoesm/atmosphere/dynamics/les/plane_operators.py"),
        ("legoesm/ocean/dynamics/spectral_pe.py",          # ocean path: untouched (not atmosphere/)
         "legoesm/ocean/dynamics/spectral_pe.py"),
        ("atmosphere/dynamics/gcm/primitive_eq_cdgrid.py",  # idempotent: already bucketed path
         "atmosphere/dynamics/gcm/primitive_eq_cdgrid.py"),
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
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]),
                    help="repo root (default: two levels up from this script)")
    ap.add_argument("--apply", action="store_true", help="move files + edit imports")
    ap.add_argument("--verify", action="store_true", help="gate only: report stragglers")
    ap.add_argument("--self-test", action="store_true", help="regex unit checks, no repo")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    repo = Path(args.repo).resolve()
    if not (repo / DYN_REL).parent.exists():
        sys.exit(f"not a legoESM repo: {repo}")

    if args.verify:
        sys.exit(1 if do_verify(repo) else 0)

    mode = "APPLY" if args.apply else "DRY RUN (use --apply to execute)"
    print(f"== reorg dynamics/ [{mode}] ==")
    print(f"repo: {repo}")
    print(f"moving {len(BUCKETS)} modules into {sorted(set(BUCKETS.values()))}")
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
        print(f"  python -c \"import {PKG}.gcm, {PKG}.les, {PKG}.crm, "
              f"{PKG}.shared, {PKG}.neural\"")


if __name__ == "__main__":
    main()
