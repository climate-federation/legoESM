"""Assemble a GLM authoring prompt for one module of the FV3 JAX lane.

The lane is authored module by module by GLM-5.3, reviewed by Claude and
codex (user directive 2026-08-14).  Every prompt so far was assembled by
hand, which is how the module-2 prompt ended up carrying a callee's
docstring for a routine it never calls.  This builds one mechanically
from four things that already exist in the tree:

  * the NumPy SPECIFICATION module (or a named subset of its routines),
  * a sibling JAX module whose head is the STYLE and CONVENTIONS to
    match,
  * the exact signature + docstring of every JAX callee the new module
    is allowed to call, extracted from the installed source rather than
    remembered (the repo's "never infer an API" rule applies to the
    prompt too -- a wrong signature in the prompt becomes a wrong call
    in the output),
  * a rules block.

Splitting matters as much as content: a whole-module call died three
times with IncompleteRead or an all-reasoning completion (jobs 9417290,
9417297, 9417348), while ~400-line parts go through.  ``--only`` selects
the spec routines for one part.

usage, from the repo root::

    python scripts/experiment/fv3_build_glm_authoring_prompt.py \
        --spec packages/core/legoesm/core/fv3_native_acoustic_3d.py \
        --sibling packages/core/legoesm/core/fv3_dsw_phase_3d.py \
        --target packages/core/legoesm/core/fv3_acoustic_3d.py \
        --callee legoesm.core.fv3_dsw_tail_3d:dsw_tail_phase_3d \
        --callee legoesm.core.fv3_cgrid_phase_3d:csw_phase_3d \
        --rules docs/atmosphere/glm_authoring_rules.txt \
        --only acoustic_substep_3d --only acoustic_loop_3d \
        --out /burg-archive/glab/users/pg2328/fv3_duo_gaps/glm/prompt_x.txt
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import sys

DOC_CHARS = 1800          # per callee; enough for a full contract, not a book


def _module_path(repo: pathlib.Path, dotted: str) -> pathlib.Path:
    """``legoesm.core.fv3_pgrad`` -> the file, searched under packages/."""
    rel = pathlib.Path(*dotted.split(".")).with_suffix(".py")
    for base in sorted((repo / "packages").glob("*")):
        cand = base / rel
        if cand.is_file():
            return cand
    cand = repo / "src" / rel
    if cand.is_file():
        return cand
    raise SystemExit(f"cannot locate module {dotted} under {repo}/packages")


def _extract(path: pathlib.Path, names: list[str]) -> str:
    """Signature + docstring of each named top-level function."""
    src = path.read_text()
    lines = src.splitlines()
    tree = ast.parse(src)
    found, out = set(), []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found.add(node.name)
            sig = "\n".join(lines[node.lineno - 1:node.body[0].lineno - 1])
            doc = (ast.get_docstring(node) or "")[:DOC_CHARS]
            out.append(f"--- {path}::{node.name} ---\n{sig}\n"
                       f'"""\n{doc}\n"""\n')
    missing = [n for n in names if n not in found]
    if missing:
        raise SystemExit(f"{path}: no top-level def for {missing}")
    return "\n".join(out)


def _whole(path: pathlib.Path, only: list[str]) -> str:
    """The spec, whole or restricted to `only` (header always kept)."""
    src = path.read_text()
    if not only:
        return src
    lines = src.splitlines()
    tree = ast.parse(src)
    first_def = min((n.lineno for n in tree.body
                     if isinstance(n, (ast.FunctionDef, ast.ClassDef))),
                    default=len(lines) + 1)
    parts = ["\n".join(lines[:first_def - 1])]
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in only:
            found.add(node.name)
            parts.append("\n".join(lines[node.lineno - 1:node.end_lineno]))
    missing = [n for n in only if n not in found]
    if missing:
        raise SystemExit(f"{path}: --only named {missing}, not found")
    return "\n\n".join(parts)


def _head(path: pathlib.Path) -> str:
    """A sibling module's docstring -- the style and conventions block."""
    tree = ast.parse(path.read_text())
    doc = ast.get_docstring(tree)
    if not doc:
        raise SystemExit(f"{path} has no module docstring to imitate")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, help="NumPy specification module")
    ap.add_argument("--sibling", required=True,
                    help="JAX module whose docstring sets the conventions")
    ap.add_argument("--target", required=True, help="file to be written")
    ap.add_argument("--callee", action="append", default=[],
                    metavar="dotted.module:symbol[,symbol...]")
    ap.add_argument("--only", action="append", default=[],
                    help="restrict the spec to these routines (repeatable)")
    ap.add_argument("--rules", help="file holding the RULES block")
    ap.add_argument("--extra", help="file holding a part-specific preamble")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    repo = pathlib.Path(__file__).resolve().parents[2]
    spec = pathlib.Path(args.spec)
    sibling = pathlib.Path(args.sibling)
    for p in (spec, sibling):
        if not p.is_file():
            raise SystemExit(f"missing {p}")

    blocks = [f"Write ONE Python file. Output ONLY source code. "
              f"No prose. No markdown fences.\n\nFILE: {args.target}\n"
              f"It is the JAX twin of the NumPy specification at the bottom. "
              f"Mirror it routine for routine, plus a make_<name>_jit() per "
              f"public routine.\n\nTHE NUMPY MODULE IS THE SPECIFICATION. "
              f"Any behavioural difference is a port bug."]
    for opt in (args.rules, args.extra):
        if opt:
            blocks.append(pathlib.Path(opt).read_text())

    api = []
    for entry in args.callee:
        dotted, _, syms = entry.partition(":")
        names = [s for s in syms.split(",") if s]
        if not names:
            raise SystemExit(f"--callee {entry!r} names no symbol")
        api.append(_extract(_module_path(repo, dotted), names))
    if api:
        blocks.append("===== CALLEE JAX API -- signatures and docstrings, "
                      "READ THEM, DO NOT INFER =====\n" + "\n".join(api))

    blocks.append(f"===== SIBLING MODULE ({sibling.name}) -- MATCH THIS "
                  f"STYLE AND THESE CONVENTIONS =====\n{_head(sibling)}")
    blocks.append(f"===== SPECIFICATION ({spec.name}) -- WHAT YOU ARE "
                  f"PORTING =====\n{_whole(spec, args.only)}")

    out = pathlib.Path(args.out)
    out.write_text("\n\n".join(blocks))
    print(f"{out}: {len(out.read_text())} chars, "
          f"{len(args.callee)} callee blocks, "
          f"{'whole spec' if not args.only else ', '.join(args.only)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
