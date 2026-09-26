"""Shared machinery for the repo-wide source-audit ratchet tests.

Used by ``test_no_hardcoded_constants`` and ``test_no_saturation_reimpl`` (and
future C-series guards) so the file-discovery, anti-vacuity sanity check, numeric
constant-folding, inline-exemption parsing, and the per-``(file, value)``
fingerprint ratchet live in exactly one place (CLAUDE.md: no duplicate
utilities).

Design notes — these answer the adversarial-review findings on the first cut:
  * Discovery is filtered to roots *under the repo* and sanity-checked against
    sentinel files, so an empty/mis-resolved ``legoesm.__path__`` fails loudly
    instead of yielding a silently-vacuous (zero-case) scan.
  * The ratchet keys on ``(file, value, count)`` — swapping one banned literal
    for a *different* banned literal in the same file goes red even though the
    raw count is unchanged.
  * Budgets are exact for non-``permanent`` files, so a leftover inflated budget
    after a cleanup also goes red (forces ratchet-down).
  * ``fold_numeric`` catches simple obfuscations (``6371 * 1000``, ``270 + 3.15``)
    that exact-AST-Constant matching alone would miss.
  * A genuine ``SyntaxError`` in a scanned file is surfaced as a failure, not
    treated as "no violations".

This module is import-only (underscore-prefixed → pytest does not collect it as
a test).
"""

from __future__ import annotations

import ast
import io
import pathlib
import tokenize


def repo_root() -> pathlib.Path:
    """``<repo>`` — this file lives at ``<repo>/tests/_ratchet_audit.py``."""
    return pathlib.Path(__file__).resolve().parents[1]


def _legoesm_namespace_roots() -> list[pathlib.Path]:
    import legoesm

    return [pathlib.Path(p) for p in legoesm.__path__]


def production_roots() -> list[pathlib.Path]:
    """legoesm namespace roots that live *inside* this repo checkout.

    A site-packages (non-editable) install would put roots outside the repo;
    those are dropped so ``rel()`` never raises and we never audit a stale
    installed copy. If this drops everything, ``assert_discovery_sane`` fails.
    """
    repo = repo_root()
    roots: list[pathlib.Path] = []
    for r in _legoesm_namespace_roots():
        rr = r.resolve()
        try:
            rr.relative_to(repo)
        except ValueError:
            continue
        roots.append(rr)
    return roots


def _scan_roots() -> list[pathlib.Path]:
    repo = repo_root()
    return production_roots() + [repo / "tests", repo / "scripts"]


def discover_py_files() -> list[pathlib.Path]:
    """Every production-source, test, and script ``.py`` file to audit.

    De-duplicated by resolved path; ``__pycache__`` and throwaway
    ``scripts/tmp`` probes excluded.

    Exclusions are matched against the path **relative to the repo root**,
    never the absolute checkout path: matching on ``rp.parts`` (absolute)
    meant any checkout living under a directory literally named ``tmp`` (every
    worktree used this week did, e.g. ``/tmp/wt-...``) put the exact part
    ``"tmp"`` in every file's parts tuple, so ``"tmp" in parts and "scripts" in
    parts`` was true for *every* file under ``scripts/`` — silently excluding
    all of ``scripts/`` from every ratchet built on this discovery, in every
    such checkout. Relative-path matching makes only the literal top-level
    ``scripts/tmp/`` excluded, regardless of where the checkout lives.
    """
    repo = repo_root()
    seen: set[pathlib.Path] = set()
    out: list[pathlib.Path] = []
    for root in _scan_roots():
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            rp = path.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            rel_parts = rp.relative_to(repo).parts
            if "__pycache__" in rel_parts:
                continue
            if rel_parts[:2] == ("scripts", "tmp"):
                continue
            # Vendored 3rd-party backend (BSD-3 CLM-ML-JAX, not written to legoESM's
            # constant/coefficient/style conventions) — audited upstream, not here.
            # Scoped to the exact vendored path *relative to the repo root* (not
            # any dir merely named clm_ml_backend, and not matched against the
            # absolute checkout path).
            if "/".join(rel_parts).startswith(
                "packages/land/legoesm/land/canopy/clm_ml_backend/"
            ):
                continue
            out.append(rp)
    return out


def rel(path: pathlib.Path) -> str:
    return path.resolve().relative_to(repo_root()).as_posix()


# One sentinel production file per federation package. If any is absent the
# corresponding namespace root is missing and the scan would be partially
# vacuous (sentinels covering only core/ocean could pass while atmosphere/land/…
# silently dropped — round-2 finding).
_SENTINELS = (
    "packages/core/legoesm/constants.py",
    "packages/core/legoesm/thermo.py",
    "packages/atmosphere/legoesm/atmosphere/physics/_shared.py",
    "packages/ocean/legoesm/ocean/eos.py",
    "packages/land/legoesm/land/slab_land.py",
    "packages/ice/legoesm/ice/sea_ice.py",
    "packages/coupler/legoesm/coupler/coupler.py",
    "packages/ml/legoesm/ml/loss.py",
    "packages/tools/legoesm/diagnostics/column_integrals.py",
)
# A misresolved namespace typically yields a tiny scan; a healthy tree is ~1800.
_MIN_FILES = 500


def assert_discovery_sane(files: list[pathlib.Path]) -> None:
    """Fail loudly if file discovery is empty/mis-resolved (anti-vacuity)."""
    assert len(files) >= _MIN_FILES, (
        f"only {len(files)} .py files discovered (expected >= {_MIN_FILES}); "
        f"legoesm.__path__ likely mis-resolved. production_roots="
        f"{[str(r) for r in production_roots()]}"
    )
    rels = {rel(f) for f in files}
    missing = [s for s in _SENTINELS if s not in rels]
    assert not missing, (
        f"scan is missing sentinel production files {missing} — a legoesm "
        f"namespace root resolved wrong / is absent, audit would be vacuous. "
        f"production_roots={[str(r) for r in production_roots()]}"
    )


def canonical_source_path(rel_path: str) -> pathlib.Path:
    """Resolve a repo-relative path to its absolute form (for exact whitelist
    matching — avoids basename-based exemptions that would exempt an unrelated
    file of the same name elsewhere in the tree)."""
    return (repo_root() / rel_path).resolve()


def fold_numeric(node: ast.AST) -> float | None:
    """Fold a pure-numeric expression (``+ - * /`` and unary ``+/-``) to a float,
    or ``None`` if any leaf is non-numeric. Catches ``6371 * 1000`` /
    ``270 + 3.15`` style obfuscation. ``bool`` is excluded (it is an ``int``
    subclass but never a physical-constant literal).

    A pathologically deep expression (~1000 nested terms) would exceed Python's
    recursion limit; we degrade to ``None`` (un-foldable) rather than crashing the
    audit — such a node still has its banned *leaf* literals caught by the
    iterative ``iter_banned_numeric`` descent."""
    try:
        return _fold_numeric_impl(node)
    except RecursionError:
        return None


def _fold_numeric_impl(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            return None
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _fold_numeric_impl(node.operand)
        if v is None:
            return None
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ):
        left = _fold_numeric_impl(node.left)
        right = _fold_numeric_impl(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            return None
        return left / right
    return None


def iter_banned_numeric(
    tree: ast.AST, banned: frozenset[float]
) -> list[tuple[ast.AST, float]]:
    """Every *maximal* pure-numeric (sub)expression whose folded value is banned.

    Top-down: when a node folds to a banned value it is recorded and its children
    are NOT visited, so a nested expression (``(270 + 3.15) + 0``) is counted once
    at the outermost banned level instead of double-counting inner folds
    (round-2 finding). Non-banned numeric expressions (``283.0 - 273.15`` →
    9.85) are still descended into, so a banned sub-literal is caught."""
    out: list[tuple[ast.AST, float]] = []
    # Iterative (explicit stack) so a deeply-nested AST cannot blow the Python
    # recursion limit; ``fold_numeric`` is itself recursion-guarded.
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        folded = fold_numeric(node)
        if folded is not None and folded in banned:
            out.append((node, folded))
            continue  # maximal: do not descend into a banned-folding expression
        stack.extend(ast.iter_child_nodes(node))
    return out


def fold_set(node: ast.AST) -> set[float]:
    """Folded values of every numeric (sub)expression reachable from ``node``.

    Lets a coefficient written as ``240 + 3.5`` be recognised as ``243.5`` —
    closes the aliased/folded-coefficient gap in structural detectors. (A pure
    variable alias ``B = 243.5; ... + B`` is still out of scope — would need
    dataflow.)"""
    vals: set[float] = set()
    for d in ast.walk(node):
        f = fold_numeric(d)
        if f is not None:
            vals.add(f)
    return vals


def normalized_line(src_lines: list[str], lineno: int) -> str:
    """The hit's source line with whitespace collapsed — the site fingerprint.
    Robust to reindentation/line-number shifts; sensitive to the surrounding code
    so two different sites with the same literal get distinct fingerprints."""
    if 1 <= lineno <= len(src_lines):
        return " ".join(src_lines[lineno - 1].split())
    return f"<line {lineno}>"


def comment_tagged_lines(src: str, tag: str) -> set[int]:
    """Line numbers carrying a real ``# <tag>-ok`` *comment* (tokenised, so a
    ``"# const-ok"`` substring inside a string literal does NOT exempt the line —
    round-2 LOW finding)."""
    out: set[int] = set()
    needle = f"{tag}-ok"
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT and needle in tok.string:
                out.add(tok.start[0])
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Best-effort: a tokeniser hiccup must not silently exempt lines.
        return out
    return out


def fingerprint_subset_errors(
    observed: dict[str, int],
    allowed: dict[str, int],
    *,
    permanent: bool,
) -> list[str]:
    """Ratchet ``observed`` site-fingerprints (normalised source line → count)
    against the recorded ``allowed`` budget for one file.

    Keying on the *source-line context* (not just the bare value/count) means a
    same-value relocation — delete one sanctioned ``283.0 - 273.15`` and add a
    new bad ``T - 273.15`` elsewhere — produces a *different* fingerprint with no
    allowance and goes red (round-2 HIGH finding). Removals shrink ``observed``
    and stay a subset, so a cleanup needs no budget edit. For non-``permanent``
    files a leftover allowance with no matching site is flagged (ratchet down)."""
    errors: list[str] = []
    for fp, count in sorted(observed.items()):
        cap = allowed.get(fp, 0)
        if count > cap:
            errors.append(f"new/extra site (x{count}, budget {cap}): {fp}")
    if not permanent:
        for fp, cap in sorted(allowed.items()):
            if observed.get(fp, 0) < cap:
                errors.append(f"stale budget (allowed {cap}, gone): {fp} — remove it")
    return errors
