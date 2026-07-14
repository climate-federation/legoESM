"""Ratchet: no cross-module import of private (``_``-prefixed) symbols in
production code.

CLAUDE.md: *"No import of private (``_``-prefixed) symbols across modules.
Promote (drop underscore + ``__init__.py`` re-export) or factor public
wrapper."*  A private name is an implementation detail: importing it from
another module couples the importer to internals that may be renamed/removed
without notice, defeats the public-API surface audits, and (historically) hid
mutable-singleton reads (``grids.halo._halo_backend``) that have public
accessors (``get_halo_backend``).  The 2026-06 sweep promoted ~95 symbols
(``_is_distributed`` → ``is_distributed``, ``_get_sendrecv_vjp`` →
``get_sendrecv_vjp``, ``_TPU_XLA_FLAGS`` → ``TPU_XLA_FLAGS``, Wright-EOS
coefficients, FV3 cdgrid operators, halo helpers, …) and converted the
mutable-global reads to accessor calls.

Mechanism:
  * AST scan of every production ``.py`` under the legoesm namespace roots
    (``src/legoesm`` + ``packages/*/legoesm`` — via
    ``_ratchet_audit.production_roots()``) for ``from legoesm[.X] import _y``
    where the *imported name* ``_y`` is underscore-prefixed.  **Function-scope
    deferred imports count too** (``ast.walk`` visits nested scopes).
  * Dunder names (``__version__``) are exempt: they are conventional public
    metadata, and the CLAUDE.md audit grep (``\\b_[a-z]``) never matched them.
  * Private names in the *module path* are NOT flagged (the blessed
    ``atmosphere.physics._shared`` / ``lateral_mixing._gm_redi_common``
    shared-module pattern): only the imported symbol's own name matters.
  * ``import public_name as _alias`` is NOT flagged (the alias is local).
  * Relative imports (``from ._sibling import _x``) are out of scope here —
    this ratchet mirrors the CLAUDE.md absolute-import audit grep.  Tests and
    scripts are also out of scope (white-box tests legitimately reach into
    internals).

Shrink-only allowlist: every entry must still be a real violation (a stale
entry fails the test, forcing the list to shrink as fixes land) and no
violation may exist outside the list (a new private cross-import fails).

A tripwire, not a proof — self-tests below prove the scanner is non-vacuous
(synthetic module/function-scope violations are flagged; public-alias, dunder
and module-path-private forms are not).
"""

from __future__ import annotations

import ast
import pathlib

from tests import _ratchet_audit as ra

# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


def private_legoesm_imports(src: str, filename: str = "<synthetic>"):
    """Return ``[(lineno, module, name), ...]`` for every absolute
    ``from legoesm[.X] import _name`` in *src* whose imported name is
    underscore-prefixed (dunders exempt).  Function-scope imports included.

    Raises ``SyntaxError`` for unparseable source (surfaced as a test failure,
    never treated as "no violations").
    """
    tree = ast.parse(src, filename=filename)
    hits: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level != 0:
            continue
        mod = node.module or ""
        if mod != "legoesm" and not mod.startswith("legoesm."):
            continue
        for alias in node.names:
            name = alias.name
            if not name.startswith("_"):
                continue
            if name.startswith("__") and name.endswith("__"):
                continue  # dunder metadata (__version__)
            hits.append((node.lineno, mod, name))
    return hits


def discover_violations() -> dict[tuple[str, str, str], int]:
    """``{(rel_file, module, name): count}`` over all production roots."""
    files: list[pathlib.Path] = []
    for root in ra.production_roots():
        files.extend(
            p for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts
        )
    assert len(files) >= 400, (
        f"only {len(files)} production .py files discovered — legoesm "
        f"namespace roots mis-resolved; scan would be vacuous. roots="
        f"{[str(r) for r in ra.production_roots()]}"
    )
    rels = {ra.rel(f) for f in files}
    sentinels = (
        # post src/->packages PEP-420 migration: the legacy ``src/legoesm/cli.py``
        # root is gone; pin a real CLI module under the ml namespace root instead.
        "packages/ml/legoesm/ml/s2s/sfno_slab/cli.py",
        "packages/core/legoesm/grids/halo.py",
        "packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_cdgrid.py",
        "packages/ocean/legoesm/ocean/eos.py",
        "packages/coupler/legoesm/driver/physics_pipeline.py",
        "packages/ml/legoesm/training/era5_to_state.py",
        "packages/ice/legoesm/ice/rheology.py",
    )
    missing = [s for s in sentinels if s not in rels]
    assert not missing, (
        f"scan is missing sentinel files {missing} — a namespace root is "
        f"absent; this ratchet would be partially vacuous."
    )
    out: dict[tuple[str, str, str], int] = {}
    for f in files:
        for _lineno, mod, name in private_legoesm_imports(
            f.read_text(), filename=str(f)
        ):
            key = (ra.rel(f), mod, name)
            out[key] = out.get(key, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Allowlist — SHRINK-ONLY.  Never add an entry; remove as fixes land.
#
# EMPTY since 2026-06-10: the last 11 entries (mass_flux kernel/geometry
# helpers imported by the five profile-prognostic convection schemes, and
# physics_pipeline's _get_gwd_fn import) were fixed by promoting
# ``apply_mass_flux_kernel`` / ``compute_column_geometry`` / ``get_gwd_fn``.
# ---------------------------------------------------------------------------

ALLOWLIST: dict[tuple[str, str, str], int] = {}


# ---------------------------------------------------------------------------
# The ratchet
# ---------------------------------------------------------------------------


def test_no_private_cross_module_imports():
    violations = discover_violations()

    new = {k: v for k, v in violations.items() if k not in ALLOWLIST}
    assert not new, (
        "New cross-module import(s) of private (_-prefixed) legoesm symbols "
        "(CLAUDE.md: promote the symbol — drop the underscore at its "
        "definition and update importers — or factor a public wrapper; "
        "never add to this allowlist):\n"
        + "\n".join(
            f"  {f}: from {m} import {n} (x{c})" for (f, m, n), c in sorted(new.items())
        )
    )

    stale = {k: v for k, v in ALLOWLIST.items() if k not in violations}
    assert not stale, (
        "Stale allowlist entries (the underlying import was fixed — DELETE "
        "them so the allowlist only ever shrinks):\n"
        + "\n".join(f"  {f}: from {m} import {n}" for (f, m, n) in sorted(stale))
    )

    drifted = {
        k: (ALLOWLIST[k], violations[k])
        for k in ALLOWLIST
        if k in violations and violations[k] != ALLOWLIST[k]
    }
    assert not drifted, (
        f"Allowlisted private-import counts drifted (no new occurrences of "
        f"an allowlisted import may be added): {drifted}"
    )


# ---------------------------------------------------------------------------
# Anti-vacuity self-tests: prove the scanner flags synthetic violations and
# ignores the sanctioned forms.
# ---------------------------------------------------------------------------


def test_scanner_flags_module_scope_private_import():
    hits = private_legoesm_imports(
        "from legoesm.core.operators import _is_distributed\n"
    )
    assert hits == [(1, "legoesm.core.operators", "_is_distributed")]


def test_scanner_flags_function_scope_and_multi_alias():
    src = (
        "def f():\n"
        "    from legoesm.grids.halo import pad_halo, _mpi_topology\n"
        "    return _mpi_topology\n"
        "class C:\n"
        "    def m(self):\n"
        "        from legoesm.parallel.mesh import _N_FACES as nf\n"
        "        return nf\n"
    )
    hits = private_legoesm_imports(src)
    assert (2, "legoesm.grids.halo", "_mpi_topology") in hits
    # `as nf` does not launder the private name:
    assert (6, "legoesm.parallel.mesh", "_N_FACES") in hits
    assert len(hits) == 2  # the public pad_halo is not flagged


def test_scanner_ignores_sanctioned_forms():
    src = (
        # public name privately aliased — local convention, allowed
        "from legoesm.core.flux_limiters import van_leer_limiter as _vl\n"
        # dunder metadata — allowed
        "from legoesm._version import __version__\n"
        # private module in the path, public symbol — the _shared pattern
        "from legoesm.atmosphere.physics._shared import virtual_temperature\n"
        # relative import — out of scope for this ratchet
        "from ._sibling import _helper\n"
        # non-legoesm package — out of scope
        "from numpy.core import _exceptions\n"
    )
    assert private_legoesm_imports(src) == []


def test_scanner_surfaces_syntax_errors():
    import pytest

    with pytest.raises(SyntaxError):
        private_legoesm_imports("from legoesm import (\n")


def test_allowlist_is_empty():
    """The allowlist shrank to zero on 2026-06-10 (mass_flux helpers +
    get_gwd_fn promoted).  It is shrink-only — it must stay empty."""
    assert ALLOWLIST == {}
