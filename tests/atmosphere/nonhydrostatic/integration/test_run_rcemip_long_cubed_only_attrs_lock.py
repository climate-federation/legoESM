"""AST-locked check that ``_CUBED_ONLY_ATTRS`` matches the actual
cubed-sphere-only argparse registrations.

iter-294 (Codex iter-293 round-4 residual note): the iter-291
cubed-sphere-only misuse detection relies on
``_CUBED_ONLY_ATTRS`` being kept in sync with the CLI flags
that semantically apply only to ``--grid cubed_sphere``. If a
future PR adds a new ``--cubed-*`` or surface-flux ``--sfc-*``
flag and forgets to extend ``_CUBED_ONLY_ATTRS``, the silent-
ignore bug returns (passing the flag with --grid mpas does
nothing, no error).

This AST-based test parses scripts/run_rcemip_long.py and asserts
the ``_CUBED_ONLY_ATTRS`` literal exactly matches the set of
parser-flag attribute names that begin with the cubed-sphere or
surface-flux prefixes. A new flag triggers the lock.
"""
from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
DRIVER = REPO_ROOT / "scripts" / "run_rcemip_long.py"


# Prefixes that conceptually mean "cubed-sphere-only" in this
# driver. n-cubed-sphere is the grid-resolution knob; sfc-* are
# the bulk surface-flux knobs (cubed-sphere is the only grid
# wired for them today); cubed-* are dycore-tuning knobs.
_CUBED_PREFIXES = ("n_cubed_sphere", "sfc_", "cubed_")


def _extract_cubed_only_attrs_tuple(tree: ast.Module) -> tuple[str, ...]:
    """Return the literal value of the module-level
    ``_CUBED_ONLY_ATTRS`` assignment."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "_CUBED_ONLY_ATTRS":
                    if not isinstance(node.value, ast.Tuple):
                        raise AssertionError(
                            "_CUBED_ONLY_ATTRS RHS is not a tuple "
                            f"literal: {ast.dump(node.value)}"
                        )
                    return tuple(
                        elt.value for elt in node.value.elts
                        if isinstance(elt, ast.Constant)
                    )
    raise AssertionError(
        "module-level _CUBED_ONLY_ATTRS tuple not found in "
        f"{DRIVER}"
    )


def _extract_add_argument_dest_names(tree: ast.Module) -> set[str]:
    """Walk every ``p.add_argument(...)`` call and return the
    argparse-derived ``dest`` attr name. argparse derives ``dest``
    from the FIRST long-form (``--foo``) option string, regardless
    of position, so we scan ALL string positional args.

    iter-295 (Codex iter-294 round-1 MEDIUM): the iter-294 version
    inspected only ``node.args[0]``, missing patterns like
    ``add_argument('-c', '--cubed-hyperdiff', ...)`` where the
    long form is the second arg. That would silently bypass the
    AST lock for short-form-prefixed cubed-only flags.
    """
    dests: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match p.add_argument(...) or parser.add_argument(...).
        func = node.func
        if not (isinstance(func, ast.Attribute)
                and func.attr == "add_argument"):
            continue
        if not node.args:
            continue
        # Scan ALL positional string args for the first one
        # starting with ``--`` (argparse picks that for dest).
        for arg in node.args:
            if (isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and arg.value.startswith("--")):
                dest = arg.value[2:].replace("-", "_")
                dests.add(dest)
                break  # argparse uses the first --long-form
    return dests


def test_extract_dest_handles_short_form_prefix():
    """iter-296 (Codex iter-295 round-2 LOW): explicit lock for
    the iter-295 short+long-form extractor branch. The real
    driver doesn't currently use ``p.add_argument('-c', '--cubed-X',
    ...)`` so the live AST test exercises only the long-form-first
    path; this synthetic unit test forces the short-form branch
    to run + verifies the dest is derived from the long form.

    Without this, the iter-295 fix is correct but silently
    untested until the driver adds such a call.
    """
    snippet = "p.add_argument('-c', '--cubed-hyperdiff', type=float, default=1.0)"
    tree = ast.parse(snippet)
    dests = _extract_add_argument_dest_names(tree)
    assert dests == {"cubed_hyperdiff"}, (
        f"Expected dest 'cubed_hyperdiff' from short+long-form call; "
        f"got {dests}."
    )


def test_extract_dest_ignores_short_only_call():
    """iter-296: a call with ONLY a short-form ``-c`` and no
    ``--long`` form should yield NO dest from the extractor.
    argparse would derive ``dest='c'`` in that case but our lock
    only cares about long-form (``--``) flags."""
    snippet = "p.add_argument('-c', type=float, default=1.0)"
    tree = ast.parse(snippet)
    dests = _extract_add_argument_dest_names(tree)
    assert dests == set(), (
        f"short-form-only call should yield no extracted dest; "
        f"got {dests}."
    )


def test_extract_dest_handles_two_long_forms():
    """iter-296: when two ``--`` forms are passed (alias pattern),
    argparse uses the FIRST one for dest. iter-295 added ``break``
    after the first match to ensure this."""
    snippet = (
        "p.add_argument('--cubed-X', '--cubed-X-alias', "
        "type=float, default=1.0)"
    )
    tree = ast.parse(snippet)
    dests = _extract_add_argument_dest_names(tree)
    assert dests == {"cubed_X"}, (
        f"First --long-form should win; got {dests}."
    )


def test_cubed_only_attrs_matches_add_argument_registrations():
    """Lock that ``_CUBED_ONLY_ATTRS`` includes every parser dest
    starting with ``n_cubed_sphere``, ``sfc_``, or ``cubed_``. A
    future ``--cubed-hyperdiff`` (or similar) added without
    extending the tuple fires this test instead of slipping past
    the iter-291 misuse detection.
    """
    tree = ast.parse(DRIVER.read_text())
    cubed_only_attrs = set(_extract_cubed_only_attrs_tuple(tree))
    all_dests = _extract_add_argument_dest_names(tree)
    cubed_prefix_dests = {
        d for d in all_dests
        if any(d.startswith(p) or d == p.rstrip("_")
               for p in _CUBED_PREFIXES)
    }
    missing = cubed_prefix_dests - cubed_only_attrs
    extra = cubed_only_attrs - cubed_prefix_dests
    assert not missing, (
        f"_CUBED_ONLY_ATTRS missing dest(s): {sorted(missing)}. "
        f"Either add to the tuple or rename the CLI flag to a "
        f"non-cubed/sfc prefix."
    )
    assert not extra, (
        f"_CUBED_ONLY_ATTRS contains attrs that don't correspond "
        f"to any p.add_argument() registration: {sorted(extra)}. "
        f"Stale tuple entries (CLI flag was renamed/removed?)."
    )
